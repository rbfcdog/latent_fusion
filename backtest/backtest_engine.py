import numpy as np
import pandas as pd


def _build_price_frame(data_dict, price_col="Close", date_col="Date", start=None, end=None, fill_method="ffill"):
    series_list = []
    for symbol, df in data_dict.items():
        if price_col not in df.columns:
            raise ValueError(f"Missing price column '{price_col}' for symbol '{symbol}'")
        if date_col in df.columns:
            dates = pd.to_datetime(df[date_col], errors="coerce")
            s = pd.Series(df[price_col].values, index=dates, name=symbol)
        else:
            dates = pd.to_datetime(df.index, errors="coerce")
            s = pd.Series(df[price_col].values, index=dates, name=symbol)
        s = s.dropna()
        series_list.append(s)
    if not series_list:
        raise ValueError("No valid price series found")
    price_df = pd.concat(series_list, axis=1).sort_index()
    if start is not None:
        price_df = price_df.loc[pd.to_datetime(start):]
    if end is not None:
        price_df = price_df.loc[:pd.to_datetime(end)]
    if fill_method == "ffill":
        price_df = price_df.ffill()
    elif fill_method == "bfill":
        price_df = price_df.bfill()
    elif fill_method == "ffill_bfill":
        price_df = price_df.ffill().bfill()
    elif fill_method is None or fill_method == "none":
        pass
    else:
        raise ValueError("fill_method must be 'ffill', 'bfill', 'ffill_bfill', or None")
    return price_df


def _to_weights(obj, symbols):
    if isinstance(obj, dict):
        w = np.array([obj.get(sym, 0.0) for sym in symbols], dtype=float)
        return w
    if isinstance(obj, pd.Series):
        return obj.reindex(symbols).fillna(0.0).values.astype(float)
    arr = np.asarray(obj, dtype=float)
    if arr.ndim != 1 or arr.shape[0] != len(symbols):
        raise ValueError("Weights must be a 1D array with length equal to number of symbols")
    return arr


def _normalize_weights(w, max_gross=1.0, allow_short=True):
    w = np.where(np.isfinite(w), w, 0.0).astype(float)
    if not allow_short:
        w[w < 0] = 0.0
        gross = w.sum()
        if gross > max_gross and gross > 0:
            w = w * (max_gross / gross)
        return w
    gross = np.sum(np.abs(w))
    if gross > max_gross and gross > 0:
        w = w * (max_gross / gross)
    return w


def _call_model(model, state):
    if hasattr(model, "allocate") and callable(model.allocate):
        return model.allocate(state)
    if callable(model):
        return model(state)
    raise TypeError("Model must be callable or implement allocate(state)")


class BacktestEngine:
    def __init__(
        self,
        data_dict,
        price_col="Close",
        date_col="Date",
        start=None,
        end=None,
        fill_method="ffill",
        trading_cost_bps=0.0,
        max_gross=1.0,
        allow_short=True,
        initial_cash=1.0,
    ):
        self.price_df = _build_price_frame(
            data_dict,
            price_col=price_col,
            date_col=date_col,
            start=start,
            end=end,
            fill_method=fill_method,
        )
        self.symbols = list(self.price_df.columns)
        self.trading_cost_bps = float(trading_cost_bps)
        self.max_gross = float(max_gross)
        self.allow_short = bool(allow_short)
        self.initial_cash = float(initial_cash)

    def run(self, model):
        prices = self.price_df
        idx = prices.index
        n = len(idx)
        if n < 2:
            raise ValueError("Not enough data to run backtest")
        weights = np.zeros(len(self.symbols), dtype=float)
        value = self.initial_cash
        records = [
            {
                "date": idx[0],
                "value": value,
                "return": 0.0,
                "gross": 0.0,
                "net": 0.0,
                "turnover": 0.0,
                "cost": 0.0,
            }
        ]
        weight_records = [
            {"date": idx[0], **{sym: 0.0 for sym in self.symbols}}
        ]
        returns_hist = prices.pct_change().fillna(0.0)
        for t in range(n - 1):
            state = {
                "t": t,
                "date": idx[t],
                "symbols": self.symbols,
                "prices": prices.iloc[: t + 1],
                "returns": returns_hist.iloc[: t + 1],
                "current_prices": prices.iloc[t],
                "current_returns": returns_hist.iloc[t],
                "weights": pd.Series(weights, index=self.symbols),
                "value": value,
            }
            target = _call_model(model, state)
            target_w = _to_weights(target, self.symbols)
            target_w = _normalize_weights(
                target_w,
                max_gross=self.max_gross,
                allow_short=self.allow_short,
            )
            turnover = np.sum(np.abs(target_w - weights))
            cost = value * (self.trading_cost_bps * 1e-4) * turnover
            value_after_cost = value - cost
            price_t = prices.iloc[t].values
            price_next = prices.iloc[t + 1].values
            valid = np.isfinite(price_t) & np.isfinite(price_next)
            ret_next = np.zeros_like(price_t, dtype=float)
            ret_next[valid] = price_next[valid] / price_t[valid] - 1.0
            portfolio_return = np.nansum(target_w * ret_next)
            value_next = value_after_cost * (1.0 + portfolio_return)
            period_return = value_next / value - 1.0
            gross = float(np.sum(np.abs(target_w)))
            net = float(np.sum(target_w))
            records.append(
                {
                    "date": idx[t + 1],
                    "value": value_next,
                    "return": period_return,
                    "gross": gross,
                    "net": net,
                    "turnover": turnover,
                    "cost": cost,
                }
            )
            weight_records.append(
                {"date": idx[t], **{sym: target_w[i] for i, sym in enumerate(self.symbols)}}
            )
            weights = target_w
            value = value_next
        performance = pd.DataFrame(records).set_index("date")
        weights_df = pd.DataFrame(weight_records).set_index("date")
        return {"performance": performance, "weights": weights_df}


class EqualWeightModel:
    def __init__(self, allow_short=False):
        self.allow_short = allow_short

    def allocate(self, state):
        n = len(state["symbols"])
        if n == 0:
            return np.array([], dtype=float)
        w = np.ones(n, dtype=float) / n
        if not self.allow_short:
            return w
        return w
