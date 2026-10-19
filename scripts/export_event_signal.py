"""Exporta painel de preços e eventos sem treinar ou executar modelo."""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))

from src.fusion.event_data import SOURCE_NOTE, export_events, load_event_panel, save_event_panel


def main():
    panel = load_event_panel(cache_dir=_root / "cache", ts_dir=_root / "data/time_series")
    events_path = _root / "data/fusion_events.jsonl.gz"
    panel_path = _root / "data/fusion_event_panel.npz"
    available = {ticker: array["dates"] for ticker, array in zip(panel["tickers"], panel["arrays"])}
    coverage = export_events(_root / "data/text", panel["trading_dates"], events_path, available)
    save_event_panel(panel, panel_path, coverage)
    print(f"{panel_path}: {len(panel['tickers'])} ativos, "
          + ", ".join(f"{s}={len(rows)}" for s, rows in panel["splits"].items()))
    print(f"{events_path}: {int(coverage['counts'].sum())} eventos com manchetes verificadas")
    print("Mês da decisão | brutos no mês de publicação | exportados | ativos com eventos")
    for j, month in enumerate(coverage["months"]):
        print(f"{month} | {coverage['raw_counts'][j]} | {coverage['counts'][j]} | "
              f"{(coverage['ticker_month_counts'][:, j] > 0).sum()}")
    print("Cobertura por ativo e mês da decisão (inclui meses sem eventos):")
    print("ativo | " + " | ".join(coverage["months"]))
    for ticker, counts in zip(panel["tickers"], coverage["ticker_month_counts"]):
        print(f"{ticker} | " + " | ".join(str(value) for value in counts))
    print(SOURCE_NOTE)


if __name__ == "__main__":
    main()
