import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.fusion.backtest_metrics import portfolio_metrics
from src.fusion.llama_advanced_agent import weekly_decision_dates
from src.fusion.portfolio_execution import execute_portfolio


class AttentionColabTest(unittest.TestCase):
    def test_each_seed_uses_its_own_predictions_without_global_scores(self):
        notebook = Path(__file__).resolve().parents[1] / "notebooks/05_fusion/attention_vs_agentic_2024_2026.ipynb"
        cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
        source = next("".join(cell["source"]) for cell in cells
                      if cell["cell_type"] == "code" and "".join(cell["source"]).startswith("def curve("))
        sessions = pd.bdate_range("2025-01-02", periods=60)
        tickers = list("ABCDEFGH")
        prices = pd.DataFrame(100 * (1 + np.arange(1, 9)[None, :] / 1000) ** np.arange(60)[:, None],
                              index=sessions, columns=tickers)
        frame = pd.MultiIndex.from_product([sessions[10:50], tickers], names=["date", "ticker"]).to_frame(index=False)
        saved = pd.DataFrame({"Consenso Llama": prices.mean(axis=1) / 60,
                              "Momentum numérico Top 4": prices.iloc[:, :4].mean(axis=1) / 70,
                              "Comprar e manter": prices.mean(axis=1) / 80})
        seeds = (42, 43, 44)
        predictions, baskets = {}, {}
        for offset, seed in enumerate(seeds):
            predictions[seed] = {}
            for name, shift in (("numeric", offset), ("attention", offset + 3)):
                ranking = np.roll(np.arange(8, dtype=float), shift)
                predictions[seed][name] = np.tile(ranking, 40)
                baskets[f"{name}_{seed}"] = [tickers[i] for i in np.argsort(ranking)[-4:]]
        for stale_scores in (False, True):
            with self.subTest(stale_scores=stale_scores):
                scope = dict(np=np, pd=pd, prices=prices, TICKERS=tickers, SEEDS=seeds,
                             COST_BPS=10.0, test_frame=frame, scores_by_seed=predictions, saved=saved,
                             weekly_decision_dates=weekly_decision_dates, execute_portfolio=execute_portfolio,
                             portfolio_metrics=portfolio_metrics)
                if stale_scores:
                    scope["scores"] = predictions[44]
                exec(compile(source, str(notebook), "exec"), scope)
                for key, chosen in baskets.items():
                    curve = scope["curves"][key]
                    relative = prices.loc[curve.index, chosen] / prices.loc[curve.index[0], chosen]
                    expected = relative.mean(axis=1) / 1.001
                    np.testing.assert_allclose(curve.to_numpy(), expected.to_numpy(), rtol=1e-12, atol=1e-12)
                self.assertEqual(scope["summary"].shape, (7, 8))
                self.assertTrue(np.isfinite(scope["summary"].to_numpy()).all())


if __name__ == "__main__":
    unittest.main()
