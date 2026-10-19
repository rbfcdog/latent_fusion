import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))

from src.fusion.dataset import load_panel, save_panel

if __name__ == "__main__":
    panel = load_panel()
    out = _root / "data" / "fusion_vol_panel.npz"
    save_panel(panel, out)
    print(f"{out} ({out.stat().st_size / 1e6:.0f} MB)")
