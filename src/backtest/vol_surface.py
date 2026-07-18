from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from src.backtest.visualization import BG, PANEL, GOLD, GREEN, RED, WHITE

DEFAULT_WINDOWS = [5, 10, 15, 20, 30, 40, 50, 60, 90, 120]


def compute_vol_surface(
    df: pd.DataFrame,
    price_col: str = "close",
    lookback: int = 200,
    windows: list[int] | None = None,
) -> tuple[np.ndarray, list[int], pd.DataFrame]:
    if windows is None:
        windows = DEFAULT_WINDOWS

    tail = df.tail(lookback).copy().reset_index(drop=True)
    prices = pd.to_numeric(tail[price_col], errors="coerce")
    log_ret = np.log(prices / prices.shift(1))

    max_win = max(windows)
    n = len(tail)
    surface = np.zeros((n, len(windows)), dtype=float)

    for i in range(n):
        if i < max_win:
            continue
        for w_idx, w in enumerate(windows):
            window = log_ret.iloc[max(0, i - w + 1): i + 1]
            if len(window) > 1:
                surface[i, w_idx] = float(window.std() * np.sqrt(w))

    df_tail = tail.iloc[max_win:].reset_index(drop=True)
    surface = surface[max_win:]
    return surface, list(windows), df_tail


def _style_3d(ax, title: str) -> None:
    ax.set_facecolor(PANEL)
    ax.xaxis.set_pane_color((0, 0, 0, 0))
    ax.yaxis.set_pane_color((0, 0, 0, 0))
    ax.zaxis.set_pane_color((0, 0, 0, 0))
    ax.tick_params(colors=WHITE, labelsize=10)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_color(WHITE)
    ax.set_xlabel("Time index", color=WHITE, fontsize=12, labelpad=10)
    ax.set_ylabel("Window", color=WHITE, fontsize=12, labelpad=10)
    ax.set_zlabel("Volatility", color=WHITE, fontsize=12, labelpad=10)
    ax.set_title(title, color=WHITE, fontsize=16, fontweight="bold", pad=20)
    ax.grid(True, alpha=0.12)


def plot_vol_surface_snapshot(
    surface: np.ndarray,
    windows: list[int],
    df: pd.DataFrame,
    output_path: str,
    title: str = "Volatility Surface",
) -> None:
    fig = plt.figure(figsize=(14, 10), facecolor=BG)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(PANEL)

    n_times = surface.shape[0]
    x = np.arange(n_times)
    y = np.array(windows)
    X, Y = np.meshgrid(x, y)
    Z = surface.T

    vmax = float(np.percentile(surface, 95)) if surface.size else 1.0
    surf = ax.plot_surface(
        X,
        Y,
        Z,
        cmap="inferno",
        alpha=0.85,
        edgecolor="none",
        antialiased=True,
        vmin=0.0,
        vmax=vmax,
    )
    ax.plot_wireframe(X, Y, Z, color=GOLD, alpha=0.15, linewidth=0.3)

    if n_times > 0:
        latest = Z[:, -1]
        ax.plot(
            [x[-1]] * len(windows),
            y,
            latest,
            color=GREEN,
            linewidth=3,
            label="Latest vol curve",
        )
        ax.legend(loc="upper left", fontsize=12, labelcolor=WHITE)

    ax.set_xlim(0, n_times)
    ax.invert_xaxis()
    ax.set_ylim(min(windows), max(windows))
    ax.set_zlim(0, vmax)
    ax.view_init(elev=25, azim=45)
    _style_3d(ax, title)

    mappable = surf
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.55, pad=0.1)
    cbar.ax.tick_params(colors=WHITE, labelsize=10)
    cbar.set_label("Volatility", color=WHITE, fontsize=12)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, facecolor=BG, edgecolor="none", bbox_inches="tight")
    plt.close(fig)


def animate_vol_surface(
    df: pd.DataFrame,
    output_path: str = "vol_surface.mp4",
    fps: int = 10,
    lookback: int = 300,
    windows: list[int] | None = None,
) -> None:
    surface, wins, time_df = compute_vol_surface(
        df, lookback=lookback, windows=windows
    )
    if surface.size == 0:
        return

    n_times = surface.shape[0]
    total_frames = n_times
    frame_step = max(1, total_frames // 200)
    n_frames = max(1, total_frames // frame_step)

    vmax = float(np.percentile(surface, 95)) if surface.size else 1.0
    y = np.array(wins)

    fig = plt.figure(figsize=(14, 10), facecolor=BG)
    ax = fig.add_subplot(111, projection="3d")

    def update(frame: int):
        ax.clear()
        end_idx = min(frame * frame_step + 50, n_times)
        if end_idx < 20:
            return

        z = surface[:end_idx].T
        x = np.arange(end_idx)
        X, Y = np.meshgrid(x, y)

        surf = ax.plot_surface(
            X,
            Y,
            z,
            cmap="inferno",
            alpha=0.85,
            edgecolor="none",
            antialiased=True,
            vmin=0.0,
            vmax=vmax,
        )
        ax.plot_wireframe(X, Y, z, color=GOLD, alpha=0.15, linewidth=0.3)

        latest = z[:, -1]
        ax.plot(
            [x[-1]] * len(wins),
            y,
            latest,
            color=GREEN,
            linewidth=3,
            label="Latest vol curve",
        )
        ax.legend(loc="upper left", fontsize=12, labelcolor=WHITE)

        ax.set_xlim(0, n_times)
        ax.invert_xaxis()
        ax.set_ylim(min(wins), max(wins))
        ax.set_zlim(0, vmax)
        ax.view_init(elev=25, azim=45 + frame * 0.3)
        _style_3d(ax, "Volatility Surface Evolution")

    anim = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=int(1000 / fps),
        cache_frame_data=False,
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(path), writer="ffmpeg", fps=fps, dpi=100, bitrate=2000)
    plt.close(fig)
