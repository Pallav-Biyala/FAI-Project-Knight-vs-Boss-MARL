"""Generate Figure 1: Win Rate Frontiers and Training Reward Curves for the Proposal Study.

Reproduces docs/figures/fig_proposal_curves.png and figures/fig_proposal_curves.png.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_figure(project_root: Path | None = None) -> list[Path]:
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]

    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # ----------------- Panel (a): Win Rate vs Diagonal Alpha -----------------
    alphas = [0.5, 1.0, 1.25, 1.5, 2.0]

    # Proposal study data (2_proposal from aggregated_2x2.csv / 2_proposal_aggregated.csv)
    fixed_win = [100.0, 74.3, 1.7, 0.0, 0.0]
    fixed_std = [0.0, 2.1, 0.8, 0.0, 0.0]

    sim_win = [12.3, 0.1, 0.0, 0.0, 0.0]
    sim_std = [1.4, 0.2, 0.0, 0.0, 0.0]

    alt_win = [9.9, 0.0, 0.0, 0.0, 0.0]
    alt_std = [11.5, 0.0, 0.0, 0.0, 0.0]

    # Calibrated baselines (proposal env from calibration_baselines.csv)
    heur_win = [100.0, 100.0, 96.6, 9.4, 0.0]
    heur_ci_low = [99.2376, 99.2376, 94.6229, 7.1426, 0.0]
    heur_ci_high = [100.0, 100.0, 97.8665, 12.2765, 0.7624]

    rand_win = [83.6, 0.2, 0.0, 0.0, 0.0]
    rand_ci_low = [80.1005, 0.0353, 0.0, 0.0, 0.0]
    rand_ci_high = [86.5871, 1.1241, 0.7624, 0.7624, 0.7624]

    # Plot PPO curves with +-1 std bars
    ax1.errorbar(
        alphas,
        fixed_win,
        yerr=fixed_std,
        fmt="-o",
        color="#1f77b4",
        lw=2.2,
        capsize=4,
        label="Fixed Opponent (vs Heur Boss)",
    )
    ax1.errorbar(
        alphas,
        sim_win,
        yerr=sim_std,
        fmt="-s",
        color="#2ca02c",
        lw=2.0,
        capsize=4,
        label="Simultaneous (vs Matched PPO)",
    )
    ax1.errorbar(
        alphas,
        alt_win,
        yerr=alt_std,
        fmt="-^",
        color="#ff7f0e",
        lw=2.0,
        capsize=4,
        label="Alternating (vs Matched PPO)",
    )

    # Plot baselines with Wilson confidence bands
    ax1.plot(
        alphas,
        heur_win,
        "--d",
        color="#9467bd",
        lw=1.8,
        label="Calibrated HeurKnight (vs Heur Boss)",
    )
    ax1.fill_between(
        alphas,
        heur_ci_low,
        heur_ci_high,
        color="#9467bd",
        alpha=0.18,
        label="HeurKnight 95% Wilson CI",
    )

    ax1.plot(
        alphas,
        rand_win,
        ":x",
        color="#7f7f7f",
        lw=1.5,
        label="Random Knight (vs Heur Boss)",
    )
    ax1.fill_between(
        alphas,
        rand_ci_low,
        rand_ci_high,
        color="#7f7f7f",
        alpha=0.15,
        label="RandomKnight 95% Wilson CI",
    )

    ax1.axhline(
        50.0,
        color="#d62728",
        linestyle="--",
        lw=1.2,
        alpha=0.8,
        label="50% Win Rate Threshold (α*)",
    )
    ax1.set_xlabel(
        "Diagonal Capability Multiplier α (α_H = α_D)",
        fontsize=11,
        fontweight="bold",
    )
    ax1.set_ylabel("Knight Win Rate (%)", fontsize=11, fontweight="bold")
    ax1.set_title(
        "(a) Knight Win Rate vs. Physical Asymmetry (α)",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_ylim(-5, 105)
    ax1.set_xticks(alphas)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(
        frameon=True,
        facecolor="#fbfbfb",
        edgecolor="#dddddd",
        fontsize=8.5,
        loc="upper right",
    )

    # ----------------- Panel (b): Training Reward Curves at Parity -----------------
    fixed_files = glob.glob(
        str(
            project_root
            / "experiments_unified/2_proposal/fixed_opponent/ah_1p00_ad_1p00/**/training.csv"
        ),
        recursive=True,
    )
    sim_files = glob.glob(
        str(
            project_root
            / "experiments_unified/2_proposal/simultaneous_learning/ah_1p00_ad_1p00/**/training.csv"
        ),
        recursive=True,
    )
    alt_files = glob.glob(
        str(
            project_root
            / "experiments_unified/2_proposal/alternating_learning/ah_1p00_ad_1p00/**/training.csv"
        ),
        recursive=True,
    )

    def get_mean_reward_curve(files, agent_name="knight", max_ep=1500):
        dfs = [pd.read_csv(f) for f in files]
        dfs = [d[d["agent"] == agent_name] for d in dfs]
        grid = np.linspace(30, max_ep, 100)
        interp_rewards = []
        for d in dfs:
            if len(d) > 1:
                r = np.interp(grid, d["completed_episodes"], d["rollout_reward"])
                interp_rewards.append(r)
        if interp_rewards:
            arr = np.array(interp_rewards)
            return grid, np.mean(arr, axis=0), np.std(arr, axis=0)
        return grid, np.zeros_like(grid), np.zeros_like(grid)

    # Fixed opponent (Knight only, against heuristic Boss)
    gx_f, gy_f, gs_f = get_mean_reward_curve(fixed_files, "knight", 1500)
    ax2.plot(gx_f, gy_f, color="#1f77b4", lw=2.2, label="Fixed Opponent: Knight")
    ax2.fill_between(gx_f, gy_f - gs_f, gy_f + gs_f, color="#1f77b4", alpha=0.15)

    # Simultaneous (both Knight and Boss)
    gx_sk, gy_sk, gs_sk = get_mean_reward_curve(sim_files, "knight", 750)
    gx_sb, gy_sb, gs_sb = get_mean_reward_curve(sim_files, "boss", 750)
    ax2.plot(gx_sk, gy_sk, color="#2ca02c", lw=2.0, label="Simultaneous: Knight")
    ax2.fill_between(
        gx_sk, gy_sk - gs_sk, gy_sk + gs_sk, color="#2ca02c", alpha=0.12
    )
    ax2.plot(
        gx_sb,
        gy_sb,
        color="#2ca02c",
        linestyle=":",
        lw=1.8,
        label="Simultaneous: Boss",
    )

    # Alternating (both Knight and Boss across active phases)
    gx_ak, gy_ak, gs_ak = get_mean_reward_curve(alt_files, "knight", 750)
    gx_ab, gy_ab, gs_ab = get_mean_reward_curve(alt_files, "boss", 750)
    ax2.plot(gx_ak, gy_ak, color="#ff7f0e", lw=2.0, label="Alternating: Knight")
    ax2.fill_between(
        gx_ak, gy_ak - gs_ak, gy_ak + gs_ak, color="#ff7f0e", alpha=0.12
    )
    ax2.plot(
        gx_ab,
        gy_ab,
        color="#ff7f0e",
        linestyle=":",
        lw=1.8,
        label="Alternating: Boss",
    )

    ax2.set_xlabel("Completed Training Episodes", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Mean Rollout Reward", fontsize=11, fontweight="bold")
    ax2.set_title(
        "(b) Training Reward Trajectories at Parity (1.00, 1.00)",
        fontsize=12,
        fontweight="bold",
    )
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(
        frameon=True,
        facecolor="#fbfbfb",
        edgecolor="#dddddd",
        fontsize=8.5,
        loc="lower right",
    )

    plt.tight_layout()

    out_paths = [
        project_root / "results/unified_2x2/figures/fig_proposal_curves.png",
        project_root / "docs/figures/fig_proposal_curves.png",
        project_root / "figures/fig_proposal_curves.png",
    ]
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(p), dpi=300)
        print("Saved to", p)
    plt.close()
    return out_paths


if __name__ == "__main__":
    generate_figure()
