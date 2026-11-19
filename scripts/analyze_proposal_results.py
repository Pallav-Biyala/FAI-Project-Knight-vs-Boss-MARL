"""Aggregate results.jsonl from run_proposal_experiments.py.

Produces (in <out>/analysis/):
  summary_by_config.csv   mean/std/95% t-CI of win, loss, draw, timeout, length per
                          (regime, opponent, alpha_h, alpha_d) over training seeds
  alpha_star.csv          critical asymmetry per (regime, opponent, slice) with grid
                          resolution and uncertainty (see definitions below)
  fig_win_vs_alpha.png    win-rate curves with 95% CIs (diagonal slice)
  fig_slices.png          durability / lethality slices, if present
  report_tables.md        the same tables in Markdown

Definitions (proposal Eq. 5 and reviewer feedback):
  alpha*       = max tested alpha whose mean win rate >= 0.5 (proposal definition).
  bracket      = [alpha*, next tested alpha): the true crossing is only known to lie
                 here, so alpha* is a lower bound on the grid, not the true threshold.
  alpha*_lo    = max tested alpha whose 95% CI lower bound >= 0.5 (conservative).
  alpha*_hi    = max tested alpha whose 95% CI upper bound >= 0.5 (optimistic).
  crossing_est = linear interpolation of the mean curve between the two grid points
                 straddling 0.5 (an estimate, not a measurement).
A value of "none" means no tested alpha reached a 0.5 mean win rate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Two-sided 95% Student-t critical values by degrees of freedom.
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086, 30: 2.042}


def t_crit(df: int) -> float:
    if df < 1:
        return float("nan")
    keys = sorted(T975)
    for key in keys:
        if df <= key:
            return T975[key]
    return 1.96


def to_md(frame: pd.DataFrame) -> str:
    """Minimal Markdown table writer (avoids the optional `tabulate` dependency)."""
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def load(out: Path) -> pd.DataFrame:
    rows = []
    for line in (out / "results.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.extend(json.loads(line)["rows"])
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    metrics = {
        "win": "knight_win_rate",
        "loss": "boss_win_rate",
        "draw": "draw_rate",
        "timeout": "timeout_rate",
        "length": "average_episode_length",
    }
    out = []
    for (regime, opp, ah, ad), g in df.groupby(["regime", "opponent", "alpha_h", "alpha_d"]):
        n = len(g)
        rec = {"regime": regime, "opponent": opp, "alpha_h": ah, "alpha_d": ad, "n_seeds": n}
        for name, col in metrics.items():
            mean = g[col].mean()
            std = g[col].std(ddof=1) if n > 1 else float("nan")
            half = t_crit(n - 1) * std / np.sqrt(n) if n > 1 else float("nan")
            rec[f"{name}_mean"] = mean
            rec[f"{name}_std"] = std
            rec[f"{name}_ci_lo"] = max(0.0, mean - half) if name != "length" else mean - half
            rec[f"{name}_ci_hi"] = min(1.0, mean + half) if name != "length" else mean + half
        rec["win_per_seed"] = ",".join(f"{v:.2f}" for v in g.sort_values("seed")["knight_win_rate"])
        out.append(rec)
    return pd.DataFrame(out).sort_values(["regime", "opponent", "alpha_h", "alpha_d"])


def slice_of(row) -> str | None:
    if row.alpha_h == row.alpha_d:
        return "diagonal"
    if row.alpha_d == 1.0:
        return "durability (alpha_h, alpha_d=1)"
    if row.alpha_h == 1.0:
        return "lethality (alpha_d, alpha_h=1)"
    return None


def alpha_star_table(agg: pd.DataFrame) -> pd.DataFrame:
    agg = agg.copy()
    agg["slice"] = agg.apply(slice_of, axis=1)
    agg = agg.dropna(subset=["slice"])
    recs = []
    for (regime, opp, sl), g in agg.groupby(["regime", "opponent", "slice"]):
        axis = "alpha_h" if sl.startswith("durability") else "alpha_d"
        if sl == "diagonal":
            axis = "alpha_h"
        g = g.sort_values(axis)
        alphas = g[axis].to_numpy()
        win = g["win_mean"].to_numpy()
        lo = g["win_ci_lo"].to_numpy()
        hi = g["win_ci_hi"].to_numpy()

        def best(mask):
            return float(alphas[mask].max()) if mask.any() else None

        star = best(win >= 0.5)
        star_lo = best(lo >= 0.5)
        star_hi = best(hi >= 0.5)
        bracket, crossing = "n/a", None
        if star is None:
            bracket = f"none: win < 0.5 at every tested alpha (min tested {alphas.min():g})"
        elif star == alphas.max():
            bracket = f"not bounded: still >= 0.5 at largest tested alpha {star:g}"
        else:
            nxt = float(alphas[alphas > star].min())
            bracket = f"[{star:g}, {nxt:g})"
            i = int(np.where(alphas == star)[0][0])
            w0, w1 = win[i], win[i + 1]
            if w0 != w1:
                crossing = float(alphas[i] + (w0 - 0.5) / (w0 - w1) * (alphas[i + 1] - alphas[i]))
        monotone = bool(np.all(np.diff(win) <= 1e-9))
        recs.append({
            "regime": regime, "opponent": opp, "slice": sl,
            "tested_alphas": " ".join(f"{a:g}" for a in alphas),
            "mean_win_by_alpha": " ".join(f"{w:.3f}" for w in win),
            "alpha_star": star if star is not None else "none",
            "alpha_star_lo(CI)": star_lo if star_lo is not None else "none",
            "alpha_star_hi(CI)": star_hi if star_hi is not None else "none",
            "bracket": bracket,
            "crossing_est": round(crossing, 2) if crossing is not None else "",
            "curve_monotone": monotone,
            "n_seeds": int(g["n_seeds"].min()),
        })
    return pd.DataFrame(recs)


def plot_diagonal(agg: pd.DataFrame, baselines: pd.DataFrame | None, path: Path) -> None:
    d = agg[agg.alpha_h == agg.alpha_d]
    opps = [o for o in ("heuristic", "matched", "br_boss") if o in set(d.opponent)]
    if not opps:
        return
    fig, axes = plt.subplots(1, len(opps), figsize=(5.2 * len(opps), 4.4), sharey=True, squeeze=False)
    titles = {"heuristic": "vs scripted Boss", "matched": "vs co-trained Boss",
              "br_boss": "vs fresh best-response Boss"}
    colors = {"fixed_opponent": "tab:blue", "simultaneous_learning": "tab:green",
              "alternating_learning": "tab:orange"}
    for ax, opp in zip(axes[0], opps):
        for regime, g in d[d.opponent == opp].groupby("regime"):
            g = g.sort_values("alpha_h")
            err = np.vstack([g.win_mean - g.win_ci_lo, g.win_ci_hi - g.win_mean])
            ax.errorbar(g.alpha_h, g.win_mean * 100, yerr=np.nan_to_num(err) * 100, marker="o",
                        capsize=3, label=regime, color=colors.get(regime))
        if baselines is not None and opp == "heuristic":
            for name, style in (("scripted_knight", "--"), ("random_knight", ":")):
                b = baselines[(baselines.agent == name) & (baselines.alpha_h == baselines.alpha_d)].sort_values("alpha_h")
                if len(b):
                    ax.plot(b.alpha_h, b.knight_win_rate * 100, style, color="gray", label=f"{name} (no learning)")
        ax.axhline(50, color="red", lw=0.8, ls="--")
        ax.set_title(titles[opp])
        ax.set_xlabel("alpha (alpha_h = alpha_d)")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("Knight win rate (%)  [mean, 95% CI over seeds]")
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_slices(agg: pd.DataFrame, path: Path) -> None:
    dur = agg[(agg.alpha_d == 1.0) & (agg.opponent == "heuristic")]
    let = agg[(agg.alpha_h == 1.0) & (agg.opponent == "heuristic")]
    if dur.alpha_h.nunique() < 2 and let.alpha_d.nunique() < 2:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for label, frame, col in (("durability: alpha_h (Boss health)", dur, "alpha_h"),
                              ("lethality: alpha_d (Boss damage)", let, "alpha_d")):
        for regime, g in frame.groupby("regime"):
            g = g.sort_values(col)
            if g[col].nunique() < 2:
                continue
            ax.plot(g[col], g.win_mean * 100, marker="o", label=f"{regime} - {label}")
    ax.axhline(50, color="red", lw=0.8, ls="--")
    ax.set_xlabel("capability multiplier on one Boss attribute")
    ax.set_ylabel("Knight win rate vs scripted Boss (%)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="experiments_final")
    args = parser.parse_args()
    out = Path(args.out)
    adir = out / "analysis"
    adir.mkdir(parents=True, exist_ok=True)

    df = load(out)
    agg = aggregate(df)
    star = alpha_star_table(agg)
    baselines = None
    if (out / "baselines.json").exists():
        baselines = pd.DataFrame(json.loads((out / "baselines.json").read_text(encoding="utf-8")))
        baselines.to_csv(adir / "baselines.csv", index=False)

    agg.to_csv(adir / "summary_by_config.csv", index=False)
    star.to_csv(adir / "alpha_star.csv", index=False)
    plot_diagonal(agg, baselines, adir / "fig_win_vs_alpha.png")
    plot_slices(agg, adir / "fig_slices.png")

    cols = ["regime", "opponent", "alpha_h", "alpha_d", "n_seeds", "win_mean", "win_ci_lo",
            "win_ci_hi", "loss_mean", "draw_mean", "timeout_mean", "length_mean"]
    with (adir / "report_tables.md").open("w", encoding="utf-8") as handle:
        handle.write("# Results (mean over training seeds, 95% t-CI)\n\n")
        handle.write(to_md(agg[cols].round(3)) + "\n\n# Critical asymmetry\n\n")
        handle.write(to_md(star) + "\n")
    print(star.to_string(index=False))
    print(f"\nWrote analysis to {adir}")


if __name__ == "__main__":
    main()
