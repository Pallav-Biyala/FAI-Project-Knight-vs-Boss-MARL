"""Aggregate results.jsonl from run_proposal_experiments.py.

Produces (in <out>/analysis/):
  summary_by_config.csv   mean/std/95% t-CI of win, loss, draw, timeout, length per
                          (regime, opponent, alpha_h, alpha_d) over training seeds
  alpha_star.csv          critical asymmetry per (regime, opponent, slice) with grid
                          resolution and uncertainty (see definitions below)
  fig_win_vs_alpha.png    win-rate curves with 95% CIs (diagonal slice)
  fig_slices.png          health / damage sensitivity by opponent
  fig_heatmap_heuristic.*  5x5 grid vs scripted Boss
  fig_heatmap_matched.*    5x5 grid vs matched co-trained Boss
  fig_seed_variability.*   individual seed outcomes and 95% CIs
  report_tables.md        the same tables in Markdown

Definitions (proposal Eq. 5 and reviewer feedback):
  alpha*       = max tested alpha whose mean win rate >= 0.5 (proposal definition).
  bracket      = [alpha*, next tested alpha): the true crossing is only known to lie
                 here, so alpha* is a lower bound on the grid, not the true threshold.
  alpha*_lo    = max tested alpha whose 95% CI lower bound >= 0.5 (conservative).
  alpha*_hi    = max tested alpha whose 95% CI upper bound >= 0.5 (optimistic).
  crossing_est = linear interpolation of the mean curve between the two grid points
                 straddling 0.5 (an estimate, not a measurement).
A value of "none" means no tested alpha >= 1 reached a 0.5 mean win rate.
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


def alpha_star_table(agg: pd.DataFrame) -> pd.DataFrame:
    recs = []
    slices = (
        ("diagonal", lambda g: g[g.alpha_h == g.alpha_d], "alpha_h"),
        ("durability (alpha_h, alpha_d=1)", lambda g: g[g.alpha_d == 1.0], "alpha_h"),
        ("lethality (alpha_d, alpha_h=1)", lambda g: g[g.alpha_h == 1.0], "alpha_d"),
    )
    for (regime, opp), matchup in agg.groupby(["regime", "opponent"]):
        for sl, select, axis in slices:
            all_points = select(matchup).sort_values(axis)
            if all_points.empty:
                continue
            # The proposal defines capability asymmetry for alpha >= 1. Values below
            # parity remain in the curves for context, but cannot set alpha-star.
            g = all_points[all_points[axis] >= 1.0 - 1e-9].sort_values(axis)
            alphas = g[axis].to_numpy(dtype=float)
            win = g["win_mean"].to_numpy(dtype=float)
            lo = g["win_ci_lo"].to_numpy(dtype=float)
            hi = g["win_ci_hi"].to_numpy(dtype=float)

            def best(mask):
                return float(alphas[mask].max()) if mask.any() else None

            star = best(win >= 0.5)
            star_lo = best(lo >= 0.5)
            star_hi = best(hi >= 0.5)
            bracket, crossing = "n/a", None
            if star is None:
                bracket = "none: no tested alpha >= 1 has mean win rate >= 0.5"
            elif star == alphas.max():
                bracket = f"not bounded: still >= 0.5 at largest tested alpha {star:g}"
            else:
                i = int(np.where(alphas == star)[0][0])
                nxt = float(alphas[i + 1])
                bracket = f"[{star:g}, {nxt:g})"
                w0, w1 = win[i], win[i + 1]
                if w0 != w1:
                    crossing = float(alphas[i] + (w0 - 0.5) / (w0 - w1) * (alphas[i + 1] - alphas[i]))
            monotone = bool(np.all(np.diff(win) <= 1e-9))
            recs.append({
                "regime": regime, "opponent": opp, "slice": sl,
                "threshold_domain": "alpha >= 1",
                "tested_alphas": " ".join(f"{a:g}" for a in all_points[axis]),
                "mean_win_by_alpha": " ".join(f"{w:.3f}" for w in all_points["win_mean"]),
                "alpha_star": star if star is not None else "none",
                "alpha_star_lo(CI)": star_lo if star_lo is not None else "none",
                "alpha_star_hi(CI)": star_hi if star_hi is not None else "none",
                "bracket": bracket,
                "crossing_est": round(crossing, 2) if crossing is not None else "",
                "curve_monotone_on_alpha>=1": monotone,
                "n_seeds": int(all_points["n_seeds"].min()),
            })
    return pd.DataFrame(recs)


REGIME_LABELS = {
    "fixed_opponent": "Fixed opponent",
    "simultaneous_learning": "Simultaneous",
    "alternating_learning": "Alternating",
}
REGIME_COLORS = {
    "fixed_opponent": "#3568a8",
    "simultaneous_learning": "#2a8c67",
    "alternating_learning": "#d2782c",
}


def _save_figure(fig, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_diagonal(agg: pd.DataFrame, baselines: pd.DataFrame | None, path: Path) -> None:
    d = agg[agg.alpha_h == agg.alpha_d]
    opps = [o for o in ("heuristic", "matched", "br_boss") if o in set(d.opponent)]
    if not opps:
        return
    fig, axes = plt.subplots(1, len(opps), figsize=(5.2 * len(opps), 4.4), sharey=True, squeeze=False)
    titles = {"heuristic": "vs scripted Boss", "matched": "vs co-trained Boss",
              "br_boss": "vs fresh best-response Boss"}
    colors = REGIME_COLORS
    for ax, opp in zip(axes[0], opps):
        for regime, g in d[d.opponent == opp].groupby("regime"):
            g = g.sort_values("alpha_h")
            err = np.vstack([g.win_mean - g.win_ci_lo, g.win_ci_hi - g.win_mean])
            ax.errorbar(g.alpha_h, g.win_mean * 100, yerr=np.nan_to_num(err) * 100, marker="o",
                        capsize=3, label=REGIME_LABELS.get(regime, regime), color=colors.get(regime))
        if baselines is not None and opp == "heuristic":
            for name, style in (("scripted_knight", "--"), ("random_knight", ":")):
                b = baselines[(baselines.agent == name) & (baselines.alpha_h == baselines.alpha_d)].sort_values("alpha_h")
                if len(b):
                    baseline_label = "Scripted Knight" if name == "scripted_knight" else "Random Knight"
                    ax.plot(b.alpha_h, b.knight_win_rate * 100, style, color="gray", label=baseline_label)
        ax.axhline(50, color="red", lw=0.8, ls="--")
        ax.set_title(titles[opp])
        ax.set_xlabel(r"Diagonal multiplier ($\alpha_H=\alpha_D$)")
        ax.set_ylim(-2, 102)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("Knight win rate (%) · mean and 95% CI across training seeds")
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    _save_figure(fig, path)


def plot_slices(agg: pd.DataFrame, path: Path) -> None:
    opponents = [o for o in ("heuristic", "matched") if o in set(agg.opponent)]
    if not opponents:
        return
    fig, axes = plt.subplots(1, len(opponents), figsize=(5.5 * len(opponents), 4.2),
                             sharey=True, squeeze=False)
    titles = {"heuristic": "Against scripted Boss", "matched": "Against matched co-trained Boss"}
    line_styles = {"health": "-", "damage": "--"}
    for ax, opponent in zip(axes[0], opponents):
        subset = agg[agg.opponent == opponent]
        for regime in ("fixed_opponent", "simultaneous_learning", "alternating_learning"):
            color = REGIME_COLORS[regime]
            for attribute, mask, axis in (
                ("health", subset.alpha_d == 1.0, "alpha_h"),
                ("damage", subset.alpha_h == 1.0, "alpha_d"),
            ):
                g = subset[(subset.regime == regime) & mask].sort_values(axis)
                if len(g) < 2:
                    continue
                err = np.vstack([g.win_mean - g.win_ci_lo, g.win_ci_hi - g.win_mean])
                ax.errorbar(g[axis], g.win_mean * 100, yerr=np.nan_to_num(err) * 100,
                            marker="o", ls=line_styles[attribute], color=color, capsize=2,
                            label=f"{REGIME_LABELS[regime]} · Boss {attribute}")
        ax.axhline(50, color="#555555", lw=0.9, ls=":")
        ax.set_title(titles[opponent])
        ax.set_xlabel("Multiplier on the varied Boss attribute")
        ax.set_xticks(sorted(set(subset.alpha_h) | set(subset.alpha_d)))
        ax.set_ylim(-2, 102)
        ax.grid(alpha=0.22)
    axes[0][0].set_ylabel("Knight win rate (%) · mean and 95% CI")
    axes[0][-1].legend(fontsize=7, loc="best")
    fig.tight_layout()
    _save_figure(fig, path)


def plot_heatmaps(agg: pd.DataFrame, opponent: str, regimes: list[str], path: Path) -> None:
    subset = agg[agg.opponent == opponent]
    present = [r for r in regimes if r in set(subset.regime)]
    if not present:
        return
    fig, axes = plt.subplots(1, len(present), figsize=(4.8 * len(present), 4.2),
                             squeeze=False, constrained_layout=True)
    image = None
    for ax, regime in zip(axes[0], present):
        frame = subset[subset.regime == regime]
        matrix = frame.pivot(index="alpha_d", columns="alpha_h", values="win_mean")
        hs = matrix.columns.to_numpy(dtype=float)
        ds = matrix.index.to_numpy(dtype=float)
        values = matrix.to_numpy(dtype=float)
        image = ax.imshow(values, origin="lower", aspect="equal", interpolation="nearest",
                          extent=(hs[0] - 0.125, hs[-1] + 0.125,
                                  ds[0] - 0.125, ds[-1] + 0.125),
                          vmin=0.0, vmax=1.0, cmap="viridis")
        if values.min() <= 0.5 <= values.max():
            ax.contour(hs, ds, values, levels=[0.5], colors="white", linewidths=1.3)
        for yi, damage in enumerate(ds):
            for xi, health in enumerate(hs):
                rate = values[yi, xi]
                ax.text(health, damage, f"{rate * 100:.0f}", ha="center", va="center",
                        fontsize=8, color="white" if rate < 0.48 else "black")
        ax.set_xticks(hs)
        ax.set_yticks(ds)
        ax.set_xlabel(r"Boss health multiplier ($\alpha_H$)")
        ax.set_title(REGIME_LABELS.get(regime, regime))
        ax.set_xlim(hs[0] - 0.125, hs[-1] + 0.125)
        ax.set_ylim(ds[0] - 0.125, ds[-1] + 0.125)
    axes[0][0].set_ylabel(r"Boss damage multiplier ($\alpha_D$)")
    from matplotlib.ticker import FuncFormatter
    fig.colorbar(image, ax=list(axes[0]), shrink=0.82, pad=0.035,
                 label="Knight win rate (%)", format=FuncFormatter(lambda x, _: f"{x * 100:.0f}"))
    opponent_title = "scripted Boss" if opponent == "heuristic" else "matched co-trained Boss"
    fig.suptitle(f"Knight win rate against {opponent_title}\nCell labels show mean %; white contour marks 50%",
                 fontsize=11)
    _save_figure(fig, path)


def plot_seed_variability(df: pd.DataFrame, path: Path) -> None:
    diagonal = df[df.alpha_h == df.alpha_d]
    opponents = [o for o in ("heuristic", "matched") if o in set(diagonal.opponent)]
    if not opponents:
        return
    fig, axes = plt.subplots(1, len(opponents), figsize=(5.5 * len(opponents), 4.3),
                             sharey=True, squeeze=False)
    titles = {"heuristic": "Against scripted Boss", "matched": "Against matched co-trained Boss"}
    regime_order = [r for r in ("fixed_opponent", "simultaneous_learning", "alternating_learning")
                    if r in set(diagonal.regime)]
    for ax, opponent in zip(axes[0], opponents):
        sub = diagonal[diagonal.opponent == opponent]
        alphas = sorted(sub.alpha_h.unique())
        offsets = np.linspace(-0.035, 0.035, len(regime_order)) if len(regime_order) > 1 else [0.0]
        for regime, offset in zip(regime_order, offsets):
            color = REGIME_COLORS[regime]
            means_x, means_y = [], []
            for alpha in alphas:
                group = sub[(sub.regime == regime) & (sub.alpha_h == alpha)].sort_values("seed")
                if group.empty:
                    continue
                rates = group.knight_win_rate.to_numpy(dtype=float)
                x = float(alpha + offset)
                jitter = np.linspace(-0.009, 0.009, len(rates)) if len(rates) > 1 else [0.0]
                ax.scatter(x + jitter, rates * 100, s=18, color=color, alpha=0.38,
                           edgecolors="none", zorder=2)
                mean = float(rates.mean())
                half = t_crit(len(rates) - 1) * float(rates.std(ddof=1)) / np.sqrt(len(rates)) if len(rates) > 1 else 0.0
                means_x.append(x)
                means_y.append(mean * 100)
                ax.errorbar(x, mean * 100, yerr=half * 100, marker="o", ms=4.5,
                            color=color, ecolor=color, capsize=3, lw=1.3, zorder=3)
            if means_x:
                ax.plot(means_x, means_y, color=color, lw=1.0,
                        label=REGIME_LABELS.get(regime, regime), zorder=1)
        ax.axhline(50, color="#555555", lw=0.9, ls=":")
        ax.set_title(titles[opponent])
        ax.set_xlabel(r"Diagonal alpha ($\alpha_H=\alpha_D$)")
        ax.set_xticks(alphas)
        ax.set_ylim(-3, 103)
        ax.grid(alpha=0.22)
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel("Knight win rate (%) · seed points, mean, 95% t-CI")
    fig.tight_layout()
    _save_figure(fig, path)


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
    plot_slices(agg, adir / "fig_sensitivity.png")
    plot_heatmaps(
        agg,
        "heuristic",
        ["fixed_opponent", "simultaneous_learning", "alternating_learning"],
        adir / "fig_heatmap_heuristic.png",
    )
    plot_heatmaps(
        agg,
        "matched",
        ["simultaneous_learning", "alternating_learning"],
        adir / "fig_heatmap_matched.png",
    )
    plot_seed_variability(df, adir / "fig_seed_variability.png")

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
