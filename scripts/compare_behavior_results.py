"""Compare per-seed proposal experiment results and behavior diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as result_file:
        for line in result_file:
            if line.strip():
                rows.extend(json.loads(line).get("rows", []))
    return rows


def avg(rows: list[dict], field: str) -> float:
    return mean(float(row.get(field, 0.0)) for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--regime", default="alternating_learning")
    args = parser.parse_args()

    before = [
        row for row in load_rows(args.before) if row.get("regime") == args.regime
    ]
    after = [
        row for row in load_rows(args.after) if row.get("regime") == args.regime
    ]
    opponents = sorted(
        {row["opponent"] for row in before}
        & {row["opponent"] for row in after}
    )
    if not opponents:
        raise SystemExit(
            f"No shared opponent labels found for regime {args.regime!r}."
        )

    print(f"Training regime: {args.regime}")

    columns = (
        ("Opponent", 11),
        ("Old W/L/D", 19),
        ("New W/L/D", 19),
        ("Win SD new", 11),
        ("New seed rates", 29),
        ("Timeout old->new", 17),
        ("K dmg old->new", 24),
        ("Steps old->new", 19),
        ("Invalid/step old->new", 23),
        ("S2 hit/block old->new", 24),
        ("S2 escape old->new", 21),
        ("Heal noop/use old->new", 25),
        ("Attack/hit old->new", 22),
    )
    print(" ".join(title.ljust(width) for title, width in columns))
    for opponent in opponents:
        old = [row for row in before if row["opponent"] == opponent]
        new = [row for row in after if row["opponent"] == opponent]
        old_rates = [float(row["knight_win_rate"]) for row in old]
        new_rates = [float(row["knight_win_rate"]) for row in new]
        new_sd = stdev(new_rates) if len(new_rates) > 1 else 0.0
        old_wld = "/".join(
            f"{avg(old, field):.3f}"
            for field in ("knight_win_rate", "boss_win_rate", "draw_rate")
        )
        new_wld = "/".join(
            f"{avg(new, field):.3f}"
            for field in ("knight_win_rate", "boss_win_rate", "draw_rate")
        )
        values = (
            opponent,
            old_wld,
            new_wld,
            f"{new_sd:.3f}",
            ",".join(f"{rate:.2f}" for rate in new_rates),
            f"{avg(old, 'timeout_rate'):.3f}->{avg(new, 'timeout_rate'):.3f}",
            f"{avg(old, 'average_knight_damage_dealt'):.1f}/"
            f"{avg(old, 'average_knight_damage_received'):.1f}->"
            f"{avg(new, 'average_knight_damage_dealt'):.1f}/"
            f"{avg(new, 'average_knight_damage_received'):.1f}",
            f"{avg(old, 'average_episode_length'):.1f}->"
            f"{avg(new, 'average_episode_length'):.1f}",
            f"{avg(old, 'invalid_knight_moves_per_step'):.4f}->"
            f"{avg(new, 'invalid_knight_moves_per_step'):.4f}",
            f"{avg(old, 'special2_hit_rate'):.3f}/"
            f"{avg(old, 'special2_block_rate'):.3f}->"
            f"{avg(new, 'special2_hit_rate'):.3f}/"
            f"{avg(new, 'special2_block_rate'):.3f}",
            f"{avg(old, 'average_special2_escapes'):.2f}->"
            f"{avg(new, 'average_special2_escapes'):.2f}",
            f"{avg(old, 'average_knight_heal_noops'):.2f}/"
            f"{avg(old, 'average_knight_heals'):.2f}->"
            f"{avg(new, 'average_knight_heal_noops'):.2f}/"
            f"{avg(new, 'average_knight_heals'):.2f}",
            f"{avg(old, 'average_knight_attack_attempts'):.1f}/"
            f"{avg(old, 'average_knight_attack_hits'):.1f}->"
            f"{avg(new, 'average_knight_attack_attempts'):.1f}/"
            f"{avg(new, 'average_knight_attack_hits'):.1f}",
        )
        print(
            " ".join(
                str(value)[:width].ljust(width)
                for value, (_, width) in zip(values, columns)
            )
        )


if __name__ == "__main__":
    main()
