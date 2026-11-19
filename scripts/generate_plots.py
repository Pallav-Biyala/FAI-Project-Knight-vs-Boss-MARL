"""Generate analysis plots from saved training/evaluation CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from knight_boss.visualization.plots import (
    plot_evaluation_analysis,
    plot_training_analysis,
    plot_validation_analysis,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-csv", action="append", default=[])
    parser.add_argument("--validation-csv", action="append", default=[])
    parser.add_argument("--training-manifest", action="append", default=[])
    parser.add_argument("--evaluation-csv", action="append", default=[])
    parser.add_argument("--output", default="results/plots")
    args = parser.parse_args()

    training_csvs = list(args.training_csv)
    validation_csvs = list(args.validation_csv)
    for manifest in args.training_manifest:
        with Path(manifest).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                run_dir = Path(row["training_directory"])
                t_path = run_dir / "training.csv"
                v_path = run_dir / "validation.csv"
                if t_path.is_file():
                    training_csvs.append(str(t_path))
                if v_path.is_file():
                    validation_csvs.append(str(v_path))

    if not training_csvs and not validation_csvs and not args.evaluation_csv:
        parser.error(
            "Provide at least one --training-csv, --validation-csv, "
            "--training-manifest, or --evaluation-csv."
        )
    output = Path(args.output)
    outputs = []
    if training_csvs:
        outputs.extend(
            plot_training_analysis(training_csvs, output / "training")
        )
    if validation_csvs:
        outputs.extend(
            plot_validation_analysis(validation_csvs, output / "validation")
        )
    if args.evaluation_csv:
        outputs.extend(
            plot_evaluation_analysis(args.evaluation_csv, output / "evaluation")
        )
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
