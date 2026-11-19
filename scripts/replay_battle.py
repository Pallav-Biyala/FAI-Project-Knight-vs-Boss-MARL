"""Replay a previously recorded battle in the terminal."""

import argparse

from knight_boss.visualization.battle_replay import play_replay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay")
    parser.add_argument("--delay", type=float, default=0.0)
    args = parser.parse_args()
    play_replay(args.replay, delay_seconds=args.delay)


if __name__ == "__main__":
    main()
