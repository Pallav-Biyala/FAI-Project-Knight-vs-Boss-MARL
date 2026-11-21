"""Run the proposal experiments end to end (resumable, parallel).

For every (regime, alpha_h, alpha_d, seed, reward_mode) job this script:
  1. trains the Knight (and Boss, for co-learning regimes);
  2. evaluates the FINAL Knight checkpoint against:
       (a) the scripted heuristic Boss,
       (b) the co-trained Boss (simultaneous / alternating only),
       (c) a freshly trained PPO best-response Boss;
  3. appends one JSON line per finished job to <out>/results.jsonl.

Re-running the same command skips jobs already present in results.jsonl.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import isfinite
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_SEED = 100000
DEFAULT_ALPHA_GRID = (0.5, 0.75, 1.0, 1.25, 1.5)


def job_key(
    regime: str,
    alpha_h: float,
    alpha_d: float,
    seed: int,
    reward_mode: str,
) -> str:
    return f"{regime}|{alpha_h:g}|{alpha_d:g}|{reward_mode}|s{seed}"

def _make_env_factory(
    game_config,
    alpha_h,
    alpha_d,
    reward_mode,
    shaping_weight,
    gamma,
):
    from knight_boss.marl.proposal_env import ProposalCombatEnv

    def factory():
        return ProposalCombatEnv(
            game_config,
            alpha_h=alpha_h,
            alpha_d=alpha_d,
            reward_mode=reward_mode,
            shaping_weight=shaping_weight,
            gamma=gamma,
        )

    return factory


def _load_ppo(path, observation_size, ppo_config, n_actions=8):
    from knight_boss.agents.ppo_agent import PPOAgent
    from knight_boss.training.checkpointing import load_checkpoint

    agent = PPOAgent(
        observation_size,
        ppo_config,
        seed=0,
        n_actions=n_actions,
    )
    load_checkpoint(agent, path, load_optimizer=False)
    return agent


def _evaluate(
    knight,
    boss,
    game_config,
    alpha_h,
    alpha_d,
    episodes,
    reward_mode,
    shaping_weight,
    gamma,
):
    from knight_boss.evaluation.evaluator import evaluate_policies

    _, summary = evaluate_policies(
        {"knight": knight, "boss": boss},
        game_config,
        episodes=episodes,
        seed=EVAL_SEED,
        alpha_h=alpha_h,
        alpha_d=alpha_d,
        deterministic=False,
        env_factory=_make_env_factory(
            game_config,
            alpha_h,
            alpha_d,
            reward_mode,
            shaping_weight,
            gamma,
        ),
    )

    return summary.to_dict()


def _train_best_response_boss(
    knight,
    game_config,
    ppo_config,
    alpha_h,
    alpha_d,
    seed,
    steps,
    reward_mode,
    shaping_weight,
):
    """Train a fresh PPO Boss against a frozen Knight."""
    from knight_boss.agents.ppo_agent import PPOAgent
    from knight_boss.marl.observations import observation_to_vector
    from knight_boss.training.trainer import MultiAgentPPOTrainer

    env = _make_env_factory(
        game_config,
        alpha_h,
        alpha_d,
        reward_mode,
        shaping_weight,
        ppo_config.gamma,
    )()

    observations, _ = env.reset(seed=seed)

    size = observation_to_vector(observations["knight"]).size

    boss = PPOAgent(
        size,
        ppo_config,
        seed=seed + 5000,
        n_actions=8,
    )

    trainer = MultiAgentPPOTrainer(
        env,
        ppo_config,
        seed=seed + 1000,
        policies={"knight": knight, "boss": boss},
    )

    while trainer.total_environment_steps < steps:
        trainer.train_rollout(
            steps=min(
                ppo_config.rollout_steps,
                steps - trainer.total_environment_steps,
            ),
            trainable_agents=("boss",),
        )

    return boss


def run_job(job: dict) -> dict:
    """Worker: train and evaluate one experiment job."""
    import torch

    torch.set_num_threads(1)

    from knight_boss.game.config import GameConfig
    from knight_boss.marl.observations import observation_to_vector
    from knight_boss.marl.proposal_env import ProposalHeuristicBoss
    from knight_boss.training.config import PPOConfig
    from knight_boss.training.regimes import TrainingRunConfig, run_training

    start = time.time()

    regime = job["regime"]
    ah = job["alpha_h"]
    ad = job["alpha_d"]
    seed = job["seed"]

    reward_mode = job["reward_mode"]
    shaping = job["shaping_weight"]

    game_config = GameConfig.from_yaml(
        PROJECT_ROOT / "configs" / "game.yaml"
    )
    ppo_config = PPOConfig.from_yaml(
        PROJECT_ROOT / "configs" / "ppo.yaml"
    )

    run_root = (
        Path(job["out"])
        / "runs"
        / job["key"].replace("|", "_")
    )

    run_config = TrainingRunConfig(
        regime=regime,
        training_episodes=1,
        checkpoint_interval_updates=50,
        validation_interval_updates=25,
        validation_episodes=20,
        validation_seed=90000,
        validation_alpha_h=ah,
        validation_alpha_d=ad,
        output_root=str(run_root),
        opponent="heuristic" if regime == "fixed_opponent" else None,
        phase_episodes=None,
        reward_mode=reward_mode,
        shaping_weight=shaping,
        training_steps=job["steps"],
        phase_steps=(
            job["phase_steps"]
            if regime == "alternating_learning"
            else None
        ),
        population_size=job["population_size"],
        phase_updates=job["phase_updates"],
        snapshot_interval_updates=job["snapshot_interval_updates"],
    )

    summary = run_training(
        run_config,
        game_config,
        ppo_config,
        seed=seed,
        alpha_h=ah,
        alpha_d=ad,
        env_factory=lambda a, d: _make_env_factory(
            game_config,
            a,
            d,
            reward_mode,
            shaping,
            ppo_config.gamma,
        )(),
        heuristic_factory=lambda s: ProposalHeuristicBoss(
            rows=game_config.rows,
            columns=game_config.columns,
            seed=s,
        ),
    )

    run_dir = Path(summary["run_directory"])
    ckpt_dir = run_dir / "checkpoints"

    probe_obs, _ = _make_env_factory(
        game_config,
        ah,
        ad,
        reward_mode,
        shaping,
        ppo_config.gamma,
    )().reset(seed=0)

    size = observation_to_vector(probe_obs["knight"]).size

    # Always load the FINAL Knight checkpoint.
    knight = _load_ppo(
        ckpt_dir / "knight-final.pt",
        size,
        ppo_config,
    )

    base = {
        "key": job["key"],
        "regime": regime,
        "alpha_h": ah,
        "alpha_d": ad,
        "seed": seed,
        "train_steps": summary["environment_steps"],
        "train_episodes": summary["completed_episodes"],
        "run_dir": str(run_dir),
        "reward_mode": reward_mode,
        "population_size": job["population_size"],
        "phase_updates": job["phase_updates"],
        "snapshot_interval_updates": job["snapshot_interval_updates"],
    }

    rows = []

    # ---------------------------------------------------------
    # 1. Evaluate against scripted heuristic Boss.
    # ---------------------------------------------------------
    heuristic = ProposalHeuristicBoss(
        rows=game_config.rows,
        columns=game_config.columns,
        seed=seed + 1,
    )

    rows.append(
        {
            **base,
            "opponent": "heuristic",
            **_evaluate(
                knight,
                heuristic,
                game_config,
                ah,
                ad,
                job["eval_episodes"],
                reward_mode,
                shaping,
                ppo_config.gamma,
            ),
        }
    )

    # ---------------------------------------------------------
    # 2. Evaluate against co-trained Boss.
    # ---------------------------------------------------------
    if regime != "fixed_opponent":
        boss = _load_ppo(
            ckpt_dir / "boss-final.pt",
            size,
            ppo_config,
        )

        rows.append(
            {
                **base,
                "opponent": "matched",
                **_evaluate(
                    knight,
                    boss,
                    game_config,
                    ah,
                    ad,
                    job["eval_episodes"],
                    reward_mode,
                    shaping,
                    ppo_config.gamma,
                ),
            }
        )

    # ---------------------------------------------------------
    # 3. Train and evaluate against a fresh best-response Boss.
    # ---------------------------------------------------------
    if job["br_steps"] > 0:
        br = _train_best_response_boss(
            knight,
            game_config,
            ppo_config,
            ah,
            ad,
            seed,
            job["br_steps"],
            reward_mode,
            shaping,
        )

        rows.append(
            {
                **base,
                "opponent": "br_boss",
                "br_steps": job["br_steps"],
                **_evaluate(
                    knight,
                    br,
                    game_config,
                    ah,
                    ad,
                    job["eval_episodes"],
                    reward_mode,
                    shaping,
                    ppo_config.gamma,
                ),
            }
        )

    return {
        "key": job["key"],
        "elapsed_s": round(time.time() - start, 1),
        "rows": rows,
    }


def run_baselines(
    out: Path,
    alphas,
    eval_episodes: int,
    reward_mode: str,
    shaping: float,
) -> None:
    """Random and scripted Knight vs scripted Boss."""
    from knight_boss.game.config import GameConfig
    from knight_boss.marl.proposal_env import (
        ProposalHeuristicBoss,
        ProposalHeuristicKnight,
        ProposalRandomAgent,
    )
    from knight_boss.training.config import PPOConfig

    game_config = GameConfig.from_yaml(
        PROJECT_ROOT / "configs" / "game.yaml"
    )
    ppo_config = PPOConfig.from_yaml(
        PROJECT_ROOT / "configs" / "ppo.yaml"
    )

    rows = []

    for name, ctor in (
        ("random_knight", ProposalRandomAgent),
        ("scripted_knight", ProposalHeuristicKnight),
    ):
        for ah, ad in alphas:
            knight = ctor(
                rows=game_config.rows,
                columns=game_config.columns,
                seed=1,
            )

            boss = ProposalHeuristicBoss(
                rows=game_config.rows,
                columns=game_config.columns,
                seed=2,
            )

            rows.append(
                {
                    "agent": name,
                    "alpha_h": ah,
                    "alpha_d": ad,
                    **_evaluate(
                        knight,
                        boss,
                        game_config,
                        ah,
                        ad,
                        eval_episodes,
                        reward_mode,
                        shaping,
                        ppo_config.gamma,
                    ),
                }
            )

    with (out / "baselines.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(rows, handle, indent=1)


def build_alpha_pairs(args) -> list[tuple[float, float]]:
    """Build either a full Cartesian alpha grid or requested legacy slices."""
    alpha_grid = getattr(args, "alpha_grid", None)
    diagonal = getattr(args, "diagonal", None)
    durability = getattr(args, "durability", None)
    lethality = getattr(args, "lethality", None)
    has_legacy_slices = any(
        value is not None for value in (diagonal, durability, lethality)
    )

    if alpha_grid is not None and has_legacy_slices:
        raise ValueError(
            "Use --alpha-grid by itself, or use --diagonal/--durability/--lethality."
        )

    if alpha_grid is not None:
        values = tuple(sorted(set(float(value) for value in alpha_grid)))
    elif has_legacy_slices:
        values = ()
    else:
        values = DEFAULT_ALPHA_GRID

    if values:
        if any(not isfinite(value) or value <= 0 for value in values):
            raise ValueError("Alpha values must be positive and finite.")
        if alpha_grid is not None or not has_legacy_slices:
            return sorted((alpha_h, alpha_d) for alpha_h in values for alpha_d in values)

    pairs = {(a, a) for a in (diagonal or [])}
    pairs |= {(a, 1.0) for a in (durability or [])}
    pairs |= {(1.0, a) for a in (lethality or [])}
    if any(
        not isfinite(value) or value <= 0
        for pair in pairs
        for value in pair
    ):
        raise ValueError("Alpha values must be positive and finite.")
    if not pairs:
        raise ValueError("At least one alpha pair must be selected.")

    return sorted(pairs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--out", default="experiments_final")

    parser.add_argument(
        "--regimes",
        nargs="+",
        default=[
            "fixed_opponent",
            "simultaneous_learning",
            "alternating_learning",
            "population_self_play",
        ],
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
    )

    parser.add_argument(
        "--alpha-grid",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Values for both axes of a full Cartesian alpha_h x alpha_d grid. "
            f"Default: {list(DEFAULT_ALPHA_GRID)}."
        ),
    )

    parser.add_argument(
        "--diagonal",
        nargs="*",
        type=float,
        default=None,
        help="Legacy diagonal slice (alpha_h = alpha_d).",
    )

    parser.add_argument(
        "--durability",
        nargs="*",
        type=float,
        default=None,
        help="alpha_h values with alpha_d = 1.",
    )

    parser.add_argument(
        "--lethality",
        nargs="*",
        type=float,
        default=None,
        help="alpha_d values with alpha_h = 1.",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=300000,
        help="Environment steps per run.",
    )

    parser.add_argument(
        "--phase-steps",
        type=int,
        default=50000,
        help="Alternating: steps per phase.",
    )

    parser.add_argument(
        "--br-steps",
        type=int,
        default=100000,
        help="Best-response Boss training steps (0 = skip).",
    )

    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--shaping-weight",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--population-size",
        type=int,
        default=10,
        help="Population self-play: number of frozen Boss snapshots kept.",
    )

    parser.add_argument(
        "--phase-updates",
        type=int,
        default=5,
        help="Population self-play: PPO updates per Knight/Boss phase.",
    )

    parser.add_argument(
        "--snapshot-interval-updates",
        type=int,
        default=5,
        help="Population self-play: snapshot the Boss every N updates.",
    )

    parser.add_argument(
        "--reward-mode",
        choices=["terminal", "shaped", "draw_penalty", "combat", "tactical"],
        default="shaped",
        help="Reward function used during training.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny run to check that the pipeline works.",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Smoke-test configuration.
    # ---------------------------------------------------------
    if args.smoke:
        args.steps = 2048
        args.phase_steps = 1024
        args.br_steps = 1024
        args.eval_episodes = 10
        args.seeds = [0]
        args.alpha_grid = None
        args.diagonal = [1.0, 1.5]
        args.durability = None
        args.lethality = None
        # Keep population phases short enough to exercise the alternation.
        args.population_size = 3
        args.phase_updates = 1
        args.snapshot_interval_updates = 1
        args.out = f"experiments_smoke_{args.reward_mode}"

    out = Path(args.out)

    (out / "runs").mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = out / "results.jsonl"

    # ---------------------------------------------------------
    # Load completed jobs for resumability.
    # ---------------------------------------------------------
    done = set()

    if results_path.exists():
        for line in results_path.read_text(
            encoding="utf-8"
        ).splitlines():
            if line.strip():
                done.add(json.loads(line)["key"])

    pairs = build_alpha_pairs(args)

    # ---------------------------------------------------------
    # Build experiment jobs.
    # ---------------------------------------------------------
    jobs = []

    for regime in args.regimes:
        for ah, ad in pairs:
            for seed in args.seeds:
                key = job_key(
                    regime,
                    ah,
                    ad,
                    seed,
                    args.reward_mode,
                )

                if key in done:
                    continue

                jobs.append(
                    {
                        "key": key,
                        "regime": regime,
                        "alpha_h": ah,
                        "alpha_d": ad,
                        "seed": seed,
                        "steps": args.steps,
                        "phase_steps": args.phase_steps,
                        "br_steps": args.br_steps,
                        "eval_episodes": args.eval_episodes,
                        "shaping_weight": args.shaping_weight,
                        "reward_mode": args.reward_mode,
                        "population_size": args.population_size,
                        "phase_updates": args.phase_updates,
                        "snapshot_interval_updates": args.snapshot_interval_updates,
                        "out": str(out),
                    }
                )

    total = len(jobs) + len(done)

    print(
        f"{len(jobs)} jobs to run "
        f"({len(done)} already done, {total} total); "
        f"alpha pairs: {pairs}; "
        f"reward mode: {args.reward_mode}; "
        f"workers: {args.workers}",
        flush=True,
    )

    # ---------------------------------------------------------
    # Run baselines.
    # ---------------------------------------------------------
    if not (out / "baselines.json").exists():
        run_baselines(
            out,
            pairs,
            args.eval_episodes,
            args.reward_mode,
            args.shaping_weight,
        )

        print("Baselines written.", flush=True)

    # ---------------------------------------------------------
    # Execute jobs in parallel.
    # ---------------------------------------------------------
    finished = 0

    with ProcessPoolExecutor(
        max_workers=args.workers
    ) as pool:

        futures = {
            pool.submit(run_job, job): job
            for job in jobs
        }

        for future in as_completed(futures):
            job = futures[future]

            try:
                result = future.result()

            except Exception as error:
                print(
                    f"FAILED {job['key']}: {error!r}",
                    flush=True,
                )
                continue

            with results_path.open(
                "a",
                encoding="utf-8",
            ) as handle:
                handle.write(json.dumps(result) + "\n")

            finished += 1

            print(
                f"[{finished}/{len(jobs)}] "
                f"{result['key']} done in "
                f"{result['elapsed_s']}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
