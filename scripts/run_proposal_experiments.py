"""Run the proposal experiments end to end (resumable, parallel).

For every (regime, alpha_h, alpha_d, seed) job this script:
  1. trains the Knight (and Boss, for co-learning regimes) with the regime's
     schedule and an environment-step budget that is identical across regimes;
  2. evaluates the FINAL Knight checkpoint (never a validation-selected one) against
       (a) the scripted heuristic Boss,
       (b) the co-trained Boss (simultaneous / alternating only),
       (c) a freshly trained PPO "best-response" Boss against the frozen Knight;
  3. appends one JSON line per finished job to <out>/results.jsonl.

Re-running the same command skips jobs already present in results.jsonl, so an
interrupted run can simply be restarted.

Example (smoke test, ~minutes):   python scripts/run_proposal_experiments.py --smoke
Example (proposal grid):          python scripts/run_proposal_experiments.py --workers 5
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_SEED = 100000


def job_key(regime: str, alpha_h: float, alpha_d: float, seed: int) -> str:
    return f"{regime}|{alpha_h:g}|{alpha_d:g}|s{seed}"


def _make_env_factory(game_config, alpha_h, alpha_d, shaping_weight):
    from knight_boss.marl.proposal_env import ProposalCombatEnv

    def factory():
        return ProposalCombatEnv(
            game_config,
            alpha_h=alpha_h,
            alpha_d=alpha_d,
            reward_mode="shaped",
            shaping_weight=shaping_weight,
        )

    return factory


def _load_ppo(path, observation_size, ppo_config, n_actions=8):
    from knight_boss.agents.ppo_agent import PPOAgent
    from knight_boss.training.checkpointing import load_checkpoint

    agent = PPOAgent(observation_size, ppo_config, seed=0, n_actions=n_actions)
    load_checkpoint(agent, path, load_optimizer=False)
    return agent


def _evaluate(knight, boss, game_config, alpha_h, alpha_d, episodes, shaping_weight):
    from knight_boss.evaluation.evaluator import evaluate_policies

    _, summary = evaluate_policies(
        {"knight": knight, "boss": boss},
        game_config,
        episodes=episodes,
        seed=EVAL_SEED,
        alpha_h=alpha_h,
        alpha_d=alpha_d,
        deterministic=False,
        env_factory=_make_env_factory(game_config, alpha_h, alpha_d, shaping_weight),
    )
    return summary.to_dict()


def _train_best_response_boss(
    knight, game_config, ppo_config, alpha_h, alpha_d, seed, steps, shaping_weight
):
    """Train a fresh PPO Boss against a frozen Knight (empirical best response)."""
    from knight_boss.agents.ppo_agent import PPOAgent
    from knight_boss.marl.observations import observation_to_vector
    from knight_boss.training.trainer import MultiAgentPPOTrainer

    env = _make_env_factory(game_config, alpha_h, alpha_d, shaping_weight)()
    observations, _ = env.reset(seed=seed)
    size = observation_to_vector(observations["knight"]).size
    boss = PPOAgent(size, ppo_config, seed=seed + 5000, n_actions=8)
    trainer = MultiAgentPPOTrainer(
        env, ppo_config, seed=seed + 1000, policies={"knight": knight, "boss": boss}
    )
    while trainer.total_environment_steps < steps:
        trainer.train_rollout(
            steps=min(ppo_config.rollout_steps, steps - trainer.total_environment_steps),
            trainable_agents=("boss",),
        )
    return boss


def run_job(job: dict) -> dict:
    """Worker: train + evaluate one (regime, alpha, seed). Must stay picklable."""
    import torch

    torch.set_num_threads(1)
    from knight_boss.game.config import GameConfig
    from knight_boss.marl.observations import observation_to_vector
    from knight_boss.marl.proposal_env import ProposalHeuristicBoss
    from knight_boss.training.config import PPOConfig
    from knight_boss.training.regimes import TrainingRunConfig, run_training

    start = time.time()
    regime, ah, ad, seed = job["regime"], job["alpha_h"], job["alpha_d"], job["seed"]
    game_config = GameConfig.from_yaml(PROJECT_ROOT / "configs" / "game.yaml")
    ppo_config = PPOConfig.from_yaml(PROJECT_ROOT / "configs" / "ppo.yaml")
    shaping = job["shaping_weight"]
    run_root = Path(job["out"]) / "runs" / job["key"].replace("|", "_")

    run_config = TrainingRunConfig(
        regime=regime,
        training_episodes=1,  # unused: the step budget below takes precedence
        checkpoint_interval_updates=50,
        validation_interval_updates=25,
        validation_episodes=20,
        validation_seed=90000,
        validation_alpha_h=ah,
        validation_alpha_d=ad,
        output_root=str(run_root),
        opponent="heuristic" if regime == "fixed_opponent" else None,
        phase_episodes=None,
        reward_mode="shaped",
        shaping_weight=shaping,
        training_steps=job["steps"],
        phase_steps=job["phase_steps"] if regime == "alternating_learning" else None,
    )
    summary = run_training(
        run_config,
        game_config,
        ppo_config,
        seed=seed,
        alpha_h=ah,
        alpha_d=ad,
        env_factory=lambda a, d: _make_env_factory(game_config, a, d, shaping)(),
        heuristic_factory=lambda s: ProposalHeuristicBoss(
            rows=game_config.rows, columns=game_config.columns, seed=s
        ),
    )
    run_dir = Path(summary["run_directory"])
    ckpt_dir = run_dir / "checkpoints"
    probe_obs, _ = _make_env_factory(game_config, ah, ad, shaping)().reset(seed=0)
    size = observation_to_vector(probe_obs["knight"]).size
    knight = _load_ppo(ckpt_dir / "knight-final.pt", size, ppo_config)

    base = {
        "key": job["key"],
        "regime": regime,
        "alpha_h": ah,
        "alpha_d": ad,
        "seed": seed,
        "train_steps": summary["environment_steps"],
        "train_episodes": summary["completed_episodes"],
        "run_dir": str(run_dir),
    }
    rows = []

    heuristic = ProposalHeuristicBoss(
        rows=game_config.rows, columns=game_config.columns, seed=seed + 1
    )
    rows.append(
        {
            **base,
            "opponent": "heuristic",
            **_evaluate(knight, heuristic, game_config, ah, ad, job["eval_episodes"], shaping),
        }
    )
    if regime != "fixed_opponent":
        boss = _load_ppo(ckpt_dir / "boss-final.pt", size, ppo_config)
        rows.append(
            {
                **base,
                "opponent": "matched",
                **_evaluate(knight, boss, game_config, ah, ad, job["eval_episodes"], shaping),
            }
        )
    if job["br_steps"] > 0:
        br = _train_best_response_boss(
            knight, game_config, ppo_config, ah, ad, seed, job["br_steps"], shaping
        )
        rows.append(
            {
                **base,
                "opponent": "br_boss",
                "br_steps": job["br_steps"],
                **_evaluate(knight, br, game_config, ah, ad, job["eval_episodes"], shaping),
            }
        )
    return {"key": job["key"], "elapsed_s": round(time.time() - start, 1), "rows": rows}


def run_baselines(out: Path, alphas, eval_episodes: int, shaping: float) -> None:
    """Random and scripted Knight vs scripted Boss (proposal Sec. 4.5)."""
    from knight_boss.game.config import GameConfig
    from knight_boss.marl.proposal_env import (
        ProposalHeuristicBoss,
        ProposalHeuristicKnight,
        ProposalRandomAgent,
    )

    game_config = GameConfig.from_yaml(PROJECT_ROOT / "configs" / "game.yaml")
    rows = []
    for name, ctor in (
        ("random_knight", ProposalRandomAgent),
        ("scripted_knight", ProposalHeuristicKnight),
    ):
        for ah, ad in alphas:
            knight = ctor(rows=game_config.rows, columns=game_config.columns, seed=1)
            boss = ProposalHeuristicBoss(
                rows=game_config.rows, columns=game_config.columns, seed=2
            )
            rows.append(
                {
                    "agent": name,
                    "alpha_h": ah,
                    "alpha_d": ad,
                    **_evaluate(knight, boss, game_config, ah, ad, eval_episodes, shaping),
                }
            )
    with (out / "baselines.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=1)


def build_alpha_pairs(args) -> list[tuple[float, float]]:
    pairs = {(a, a) for a in args.diagonal}
    pairs |= {(a, 1.0) for a in args.durability}
    pairs |= {(1.0, a) for a in args.lethality}
    return sorted(pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="experiments_final")
    parser.add_argument("--regimes", nargs="+",
                        default=["fixed_opponent", "simultaneous_learning", "alternating_learning"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--diagonal", nargs="*", type=float, default=[1.0, 1.5, 2.0, 3.0],
                        help="alpha_h = alpha_d values (the proposal's single alpha).")
    parser.add_argument("--durability", nargs="*", type=float, default=[],
                        help="alpha_h values with alpha_d = 1 (isolates Boss health).")
    parser.add_argument("--lethality", nargs="*", type=float, default=[],
                        help="alpha_d values with alpha_h = 1 (isolates Boss damage).")
    parser.add_argument("--steps", type=int, default=300000, help="Env steps per run, same for all regimes.")
    parser.add_argument("--phase-steps", type=int, default=50000, help="Alternating: steps per phase (N).")
    parser.add_argument("--br-steps", type=int, default=100000, help="Best-response Boss training steps (0 = skip).")
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--shaping-weight", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true", help="Tiny run to check the pipeline works.")
    args = parser.parse_args()

    if args.smoke:
        args.steps, args.phase_steps, args.br_steps = 2048, 1024, 1024
        args.eval_episodes, args.seeds, args.diagonal = 10, [0], [1.0, 1.5]
        args.durability, args.lethality, args.out = [], [], "experiments_smoke"

    out = Path(args.out)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    results_path = out / "results.jsonl"
    done = set()
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["key"])

    pairs = build_alpha_pairs(args)
    jobs = []
    for regime in args.regimes:
        for ah, ad in pairs:
            for seed in args.seeds:
                key = job_key(regime, ah, ad, seed)
                if key in done:
                    continue
                jobs.append(
                    {
                        "key": key, "regime": regime, "alpha_h": ah, "alpha_d": ad,
                        "seed": seed, "steps": args.steps, "phase_steps": args.phase_steps,
                        "br_steps": args.br_steps, "eval_episodes": args.eval_episodes,
                        "shaping_weight": args.shaping_weight, "out": str(out),
                    }
                )
    total = len(jobs) + len(done)
    print(f"{len(jobs)} jobs to run ({len(done)} already done, {total} total); "
          f"alpha pairs: {pairs}; workers: {args.workers}", flush=True)

    if not (out / "baselines.json").exists():
        run_baselines(out, pairs, args.eval_episodes, args.shaping_weight)
        print("Baselines written.", flush=True)

    finished = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_job, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
            except Exception as error:  # keep going; failed jobs are re-tried on re-run
                print(f"FAILED {job['key']}: {error!r}", flush=True)
                continue
            with results_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result) + "\n")
            finished += 1
            print(f"[{finished}/{len(jobs)}] {result['key']} done in {result['elapsed_s']}s", flush=True)


if __name__ == "__main__":
    main()
