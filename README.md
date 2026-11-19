# Can a Weaker Agent Learn to Win?

A headless research framework for studying asymmetric adaptation in
simultaneous Knight-versus-Boss combat. The Boss's health multiplier
(`Alpha_H`) and damage multiplier (`Alpha_D`) are independent throughout the
game engine, PPO training, parameter sweep, evaluation, and result records.

The repository implements the complete research system: simulator, PettingZoo
Parallel API environment, baselines, independent PPO training, three training
regimes, parameter-grid runner, statistical evaluation, plots, and JSON battle replay.
The framework runs fully on CPU without requiring a GPU.

## Installation

Python 3.10 or later is required. From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

The package uses NumPy, PyTorch, Gymnasium, PettingZoo, PyYAML, Pandas,
Matplotlib, and Pytest. PPO training and game simulation run headlessly;
Matplotlib writes plots to files.

## Run a game and baselines

Play a heuristic Knight against a seeded random Boss:

```powershell
python scripts/run_game.py --knight heuristic --boss random --seed 7
```

Run both deterministic heuristic policies with an asymmetric Boss and render each turn:

```powershell
python scripts/run_game.py --knight heuristic --boss heuristic `
  --alpha-h 1.5 --alpha-d 2.0 --seed 7 --render
```

Play a match with a trained PPO checkpoint and record the battle:

```powershell
python scripts/run_game.py `
  --knight-checkpoint runs\fixed_opponent\<run-id>\checkpoints\knight-final.pt `
  --boss heuristic --seed 5 --record runs\knight_victory.json
```

Use `--record path\to\battle.json` to save a replay. Play it back at any time with:

```powershell
python scripts/replay_battle.py runs\knight_victory.json
```

`configs/game.yaml` is the canonical game-stat and rules configuration.
Boss health is `Alpha_H × Knight maximum health`; Boss basic and heavy damage
are independently scaled by `Alpha_D`.

## Train PPO

PPO hyperparameters are in `configs/ppo.yaml`; the selected regime's training
configuration is in its own YAML file. For example:

```powershell
python scripts/train.py --regime fixed_opponent --seed 7
python scripts/train.py --regime simultaneous_learning --seed 7
python scripts/train.py --regime alternating_learning --seed 7
```

You can also override the number of training episodes directly from CLI using `--episodes`:

```powershell
python scripts/train.py --regime fixed_opponent --episodes 100 --seed 7
```

The three regimes are:

| Regime | Trainable policies |
|---|---|
| `fixed_opponent` | Knight PPO; Boss fixed heuristic |
| `simultaneous_learning` | Knight and Boss PPO update from shared simultaneous battle rollouts |
| `alternating_learning` | Knight and Boss PPO update in alternating frozen-policy phases |

Use `--alpha-h` and `--alpha-d` to train at a specific capability
configuration. Each run writes resolved JSON metadata, a per-update CSV log,
per-agent checkpoints, and a distinct periodic validation CSV under a unique
run directory. Validation uses a fixed, separately seeded capability
configuration for checkpoint comparison; it is not used as final evaluation.
Frozen agents act but their policy and optimizer are not updated.

The project's PyTorch PPO loop is deliberately modular and documented. A
centralized collector is used to choose both policies' actions from one
pre-step state and keep each independent policy's observations, actions,
rewards, values, and log probabilities aligned.

## Run the capability matrix

Inspect the configured matrix without starting training:

```powershell
python scripts/run_sweep.py --dry-run
```

Run the complete matrix:

```powershell
python scripts/run_sweep.py
```

The default `configs/experiments.yaml` defines 11 Alpha_H values × 11
Alpha_D values (121 pairs), five independent seeds, and three regimes: 1,815
training runs. The runner groups runs by experiment and incrementally writes a
manifest with **training** multipliers and model/run metadata. It is a large
workload and should be launched deliberately.

## Evaluate saved policies

Evaluate the two PPO checkpoints from a simultaneous-learning run:

```powershell
python scripts/evaluate.py `
  --knight-checkpoint runs\simultaneous_learning\<run-id>\checkpoints\knight-final.pt `
  --boss-checkpoint runs\simultaneous_learning\<run-id>\checkpoints\boss-final.pt
```

For the fixed-heuristic regime, use `--boss-opponent heuristic` instead of a
Boss checkpoint:

```powershell
python scripts/evaluate.py `
  --knight-checkpoint runs\fixed_opponent\<run-id>\checkpoints\knight-final.pt `
  --boss-opponent heuristic --episodes 100 --seed 12345
```

The default is 500 evaluation episodes; change the count and
seed with `--episodes` and `--seed`. `--alpha-h` / `--alpha-d` evaluate a
different configuration. `--sweep` evaluates all 121 grid points, enabling
cross-configuration generalization analysis.

Evaluation is separate from training and does not update or mutate source
policies. Each evaluation saves summary and episode CSVs plus JSON metadata.
Win-rate uncertainty is reported with Wilson intervals. Across-checkpoint
comparisons use training seeds—not individual matches—as independent units.
Threshold estimates are empirical grid results, retain non-monotone
neighbors, and are not theoretical critical values.

## Generate plots

All graphs are made from saved training/evaluation CSV files:

```powershell
python scripts/generate_plots.py --training-csv runs\<run-id>\training.csv
python scripts/generate_plots.py --validation-csv runs\<run-id>\validation.csv
python scripts/generate_plots.py --evaluation-csv results\evaluation-<id>\summaries.csv
```

Repeat either option to compare runs. Plots include rollout reward, win rates,
episode length, capability heatmaps, regime comparisons, and threshold
curves; threshold estimates are also exported to CSV and JSON.

## Tests

```powershell
python -m pytest -q
```

Tests cover game mechanics, PettingZoo Parallel API compatibility, reward and
termination semantics, baselines, simultaneous PPO rollout collection,
independent/frozen updates, checkpoint restoration, training regimes,
asymmetric parameter grids, statistics, evaluation, plots, and battle replay.

## Project documentation

- [Game rules](docs/game_rules.md)
- [Environment interface](docs/environment_interface.md)
- [Baseline agents](docs/baselines.md)
- [PPO rollout and checkpointing](docs/ppo_training.md)
- [Training regimes](docs/training_regimes.md)
- [Asymmetric experiments](docs/asymmetric_experiments.md)
- [Evaluation and analysis](docs/evaluation_analysis.md)
- [Architecture](docs/architecture.md)
- [Design decisions](docs/design_decisions.md)
- [Experimental protocol](docs/experimental_protocol.md)
- [Reproducibility](docs/reproducibility.md)
- [CS 329 Complete Research Report](docs/research_report_complete.md)
- [CS 329 Complete Research Report (PDF)](docs/CS329_Knight_vs_Boss_Research_Report.pdf)
