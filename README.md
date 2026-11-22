# Can a Weaker Agent Learn to Win?
### Asymmetric Co-Adaptation in Multi-Agent Combat

A headless Multi-Agent Reinforcement Learning (MARL) research framework for studying asymmetric adaptation in simultaneous Knight-versus-Boss combat.

The central research question is:

**Can a mechanically weaker Knight learn to defeat a stronger Boss, particularly when both agents are capable of learning?**

The Boss's health multiplier (`alpha_h`) and damage multiplier (`alpha_d`) are independent parameters. The framework implements the combat simulator, PettingZoo Parallel API environment, heuristic and random baselines, independent PPO agents, three training regimes, a capability-grid experiment runner, statistical evaluation, and result visualization.

All experiments are designed to run on CPU; a GPU is not required.

---

## 1. Research Overview

The Boss's capabilities are controlled by two independent multipliers:

- **Health multiplier (`alpha_h`)**: scales the Boss's maximum health.
- **Damage multiplier (`alpha_d`)**: scales the Boss's attack damage.

The experiment investigates a full Cartesian grid:

$\alpha_H,\alpha_D \in \{0.5,0.75,1.0,1.25,1.5\}$

This produces 25 capability configurations.

Three training regimes are compared:

| Regime | Training procedure |
|---|---|
| `fixed_opponent` | Knight PPO trains against a fixed heuristic Boss. |
| `simultaneous_learning` | Knight and Boss PPO policies learn simultaneously. |
| `alternating_learning` | Knight trains against a frozen Boss, followed by Boss training against the frozen Knight. |

The completed experiment consists of:

- 25 capability configurations.
- 5 independent training seeds (0–4).
- 3 training regimes.
- **375 total training runs.**
- 500 evaluation games per final Knight checkpoint against the heuristic Boss.
- Additional paired learned-Boss evaluations for simultaneous and alternating regimes.

The alternating regime uses 40,000 environment steps for each agent phase, giving it 80,000 total steps. Fixed and simultaneous regimes use 40,000 steps each. The regimes are therefore not compute-matched.

---

## 2. Installation

Python 3.10 or later is required.

Clone the repository and enter the project directory:

```bash
git clone https://github.com/Pallav-Biyala/FAI-Project-Knight-vs-Boss-MARL.git
cd FAI-Project-Knight-vs-Boss-MARL
```

Create and activate a virtual environment.

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project and test dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

The project dependencies are declared in `pyproject.toml`:

- NumPy
- PyTorch
- Gymnasium
- PettingZoo
- PyYAML
- Pandas
- Matplotlib
- Pytest

---

## 3. Environment and PPO Configuration

The canonical configurations are:

- `configs/game.yaml` — board dimensions, initial positions, combat statistics, and game rules.
- `configs/ppo.yaml` — PPO hyperparameters and network architecture.

The game uses a 9×9 grid with simultaneous agent actions and a maximum episode length of 300 steps.

The Knight has 150 HP, basic attacks, healing, and dodge mechanics. The Boss's health and damage scale independently with the two capability multipliers.

The PPO implementation is project-local and uses independent actor–critic policies.

The main PPO settings are:

| Parameter | Value |
|---|---:|
| Learning rate | 0.0003 |
| Discount factor (`gamma`) | 0.995 |
| GAE lambda | 0.95 |
| PPO clip range | 0.2 |
| Entropy coefficient | 0.02 |
| Value coefficient | 0.5 |
| Update epochs | 10 |
| Batch size | 64 |
| Rollout steps | 1024 |
| Hidden layers | [128, 128] |
| Maximum gradient norm | 0.5 |
| Device | CPU |

The final experiment uses the `tactical` reward mode with shaping weight 0.5.

The reward design includes terminal outcomes, draw penalties, health-based shaping, and tactical event rewards. It is not strictly zero-sum.

---

## 4. Reproducing the Final Experiments

The completed experiment can be reproduced using:

`scripts/run_proposal_experiments.py`

The runner supports a full Cartesian capability grid, multiple training regimes, independent seeds, parallel execution, and resumable jobs.

### Important

Use a **new output directory** when reproducing the experiments. The repository already contains the completed result files, and the runner skips jobs whose keys are already present in `results.jsonl`.

The commands below reproduce the three regimes in the same output directory(reproduction_run).

### 4.1 Fixed-opponent training

```bash
python scripts/run_proposal_experiments.py \
  --out reproduction_run \
  --regimes fixed_opponent \
  --alpha-grid 0.5 0.75 1.0 1.25 1.5 \
  --seeds 0 1 2 3 4 \
  --steps 40000 \
  --br-steps 0 \
  --eval-episodes 500 \
  --reward-mode tactical \
  --shaping-weight 0.5 \
  --workers 1
```

### 4.2 Simultaneous learning

```bash
python scripts/run_proposal_experiments.py \
  --out reproduction_run \
  --regimes simultaneous_learning \
  --alpha-grid 0.5 0.75 1.0 1.25 1.5 \
  --seeds 0 1 2 3 4 \
  --steps 40000 \
  --br-steps 0 \
  --eval-episodes 500 \
  --reward-mode tactical \
  --shaping-weight 0.5 \
  --workers 1
```

### 4.3 Alternating learning

```bash
python scripts/run_proposal_experiments.py \
  --out reproduction_run \
  --regimes alternating_learning \
  --alpha-grid 0.5 0.75 1.0 1.25 1.5 \
  --seeds 0 1 2 3 4 \
  --steps 80000 \
  --phase-steps 40000 \
  --br-steps 0 \
  --eval-episodes 500 \
  --reward-mode tactical \
  --shaping-weight 0.5 \
  --workers 1
```

The alternating regime performs one Knight training phase followed by one Boss training phase. The Knight does not undergo another adaptation phase afterward.

## 5. Statistical Analysis and Visualization

After all three regimes have completed, regenerate the analysis using:

```bash
python scripts/analyze_proposal_results.py --out reproduction_run
```

The analysis script reads `results.jsonl` and generates:

- Mean and standard deviation of evaluation metrics across training seeds.
- 95% Student-t confidence intervals.
- Capability-grid summary tables.
- Empirical critical asymmetry (`alpha*`) estimates.
- Diagonal capability comparisons.
- Health-only and damage-only sensitivity plots.
- Heuristic-opponent and matched learned-opponent heatmaps.
- Training-seed variability plots.
- Markdown result tables.

The five independent training seeds are treated as the statistical units for PPO confidence intervals. The 500 evaluation games per checkpoint are not treated as 500 independent training runs.

The empirical `alpha*` value is the largest tested multiplier at or above parity for which the mean Knight win rate is at least 50%. It is a finite-grid estimate, not a theoretical threshold.

---

### Output structure

The runner creates an output directory containing:

```text
reproduction_run/
├── results.jsonl
├── baselines.json
├── runs/
│   ├── ...
│   └── ...
└── analysis/
    ├── summary_by_config.csv
    ├── alpha_star.csv
    ├── baselines.csv
    ├── report_tables.md
    ├── fig_win_vs_alpha.png
    ├── fig_win_vs_alpha.pdf
    ├── fig_sensitivity.png
    ├── fig_sensitivity.pdf
    ├── fig_heatmap_heuristic.png
    ├── fig_heatmap_heuristic.pdf
    ├── fig_heatmap_matched.png
    ├── fig_heatmap_matched.pdf
    ├── fig_seed_variability.png
    └── fig_seed_variability.pdf
```

`results.jsonl` stores completed experiment results. `baselines.json` stores the random and scripted Knight baseline evaluations. The `runs/` directory contains training artifacts such as checkpoints and logs generated during execution.

---

## 6. Results and Visualizations

The plots below summarize the completed 5×5 capability-grid experiment. Win rates and confidence intervals are aggregated across the five independent training seeds. In learned-Boss evaluations, the simultaneous regime's Boss was co-trained with its Knight; the alternating regime's Boss was trained against the frozen Knight.

### Win rate and training-regime comparison

![Knight win rate versus Boss capability multiplier](experiments_action_masked_alpha_fullgrid_v1/analysis/fig_win_vs_alpha.png)

*Diagonal win-rate curves compare fixed-opponent, simultaneous, and alternating learning. Panels distinguish heuristic and paired learned-Boss evaluations.*

### Capability heatmap

![Knight win-rate heatmap against paired learned Bosses](experiments_action_masked_alpha_fullgrid_v1/analysis/fig_heatmap_matched.png)

*The learned-opponent grid shows simultaneous and alternating performance. The paired Boss is co-trained in simultaneous learning and trained against the frozen Knight in alternating learning.*

### Health and damage sensitivity

![Knight win-rate sensitivity to Boss health and damage](experiments_action_masked_alpha_fullgrid_v1/analysis/fig_sensitivity.png)

*Health-only and damage-only slices show how changing each Boss capability affects Knight win rates across training regimes and opponent types.*

### Seed variability

![Variation in Knight win rates across training seeds](experiments_action_masked_alpha_fullgrid_v1/analysis/fig_seed_variability.png)

*Seed-level diagonal results show the mean and 95% Student-t interval across five training seeds, highlighting run-to-run variability.*

### Summary of findings

Some key findings from the completed experiment:

- Against the heuristic Boss, fixed-opponent training achieves an average Knight win rate of **86.0%** at the strongest tested diagonal configuration, `(1.5, 1.5)`.
- At parity, simultaneous learning achieves a **98.1%** mean Knight win rate against the heuristic Boss.
- At parity, alternating learning achieves a **91.4%** mean Knight win rate against the heuristic Boss.
- Against its paired co-trained Boss, simultaneous learning achieves a **71.8%** mean win rate at parity, with a wide 95% confidence interval of 30.9–100.0%.
- Against the Boss trained against its frozen Knight, alternating learning achieves a **32.7%** mean win rate at parity, with a 95% confidence interval of 8.0–57.4%.
- At `(1.5, 1.5)`, the paired learned-Boss win rates fall to 3.6% for simultaneous learning and 1.2% for alternating learning.

These results suggest that an underdog agent can learn to win under certain conditions. They do not establish that it can reliably defeat arbitrary or freshly optimized opponents.

The experiment has important limitations, including five training seeds, substantial seed variability, unequal training budgets, a one-switch alternating schedule, and the absence of a fresh best-response Boss evaluation.

---

## 7. Tests

Run the test suite from the project root:

```bash
python -m pytest -q
```

The tests cover the game environment, agent behavior, PPO training components, training regimes, asymmetric capability handling, evaluation, statistics, and supporting experiment functionality.

---

## 8. Repository Structure

```text
.
├── configs/
│   ├── game.yaml
│   └── ppo.yaml
├── docs/
│   ├── final_report.tex
│   ├── final_report.pdf
│   └── PRIMEarxiv.sty
├── experiments_action_masked_alpha_fullgrid_v1/
│   ├── results.jsonl
│   ├── baselines.json
│   └── analysis/
├── scripts/
│   ├── run_proposal_experiments.py
│   └── analyze_proposal_results.py
├── src/
│   └── knight_boss/
├── tests/
├── pyproject.toml
└── README.md
```

---

## 9. Reproducibility Notes

The repository preserves the final experiment's source code, configuration files, aggregate JSONL results, baseline evaluations, statistical summaries, plots, and research report.

Individual training checkpoints and the full `runs/` directories are not included in the repository. Running the experiment commands therefore retrains the policies and regenerates the results rather than restoring the original trained checkpoints.

The dependency declarations in `pyproject.toml` specify minimum supported versions rather than a fully pinned environment. Differences in dependency versions or execution environments may therefore introduce small numerical variations.

The runner uses resumable job keys. For a clean reproduction, use a fresh output directory and retain the same experiment parameters across all three commands.

---

## 10. Final Research Report

The complete research report contains the mathematical formulation, environment design, training methodology, experimental protocol, statistical analysis, results, limitations, and team contributions.

- [Final Research Report (PDF)](docs/final_report.pdf)
- [LaTeX Source (Overleaf)](https://www.overleaf.com/project/6b01b486a5a3f1c8adf6f75a/share#f56d935df381ccd38b1b54ebb249bba4905526b2d3dac873)
- [LaTeX Source (GitHub)](docs/final_report.tex)
- [LaTeX Style File](docs/PRIMEarxiv.sty)

---

# 11. Acknowledgements

This project was developed as part of the course **Foundations of AI: Multi-Agent Systems (CS 329)** under the guidance of **Prof. Manisha Padala**.

We sincerely thank the professor for her guidance, valuable feedback, and support throughout the development of this project.

