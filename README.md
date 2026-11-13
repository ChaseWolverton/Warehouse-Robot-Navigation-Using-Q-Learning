# Warehouse Robot Navigation Using Q-Learning — Milestone 2

This milestone implements and tests a custom 6×6 warehouse environment. **No Q-learning agent is trained yet**; that is planned for the final project.

## Ubuntu setup

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip git
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run tests and demonstration

From the repository root:

```bash
python -m pytest tests/ -v
python demo.py
python make_figures.py
```

`demo.py` prints observations, positions, battery, rewards, and delivery status, and writes `figures/sample_output.txt`. `make_figures.py` creates `figures/environment.png` and `figures/code_excerpt.png`. Tests cover reset/seed, movement, blocked moves, charging, pickup/delivery, battery failure, time limit, and randomized obstacles/packages.

## Environment specification

- Grid: 6×6. Start (5,0), pickup (2,1), delivery (0,5), charger (5,2); fixed walls at (1,1), (1,2), (3,3), (4,3).
- Actions: `0=up`, `1=down`, `2=left`, `3=right`, `4=recharge`.
- Observation: `[row, col, battery, carrying, package_row, package_col, obstacle_at_2_3, obstacle_at_3_4]`.
- Battery: movement attempts cost one unit; recharge works only at the charger; a recharge attempt elsewhere is penalized.
- Obstacles: two candidate cells independently blocked with configurable probability after each nonterminal step. Never cover special cells or the robot's current cell.
- Rewards: −1 per step; −2 additional for blocked moves or recharge away from charger; +30 on delivery; −5 additional when battery is exhausted.
- Pickup occurs automatically upon entering the package cell. Delivery terminates the episode. Battery exhaustion terminates it unsuccessfully. A 100-step limit truncates the episode.
- `reset(seed=...)` makes stochastic behavior reproducible. `randomize_package=True` enables optional package randomization for future generalization experiments.

## Next milestone

Implement tabular Q-learning, random and shortest-path baselines, multiple-seed evaluation, and tests on changed obstacle configurations. Metrics: delivery rate, steps per successful delivery, battery remaining, and battery failures.

## Milestone 3 — Q-learning and evaluation

From the repository root, activate `.venv` and install requirements as above.

```bash
python -m pytest tests/ -v
python train.py --episodes 20000 --seed 7
python evaluate.py --episodes 100 --agent-seed 7
python plot_milestone3.py --seed 7
```

The training script uses sparse tabular Q-learning (alpha=0.15, gamma=0.97),
random tie-breaking, and epsilon decreasing linearly from 1.0 to 0.05 over
80% of training, then remaining at 0.05. It uses battery=30 and stochastic
obstacles (probability 0.15). Training includes an optional +8 pickup bonus
(`--no-shaping` disables it); logged rewards and evaluation use the original
environment reward. Evaluation uses fresh episodes and no Q updates, comparing
greedy Q-learning with random actions and a BFS heuristic. The heuristic is
provided as a reference, not as a learned policy.

Generated files: `results/train_seed7.csv`, `results/agent_seed7.npz`,
`results/evaluation_seed7.csv`, `results/milestone3_summary.csv`,
`figures/milestone3_training.png`, and `figures/milestone3_baselines.png`.

Run more seeds (e.g. 11 and 19) by changing `--seed` for training and
`--agent-seed` for evaluation. Do not claim successful learning without
checking the resulting CSV and plots. The saved Q-table is a generated artifact
and can be omitted from the GitHub repository if it is large.
