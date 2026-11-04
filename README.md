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
