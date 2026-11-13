"""Evaluate trained, random, and shortest-path heuristic policies."""
import argparse
from collections import deque
import csv
from pathlib import Path
import numpy as np
from warehouse_robot.agent import QLearningAgent
from warehouse_robot.environment import WarehouseEnv


def heuristic_action(env):
    """BFS to pickup/drop-off, detouring around currently observed obstacles."""
    if env.battery < 8 and env.position == env.charger:
        return 4
    target = env.delivery if env.carrying else env.package
    if env.battery < 8 and not env.carrying:
        target = env.charger
    queue = deque([(env.position, None)])
    seen = {env.position}
    while queue:
        position, first = queue.popleft()
        if position == target:
            return 4 if first is None and position == env.charger else (first if first is not None else 0)
        for action, (dr, dc) in env.ACTIONS.items():
            nxt = (position[0] + dr, position[1] + dc)
            if 0 <= nxt[0] < 6 and 0 <= nxt[1] < 6 and nxt not in env.walls and nxt not in env.obstacles and nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, action if first is None else first))
    return 4 if env.position == env.charger else 0


def evaluate(episodes=100, seed=2026, output='results', battery=30,
             obstacle_probability=0.15, agent_seed=7):
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    agent = QLearningAgent(seed=seed)
    agent.load(folder / f'agent_seed{agent_seed}.npz')
    rows = []
    for policy in ('q_learning', 'random', 'heuristic'):
        env = WarehouseEnv(max_battery=battery, obstacle_probability=obstacle_probability)
        rng = np.random.default_rng(seed + 99)
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)  # identical environment seeds for each policy
            total = 0.0
            while True:
                if policy == 'q_learning':
                    action = agent.act(obs, epsilon=0.0)
                elif policy == 'random':
                    action = int(rng.integers(env.action_space.n))
                else:
                    action = heuristic_action(env)
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                if terminated or truncated:
                    break
            rows.append({'policy': policy, 'episode': episode + 1, 'reward': total,
                         'success': int(info['success']), 'steps': info['steps'],
                         'battery': info['battery'], 'reason': info['reason']})
        subset = rows[-episodes:]
        successful = [r for r in subset if r['success']]
        print(f"{policy:>11}: success={np.mean([r['success'] for r in subset]):.1%}, "
              f"mean_reward={np.mean([r['reward'] for r in subset]):.2f}, "
              f"mean_steps={np.mean([r['steps'] for r in subset]):.1f}, "
              f"steps_if_success={np.mean([r['steps'] for r in successful]):.1f}" if successful else
              f"{policy:>11}: success=0.0%, mean_reward={np.mean([r['reward'] for r in subset]):.2f}, no successful deliveries")
    path = folder / f'evaluation_seed{agent_seed}.csv'
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('Saved:', path)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--agent-seed', type=int, default=7)
    p.add_argument('--battery', type=int, default=30)
    p.add_argument('--obstacle-probability', type=float, default=0.15)
    p.add_argument('--output', default='results')
    a = p.parse_args()
    evaluate(a.episodes, a.seed, a.output, a.battery, a.obstacle_probability, a.agent_seed)
