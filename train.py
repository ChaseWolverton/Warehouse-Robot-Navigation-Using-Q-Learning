"""Train a sparse Q-learning agent and log per-episode metrics."""
import argparse
import csv
from pathlib import Path
import numpy as np
from warehouse_robot.agent import QLearningAgent
from warehouse_robot.environment import WarehouseEnv


def train(episodes=20000, seed=7, obstacle_probability=0.15, battery=30,
          output='results', shaping=True):
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    env = WarehouseEnv(max_battery=battery, obstacle_probability=obstacle_probability)
    agent = QLearningAgent(seed=seed)
    rows = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed * 1000000 + episode)
        epsilon = max(0.05, 1.0 - 0.95 * episode / max(1, int(episodes * 0.8)))
        total_reward = 0.0
        for step in range(env.max_steps):
            action = agent.act(obs, epsilon)
            nxt, reward, terminated, truncated, info = env.step(action)
            # Optional training-only pickup bonus to help sparse reward exploration.
            # Evaluation always uses the environment's unmodified reward.
            training_reward = reward
            if shaping and not bool(obs[3]) and bool(nxt[3]):
                training_reward += 8.0
            agent.update(obs, action, training_reward, nxt, terminated)
            total_reward += reward
            obs = nxt
            if terminated or truncated:
                break
        rows.append({'episode': episode + 1, 'reward': total_reward,
                     'success': int(info['success']), 'steps': info['steps'],
                     'battery': info['battery'], 'reason': info['reason'],
                     'epsilon': epsilon})
        if (episode + 1) % 1000 == 0:
            recent = rows[-1000:]
            print(f"Episode {episode+1:>6}: success={np.mean([r['success'] for r in recent]):.1%} "
                  f"mean_reward={np.mean([r['reward'] for r in recent]):.2f} "
                  f"Q_states={len(agent.q)}", flush=True)
    csv_path = folder / f'train_seed{seed}.csv'
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    agent.save(folder / f'agent_seed{seed}.npz')
    print('Saved:', csv_path)
    return agent

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=20000)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--obstacle-probability', type=float, default=0.15)
    p.add_argument('--battery', type=int, default=30)
    p.add_argument('--output', default='results')
    p.add_argument('--no-shaping', action='store_true')
    a = p.parse_args()
    train(a.episodes, a.seed, a.obstacle_probability, a.battery, a.output, not a.no_shaping)
