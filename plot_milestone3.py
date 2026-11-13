"""Create Milestone 3 figures and a numeric summary from actual CSV files."""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('--seed', type=int, default=7)
p.add_argument('--output', default='results')
a = p.parse_args()
folder = Path(a.output)
figures = Path('figures')
figures.mkdir(exist_ok=True)

def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))

training = read_csv(folder / f'train_seed{a.seed}.csv')
window = 500
n = len(training) // window
x = np.arange(1, n+1) * window
fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
for ax, metric, title in [(axes[0], 'success', 'Training success rate'), (axes[1], 'reward', 'Training episode reward')]:
    y = [np.mean([float(r[metric]) for r in training[i*window:(i+1)*window]]) for i in range(n)]
    ax.plot(x, y, marker='o', markersize=2)
    ax.set_ylabel('Success fraction' if metric == 'success' else 'Mean reward')
    ax.set_title(title)
    ax.grid(alpha=.3)
axes[-1].set_xlabel('Training episode (end of 500-episode block)')
fig.tight_layout()
fig.savefig(figures / 'milestone3_training.png', dpi=160)
plt.close(fig)

rows = read_csv(folder / f'evaluation_seed{a.seed}.csv')
policies = ['q_learning', 'random', 'heuristic']
metrics = {}
for policy in policies:
    sub = [r for r in rows if r['policy'] == policy]
    successes = [r for r in sub if int(r['success'])]
    metrics[policy] = {
        'episodes': len(sub),
        'success_rate': np.mean([int(r['success']) for r in sub]),
        'mean_reward': np.mean([float(r['reward']) for r in sub]),
        'mean_steps': np.mean([int(r['steps']) for r in sub]),
        'steps_if_success': np.mean([int(r['steps']) for r in successes]) if successes else float('nan'),
        'battery_failures': sum(r['reason'] == 'battery_empty' for r in sub)
    }
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(policies, [metrics[p]['success_rate'] * 100 for p in policies])
ax.set_ylim(0, 100)
ax.set_ylabel('Successful deliveries (%)')
ax.set_title('Evaluation: greedy Q-learning vs. baselines')
for i, policy in enumerate(policies):
    ax.text(i, metrics[policy]['success_rate']*100+2, f"{metrics[policy]['success_rate']:.1%}", ha='center')
fig.tight_layout()
fig.savefig(figures / 'milestone3_baselines.png', dpi=160)
plt.close(fig)
with (folder / 'milestone3_summary.csv').open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['policy', *next(iter(metrics.values())).keys()])
    for policy in policies:
        w.writerow([policy, *metrics[policy].values()])
print('Last 500 training episodes: success =', np.mean([int(r['success']) for r in training[-500:]]))
for policy, data in metrics.items():
    print(policy, data)
print('Figures saved to figures/milestone3_training.png and figures/milestone3_baselines.png')
