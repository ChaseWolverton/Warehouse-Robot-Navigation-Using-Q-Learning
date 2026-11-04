"""Generate reproducible evidence images for the milestone report."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from warehouse_robot import WarehouseEnv


def main():
    Path("figures").mkdir(exist_ok=True)
    env = WarehouseEnv(obstacle_probability=1)
    env.reset(seed=7)
    fig, ax = plt.subplots(figsize=(6.4, 5.7))
    for r in range(6):
        for c in range(6):
            cell = (r, c)
            color, label = "#f7f9fc", ""
            if cell in env.walls: color, label = "#374151", "WALL"
            elif cell in env.obstacles: color, label = "#f7c9c9", "TEMP"
            elif cell == env.charger: color, label = "#d3e8fc", "CHARGE"
            elif cell == env.delivery: color, label = "#d1f2d8", "DROP"
            elif cell == env.package: color, label = "#fff0b8", "PICKUP"
            elif cell == env.position: color, label = "#d9ccfb", "ROBOT"
            ax.add_patch(Rectangle((c, 5-r), 1, 1, facecolor=color, edgecolor="#aeb7c2"))
            ax.text(c+.5, 5-r+.5, label, ha="center", va="center", fontsize=8, weight="bold")
    ax.set(xlim=(0,6), ylim=(0,6), xticks=[i+.5 for i in range(6)], yticks=[i+.5 for i in range(6)])
    ax.set_xticklabels(range(6)); ax.set_yticklabels(range(5,-1,-1))
    ax.set_aspect("equal"); ax.set_title("6×6 warehouse: example obstacle configuration")
    fig.tight_layout(); fig.savefig("figures/environment.png", dpi=170); plt.close(fig)
    code = Path("warehouse_robot/environment.py").read_text().splitlines()
    start = next(i for i,s in enumerate(code) if s.startswith("    def step("))
    excerpt = code[start:start+29]
    fig, ax = plt.subplots(figsize=(11, 6.6))
    ax.axis("off")
    ax.text(.02, .98, "environment.py — step() transition and reward logic", transform=ax.transAxes,
            va="top", fontsize=13, weight="bold")
    ax.text(.02, .91, "\n".join(f"{start+i+1:3d}  {line}" for i,line in enumerate(excerpt)),
            transform=ax.transAxes, va="top", family="monospace", fontsize=9)
    fig.tight_layout(); fig.savefig("figures/code_excerpt.png", dpi=170); plt.close(fig)

if __name__ == "__main__":
    main()
