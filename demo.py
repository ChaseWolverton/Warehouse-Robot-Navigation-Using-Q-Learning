"""Reproducible demonstration and sample-output generator (no RL training)."""
from pathlib import Path
from warehouse_robot import WarehouseEnv


def main():
    env = WarehouseEnv(max_battery=30, obstacle_probability=0.15)
    obs, info = env.reset(seed=7)
    lines = ["RESET", f"Observation: {obs.tolist()}", env.render()]
    # Show deterministic mechanics with no temporary obstacles for the route demo.
    env = WarehouseEnv(max_battery=30, obstacle_probability=0)
    obs, _ = env.reset(seed=7)
    lines += ["", "DETERMINISTIC ROUTE DEMO (obstacle_probability=0)"]
    route = [0, 0, 0, 3, 3, 3, 3, 3, 0, 0]
    for i, action in enumerate(route, 1):
        obs, reward, terminated, truncated, info = env.step(action)
        lines.append(f"Step {i:02d}: {env.ACTION_NAMES[action]:8} pos={env.position} "
                     f"battery={env.battery:2d} carrying={env.carrying} "
                     f"reward={reward:5.1f} reason={info['reason']}")
        if i in (4, len(route)):
            lines.append(env.render())
        if terminated or truncated:
            break
    lines.append(f"Delivery success: {info['success']}")
    output = "\n".join(lines) + "\n"
    print(output)
    Path("figures").mkdir(exist_ok=True)
    Path("figures/sample_output.txt").write_text(output)


if __name__ == "__main__":
    main()
