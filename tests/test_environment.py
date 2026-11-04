import numpy as np
import pytest
from warehouse_robot import WarehouseEnv


def test_reset_reproducible_and_valid():
    env = WarehouseEnv(obstacle_probability=0.5)
    a, _ = env.reset(seed=7)
    b, _ = env.reset(seed=7)
    assert np.array_equal(a, b)
    assert env.observation_space.contains(a)
    assert tuple(a[:2]) == (5, 0)
    assert a[2] == 12 and a[3] == 0


def test_move_and_blocked_move():
    env = WarehouseEnv(obstacle_probability=0)
    env.reset(seed=1)
    obs, reward, term, trunc, _ = env.step(3)
    assert tuple(obs[:2]) == (5, 1) and obs[2] == 11 and reward == -1
    obs, reward, _, _, _ = env.step(1)  # out of bounds
    assert tuple(obs[:2]) == (5, 1) and obs[2] == 10 and reward == -3


def test_charging_and_invalid_recharge():
    env = WarehouseEnv(obstacle_probability=0)
    env.reset(seed=1)
    _, reward, _, _, _ = env.step(4)
    assert reward == -3 and env.battery == 12
    env.step(3)
    env.step(3)
    assert env.position == env.charger and env.battery == 10
    _, reward, _, _, _ = env.step(4)
    assert env.battery == 12 and reward == -1


def test_pickup_and_delivery():
    env = WarehouseEnv(max_battery=30, obstacle_probability=0)
    env.reset(seed=1)
    # (5,0) -> (2,1): up up up right; then route via row 2 to delivery
    for a in (0, 0, 0, 3):
        obs, reward, term, trunc, info = env.step(a)
    assert env.position == env.package and env.carrying
    # (2,1) -> (2,5) via (2,2),(2,3),(2,4),(2,5), then up up
    for a in (3, 3, 3, 3, 0, 0):
        obs, reward, term, trunc, info = env.step(a)
    assert term and not trunc and info["success"] and info["reason"] == "delivered"
    assert reward == 29


def test_battery_failure():
    env = WarehouseEnv(max_battery=2, obstacle_probability=0)
    env.reset(seed=1)
    env.step(0)
    _, reward, term, trunc, info = env.step(0)
    assert term and not trunc and reward == -6 and info["reason"] == "battery_empty"
    with pytest.raises(RuntimeError):
        env.step(0)


def test_time_limit():
    env = WarehouseEnv(max_steps=2, obstacle_probability=0)
    env.reset(seed=1)
    env.step(4)
    _, _, term, trunc, info = env.step(4)
    assert not term and trunc and info["reason"] == "time_limit"


def test_dynamic_obstacles_never_cover_special_cells():
    env = WarehouseEnv(obstacle_probability=1)
    env.reset(seed=1)
    assert env.obstacles == set(env.dynamic_cells)
    assert not env.obstacles.intersection({env.start, env.charger, env.delivery, env.package})
    assert env.observation_space.contains(env._observation())


def test_randomized_package():
    env = WarehouseEnv(randomize_package=True)
    a, _ = env.reset(seed=42)
    b, _ = env.reset(seed=42)
    assert np.array_equal(a, b)
    assert env.package not in env.walls
