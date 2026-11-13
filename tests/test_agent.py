import numpy as np
from warehouse_robot.agent import QLearningAgent

def test_terminal_update():
    agent = QLearningAgent(alpha=0.5, gamma=0.9)
    state = np.array([5,0,30,0,2,1,0,0])
    agent.update(state, 1, 10.0, state, True)
    assert agent.q[agent.key(state)][1] == 5.0

def test_bootstrap_update():
    agent = QLearningAgent(alpha=1.0, gamma=0.9)
    s = np.array([5,0,30,0,2,1,0,0]); t = s.copy(); t[0] = 4
    agent.q[agent.key(t)][2] = 8.0
    agent.update(s, 1, -1.0, t, False)
    assert np.isclose(agent.q[agent.key(s)][1], 6.2)

def test_valid_action():
    agent = QLearningAgent()
    s = np.array([5,0,30,0,2,1,0,0])
    assert 0 <= agent.act(s, epsilon=1.0) < 5
