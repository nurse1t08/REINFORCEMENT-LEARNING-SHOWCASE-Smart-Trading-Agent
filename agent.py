import numpy as np


ACTION_COUNT = 6


def choose_action(q_table, state_id, epsilon):
    if np.random.rand() < epsilon:
        return int(np.random.randint(0, ACTION_COUNT))
    values = q_table[state_id]
    best_value = np.max(values)
    best_actions = np.flatnonzero(np.isclose(values, best_value))
    return int(np.random.choice(best_actions))


def update_q(q_table, state_id, action, reward, next_state_id, alpha, gamma):
    q_table[state_id, action] = q_table[state_id, action] + alpha * (
        reward + gamma * np.max(q_table[next_state_id]) - q_table[state_id, action]
    )
