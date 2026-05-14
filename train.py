import numpy as np

from agent import ACTION_COUNT, choose_action, update_q
from env import (
    BUDGET_BUCKETS,
    GRID_SIZE,
    INVENTORY_BUCKETS,
    PREV_MOVE_STATES,
    TARGET_DIRECTIONS,
    VISITED_MASK_STATES,
    apply_action,
    choose_target_shop,
    create_env,
    get_state_id,
    reset_env,
    should_sell,
)


MODEL_VERSION = "smart-trading-agent-v7"
STATE_COUNT = GRID_SIZE * GRID_SIZE * BUDGET_BUCKETS * TARGET_DIRECTIONS * INVENTORY_BUCKETS * PREV_MOVE_STATES * VISITED_MASK_STATES


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def train(episodes=2600, alpha=0.18, gamma=0.96, epsilon=1.0):
    env = create_env()
    q_table = np.zeros((STATE_COUNT, ACTION_COUNT))
    rewards = []
    epsilon_value = epsilon

    for _ in range(episodes):
        state = reset_env(env)
        total_reward = 0

        while not state["done"]:
            state_id = get_state_id(env, state)
            action = choose_action(q_table, state_id, epsilon_value)
            next_state, reward, done = apply_action(env, state, action)
            next_state_id = get_state_id(env, next_state)
            shaped_reward = reward

            target_shop = choose_target_shop(env, state)

            if not should_sell(env, state) and target_shop is not None:
                before = manhattan(state["position"], target_shop)
                after = manhattan(next_state["position"], target_shop)
                shaped_reward += (before - after) * 0.25

            if should_sell(env, state):
                before = manhattan(state["position"], env["market"])
                after = manhattan(next_state["position"], env["market"])
                shaped_reward += (before - after) * 0.28

            if next_state["inventory_count"] > state["inventory_count"]:
                units = next_state["inventory_count"] - state["inventory_count"]
                avg_price = (next_state["inventory_cost"] - state["inventory_cost"]) / max(units, 1)
                potential_profit = (env["market_price"] - avg_price) * units
                shaped_reward += potential_profit * 0.65
                shaped_reward += len(next_state["visited_shops"]) - len(state["visited_shops"])

            if next_state["profit"] > state["profit"]:
                batch_profit = next_state["profit"] - state["profit"]
                shaped_reward += batch_profit * 1.45 + next_state["last_sale_units"] * 0.4

            if action < 4 and next_state["position"] == state["position"]:
                shaped_reward -= 0.5

            if action == 4 and next_state["inventory_count"] == state["inventory_count"]:
                shaped_reward -= 1.2

            if len(next_state["visited_shops"]) > len(state["visited_shops"]) and next_state["inventory_count"] == state["inventory_count"]:
                shaped_reward -= 0.8

            if done and next_state["profit"] == state["profit"]:
                shaped_reward -= 4

            update_q(q_table, state_id, action, shaped_reward, next_state_id, alpha, gamma)
            total_reward += shaped_reward
            state = next_state

            if done:
                break

        rewards.append(total_reward)
        epsilon_value = max(0.01, epsilon_value * 0.995)

    return q_table, rewards
