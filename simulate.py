import numpy as np

from agent import choose_action
from env import apply_action, choose_target_shop, get_state_id, is_reverse_move, should_sell


def step(env, agent, state):
    def greedy_move(source, target):
        if target is None:
            return 4
        row_diff = target[0] - source[0]
        col_diff = target[1] - source[1]
        if abs(row_diff) >= abs(col_diff):
            if row_diff > 0:
                return 1
            if row_diff < 0:
                return 0
        if col_diff > 0:
            return 3
        if col_diff < 0:
            return 2
        return 4

    state_id = get_state_id(env, state)
    q_values = agent["q_table"][state_id]
    action = choose_action(agent["q_table"], state_id, 0.0)
    target_shop = choose_target_shop(env, state)
    sell_mode = should_sell(env, state)

    if sell_mode and state["position"] == env["market"]:
        action = 5
    elif target_shop is not None and state["position"] == target_shop:
        action = 4

    if target_shop is None and not sell_mode and state["inventory_count"] == 0:
        done_state = dict(state)
        done_state["done"] = True
        done_state["message"] = "No affordable profitable shops left"
        done_state["last_action"] = "Wait"
        done_state["last_reward"] = 0
        return done_state

    if action < 4 and is_reverse_move(state.get("last_move", 4), action):
        best_value = np.max(q_values)
        ranked_actions = np.argsort(q_values)[::-1]
        for candidate in ranked_actions:
            if candidate == action:
                continue
            if candidate < 4 and is_reverse_move(state.get("last_move", 4), int(candidate)):
                continue
            if best_value - q_values[candidate] <= 0.15:
                action = int(candidate)
                break

    new_state, _, _ = apply_action(env, state, action)

    stuck = (
        new_state["position"] == state["position"]
        and new_state["inventory_count"] == state["inventory_count"]
        and new_state["profit"] == state["profit"]
    )

    if stuck:
        if sell_mode:
            if state["position"] == env["market"]:
                fallback_action = 5
            else:
                fallback_action = greedy_move(state["position"], env["market"])
        else:
            if target_shop is not None and state["position"] == target_shop:
                fallback_action = 4
            else:
                fallback_action = greedy_move(state["position"], target_shop)

        if fallback_action != action:
            new_state, _, _ = apply_action(env, state, fallback_action)

    return new_state
