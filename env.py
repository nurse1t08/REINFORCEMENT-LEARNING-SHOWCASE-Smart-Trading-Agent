import numpy as np


GRID_SIZE = 6
START_POS = (0, 0)
SHOP_LABELS = ["Tech Hub", "Outlet", "Fashion Spot", "Gadget Box"]
SHOP_COUNT = 4
BUY_PRICE_VALUES = [3, 4, 5, 6, 7, 8, 9, 10, 11]
MARKET_PRICE_VALUES = [10, 11, 12, 13, 14, 15, 16, 17, 18]
BALANCE_VALUES = [18, 20, 22, 24, 26, 28, 30, 34, 38, 42]
SHOP_STOCK_VALUES = [1, 2, 3]
MAX_STEPS = 48
MAX_INVENTORY = 4
ACTION_NAMES = ["Up", "Down", "Left", "Right", "Buy", "Sell"]
PRICE_STATE_COUNT = 13
BUDGET_BUCKETS = 6
TARGET_DIRECTIONS = 6
INVENTORY_BUCKETS = MAX_INVENTORY + 1
PREV_MOVE_STATES = 5
VISITED_MASK_STATES = 2 ** SHOP_COUNT


def create_env():
    env = {
        "grid_size": GRID_SIZE,
        "start": START_POS,
        "mall_name": "Orchid Trade Center",
    }
    randomize_layout(env)
    return env


def randomize_layout(env):
    cells = [(row, col) for row in range(env["grid_size"]) for col in range(env["grid_size"])]
    cells = [cell for cell in cells if cell != env["start"]]
    order = np.random.permutation(len(cells))
    market = cells[int(order[0])]
    shop_positions = [cells[int(i)] for i in order[1 : SHOP_COUNT + 1]]
    shop_prices = {shop: int(np.random.choice(BUY_PRICE_VALUES)) for shop in shop_positions}
    shop_stock = {shop: int(np.random.choice(SHOP_STOCK_VALUES)) for shop in shop_positions}
    market_price = int(max(np.random.choice(MARKET_PRICE_VALUES), min(shop_prices.values()) + 2))

    env["market"] = market
    env["market_price"] = market_price
    env["shops"] = shop_prices
    env["shop_stock"] = shop_stock
    env["shop_names"] = {shop_positions[i]: SHOP_LABELS[i] for i in range(SHOP_COUNT)}
    env["shop_order"] = shop_positions
    env["shop_index"] = {shop_positions[i]: i for i in range(SHOP_COUNT)}
    env["shop_codes"] = {shop_positions[i]: f"S{i + 1}" for i in range(SHOP_COUNT)}


def reset_env(env):
    randomize_layout(env)
    balance = int(np.random.choice(BALANCE_VALUES))
    return {
        "position": env["start"],
        "balance": balance,
        "start_balance": balance,
        "inventory_count": 0,
        "inventory_cost": 0,
        "profit": 0,
        "sales": 0,
        "best_trade": 0,
        "last_trade": 0,
        "last_sale_units": 0,
        "last_sale_revenue": 0,
        "steps": 0,
        "done": False,
        "last_action": "Start",
        "last_reward": 0,
        "message": "Ready to explore the mall",
        "shop_price": get_shop_price(env, env["start"]),
        "trail": [env["start"]],
        "last_move": 4,
        "visited_shops": set(),
    }


def get_shop_price(env, position):
    return int(env["shops"].get(position, 0))


def get_budget_bucket(balance):
    return min(balance // 8, BUDGET_BUCKETS - 1)


def get_remaining_space(state):
    return max(0, MAX_INVENTORY - state.get("inventory_count", 0))


def direction_to(source, target):
    if target is None:
        return 5
    if source == target:
        return 4

    row_diff = target[0] - source[0]
    col_diff = target[1] - source[1]

    if abs(row_diff) >= abs(col_diff):
        return 1 if row_diff > 0 else 0
    return 3 if col_diff > 0 else 2


def choose_target_shop(env, state):
    candidates = []
    remaining_space = get_remaining_space(state)

    if remaining_space <= 0:
        return None

    for shop, price in env["shops"].items():
        if shop in state.get("visited_shops", set()):
            continue
        max_units = min(env["shop_stock"][shop], remaining_space, state["balance"] // price)
        if max_units > 0 and env["market_price"] > price:
            distance = abs(shop[0] - state["position"][0]) + abs(shop[1] - state["position"][1])
            candidates.append((price, distance, shop))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1], item[2][0], item[2][1]))
    return candidates[0][2]


def should_sell(env, state):
    return state.get("inventory_count", 0) > 0 and choose_target_shop(env, state) is None


def is_reverse_move(previous_move, action):
    return (
        (previous_move == 0 and action == 1)
        or (previous_move == 1 and action == 0)
        or (previous_move == 2 and action == 3)
        or (previous_move == 3 and action == 2)
    )


def get_visited_mask(env, state):
    mask = 0

    for shop in state.get("visited_shops", set()):
        index = env["shop_index"].get(shop)
        if index is not None:
            mask |= 1 << index

    return mask


def get_state_id(env, state):
    row, col = state["position"]
    position_index = row * env["grid_size"] + col
    budget_bucket = get_budget_bucket(state.get("balance", 0))
    target = env["market"] if should_sell(env, state) else choose_target_shop(env, state)
    target_direction = direction_to(state["position"], target)
    inventory_bucket = min(state.get("inventory_count", 0), MAX_INVENTORY)
    previous_move = min(state.get("last_move", 4), PREV_MOVE_STATES - 1)
    visited_mask = get_visited_mask(env, state)

    return (
        ((((position_index * BUDGET_BUCKETS + budget_bucket) * TARGET_DIRECTIONS + target_direction) * INVENTORY_BUCKETS + inventory_bucket) * PREV_MOVE_STATES + previous_move)
        * VISITED_MASK_STATES
        + visited_mask
    )


def move(position, action, grid_size):
    row, col = position

    if action == 0:
        row = max(0, row - 1)
    elif action == 1:
        row = min(grid_size - 1, row + 1)
    elif action == 2:
        col = max(0, col - 1)
    elif action == 3:
        col = min(grid_size - 1, col + 1)

    return row, col


def visit_shop(env, state, reward):
    position = state["position"]

    if position not in env["shops"] or position in state.get("visited_shops", set()):
        return reward, False

    state["visited_shops"].add(position)
    price = env["shops"][position]
    stock = env["shop_stock"][position]
    code = env["shop_codes"][position]
    free_space = get_remaining_space(state)
    quantity = min(stock, free_space, state["balance"] // price)

    if env["market_price"] <= price:
        reward -= 2.5
        state["message"] = f"Visited {code} but skipped it because the price is too high"
        return reward, True

    if quantity <= 0:
        reward -= 2.5
        state["message"] = f"Visited {code} but could not afford its batch"
        return reward, True

    total_cost = quantity * price
    state["inventory_count"] = state["inventory_count"] + quantity
    state["inventory_cost"] = state["inventory_cost"] + total_cost
    state["balance"] = state["balance"] - total_cost
    state["message"] = f"Bought {quantity} items at {code} for ${total_cost}"
    reward += quantity + (env["market_price"] - price) * quantity
    return reward, True


def sell_inventory(env, state, reward):
    if state["position"] != env["market"] or state["inventory_count"] <= 0 or not should_sell(env, state):
        return reward, False

    units = state["inventory_count"]
    revenue = units * env["market_price"]
    profit = revenue - state["inventory_cost"]
    state["balance"] = state["balance"] + revenue
    state["profit"] = state["profit"] + profit
    state["sales"] = state["sales"] + 1
    state["best_trade"] = max(state["best_trade"], profit)
    state["last_trade"] = profit
    state["last_sale_units"] = units
    state["last_sale_revenue"] = revenue
    state["inventory_count"] = 0
    state["inventory_cost"] = 0
    state["done"] = True
    state["message"] = f"Sold {units} items for ${revenue} | Profit ${profit}"
    reward += profit + units
    return reward, True


def apply_action(env, state, action):
    new_state = dict(state)
    new_state["trail"] = list(state["trail"])
    new_state["visited_shops"] = set(state.get("visited_shops", set()))
    reward = -1
    position = state["position"]
    new_state["last_move"] = state.get("last_move", 4)
    event_happened = False

    if action < 4:
        position = move(position, action, env["grid_size"])
        if is_reverse_move(state.get("last_move", 4), action):
            reward -= 1.5
        recent_trail = state["trail"][-4:]
        if position in recent_trail:
            reward -= 0.8
        new_state["last_move"] = action
        new_state["position"] = position
        reward, event_happened = visit_shop(env, new_state, reward)
        if not event_happened:
            reward, event_happened = sell_inventory(env, new_state, reward)
        if not event_happened:
            new_state["message"] = ACTION_NAMES[action]

    if action == 4:
        new_state["position"] = position
        reward, event_happened = visit_shop(env, new_state, reward)
        if not event_happened and state["position"] in env["shops"] and state["position"] in state.get("visited_shops", set()):
            reward -= 3
            new_state["message"] = "This shop was already visited"
        else:
            if not event_happened:
                reward -= 4
                new_state["message"] = "No new shop batch here"

    elif action == 5:
        new_state["position"] = position
        reward, event_happened = sell_inventory(env, new_state, reward)
        if event_happened:
            pass
        elif state["position"] != env["market"]:
            reward -= 4
            new_state["message"] = "Not at market"
        elif state["inventory_count"] <= 0:
            reward -= 4
            new_state["message"] = "No items to sell"
        else:
            reward -= 3
            new_state["message"] = "Keep buying until no profitable shop remains"

    new_state["position"] = position
    if not new_state["trail"] or new_state["trail"][-1] != position:
        new_state["trail"].append(position)
    if len(new_state["trail"]) > 14:
        new_state["trail"] = new_state["trail"][-14:]

    new_state["steps"] = state["steps"] + 1
    new_state["done"] = new_state.get("done", False) or new_state["steps"] >= MAX_STEPS
    new_state["last_action"] = ACTION_NAMES[action]
    new_state["last_reward"] = reward
    new_state["shop_price"] = get_shop_price(env, new_state["position"])

    if not new_state["done"] and new_state["inventory_count"] == 0 and choose_target_shop(env, new_state) is None:
        new_state["done"] = True
        new_state["message"] = "No affordable profitable shops left"
    elif new_state["done"] and new_state["steps"] >= MAX_STEPS and new_state["inventory_count"] > 0:
        new_state["message"] = "Out of time before selling"
    elif new_state["done"] and new_state["steps"] >= MAX_STEPS:
        new_state["message"] = "Max steps reached"

    return new_state, reward, new_state["done"]
