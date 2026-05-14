import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

from env import MAX_INVENTORY, get_state_id


def get_palette(dark):
    if dark:
        return {
            "figure_bg": "#08111f",
            "card_bg": "#101b2d",
            "cell_bg": "#162235",
            "shop_bg": "#184934",
            "market_bg": "#6b4f11",
            "cell_edge": "#22314a",
            "trail": "#60a5fa",
            "shop_text": "#d1fae5",
            "market_text": "#fde68a",
            "title": "#f8fafc",
            "subtle": "#94a3b8",
            "legend": "#cbd5e1",
            "agent": "#3b82f6",
        }
    return {
        "figure_bg": "#f5efe6",
        "card_bg": "#fffaf2",
        "cell_bg": "#f5efe6",
        "shop_bg": "#cbe9cf",
        "market_bg": "#f9df87",
        "cell_edge": "#d4c9b9",
        "trail": "#7c9fdc",
        "shop_text": "#14532d",
        "market_text": "#854d0e",
        "title": "#0f172a",
        "subtle": "#475569",
        "legend": "#334155",
        "agent": "#2563eb",
    }


def render_grid(env, state, dark=False):
    palette = get_palette(dark)
    cell = 78
    board_left = 34
    board_top = 92
    board_size = env["grid_size"] * cell
    total_width = board_left * 2 + board_size
    total_height = board_top + board_size + 102

    fig, ax = plt.subplots(figsize=(7.2, 7.9))
    fig.patch.set_facecolor(palette["figure_bg"])
    ax.set_facecolor(palette["figure_bg"])
    ax.set_xlim(0, total_width)
    ax.set_ylim(total_height, 0)
    ax.axis("off")

    card = FancyBboxPatch(
        (14, 14),
        total_width - 28,
        total_height - 28,
        boxstyle="round,pad=0.02,rounding_size=26",
        linewidth=0,
        facecolor=palette["card_bg"],
    )
    ax.add_patch(card)

    shops = env["shops"]
    market = env["market"]
    visited_shops = state.get("visited_shops", set())

    for row in range(env["grid_size"]):
        for col in range(env["grid_size"]):
            x = board_left + col * cell
            y = board_top + row * cell
            color = palette["cell_bg"]
            if (row, col) in shops:
                color = palette["shop_bg"]
            if (row, col) == market:
                color = palette["market_bg"]
            alpha = 0.45 if (row, col) in visited_shops else 1.0
            rect = FancyBboxPatch(
                (x + 4, y + 4),
                cell - 8,
                cell - 8,
                boxstyle="round,pad=0.02,rounding_size=12",
                linewidth=1.5,
                edgecolor=palette["cell_edge"],
                facecolor=color,
                alpha=alpha,
            )
            ax.add_patch(rect)

    trail = state.get("trail", [])
    if len(trail) > 1:
        xs = [board_left + col * cell + cell / 2 for _, col in trail]
        ys = [board_top + row * cell + cell / 2 for row, _ in trail]
        ax.plot(xs, ys, color=palette["trail"], linewidth=3, alpha=0.55, solid_capstyle="round")

    shop_codes = {}
    for index, shop in enumerate(env["shop_order"], start=1):
        shop_codes[shop] = f"S{index}"

    for shop in env["shop_order"]:
        row, col = shop
        x0 = board_left + col * cell
        y0 = board_top + row * cell
        x = x0 + cell / 2
        y = y0 + cell / 2
        badge_color = palette["shop_text"] if shop not in visited_shops else palette["cell_edge"]
        ax.add_patch(
            FancyBboxPatch(
                (x0 + 8, y0 + 8),
                28,
                20,
                boxstyle="round,pad=0.02,rounding_size=6",
                linewidth=0,
                facecolor=badge_color,
            )
        )
        ax.text(x0 + 22, y0 + 18, shop_codes[shop], ha="center", va="center", color=palette["card_bg"], fontsize=8.5, fontweight="bold")
        stock = env["shop_stock"][shop]
        label = f"${shops[shop]} x{stock}"
        if shop in visited_shops:
            label = "USED"
        ax.text(x, y + 10, label, ha="center", va="center", color=palette["shop_text"], fontsize=11.5, fontweight="bold")

    mx0 = board_left + market[1] * cell
    my0 = board_top + market[0] * cell
    mx = mx0 + cell / 2
    my = my0 + cell / 2
    ax.add_patch(
        FancyBboxPatch(
            (mx0 + 8, my0 + 8),
            34,
            20,
            boxstyle="round,pad=0.02,rounding_size=6",
            linewidth=0,
            facecolor=palette["market_text"],
        )
    )
    ax.text(mx0 + 25, my0 + 18, "MKT", ha="center", va="center", color=palette["card_bg"], fontsize=8.5, fontweight="bold")
    ax.text(mx, my + 10, f"${env['market_price']}", ha="center", va="center", color=palette["market_text"], fontsize=14, fontweight="bold")

    row, col = state["position"]
    x = board_left + col * cell + cell / 2
    y = board_top + row * cell + cell / 2
    agent = Circle((x, y), 21, facecolor=palette["agent"], edgecolor="white", linewidth=2.5, zorder=5)
    ax.add_patch(agent)
    ax.text(x, y, "A", ha="center", va="center", color="white", fontsize=15, fontweight="bold", zorder=6)

    ax.text(34, 44, env["mall_name"], color=palette["title"], fontsize=18, fontweight="bold")
    ax.text(
        34,
        66,
        f"Balance ${state['balance']}   •   Profit ${state['profit']}   •   Items {state['inventory_count']}/{MAX_INVENTORY}   •   Visited {len(visited_shops)}/{len(env['shop_order'])}",
        color=palette["subtle"],
        fontsize=10.5,
    )

    legend_y = board_top + board_size + 42
    ax.text(34, legend_y, "Legend", color=palette["title"], fontsize=11, fontweight="bold")
    ax.add_patch(Rectangle((102, legend_y - 12), 18, 18, facecolor=palette["shop_bg"], edgecolor="none"))
    ax.text(128, legend_y + 1, "Shop", va="center", color=palette["legend"], fontsize=10)
    ax.add_patch(Rectangle((188, legend_y - 12), 18, 18, facecolor=palette["market_bg"], edgecolor="none"))
    ax.text(214, legend_y + 1, "Market", va="center", color=palette["legend"], fontsize=10)
    ax.add_patch(Circle((307, legend_y - 3), 9, facecolor=palette["agent"], edgecolor="none"))
    ax.text(328, legend_y + 1, "Agent", va="center", color=palette["legend"], fontsize=10)
    ax.text(392, legend_y + 1, "Visited shops fade after one use", va="center", color=palette["subtle"], fontsize=9.2)

    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)[..., :3]
    plt.close(fig)
    return array


def render_policy_map(q_table, env, inventory_count, balance, dark=False, visited_shops=None, last_move=4):
    palette = get_palette(dark)
    labels = {0: "↑", 1: "↓", 2: "←", 3: "→", 4: "B", 5: "S"}
    values = np.zeros((env["grid_size"], env["grid_size"]))
    actions = np.zeros((env["grid_size"], env["grid_size"]), dtype=int)
    visited = set() if visited_shops is None else set(visited_shops)

    for row in range(env["grid_size"]):
        for col in range(env["grid_size"]):
            state = {
                "position": (row, col),
                "balance": balance,
                "inventory_count": inventory_count,
                "visited_shops": visited,
                "last_move": last_move,
            }
            state_id = get_state_id(env, state)
            values[row, col] = np.max(q_table[state_id])
            actions[row, col] = int(np.argmax(q_table[state_id]))

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    fig.patch.set_facecolor(palette["card_bg"])
    ax.set_facecolor(palette["card_bg"])
    ax.imshow(values, cmap="YlGnBu" if not dark else "viridis")
    ax.set_xticks([])
    ax.set_yticks([])

    for row in range(env["grid_size"]):
        for col in range(env["grid_size"]):
            color = "#ffffff" if values[row, col] > np.mean(values) else palette["title"]
            ax.text(col, row, labels[actions[row, col]], ha="center", va="center", fontsize=16, fontweight="bold", color=color)

    for shop in env["shops"]:
        ax.add_patch(Rectangle((shop[1] - 0.5, shop[0] - 0.5), 1, 1, fill=False, edgecolor=palette["shop_text"], linewidth=2))
    ax.add_patch(Rectangle((env["market"][1] - 0.5, env["market"][0] - 0.5), 1, 1, fill=False, edgecolor=palette["market_text"], linewidth=2.4))
    title = "Policy Without Inventory" if inventory_count == 0 else "Policy With Inventory"
    ax.set_title(title, fontsize=13, color=palette["title"])
    plt.tight_layout()

    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)[..., :3]
    plt.close(fig)
    return array
