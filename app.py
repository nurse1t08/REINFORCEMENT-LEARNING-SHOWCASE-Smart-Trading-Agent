import os
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from agent import ACTION_COUNT
from env import create_env, reset_env
from simulate import step
from train import MODEL_VERSION, train
from utils import render_grid, render_policy_map


MODEL_PATH = "model.pkl"


def load_model():
    loaded = None
    if os.path.exists(MODEL_PATH):
        try:
            loaded = joblib.load(MODEL_PATH)
        except Exception:
            loaded = None

    valid = (
        isinstance(loaded, dict)
        and loaded.get("version") == MODEL_VERSION
        and loaded.get("q_table") is not None
        and loaded.get("rewards") is not None
        and loaded["q_table"].shape[1] == ACTION_COUNT
    )

    if valid:
        return loaded["q_table"], loaded["rewards"]

    q_table, rewards = train()
    joblib.dump({"version": MODEL_VERSION, "q_table": q_table, "rewards": rewards}, MODEL_PATH)
    return q_table, rewards


def reset_day(message):
    st.session_state.env = create_env()
    st.session_state.state = reset_env(st.session_state.env)
    st.session_state.activity = [message]


st.set_page_config(page_title="Smart Trading Agent", layout="wide")

if "theme" not in st.session_state:
    st.session_state.theme = "Light"
if "autoplay" not in st.session_state:
    st.session_state.autoplay = False

dark_mode = st.session_state.theme == "Dark"

theme = {
    "app_bg": """
        radial-gradient(circle at top left, rgba(93, 135, 221, 0.12), transparent 28%),
        radial-gradient(circle at top right, rgba(245, 178, 69, 0.16), transparent 30%),
        linear-gradient(180deg, #f7f2e9 0%, #efe6d6 100%)
    """ if not dark_mode else """
        radial-gradient(circle at top left, rgba(59, 130, 246, 0.16), transparent 28%),
        radial-gradient(circle at top right, rgba(20, 184, 166, 0.12), transparent 30%),
        linear-gradient(180deg, #07111f 0%, #0b1728 100%)
    """,
    "text": "#142133" if not dark_mode else "#e5eefc",
    "hero_bg": "linear-gradient(135deg, rgba(255,250,242,0.97), rgba(246,238,225,0.95))" if not dark_mode else "linear-gradient(135deg, rgba(16,27,45,0.98), rgba(10,18,33,0.96))",
    "hero_title": "#0f172a" if not dark_mode else "#f8fafc",
    "hero_text": "#475569" if not dark_mode else "#94a3b8",
    "panel_bg": "rgba(255, 250, 242, 0.82)" if not dark_mode else "rgba(16, 27, 45, 0.88)",
    "panel_border": "rgba(148, 163, 184, 0.14)" if not dark_mode else "rgba(96, 165, 250, 0.14)",
    "card_bg": "rgba(255, 250, 242, 0.92)" if not dark_mode else "rgba(15, 23, 42, 0.92)",
    "card_text": "#475569" if not dark_mode else "#9fb1c9",
    "metric_bg": "rgba(255, 250, 242, 0.84)" if not dark_mode else "rgba(15, 23, 42, 0.88)",
    "metric_label": "#64748b" if not dark_mode else "#8ea3bf",
    "metric_value": "#0f172a" if not dark_mode else "#f8fafc",
    "status_bg": "linear-gradient(135deg, rgba(37,99,235,0.08), rgba(15,118,110,0.08))" if not dark_mode else "linear-gradient(135deg, rgba(37,99,235,0.16), rgba(15,118,110,0.16))",
    "status_border": "rgba(37,99,235,0.12)" if not dark_mode else "rgba(96,165,250,0.20)",
    "status_label": "#1d4ed8" if not dark_mode else "#7dd3fc",
    "status_text": "#1f2937" if not dark_mode else "#dbe7f5",
    "log_text": "#334155" if not dark_mode else "#c6d4e6",
    "log_border": "rgba(148, 163, 184, 0.14)" if not dark_mode else "rgba(96, 165, 250, 0.12)",
}

st.markdown(
    f"""
    <style>
    .stApp {{
        background: {theme["app_bg"]};
        color: {theme["text"]};
    }}
    .block-container {{
        max-width: 1180px;
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
    }}
    .hero {{
        background: {theme["hero_bg"]};
        border: 1px solid {theme["panel_border"]};
        border-radius: 28px;
        padding: 28px 30px;
        box-shadow: 0 20px 44px rgba(71, 85, 105, 0.08);
        margin-bottom: 1rem;
    }}
    .hero-kicker {{
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        color: #b45309;
        font-weight: 800;
        margin-bottom: 0.8rem;
    }}
    .hero-title {{
        font-size: 2.8rem;
        line-height: 1.02;
        color: {theme["hero_title"]};
        font-weight: 900;
        margin: 0;
    }}
    .hero-text {{
        margin-top: 0.9rem;
        max-width: 760px;
        line-height: 1.7;
        color: {theme["hero_text"]};
        font-size: 1rem;
    }}
    .panel {{
        background: {theme["panel_bg"]};
        border: 1px solid {theme["panel_border"]};
        border-radius: 24px;
        padding: 18px 18px 8px 18px;
        box-shadow: 0 16px 38px rgba(71, 85, 105, 0.06);
    }}
    .directory-card {{
        background: {theme["card_bg"]};
        border: 1px solid {theme["panel_border"]};
        border-radius: 18px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }}
    .directory-title {{
        color: #14532d;
        font-size: 1rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }}
    .directory-text {{
        color: {theme["card_text"]};
        font-size: 0.93rem;
    }}
    .status-box {{
        background: {theme["status_bg"]};
        border: 1px solid {theme["status_border"]};
        border-radius: 20px;
        padding: 14px 16px;
        margin-top: 0.8rem;
        margin-bottom: 0.8rem;
    }}
    .status-label {{
        color: {theme["status_label"]};
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 800;
        font-size: 0.78rem;
        margin-bottom: 0.35rem;
    }}
    .status-text {{
        color: {theme["status_text"]};
        line-height: 1.6;
    }}
    [data-testid="stMetric"] {{
        background: {theme["metric_bg"]};
        border: 1px solid {theme["panel_border"]};
        border-radius: 18px;
        padding: 0.9rem 1rem;
    }}
    [data-testid="stMetricLabel"] {{
        color: {theme["metric_label"]};
    }}
    [data-testid="stMetricValue"] {{
        color: {theme["metric_value"]};
    }}
    div.stButton > button:first-child {{
        border-radius: 999px;
        border: none;
        background: linear-gradient(135deg, #2563eb, #0f766e);
        color: white;
        font-weight: 800;
        padding: 0.75rem 1rem;
        box-shadow: 0 10px 28px rgba(37, 99, 235, 0.18);
    }}
    div.stButton > button:first-child:hover {{
        color: white;
    }}
    .log-row {{
        color: {theme["log_text"]};
        padding: 0.35rem 0;
        border-bottom: 1px solid {theme["log_border"]};
        font-size: 0.95rem;
    }}
    h1, h2, h3, p, span, div, label {{
        color: {theme["text"]};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


if "env" not in st.session_state:
    q_table, rewards = load_model()
    st.session_state.agent = {"q_table": q_table}
    st.session_state.rewards = rewards
    st.session_state.env = create_env()
    st.session_state.state = reset_env(st.session_state.env)
    st.session_state.activity = ["Model loaded and ready"]
    st.session_state.autoplay = False


st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">Reinforcement Learning Showcase</div>
        <div class="hero-title">Smart Trading Agent</div>
        <div class="hero-text">
            A Q-learning courier moves through a premium mall, scouts the best store price, buys inventory, and races to the market to lock in profit.
            Each click advances the simulation and reveals how the learned policy behaves in a live trading day.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

theme_col1, theme_col2 = st.columns([0.82, 0.18])
with theme_col2:
    st.selectbox("Theme", ["Light", "Dark"], key="theme")

button_cols = st.columns(5)
train_clicked = button_cols[0].button("Train Fresh Model", use_container_width=True)
step_clicked = button_cols[1].button("Next Step", use_container_width=True)
fast_clicked = button_cols[2].button("Play 5 Steps", use_container_width=True)
auto_clicked = button_cols[3].button("Auto Play", use_container_width=True)
reset_clicked = button_cols[4].button("Reset Day", use_container_width=True)
stop_auto_clicked = st.button("Stop Auto", use_container_width=True)

if train_clicked:
    with st.spinner("Training a sharper policy..."):
        q_table, rewards = train()
        joblib.dump({"version": MODEL_VERSION, "q_table": q_table, "rewards": rewards}, MODEL_PATH)
    st.session_state.agent = {"q_table": q_table}
    st.session_state.rewards = rewards
    st.session_state.autoplay = False
    reset_day("Fresh model trained")

if step_clicked and not st.session_state.state["done"]:
    st.session_state.autoplay = False
    st.session_state.state = step(st.session_state.env, st.session_state.agent, st.session_state.state)
    st.session_state.activity = [st.session_state.state["message"]] + st.session_state.activity[:7]

if fast_clicked and not st.session_state.state["done"]:
    st.session_state.autoplay = False
    for _ in range(5):
        if st.session_state.state["done"]:
            break
        st.session_state.state = step(st.session_state.env, st.session_state.agent, st.session_state.state)
    st.session_state.activity = [st.session_state.state["message"]] + st.session_state.activity[:7]

if auto_clicked and not st.session_state.state["done"]:
    st.session_state.autoplay = True
    st.session_state.activity = ["Auto play started"] + st.session_state.activity[:7]

if stop_auto_clicked:
    st.session_state.autoplay = False
    st.session_state.activity = ["Auto play stopped"] + st.session_state.activity[:7]

if reset_clicked:
    st.session_state.autoplay = False
    reset_day("New trading day created")


metric_cols = st.columns(5)
metric_cols[0].metric("Balance", f"${st.session_state.state['balance']}")
metric_cols[1].metric("Inventory", f"{st.session_state.state['inventory_count']} items")
metric_cols[2].metric("Inventory Cost", f"${st.session_state.state['inventory_cost']}")
metric_cols[3].metric("Total Profit", f"${st.session_state.state['profit']}")
metric_cols[4].metric("Training Episodes", str(len(st.session_state.rewards)))

main_left, main_right = st.columns([1.15, 0.85], gap="large")

with main_left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.image(render_grid(st.session_state.env, st.session_state.state, dark=dark_mode), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with main_right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    reward_tab, policy_tab = st.tabs(["Learning Curve", "Policy View"])
    with reward_tab:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        rewards = np.array(st.session_state.rewards, dtype=float)
        x = np.arange(1, len(rewards) + 1)
        fig.patch.set_facecolor("#fffaf2" if not dark_mode else "#101b2d")
        ax.set_facecolor("#fffaf2" if not dark_mode else "#101b2d")
        ax.plot(x, rewards, color="#8dbefb" if not dark_mode else "#60a5fa", linewidth=1.2, alpha=0.23, label="Reward")
        if len(rewards) >= 30:
            window = 30
            smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")
            sx = np.arange(window, len(rewards) + 1)
            ax.plot(sx, smooth, color="#0f766e" if not dark_mode else "#2dd4bf", linewidth=3.2, label="Moving Average")
            ax.fill_between(sx, smooth, np.min(smooth), color="#99f6e4" if not dark_mode else "#134e4a", alpha=0.18)
        ax.set_title("Total Reward per Episode", fontsize=15, color="#0f172a" if not dark_mode else "#f8fafc")
        ax.set_xlabel("Episode", color="#475569" if not dark_mode else "#94a3b8")
        ax.set_ylabel("Reward", color="#475569" if not dark_mode else "#94a3b8")
        ax.grid(alpha=0.24, color="#94a3b8" if not dark_mode else "#334155")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(colors="#64748b" if not dark_mode else "#94a3b8")
        ax.legend(frameon=False, loc="upper left")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with policy_tab:
        policy_left, policy_right = st.columns(2)
        with policy_left:
            st.image(
                render_policy_map(
                    st.session_state.agent["q_table"],
                    st.session_state.env,
                    0,
                    st.session_state.state["balance"],
                    dark=dark_mode,
                    visited_shops=st.session_state.state["visited_shops"],
                    last_move=st.session_state.state["last_move"],
                ),
                use_container_width=True,
            )
            st.caption("Policy when the basket is empty")
        with policy_right:
            carrying_preview = max(1, min(2, st.session_state.state["inventory_count"]))
            st.image(
                render_policy_map(
                    st.session_state.agent["q_table"],
                    st.session_state.env,
                    carrying_preview,
                    st.session_state.state["balance"],
                    dark=dark_mode,
                    visited_shops=st.session_state.state["visited_shops"],
                    last_move=st.session_state.state["last_move"],
                ),
                use_container_width=True,
            )
            st.caption("Policy when the basket already has items")
    st.markdown("</div>", unsafe_allow_html=True)

bottom_left, bottom_right = st.columns([0.9, 1.1], gap="large")

with bottom_left:
    st.markdown("### Store Directory")
    shop_prices = st.session_state.env["shops"]
    cheapest_price = min(shop_prices.values())
    for shop in st.session_state.env["shop_order"]:
        visited = shop in st.session_state.state["visited_shops"]
        badge = "  •  Cheapest" if shop_prices[shop] == cheapest_price else ""
        status = "  •  Visited" if visited else ""
        st.markdown(
            f"""
            <div class="directory-card">
                <div class="directory-title">{st.session_state.env['shop_names'][shop]}</div>
                <div class="directory-text">Location {shop}  •  Buy price ${shop_prices[shop]}  •  Batch x{st.session_state.env['shop_stock'][shop]}{badge}{status}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown(
        f"""
            <div class="status-box">
                <div class="status-label">Market</div>
                <div class="status-text">The market is currently at <strong>{st.session_state.env['market']}</strong> and buys every item for <strong>${st.session_state.env['market_price']}</strong>. The agent can use each shop only once, keeps buying the cheapest affordable profitable batch, and sells the whole basket in one final trade.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with bottom_right:
    st.markdown("### Control Center")
    st.markdown(
        f"""
        <div class="status-box">
            <div class="status-label">Live Status</div>
            <div class="status-text">
                Last action: <strong>{st.session_state.state['last_action']}</strong><br>
                Last reward: <strong>{st.session_state.state['last_reward']}</strong><br>
                Inventory units: <strong>{st.session_state.state['inventory_count']}</strong><br>
                Visited shops: <strong>{len(st.session_state.state['visited_shops'])}</strong><br>
                Last batch revenue: <strong>${st.session_state.state['last_sale_revenue']}</strong><br>
                Last batch profit: <strong>${st.session_state.state['last_trade']}</strong><br>
                Current shop price: <strong>${st.session_state.state['shop_price']}</strong><br>
                Message: <strong>{st.session_state.state['message']}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### Activity Log")
    for item in st.session_state.activity[:8]:
        st.markdown(f'<div class="log-row">{item}</div>', unsafe_allow_html=True)
    if st.session_state.state["done"]:
        st.info("This trading day is finished. Press Reset Day to open a new route through the mall.")

if st.session_state.autoplay:
    if st.session_state.state["done"]:
        st.session_state.autoplay = False
        st.session_state.activity = ["Auto play finished the trading day"] + st.session_state.activity[:7]
        st.rerun()
    else:
        time.sleep(0.35)
        st.session_state.state = step(st.session_state.env, st.session_state.agent, st.session_state.state)
        st.session_state.activity = [st.session_state.state["message"]] + st.session_state.activity[:7]
        if "Sold" in st.session_state.state["message"]:
            st.session_state.autoplay = False
            st.session_state.activity = ["Auto play stopped after sale"] + st.session_state.activity[:7]
            st.rerun()
        elif st.session_state.state["done"]:
            st.session_state.autoplay = False
            st.session_state.activity = ["Auto play stopped at end of day"] + st.session_state.activity[:7]
            st.rerun()
        else:
            st.rerun()
