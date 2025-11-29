# app.py
from __future__ import annotations
import os
import sys
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import logging

# Increase recursion safety
sys.setrecursionlimit(3000)

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

# Local modules
from src.database import InvestmentDatabase
from src.data_preprocessing import (
    load_and_preprocess_data,
    compute_spending_features,
    prepare_ml_features,
    create_sample_users_dataset,
)
from src.allocation_engine import AllocationEngine, check_batch_trigger
from src.portfolio_simulator import MarketDataSimulator, PortfolioSimulator
from src.user_profiling import UserProfiler

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("microinvest")

st.set_page_config(page_title="Smart Investment Round-Up", page_icon="💰", layout="wide")
np.random.seed(42)

# Session state initialization
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.db = InvestmentDatabase()
    st.session_state.market = None
    st.session_state.portfolio_sim = None
    st.session_state.allocator = AllocationEngine()
    st.session_state.profiler = None
    st.session_state.transactions_df = None
    st.session_state.selected_user = None
    st.session_state.feature_matrix = None
    st.session_state.user_id_order = []

# plotting helpers (matplotlib)
def plot_pie(labels, values, title=""):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title(title)
    ax.axis("equal")
    st.pyplot(fig)
    plt.close(fig)

def plot_scatter(x, y, c=None, labels=None, title="", xlabel="", ylabel=""):
    fig, ax = plt.subplots(figsize=(6, 4))
    sc = ax.scatter(x, y, c=c, s=80, cmap="tab10", alpha=0.9)
    if labels is not None:
        for xi, yi, lab in zip(x, y, labels):
            ax.annotate(lab, (xi, yi), fontsize=8, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    st.pyplot(fig)
    plt.close(fig)

def plot_line(dates, values, title="", xlabel="", ylabel=""):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(dates, values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.autofmt_xdate()
    st.pyplot(fig)
    plt.close(fig)

# cached loader
@st.cache_data
def load_transactions_cached(path: str):
    return load_and_preprocess_data(path, num_users=10)

# core functions
def initialize_system():
    data_path = os.path.join("data", "raw", "transactions.csv")
    if not os.path.exists(data_path):
        st.error("Place transactions.csv in data/raw/ and re-run.")
        return

    df = load_transactions_cached(data_path)
    st.session_state.transactions_df = df

    db = st.session_state.db
    user_ids = create_sample_users_dataset(df, db)

    # feed recent transactions into DB
    for uid in user_ids:
        user_trans = df[df["user_id"] == uid].tail(20)
        for _, trans in user_trans.iterrows():
            db.add_transaction(uid, trans["amount"], trans.get("merchant", ""), trans.get("category", ""), trans["timestamp"])

    st.session_state.market = MarketDataSimulator()
    st.session_state.portfolio_sim = PortfolioSimulator(st.session_state.market)

    # build features + fit profiler
    build_and_fit_profiler(user_ids)

    st.session_state.initialized = True
    st.success(f"System initialized with {len(user_ids)} users.")

def build_and_fit_profiler(user_ids: Optional[List[str]] = None):
    # compute features for available users from transactions_df
    df = st.session_state.transactions_df
    if df is None:
        st.warning("No transactions loaded.")
        return

    if user_ids is None:
        user_ids = df["user_id"].unique().tolist()

    features = []
    valid_ids = []
    for uid in user_ids:
        feat = compute_spending_features(df, uid)
        if feat:
            vec = prepare_ml_features(feat)
            if vec is not None:
                features.append(vec)
                valid_ids.append(uid)

    if len(features) == 0:
        st.warning("Not enough data to train profiler.")
        st.session_state.profiler = None
        st.session_state.feature_matrix = None
        st.session_state.user_id_order = []
        return

    X = np.array(features)
    profiler = UserProfiler(n_clusters=3, random_state=42)
    profiles = profiler.fit(X, valid_ids)
    st.session_state.profiler = profiler
    st.session_state.feature_matrix = X
    st.session_state.user_id_order = list(valid_ids)

    # store profiles into db
    db = st.session_state.db
    for uid, pdata in profiles.items():
        db.set_user_profile(uid, pdata["profile"], pdata["risk_score"])

    st.success("Profiler trained and user profiles updated.")

def retrain_models_button():
    if st.button("Train models again (rebuild profiler)"):
        user_ids = list(st.session_state.db.users.keys())
        build_and_fit_profiler(user_ids)

def save_profiler_button():
    profiler = st.session_state.profiler
    if profiler is None:
        st.info("Profiler not fitted; nothing to save.")
        return
    if st.button("Save profiler to disk"):
        os.makedirs("models", exist_ok=True)
        profiler.save_model("models/user_profiler.pkl")
        st.success("Profiler saved to models/user_profiler.pkl")

def load_profiler_button():
    if st.button("Load saved profiler"):
        profiler = UserProfiler()
        loaded = profiler.load_model("models/user_profiler.pkl")
        if loaded:
            st.session_state.profiler = profiler
            # if feature matrix exists, re-apply labels to db
            if st.session_state.feature_matrix is not None and st.session_state.user_id_order:
                # produce labels using profiler's chosen model info if possible
                labels = profiler.get_labels_for_users(st.session_state.feature_matrix)
                for uid, lab in zip(st.session_state.user_id_order, labels):
                    profile = profiler.cluster_mapping.get(int(lab), "Moderate")
                    st.session_state.db.set_user_profile(uid, profile, 0.5)
            st.success("Loaded profiler from models/user_profiler.pkl")
        else:
            st.error("No saved profiler found at models/user_profiler.pkl")

def add_new_transaction(user_id: str, amount: float, merchant: str, category: str):
    db = st.session_state.db
    tx = db.add_transaction(user_id, amount, merchant, category)
    st.success(f"Transaction added. Spare change: ₹{tx['spare_change']:.2f}")
    # trigger investment automatically if threshold reached (no st.rerun)
    if check_batch_trigger(db, user_id):
        st.info("Wallet threshold reached — executing batch investment.")
        execute_batch_investment(user_id)

def execute_batch_investment(user_id: str):
    db = st.session_state.db
    wallet = db.get_wallet_balance(user_id)
    if wallet < db.users[user_id]["threshold"]:
        st.warning("Wallet below threshold.")
        return
    profile = db.users[user_id]["profile"]
    allocation = st.session_state.allocator.get_allocation(user_id, profile, wallet)
    result = st.session_state.portfolio_sim.execute_investment(db, user_id, allocation)
    if result:
        st.success(f"Investment executed: ₹{result['amount']:.2f}")
        st.info(f"Fees: ₹{result['fees']:.2f} | Net: ₹{result['net_investment']:.2f}")
    else:
        st.error("Investment failed.")

# UI layout
st.title("💰 Smart Investment Round-Up System — Enhanced")
st.markdown("Round up spare change → Auto-invest → ML-driven profiles (custom models).")

with st.sidebar:
    st.header("System")
    if not st.session_state.initialized:
        if st.button("Initialize System"):
            initialize_system()
    else:
        st.success("System active")
        if st.button("Update ML allocations (simple)"):
            db = st.session_state.db
            for uid in list(db.users.keys()):
                perf = st.session_state.portfolio_sim.get_asset_performance(db, uid, days=30)
                if perf:
                    st.session_state.allocator.update_weights(uid, db.users[uid]["profile"], perf)
            st.success("Allocations updated.")

        st.divider()
        st.header("Profiler controls")
        retrain_models_button()
        save_profiler_button()
        load_profiler_button()
        st.divider()
        st.header("User selection")
        user_list = list(st.session_state.db.users.keys())
        if user_list:
            user_names = [f"{st.session_state.db.users[uid]['name']} ({uid[:8]}...)" for uid in user_list]
            selected_idx = st.selectbox("Choose User", range(len(user_list)), format_func=lambda i: user_names[i])
            st.session_state.selected_user = user_list[selected_idx]
        else:
            st.info("No users — initialize first.")

# Main area
if not st.session_state.initialized:
    st.info("Click 'Initialize System' in the sidebar to load data and start.")
else:
    user_id = st.session_state.selected_user
    db = st.session_state.db

    if user_id:
        # header
        user_info = db.users[user_id]
        user_profile = db.user_profiles.get(user_id, {"profile": "Moderate", "risk_score": 0.5})
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("User", user_info["name"])
        with c2:
            st.metric("Risk Profile", user_profile["profile"])
        with c3:
            st.metric("Risk Score", f"{user_profile['risk_score']:.2f}")
        with c4:
            st.metric("Wallet Balance", f"₹{db.wallets[user_id]['balance']:.2f}")

        st.divider()
        tabs = st.tabs(["Dashboard", "Add Transaction", "Portfolio", "Performance", "Settings"])

        # Dashboard
        with tabs[0]:
            st.subheader("Portfolio Overview")
            pdata = st.session_state.portfolio_sim.calculate_portfolio_value(db, user_id)
            wallet = db.wallets[user_id]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Wallet", f"₹{wallet['balance']:.2f}")
            with col2:
                st.metric("Total Invested", f"₹{wallet['total_invested']:.2f}")
            with col3:
                st.metric("Rounded Up", f"₹{wallet['total_rounded_up']:.2f}")

            st.subheader("Asset Breakdown")
            breakdown = pdata.get("asset_breakdown", {})
            if breakdown and sum(breakdown.values()) > 0:
                labels = [k.capitalize() for k in breakdown.keys()]
                values = list(breakdown.values())
                plot_pie(labels, values, title="Asset Distribution")
            else:
                st.info("No investments yet.")

        # Add Transaction
        with tabs[1]:
            st.subheader("Add Transaction")
            a1, a2 = st.columns(2)
            with a1:
                amount = st.number_input("Amount (₹)", min_value=1.0, value=100.0, step=10.0)
                category = st.selectbox("Category", ["grocery_pos", "gas_transport", "misc_net", "shopping_net", "entertainment", "food_dining", "personal_care", "health_fitness"])
            with a2:
                merchant = st.text_input("Merchant", value="Sample Merchant")
                round_rule = user_info.get("round_up_rule", 50)
                rounded = np.ceil(amount / round_rule) * round_rule
                spare = rounded - amount
                st.info(f"Round-up to ₹{rounded:.2f}")
                st.success(f"Spare change: ₹{spare:.2f}")

            if st.button("Add Transaction"):
                add_new_transaction(user_id, float(amount), merchant, category)

        # Portfolio
        with tabs[2]:
            st.subheader("Holdings")
            holdings = db.get_portfolio(user_id)
            current_prices = st.session_state.market.get_all_prices()
            rows = []
            for asset, h in holdings.items():
                units = h.get("units", 0)
                invested = h.get("invested", 0)
                value = units * current_prices.get(asset, 0)
                pl = value - invested
                pl_pct = (pl / invested * 100) if invested > 0 else 0
                rows.append({
                    "Asset": asset.capitalize(),
                    "Units": f"{units:.4f}",
                    "Invested (₹)": f"₹{invested:.2f}",
                    "Current Value (₹)": f"₹{value:.2f}",
                    "P/L (₹)": f"₹{pl:.2f}",
                    "P/L (%)": f"{pl_pct:.2f}%"
                })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info("No holdings yet.")
            st.subheader("Investment History")
            invs = [inv for inv in db.investments if inv["user_id"] == user_id]
            if invs:
                rows = []
                for inv in invs[-10:]:
                    rows.append({
                        "Date": inv["timestamp"].strftime("%Y-%m-%d %H:%M") if hasattr(inv["timestamp"], "strftime") else str(inv["timestamp"]),
                        "Equity": f"₹{inv['allocation'].get('equity', 0):.2f}",
                        "Gold": f"₹{inv['allocation'].get('gold', 0):.2f}",
                        "FD": f"₹{inv['allocation'].get('fd', 0):.2f}",
                        "Liquid": f"₹{inv['allocation'].get('liquid', 0):.2f}",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info("No investments made yet.")

        # Performance
        with tabs[3]:
            st.subheader("Asset Performance (30d)")
            perf = st.session_state.portfolio_sim.get_asset_performance(db, user_id, days=30)
            if perf:
                dfp = pd.DataFrame([{"Asset": k.capitalize(), "Returns (%)": float(v)} for k, v in perf.items()])
                st.dataframe(dfp, use_container_width=True, hide_index=True)
            else:
                st.info("No performance data.")

            st.divider()
            st.subheader("Allocation Explanation")
            st.text(st.session_state.allocator.get_allocation_explanation(user_id, user_profile["profile"]))

            st.divider()
            st.subheader("Model Comparison & PCA")
            profiler = st.session_state.profiler
            if profiler and getattr(profiler, "is_fitted", False):
                info = profiler.model_info
                st.write(f"Chosen model: **{info.get('chosen')}**")
                scores = info.get("scores", {})
                st.write(f"Silhouette — KMeans: {scores.get('kmeans', 0):.4f} | Agglomerative: {scores.get('agglomerative', 0):.4f}")
                mapping = info.get("mapping", {})
                if mapping:
                    st.dataframe(pd.DataFrame([{"Cluster": int(k), "Profile": v} for k, v in mapping.items()]), use_container_width=True, hide_index=True)

                # PCA scatter
                X = st.session_state.feature_matrix
                user_order = st.session_state.user_id_order
                if X is not None and user_order:
                    proj = profiler.get_pca_projection(X)
                    # get labels for users
                    labels = profiler.get_labels_for_users(X)
                    # Map profile labels to color numbers
                    unique_clusters = sorted(list(set(labels)))
                    color_map = {c: i for i, c in enumerate(unique_clusters)}
                    colors = [color_map[int(l)] for l in labels]
                    # Show interactive scatter
                    st.markdown("**PCA (2D) projection — users colored by cluster**")
                    plot_scatter(proj[:, 0], proj[:, 1], c=colors, labels=user_order, title="PCA projection of users", xlabel="PC1", ylabel="PC2")
                else:
                    st.info("No feature matrix to show PCA.")
            else:
                st.info("Profiler not fitted.")

        # Settings
        with tabs[4]:
            st.subheader("User Settings")
            a1, a2 = st.columns(2)
            with a1:
                new_round = st.selectbox("Round-up Rule", [10, 20, 50, 100], index=[10,20,50,100].index(int(user_info.get("round_up_rule",50))))
                if st.button("Update Round-up"):
                    db.users[user_id]["round_up_rule"] = int(new_round)
                    st.success(f"Round-up rule updated to ₹{new_round}")
            with a2:
                new_thresh = st.number_input("Investment Threshold (₹)", min_value=50.0, max_value=1000.0, value=float(user_info.get("threshold",100.0)), step=50.0)
                if st.button("Update Threshold"):
                    db.users[user_id]["threshold"] = float(new_thresh)
                    st.success(f"Threshold updated to ₹{new_thresh}")

            st.divider()
            st.subheader("Quick Actions")
            q1, q2 = st.columns(2)
            with q1:
                if st.button("Trigger Batch Investment"):
                    execute_batch_investment(user_id)
            with q2:
                if st.button("Reset Wallet"):
                    db.wallets[user_id]["balance"] = 0.0
                    st.success("Wallet reset.")

st.divider()
st.markdown("<div style='text-align:center;color:#666;padding:12px;'>MicroInvestment — Round up & invest</div>", unsafe_allow_html=True)
