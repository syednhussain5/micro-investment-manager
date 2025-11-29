# src/database.py
import os
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np

class InvestmentDatabase:
    def __init__(self):
        self.users: Dict[str, dict] = {}
        self.transactions: List[dict] = []
        self.wallets: Dict[str, dict] = {}
        self.portfolios: Dict[str, dict] = {}
        self.investments: List[dict] = []
        self.user_profiles: Dict[str, dict] = {}

    def add_user(self, user_id: str, name: str, initial_balance: float = 10000.0) -> dict:
        if user_id not in self.users:
            self.users[user_id] = {
                'user_id': user_id,
                'name': name,
                'created_at': datetime.now(),
                'round_up_rule': 50,
                'threshold': 100,
                'profile': 'Moderate'
            }
            self.wallets[user_id] = {'balance': 0.0, 'total_rounded_up': 0.0, 'total_invested': 0.0}
            self.portfolios[user_id] = {
                'equity': {'units': 0.0, 'invested': 0.0},
                'gold': {'units': 0.0, 'invested': 0.0},
                'fd': {'units': 0.0, 'invested': 0.0},
                'liquid': {'units': 0.0, 'invested': 0.0}
            }
        return self.users[user_id]

    def add_transaction(self, user_id: str, amount: float, merchant: str = "", category: str = "", timestamp: Optional[datetime] = None) -> dict:
        if timestamp is None:
            timestamp = datetime.now()
        if user_id not in self.users:
            self.add_user(user_id, name=f"User {str(user_id)[:8]}")
        round_up = int(self.users[user_id].get('round_up_rule', 50))
        try:
            rounded_amount = float(np.ceil(float(amount) / round_up) * round_up)
        except Exception:
            rounded_amount = float(amount)
        spare = max(0.0, rounded_amount - float(amount))
        t = {
            'trans_id': len(self.transactions),
            'user_id': user_id,
            'amount': float(amount),
            'merchant': merchant,
            'category': category,
            'timestamp': timestamp,
            'spare_change': float(spare),
            'rounded_amount': float(rounded_amount)
        }
        self.transactions.append(t)
        self.wallets.setdefault(user_id, {'balance': 0.0, 'total_rounded_up': 0.0, 'total_invested': 0.0})
        self.wallets[user_id]['balance'] += spare
        self.wallets[user_id]['total_rounded_up'] += spare
        return t

    def get_user_transactions(self, user_id: str, days: int = 30) -> List[dict]:
        user_trans = [t for t in self.transactions if t['user_id'] == user_id]
        cutoff = datetime.now() - timedelta(days=days)
        return [t for t in user_trans if t['timestamp'] > cutoff]

    def get_wallet_balance(self, user_id: str) -> float:
        return float(self.wallets.get(user_id, {}).get('balance', 0.0))

    def deduct_from_wallet(self, user_id: str, amount: float) -> bool:
        if self.wallets.get(user_id, {}).get('balance', 0.0) >= amount:
            self.wallets[user_id]['balance'] -= amount
            self.wallets[user_id]['total_invested'] += amount
            return True
        return False

    def add_investment(self, user_id: str, allocation: dict, prices: dict, timestamp: Optional[datetime] = None) -> dict:
        if timestamp is None:
            timestamp = datetime.now()
        if user_id not in self.portfolios:
            self.portfolios[user_id] = {
                'equity': {'units': 0.0, 'invested': 0.0},
                'gold': {'units': 0.0, 'invested': 0.0},
                'fd': {'units': 0.0, 'invested': 0.0},
                'liquid': {'units': 0.0, 'invested': 0.0}
            }
        inv = {
            'inv_id': len(self.investments),
            'user_id': user_id,
            'timestamp': timestamp,
            'allocation': allocation.copy(),
            'prices': prices.copy()
        }
        for asset, amount in allocation.items():
            try:
                if amount > 0 and asset in prices:
                    units = float(amount) / float(prices[asset]) if float(prices[asset]) != 0 else 0.0
                    self.portfolios[user_id].setdefault(asset, {'units': 0.0, 'invested': 0.0})
                    self.portfolios[user_id][asset]['units'] += units
                    self.portfolios[user_id][asset]['invested'] += float(amount)
            except Exception:
                continue
        self.investments.append(inv)
        return inv

    def get_portfolio(self, user_id: str) -> dict:
        return self.portfolios.get(user_id, {
            'equity': {'units': 0.0, 'invested': 0.0},
            'gold': {'units': 0.0, 'invested': 0.0},
            'fd': {'units': 0.0, 'invested': 0.0},
            'liquid': {'units': 0.0, 'invested': 0.0}
        })

    def get_portfolio_value(self, user_id: str, current_prices: dict):
        portfolio = self.get_portfolio(user_id)
        total = 0.0
        asset_vals = {}
        for asset, h in portfolio.items():
            price = current_prices.get(asset, 0.0)
            try:
                val = float(h.get('units', 0.0)) * float(price)
            except Exception:
                val = 0.0
            asset_vals[asset] = val
            total += val
        return total, asset_vals

    def set_user_profile(self, user_id: str, profile: str, risk_score: float):
        self.user_profiles[user_id] = {
            'profile': profile,
            'risk_score': float(risk_score),
            'updated_at': datetime.now()
        }
        if user_id in self.users:
            self.users[user_id]['profile'] = profile

    def save_to_file(self, filepath: str = 'data/processed/database.pkl'):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'users': self.users,
                'transactions': self.transactions,
                'wallets': self.wallets,
                'portfolios': self.portfolios,
                'investments': self.investments,
                'user_profiles': self.user_profiles
            }, f)

    def load_from_file(self, filepath: str = 'data/processed/database.pkl') -> bool:
        if not os.path.exists(filepath):
            return False
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.users = data.get('users', {})
        self.transactions = data.get('transactions', [])
        self.wallets = data.get('wallets', {})
        self.portfolios = data.get('portfolios', {})
        self.investments = data.get('investments', [])
        self.user_profiles = data.get('user_profiles', {})
        return True
