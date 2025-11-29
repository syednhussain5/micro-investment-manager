# src/allocation_engine.py
import numpy as np
from datetime import datetime

class AllocationEngine:
    def __init__(self):
        self.baseline_allocations = {
            'Conservative': {'equity': 0.10, 'gold': 0.30, 'fd': 0.50, 'liquid': 0.10},
            'Moderate': {'equity': 0.40, 'gold': 0.25, 'fd': 0.25, 'liquid': 0.10},
            'Aggressive': {'equity': 0.65, 'gold': 0.20, 'fd': 0.10, 'liquid': 0.05}
        }
        self.learning_rate = 0.03
        self.max_shift = 0.10
        self.min_liquid = 0.05
        self.max_equity = 0.80
        self.custom_allocations = {}
        self.performance_history = {}

    def get_allocation(self, user_id: str, profile: str, wallet_amount: float):
        if user_id in self.custom_allocations:
            weights = self.custom_allocations[user_id].copy()
        else:
            weights = self.baseline_allocations.get(profile, self.baseline_allocations['Moderate']).copy()

        allocation = {}
        for asset, weight in weights.items():
            allocation[asset] = round(wallet_amount * weight, 2)

        total = sum(allocation.values())
        if total != wallet_amount:
            diff = wallet_amount - total
            allocation['liquid'] = allocation.get('liquid', 0) + diff

        return allocation

    def update_weights(self, user_id: str, profile: str, performance_data: dict):
        if user_id in self.custom_allocations:
            current_weights = self.custom_allocations[user_id].copy()
        else:
            current_weights = self.baseline_allocations.get(profile, self.baseline_allocations['Moderate']).copy()

        self.performance_history.setdefault(user_id, []).append({'timestamp': datetime.now(), 'performance': performance_data.copy()})

        avg_perf = np.mean(list(performance_data.values())) if performance_data else 0.0
        new_weights = {}
        adjustments = {}

        for asset, current_weight in current_weights.items():
            if asset in performance_data:
                perf_delta = performance_data[asset] - avg_perf
                adj = self.learning_rate * perf_delta / 100.0
                adj = float(np.clip(adj, -self.max_shift, self.max_shift))
                new_w = current_weight * (1 + adj)
                adjustments[asset] = adj
                new_weights[asset] = new_w
            else:
                new_weights[asset] = current_weight

        total_w = sum(new_weights.values()) if sum(new_weights.values()) > 0 else 1.0
        for a in new_weights:
            new_weights[a] /= total_w

        new_weights = self._apply_constraints(new_weights)
        self.custom_allocations[user_id] = new_weights
        return new_weights, adjustments

    def _apply_constraints(self, weights: dict):
        if weights.get('liquid', 0) < self.min_liquid:
            deficit = self.min_liquid - weights.get('liquid', 0)
            weights['liquid'] = self.min_liquid
            other = [a for a in weights if a != 'liquid']
            total_other = sum(weights[a] for a in other) if other else 1.0
            if total_other > 0:
                for a in other:
                    weights[a] -= (weights[a] / total_other) * deficit

        if weights.get('equity', 0) > self.max_equity:
            excess = weights['equity'] - self.max_equity
            weights['equity'] = self.max_equity
            other = [a for a in weights if a != 'equity']
            for a in other:
                weights[a] += excess / len(other) if len(other) > 0 else 0.0

        for a in weights:
            weights[a] = max(0.0, float(weights[a]))

        total = sum(weights.values()) if sum(weights.values()) > 0 else 1.0
        for a in weights:
            weights[a] /= total

        return weights

    def get_allocation_explanation(self, user_id: str, profile: str) -> str:
        if user_id in self.custom_allocations:
            weights = self.custom_allocations[user_id]
            explanation = f"ML-adjusted allocation for {profile}:\n"
            hist = self.performance_history.get(user_id, [])
            if hist:
                last = hist[-1]['performance']
                best = max(last.items(), key=lambda x: x[1])[0]
                explanation += f"Recently, {best} performed well so its weight was adjusted.\n"
        else:
            weights = self.baseline_allocations.get(profile, self.baseline_allocations['Moderate'])
            explanation = f"Standard {profile} allocation:\n"
        for a, w in weights.items():
            explanation += f" • {a.capitalize()}: {w*100:.1f}%\n"
        return explanation

def check_batch_trigger(db, user_id: str) -> bool:
    wallet_balance = db.get_wallet_balance(user_id)
    threshold = db.users.get(user_id, {}).get('threshold', 100)
    return wallet_balance >= threshold
