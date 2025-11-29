# src/portfolio_simulator.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional

class MarketDataSimulator:
    def __init__(self):
        self.current_prices = {'equity': 100.0, 'gold': 50000.0, 'fd': 100.0, 'liquid': 100.0}
        self.expected_returns = {'equity': 0.12, 'gold': 0.08, 'fd': 0.065, 'liquid': 0.04}
        self.volatility = {'equity': 0.18, 'gold': 0.12, 'fd': 0.01, 'liquid': 0.005}
        self.start_date = datetime(2023, 1, 1)
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self._generate_synthetic_history(days=730)

    def _generate_synthetic_history(self, days: int = 730):
        dates = [self.start_date + timedelta(days=i) for i in range(days)]
        for asset, base in self.current_prices.items():
            prices = [base]
            daily_return = self.expected_returns.get(asset, 0.05) / 252.0
            daily_vol = self.volatility.get(asset, 0.1) / np.sqrt(252.0)
            for _ in range(days - 1):
                shock = np.random.normal(0, daily_vol)
                newp = max(prices[-1] * (1 + daily_return + shock), prices[-1] * 0.4)
                prices.append(newp)
            self.historical_data[asset] = pd.DataFrame({'date': dates, 'price': prices})
            self.current_prices[asset] = prices[-1]

    def get_price(self, asset: str, date: Optional[datetime] = None) -> float:
        if date is None:
            return float(self.current_prices.get(asset, 0.0))
        if asset not in self.historical_data:
            return float(self.current_prices.get(asset, 0.0))
        df = self.historical_data[asset]
        idx = (df['date'] - date).abs().idxmin()
        return float(df.iloc[int(idx)]['price'])

    def get_all_prices(self, date: Optional[datetime] = None) -> Dict[str, float]:
        return {asset: self.get_price(asset, date) for asset in self.current_prices.keys()}

    def calculate_returns(self, asset: str, start_date: datetime, end_date: datetime) -> float:
        try:
            s = self.get_price(asset, start_date)
            e = self.get_price(asset, end_date)
            if s == 0:
                return 0.0
            return ((e - s) / s) * 100.0
        except Exception:
            return 0.0

    def get_performance_data(self, days_back: int = 30) -> Dict[str, float]:
        end = datetime.now()
        start = end - timedelta(days=days_back)
        return {asset: self.calculate_returns(asset, start, end) for asset in self.current_prices.keys()}

class PortfolioSimulator:
    def __init__(self, market_simulator: MarketDataSimulator):
        self.market = market_simulator
        self.transaction_fee = 0.01
        self.tax_rate = 0.10

    def execute_investment(self, db, user_id: str, allocation: dict, timestamp: Optional[datetime] = None):
        if timestamp is None:
            timestamp = datetime.now()
        total = float(sum(allocation.values()))
        if total <= 0:
            return None
        if not db.deduct_from_wallet(user_id, total):
            return None
        fees = total * self.transaction_fee
        net = total - fees
        adjusted = {asset: float(amount) * (net / total) if total > 0 else 0.0 for asset, amount in allocation.items()}
        prices = self.market.get_all_prices(timestamp)
        inv = db.add_investment(user_id, adjusted, prices, timestamp)
        return {'investment_id': inv['inv_id'], 'amount': total, 'fees': fees, 'net_investment': net, 'allocation': adjusted, 'prices': prices, 'timestamp': timestamp}

    def calculate_portfolio_value(self, db, user_id: str, date: Optional[datetime] = None):
        prices = self.market.get_all_prices(date)
        total, asset_vals = db.get_portfolio_value(user_id, prices)
        total_invested = db.wallets.get(user_id, {}).get('total_invested', 0.0)
        profit_loss = total - total_invested
        profit_loss_pct = (profit_loss / total_invested * 100.0) if total_invested > 0 else 0.0
        unrealized = profit_loss
        tax = max(unrealized, 0.0) * self.tax_rate
        net = total - tax
        breakdown = {asset: (val / total * 100.0) if total > 0 else 0.0 for asset, val in asset_vals.items()}
        return {'total_value': total, 'net_value': net, 'total_invested': total_invested, 'profit_loss': profit_loss, 'profit_loss_pct': profit_loss_pct, 'tax_liability': tax, 'asset_values': asset_vals, 'asset_breakdown': breakdown}

    def get_portfolio_history(self, db, user_id: str, days: int = 30):
        end = datetime.now()
        start = end - timedelta(days=days)
        dates, vals = [], []
        cur = start
        while cur <= end:
            pv = self.calculate_portfolio_value(db, user_id, cur)
            dates.append(cur)
            vals.append(pv['total_value'])
            cur += timedelta(days=1)
        return pd.DataFrame({'date': dates, 'value': vals})

    def get_asset_performance(self, db, user_id: str, days: int = 30):
        perf = {}
        port = db.get_portfolio(user_id)
        end = datetime.now()
        start = end - timedelta(days=days)
        for asset, holding in port.items():
            if holding.get('units', 0) > 0:
                perf[asset] = self.market.calculate_returns(asset, start, end)
        return perf
