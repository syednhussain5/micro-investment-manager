# src/data_preprocessing.py
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List
import warnings

warnings.filterwarnings("ignore")

def load_and_preprocess_data(filepath: str, num_users: int = 10) -> pd.DataFrame:
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, low_memory=False)

    columns_needed = ['trans_date_trans_time', 'cc_num', 'merchant', 'category', 'amt', 'first', 'last', 'gender', 'city', 'state', 'job', 'dob']
    missing = [c for c in columns_needed if c not in df.columns]
    if missing:
        print(f"Warning: Missing columns {missing}")
        columns_needed = [c for c in columns_needed if c in df.columns]

    df = df[columns_needed].copy()
    df.rename(columns={'trans_date_trans_time': 'timestamp', 'cc_num': 'user_id', 'amt': 'amount'}, inplace=True)

    # Convert timestamp - try common formats
    print("Converting timestamps...")
    df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True, errors='coerce')
    if df['timestamp'].isna().all():
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    initial = len(df)
    df = df.dropna(subset=['timestamp'])
    print(f"Removed {initial - len(df)} rows with invalid timestamps")

    # choose top users by transaction count
    print(f"Selecting top {num_users} users...")
    user_counts = df['user_id'].value_counts()
    top_users = user_counts.head(num_users).index.tolist()
    df = df[df['user_id'].isin(top_users)].copy()

    df['user_id'] = df['user_id'].astype(str)
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)

    if 'first' in df.columns and 'last' in df.columns:
        df['name'] = df['first'].astype(str) + ' ' + df['last'].astype(str)
    else:
        df['name'] = 'User ' + df['user_id'].astype(str)

    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df = df.dropna(subset=['amount'])
    df['amount'] = df['amount'].abs()
    df = df[df['amount'] > 0].copy()

    print(f"Loaded {len(df)} transactions for {df['user_id'].nunique()} users")
    return df

def extract_user_info(df: pd.DataFrame) -> pd.DataFrame:
    agg = {'name': 'first', 'user_id': 'first'}
    for col in ['gender', 'city', 'state', 'job', 'dob']:
        if col in df.columns:
            agg[col] = 'first'
    user_info = df.groupby('user_id').agg(agg).reset_index(drop=True)
    return user_info

def compute_spending_features(transactions_df: pd.DataFrame, user_id: str) -> dict:
    user_trans = transactions_df[transactions_df['user_id'] == user_id].copy()
    if len(user_trans) == 0:
        return None
    user_trans['month'] = user_trans['timestamp'].dt.to_period('M')
    monthly_spend = user_trans.groupby('month')['amount'].sum()
    if 'category' in user_trans.columns:
        category_spend = user_trans.groupby('category')['amount'].sum()
        total_spend = category_spend.sum()
        category_pct = (category_spend / total_spend * 100).to_dict() if total_spend > 0 else {}
    else:
        category_pct = {}
        total_spend = user_trans['amount'].sum()

    date_range = (user_trans['timestamp'].max() - user_trans['timestamp'].min()).days
    features = {
        'avg_monthly_spend': monthly_spend.mean() if len(monthly_spend) > 0 else 0,
        'std_monthly_spend': monthly_spend.std() if len(monthly_spend) > 1 else 0,
        'avg_transaction_amount': user_trans['amount'].mean(),
        'std_transaction_amount': user_trans['amount'].std(),
        'transaction_count': len(user_trans),
        'unique_merchants': user_trans['merchant'].nunique() if 'merchant' in user_trans.columns else 0,
        'unique_categories': user_trans['category'].nunique() if 'category' in user_trans.columns else 0,
        'category_percentages': category_pct,
        'total_spend': total_spend,
        'max_transaction': user_trans['amount'].max(),
        'min_transaction': user_trans['amount'].min(),
        'transactions_per_day': len(user_trans) / max(date_range, 1)
    }
    return features

def prepare_ml_features(features_dict: dict):
    if features_dict is None:
        return None
    feature_vector = [
        features_dict.get('avg_monthly_spend', 0),
        features_dict.get('std_monthly_spend', 0),
        features_dict.get('avg_transaction_amount', 0),
        features_dict.get('std_transaction_amount', 0),
        features_dict.get('transaction_count', 0),
        features_dict.get('unique_merchants', 0),
        features_dict.get('unique_categories', 0),
        features_dict.get('transactions_per_day', 0),
        features_dict.get('max_transaction', 0)
    ]
    fv = [0 if (pd.isna(x) or x is None or x == float('inf')) else x for x in feature_vector]
    return np.array(fv)

def create_sample_users_dataset(df: pd.DataFrame, db) -> List[str]:
    # Create users in DB from transactions
    users = df['user_id'].unique().tolist()
    user_ids = []
    for uid in users:
        name_row = df[df['user_id'] == uid]['name'].iloc[0] if 'name' in df.columns and not df[df['user_id'] == uid].empty else f"User {uid[:8]}"
        db.add_user(str(uid), name_row)
        user_ids.append(str(uid))
    print(f"Created {len(user_ids)} users in database")
    return user_ids
