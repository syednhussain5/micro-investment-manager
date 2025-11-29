# tests/test_clustering.py
import numpy as np
from src.user_profiling import CustomKMeans, CustomAgglomerative

def test_kmeans_simple():
    # simple separable clusters
    X = np.vstack([
        np.random.normal(loc=0.0, scale=0.1, size=(10, 2)),
        np.random.normal(loc=5.0, scale=0.1, size=(10, 2)),
        np.random.normal(loc=10.0, scale=0.1, size=(10, 2)),
    ])
    km = CustomKMeans(n_clusters=3, random_state=0, max_iter=100)
    labels = km.fit(X)
    assert len(np.unique(labels)) == 3
    assert labels.shape[0] == X.shape[0]

def test_agglomerative_simple():
    X = np.vstack([
        np.random.normal(loc=-3.0, scale=0.05, size=(5, 2)),
        np.random.normal(loc=0.0, scale=0.05, size=(5, 2)),
        np.random.normal(loc=3.0, scale=0.05, size=(5, 2)),
    ])
    ag = CustomAgglomerative(n_clusters=3)
    labels = ag.fit(X)
    assert len(np.unique(labels)) == 3
    assert labels.shape[0] == X.shape[0]
