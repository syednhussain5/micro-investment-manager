# src/user_profiling.py
from __future__ import annotations
import os
import pickle
from typing import Dict, List, Optional, Tuple
import numpy as np

def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def pairwise_distances(X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = euclidean(X[i], X[j])
            D[i, j] = d
            D[j, i] = d
    return D

class CustomKMeans:
    def __init__(self, n_clusters: int = 3, random_state: Optional[int] = None, max_iter: int = 200, tol: float = 1e-4):
        self.n_clusters = int(n_clusters)
        self.random_state = random_state
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.cluster_centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        if random_state is not None:
            np.random.seed(int(random_state))

    def _init_centroids(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        if n_samples <= self.n_clusters:
            return X.copy()
        idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        return X[idx].astype(float).copy()

    def fit(self, X: np.ndarray) -> np.ndarray:
        X = X.astype(float)
        n_samples = X.shape[0]
        centers = self._init_centroids(X)
        labels = np.zeros(n_samples, dtype=int)
        for it in range(self.max_iter):
            for i in range(n_samples):
                dists = np.linalg.norm(centers - X[i], axis=1)
                labels[i] = int(np.argmin(dists))
            new_centers = np.zeros_like(centers)
            for k in range(self.n_clusters):
                members = X[labels == k]
                if len(members) == 0:
                    new_centers[k] = X[np.random.randint(0, n_samples)]
                else:
                    new_centers[k] = members.mean(axis=0)
            shift = np.linalg.norm(centers - new_centers)
            centers = new_centers
            if shift <= self.tol:
                break
        self.cluster_centers_ = centers
        self.labels_ = labels
        return labels

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.cluster_centers_ is None:
            raise ValueError("Model not fitted")
        X = X.astype(float)
        labels = np.zeros(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            dists = np.linalg.norm(self.cluster_centers_ - X[i], axis=1)
            labels[i] = int(np.argmin(dists))
        return labels

class CustomAgglomerative:
    def __init__(self, n_clusters: int = 3):
        self.n_clusters = int(n_clusters)
        self.labels_: Optional[np.ndarray] = None
        self.children_ = []

    def fit(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        if n == 0:
            self.labels_ = np.array([], dtype=int)
            return self.labels_
        D = pairwise_distances(X)
        clusters = {i: [i] for i in range(n)}
        while len(clusters) > self.n_clusters:
            best_pair = None
            best_dist = float("inf")
            keys = list(clusters.keys())
            for i_idx in range(len(keys)):
                for j_idx in range(i_idx + 1, len(keys)):
                    a = keys[i_idx]
                    b = keys[j_idx]
                    min_d = float("inf")
                    for ai in clusters[a]:
                        for bj in clusters[b]:
                            d = D[ai, bj]
                            if d < min_d:
                                min_d = d
                    if min_d < best_dist:
                        best_dist = min_d
                        best_pair = (a, b)
            if best_pair is None:
                break
            a, b = best_pair
            clusters[a] = clusters[a] + clusters[b]
            del clusters[b]
            self.children_.append((a, b, best_dist))
        labels = np.zeros(n, dtype=int)
        for new_label, (cluster_id, members) in enumerate(clusters.items()):
            for m in members:
                labels[m] = int(new_label)
        self.labels_ = labels
        return labels

def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    X = X.astype(float)
    n = X.shape[0]
    if n <= 1 or len(np.unique(labels)) == 1:
        return 0.0
    D = pairwise_distances(X)
    unique_labels = np.unique(labels)
    members = {lab: np.where(labels == lab)[0] for lab in unique_labels}
    sil_vals = []
    for i in range(n):
        lab = labels[i]
        in_cluster = members[lab]
        if len(in_cluster) <= 1:
            a = 0.0
        else:
            a = float((D[i, in_cluster].sum()) / (len(in_cluster) - (1 if i in in_cluster else 0)))
            if i in in_cluster:
                a = float((D[i, in_cluster].sum() - 0.0) / (len(in_cluster) - 1))
        b_vals = []
        for other in unique_labels:
            if other == lab:
                continue
            other_idx = members[other]
            if len(other_idx) == 0:
                continue
            b_vals.append(D[i, other_idx].mean())
        b = min(b_vals) if b_vals else 0.0
        denom = max(a, b)
        s = 0.0 if denom == 0 else (b - a) / denom
        sil_vals.append(s)
    return float(np.mean(sil_vals))

def pca_svd(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    X = X.astype(float)
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:n_components]
    projected = Xc.dot(comps.T)
    return projected

class UserProfiler:
    def __init__(self, n_clusters: int = 3, random_state: Optional[int] = None):
        self.n_clusters = int(n_clusters)
        self.random_state = random_state
        self.is_fitted = False
        self.kmeans_model: Optional[CustomKMeans] = None
        self.agg_model: Optional[CustomAgglomerative] = None
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None
        self.cluster_labels = ["Conservative", "Moderate", "Aggressive"]
        self.model_info: Dict = {"chosen": None, "scores": {}, "kmeans": {}, "agglomerative": {}, "mapping": {}}
        self.cluster_mapping: Dict[int, str] = {}

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        if self.scaler_mean is None or self.scaler_std is None:
            self.scaler_mean = X.mean(axis=0)
            self.scaler_std = X.std(axis=0)
            self.scaler_std = np.where(self.scaler_std == 0, 1.0, self.scaler_std)
        return (X - self.scaler_mean) / self.scaler_std

    def fit(self, feature_matrix: np.ndarray, user_ids: List[str]) -> Dict[str, dict]:
        X = feature_matrix.astype(float)
        n_samples = X.shape[0]
        if n_samples == 0:
            return {}

        Xs = self._standardize(X)

        kmeans = CustomKMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        klabels = kmeans.fit(Xs)

        agg = CustomAgglomerative(n_clusters=self.n_clusters)
        alabels = agg.fit(Xs)

        s_k = silhouette_score(Xs, klabels)
        s_a = silhouette_score(Xs, alabels)

        self.kmeans_model = kmeans
        self.agg_model = agg
        self.model_info["scores"] = {"kmeans": float(s_k), "agglomerative": float(s_a)}
        self.model_info["kmeans"]["labels"] = klabels.tolist()
        self.model_info["agglomerative"]["labels"] = alabels.tolist()

        chosen = "kmeans" if s_k >= s_a else "agglomerative"
        self.model_info["chosen"] = chosen
        chosen_labels = klabels if chosen == "kmeans" else alabels

        cluster_stats: List[Tuple[int, float]] = []
        for cl in range(self.n_clusters):
            mask = chosen_labels == cl
            if mask.sum() == 0:
                cluster_stats.append((cl, 0.0))
                continue
            cluster_feats = feature_matrix[mask]
            num_vals = []
            for idx in range(min(3, cluster_feats.shape[1])):
                num_vals.append(cluster_feats[:, idx].mean())
            risk_score = float(np.mean(num_vals)) if len(num_vals) > 0 else 0.0
            cluster_stats.append((cl, risk_score))

        cluster_stats.sort(key=lambda x: x[1])
        self.cluster_mapping = {}
        for i, (cluster_id, _) in enumerate(cluster_stats):
            label = self.cluster_labels[min(i, len(self.cluster_labels) - 1)]
            self.cluster_mapping[int(cluster_id)] = label

        self.model_info["mapping"] = {int(k): v for k, v in self.cluster_mapping.items()}

        user_profiles: Dict[str, dict] = {}
        for uid, cluster in zip(user_ids, chosen_labels):
            profile = self.cluster_mapping.get(int(cluster), "Moderate")
            scores = [s for _, s in cluster_stats]
            min_score, max_score = (min(scores), max(scores)) if len(scores) > 0 else (0.0, 1.0)
            cluster_score = next((s for cid, s in cluster_stats if cid == cluster), 0.0)
            normalized = 0.5 if max_score == min_score else float((cluster_score - min_score) / (max_score - min_score))
            user_profiles[uid] = {"profile": profile, "cluster": int(cluster), "risk_score": float(normalized)}

        self.is_fitted = True
        return user_profiles

    def predict(self, feature_vector: np.ndarray) -> Dict:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X = feature_vector.astype(float)
        if self.scaler_mean is None or self.scaler_std is None:
            raise ValueError("Scaler stats missing")
        Xs = (X - self.scaler_mean) / self.scaler_std

        chosen = self.model_info.get("chosen", "kmeans")
        if chosen == "kmeans" and self.kmeans_model is not None:
            labels = self.kmeans_model.predict(Xs.reshape(1, -1))
            cl = int(labels[0])
        else:
            if self.kmeans_model is not None:
                labels = self.kmeans_model.predict(Xs.reshape(1, -1))
                cl = int(labels[0])
            else:
                cl = 0

        profile = self.cluster_mapping.get(int(cl), "Moderate")
        risk_score = 0.5
        try:
            if self.kmeans_model is not None and self.kmeans_model.cluster_centers_ is not None:
                center = self.kmeans_model.cluster_centers_[cl]
                dist = float(np.linalg.norm(Xs - center))
                risk_score = float(1.0 / (1.0 + dist))
        except Exception:
            risk_score = 0.5

        return {"profile": profile, "cluster": int(cl), "risk_score": float(risk_score)}

    def get_pca_projection(self, feature_matrix: np.ndarray) -> np.ndarray:
        return pca_svd(feature_matrix, n_components=2)

    def get_labels_for_users(self, feature_matrix: np.ndarray) -> List[int]:
        """
        Return numeric cluster labels (0..k-1) for rows in feature_matrix according
        to chosen model. Uses KMeans centers for prediction when needed.
        """
        if not self.is_fitted:
            raise ValueError("Profiler not fitted")
        Xs = (feature_matrix - self.scaler_mean) / self.scaler_std
        chosen = self.model_info.get("chosen", "kmeans")
        if chosen == "kmeans" and self.kmeans_model is not None:
            return self.kmeans_model.predict(Xs).tolist()
        elif chosen == "agglomerative":
            if self.kmeans_model is not None:
                return self.kmeans_model.predict(Xs).tolist()
            else:
                return [0] * Xs.shape[0]
        else:
            return [0] * Xs.shape[0]

    def save_model(self, filepath: str = "models/user_profiler.pkl"):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        data = {
            "n_clusters": self.n_clusters,
            "random_state": self.random_state,
            "scaler_mean": self.scaler_mean,
            "scaler_std": self.scaler_std,
            "cluster_mapping": self.cluster_mapping,
            "model_info": self.model_info,
            "is_fitted": self.is_fitted,
        }
        # store kmeans centers if available
        if self.kmeans_model is not None and getattr(self.kmeans_model, "cluster_centers_", None) is not None:
            data["kmeans_centers"] = self.kmeans_model.cluster_centers_
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    def load_model(self, filepath: str = "models/user_profiler.pkl") -> bool:
        if not os.path.exists(filepath):
            return False
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.n_clusters = data.get("n_clusters", self.n_clusters)
        self.random_state = data.get("random_state", self.random_state)
        self.scaler_mean = data.get("scaler_mean", self.scaler_mean)
        self.scaler_std = data.get("scaler_std", self.scaler_std)
        self.cluster_mapping = data.get("cluster_mapping", self.cluster_mapping)
        self.model_info = data.get("model_info", self.model_info)
        self.is_fitted = data.get("is_fitted", False)
        # rebuild a kmeans model from saved centers if present
        if "kmeans_centers" in data:
            km = CustomKMeans(n_clusters=self.n_clusters, random_state=self.random_state)
            km.cluster_centers_ = np.array(data["kmeans_centers"])
            self.kmeans_model = km
        return True
