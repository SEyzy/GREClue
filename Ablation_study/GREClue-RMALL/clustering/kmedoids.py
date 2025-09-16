
import numpy as np
from typing import List
from pyclustering.cluster.kmedoids import kmedoids as _kmedoids
from pyclustering.utils.metric import distance_metric, type_metric

def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).sum() + 1e-12))

def _init_medoids_kpp(X: np.ndarray, k: int, seed: int = 0) -> List[int]:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx0 = int(rng.integers(0, n))
    medoids = [idx0]
    # k-means++ 风格的 D^2 采样
    while len(medoids) < k:
        d2 = np.min(((X[:, None, :] - X[medoids, :][None, :, :])**2).sum(-1), axis=1)
        probs = d2 / (d2.sum() + 1e-12)
        next_idx = int(rng.choice(n, p=probs))
        if next_idx not in medoids:
            medoids.append(next_idx)
    return medoids

def _init_medoids_random(X: np.ndarray, k: int, seed: int = 0) -> List[int]:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    return rng.choice(n, size=min(k, n), replace=False).tolist()

def pam_kmedoids(X: np.ndarray, k: int, init: str = "kpp", seed: int = 0):
    """
    X: [N,D], normalized recommended.
    init: 'kpp' | 'random'
    Returns: (medoids_idx, labels, total_cost)
    """
    N = int(X.shape[0])
    if N == 0:
        return [], np.array([], dtype=int), 0.0
    k = min(int(k), N)

    if init == "random":
        init_idx = _init_medoids_random(X, k, seed=seed)
    else:
        init_idx = _init_medoids_kpp(X, k, seed=seed)

    metric = distance_metric(type_metric.EUCLIDEAN)
    km = _kmedoids(X.tolist(), init_idx, metric=metric)
    km.process()

    medoids = km.get_medoids()
    clusters = km.get_clusters()

    labels = np.empty(N, dtype=int)
    for cid, idxs in enumerate(clusters):
        for idx in idxs:
            labels[idx] = cid

    cost = 0.0
    medoid_points = {cid: X[m] for cid, m in enumerate(medoids)}
    for i in range(N):
        cost += _euclidean(X[i], medoid_points[labels[i]])

    return medoids, labels, float(cost)
