
import numpy as np
from typing import List
from pyclustering.cluster.kmedoids import kmedoids as _kmedoids
from pyclustering.utils.metric import distance_metric, type_metric

def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).sum() + 1e-12))

def pam_kmedoids(X: np.ndarray, k: int, init_medoids: List[int] = None, max_iter: int = 200):
    N = int(X.shape[0])
    if N == 0:
        return [], np.array([], dtype=int), 0.0
    k = min(int(k), N)

    init = list(init_medoids or [])
    if len(init) < k:
        import numpy as _np
        rng = _np.random.default_rng(0)
        rest = [i for i in range(N) if i not in init]
        need = k - len(init)
        if need > 0:
            extra = rng.choice(rest, size=need, replace=False).tolist()
            init += extra
    init = init[:k]

    metric = distance_metric(type_metric.EUCLIDEAN)
    km = _kmedoids(X.tolist(), init, metric=metric)
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
