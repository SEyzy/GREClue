
import numpy as np

def mountain_densities(X: np.ndarray, ra: float = 1.0) -> np.ndarray:
    D2 = ((X[:,None,:] - X[None,:,:])**2).sum(-1)
    return np.exp(-D2 / (2.0 * (ra**2))).sum(1)

def select_peaks(X: np.ndarray, k: int, ra: float = 1.0, rb: float = 1.5):
    N = X.shape[0]
    if N == 0: return []
    k = min(k, N)
    D2 = ((X[:,None,:] - X[None,:,:])**2).sum(-1)
    M = np.exp(-D2 / (2.0 * (ra**2))).sum(1)
    peaks = []
    for _ in range(k):
        p = int(np.argmax(M))
        peaks.append(p)
        M = M - M[p] * np.exp(-D2[:,p] / (2.0 * (rb**2)))
    return peaks
