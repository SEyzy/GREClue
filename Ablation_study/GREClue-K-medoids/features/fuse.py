
import numpy as np

def fuse_features(text_vec: np.ndarray, graph_vec: np.ndarray, mode: str = "concat", alpha: float = 0.5):
    if mode == "concat":
        return np.concatenate([text_vec, graph_vec], axis=-1)
    elif mode == "weighted":
        d = min(text_vec.shape[-1], graph_vec.shape[-1])
        return alpha * text_vec[:d] + (1.0 - alpha) * graph_vec[:d]
    else:
        raise ValueError(f"Unknown fusion mode: {mode}")
