
from typing import Dict, Any, List, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False

from utils.hashing import hash_text

NODE_KIND_VOCAB = {"method":0,"codeline":1,"var":2,"test":3,"other":4}

def _build_adjacency(n: int, edges: List[Tuple[int,int]]) -> np.ndarray:
    A = np.zeros((n,n), dtype=np.float32)
    for a,b in edges:
        if 1<=a<=n and 1<=b<=n:
            A[a-1,b-1] = 1.0
            A[b-1,a-1] = 1.0
    for i in range(n): A[i,i]=1.0
    Dinv = 1.0 / (A.sum(1, keepdims=True) + 1e-8)
    return A * Dinv

def _node_feature(node: Dict[str, Any], hash_dim: int = 128) -> np.ndarray:
    kind = node.get("kind","other").lower()
    kid = NODE_KIND_VOCAB.get(kind, NODE_KIND_VOCAB["other"])
    one = np.zeros(len(NODE_KIND_VOCAB), dtype=np.float32); one[kid]=1.0
    txt = str(node.get("content",""))
    h = hash_text(txt, dim=hash_dim)
    sus = np.array([float(node.get("sus",0.0))], dtype=np.float32)
    return np.concatenate([one, h, sus], axis=0)  # 5 + 128 + 1 = 134

class _SageLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin_neigh = torch.nn.Linear(in_dim, out_dim, bias=False)
        self.lin_self = torch.nn.Linear(in_dim, out_dim, bias=True)
    def forward(self, X, Ahat):
        neigh = torch.matmul(Ahat, X)
        h = self.lin_neigh(neigh) + self.lin_self(X)
        return torch.relu(h)

class GraphEncoder(torch.nn.Module):
    def __init__(self, in_dim=134, hidden=128, out_dim=256):
        super().__init__()
        self.g1 = _SageLayer(in_dim, hidden)
        self.g2 = _SageLayer(hidden, out_dim)
    def forward(self, X, Ahat):
        h1 = self.g1(X, Ahat)
        h2 = self.g2(h1, Ahat)
        g = h2.mean(0)
        return F.normalize(g, p=2, dim=0)

class GNNEmbedder:
    def __init__(self, in_dim=134, hidden=128, out_dim=256, device: str = "cpu"):
        if not _TORCH_OK:
            raise ImportError("PyTorch is required for GNNEmbedder")
        self.model = GraphEncoder(in_dim, hidden, out_dim).to(device)
        self.device = device

    def embed_graph(self, graph: Dict[str, Any]) -> np.ndarray:
        nodes = graph.get("nodes", []); edges = graph.get("edges", [])
        n = len(nodes)
        if n == 0: return np.zeros((256,), dtype=np.float32)
        X = np.stack([_node_feature(nd) for nd in nodes], axis=0)
        Ahat = _build_adjacency(n, edges)
        import torch
        with torch.no_grad():
            Xt = torch.from_numpy(X).to(self.device)
            At = torch.from_numpy(Ahat).to(self.device)
            g = self.model(Xt, At)
        return g.cpu().numpy().astype("float32")
