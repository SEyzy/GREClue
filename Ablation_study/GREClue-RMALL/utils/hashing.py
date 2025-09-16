
import numpy as np
import hashlib

def _hash_to_int(s: str, mod: int) -> int:
    h = hashlib.blake2b(s.encode('utf-8'), digest_size=8).hexdigest()
    return int(h, 16) % mod

def feature_hash(tokens, dim: int = 256):
    v = np.zeros(dim, dtype=np.float32)
    for t in tokens:
        idx = _hash_to_int(t, dim)
        v[idx] += 1.0
    n = np.linalg.norm(v) + 1e-8
    return v / n

def hash_text(text: str, dim: int = 256):
    toks, buf = [], []
    for ch in text:
        if ch.isalnum() or ch in ['#','.','_','$']:
            buf.append(ch)
        else:
            if buf:
                toks.append(''.join(buf).lower()); buf=[]
    if buf:
        toks.append(''.join(buf).lower())
    return feature_hash(toks, dim=dim)
