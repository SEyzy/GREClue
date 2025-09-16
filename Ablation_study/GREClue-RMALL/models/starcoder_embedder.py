
from typing import List, Dict, Any
import numpy as np

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    _TRANSFORMERS_OK = True
except Exception:
    _TRANSFORMERS_OK = False
    torch = None

from utils.hashing import hash_text

class StarCoderEmbedder:
    def __init__(self, model_name: str = "bigcode/starcoder2-3b", device: str = "cpu", max_length: int = 512):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.ok = False
        if _TRANSFORMERS_OK:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
                self.model.to(device); self.model.eval()
                self.ok = True
            except Exception as e:
                print(f"[StarCoderEmbedder] load failed: {e}")
                self.ok = False
        else:
            print("[StarCoderEmbedder] transformers not available; using hashing fallback.")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts: return np.zeros((0,256), dtype=np.float32)
        if self.ok:
            with torch.no_grad():
                embs = []
                for t in texts:
                    toks = self.tokenizer(t, truncation=True, max_length=self.max_length, return_tensors="pt")
                    toks = {k:v.to(self.device) for k,v in toks.items()}
                    out = self.model(**toks)
                    last = out.last_hidden_state
                    attn = toks.get("attention_mask", None)
                    if attn is not None:
                        mask = attn.unsqueeze(-1)
                        vec = (last * mask).sum(1) / mask.sum(1).clamp(min=1)
                        vec = vec.squeeze(0)
                    else:
                        vec = last.mean(1).squeeze(0)
                    v = vec.cpu().numpy().astype("float32")
                    n = np.linalg.norm(v) + 1e-8
                    embs.append(v / n)
                return np.vstack(embs)
        else:
            return np.vstack([hash_text(t, dim=256).astype("float32") for t in texts])

    def embed_suspect_list(self, items: List[Dict[str, Any]]) -> np.ndarray:
        parts = []
        for it in items:
            sig = it.get("signature","")
            line = it.get("line","")
            score = it.get("score",0.0)
            parts.append(f"{sig}#{line}|{score}")
        text = "\n".join(parts)
        return self.embed_texts([text])[0]
