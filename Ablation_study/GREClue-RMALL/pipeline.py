
import os, argparse, numpy as np
from tqdm import tqdm
from utils.parsers import parse_suspect_list
from models.starcoder_embedder import StarCoderEmbedder
from clustering.kmedoids import pam_kmedoids

def collect_suspects(data_dir: str):
    files = os.listdir(data_dir)
    sus = [f for f in files if f.startswith("suspects_") and f.endswith(".txt")]
    sus.sort()
    return [os.path.join(data_dir, f) for f in sus]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--model", type=str, default="bigcode/starcoder2-3b")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--init", type=str, default="kpp", choices=["kpp","random"])
    args = ap.parse_args()

    paths = collect_suspects(args.data_dir)
    if not paths:
        print(f"No suspect files in {args.data_dir}. Expect suspects_*.txt")
        return

    print(f"Found {len(paths)} suspect files. Embedding with StarCoder/hash...")
    star = StarCoderEmbedder(model_name=args.model, device=args.device)

    X, names = [], []
    for p in tqdm(paths):
        items = parse_suspect_list(open(p,"r",encoding="utf-8").read())
        vec = star.embed_suspect_list(items)
        X.append(vec); names.append(os.path.basename(p).replace("suspects_","").replace(".txt",""))
    X = np.vstack(X).astype("float32")
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    print(f"Clustering with K-medoids (pyclustering), k={args.k}, init={args.init} ...")
    medoids, labels, cost = pam_kmedoids(X, k=args.k, init=args.init, seed=0)
    print("Medoids:", medoids, "cost:", round(cost,4))

    os.makedirs("outputs", exist_ok=True)
    out = os.path.join("outputs", "assignments.csv")
    with open(out,"w",encoding="utf-8") as f:
        f.write("name,cluster\n")
        for n, lbl in zip(names, labels):
            f.write(f"{n},{int(lbl)}\n")
    print(f"Saved to {out}")

if __name__ == "__main__":
    main()
