
import os, argparse, numpy as np
from utils.parsers import parse_suspect_list, parse_graph_block, parse_graph_json
from models.starcoder_embedder import StarCoderEmbedder
from models.gnn_embedder import GNNEmbedder
from features.fuse import fuse_features
from clustering.mountain import select_peaks
from clustering.kmedoids import pam_kmedoids

def find_pairs(data_dir: str):
    files = os.listdir(data_dir)
    suspects = [f for f in files if f.startswith("suspects_") and f.endswith(".txt")]
    pairs = []
    for s in sorted(suspects):
        stem = s[len("suspects_"):-len(".txt")]
        gtxt = f"graph_{stem}.txt"; gjson = f"graph_{stem}.json"
        if gtxt in files:
            pairs.append((os.path.join(data_dir, s), os.path.join(data_dir, gtxt)))
        elif gjson in files:
            pairs.append((os.path.join(data_dir, s), os.path.join(data_dir, gjson)))
    return pairs

def load_graph(path: str):
    txt = open(path, "r", encoding="utf-8").read()
    if path.endswith(".json"): return parse_graph_json(txt)
    return parse_graph_block(txt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--model", type=str, default="bigcode/starcoder2-3b")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--fusion", type=str, default="concat", choices=["concat","weighted"])
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--ra", type=float, default=1.0)
    ap.add_argument("--rb", type=float, default=1.5)
    args = ap.parse_args()

    pairs = find_pairs(args.data_dir)
    if not pairs:
        print(f"No pairs found in {args.data_dir}. Expected suspects_*.txt + graph_*.txt|json")
        return

    print(f"Found {len(pairs)} pairs. Extracting features...")

    star = StarCoderEmbedder(model_name=args.model, device=args.device)
    gnn = GNNEmbedder(device=args.device)

    fused, names = [], []
    for s_path, g_path in pairs:
        s_items = parse_suspect_list(open(s_path,"r",encoding="utf-8").read())
        s_vec = star.embed_suspect_list(s_items)
        g_vec = gnn.embed_graph(load_graph(g_path))
        z = fuse_features(s_vec, g_vec, mode=args.fusion, alpha=args.alpha)
        fused.append(z); names.append(os.path.basename(s_path).replace("suspects_","").replace(".txt",""))

    X = np.vstack(fused).astype("float32")
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    print("Selecting peaks via mountain function...")
    peaks = select_peaks(X, k=args.k, ra=args.ra, rb=args.rb)
    print("Initial medoids (peaks):", peaks)

    medoids, labels, cost = pam_kmedoids(X, k=args.k, init_medoids=peaks)
    print("Final medoids:", medoids, "cost:", round(cost,4))

    os.makedirs("outputs", exist_ok=True)
    out = os.path.join("outputs", "assignments.csv")
    with open(out,"w",encoding="utf-8") as f:
        f.write("name,cluster\n")
        for n, lbl in zip(names, labels):
            f.write(f"{n},{int(lbl)}\n")
    print(f"Saved cluster assignments to {out}")

if __name__ == "__main__":
    main()
