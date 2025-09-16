import argparse
from pathlib import Path
from collections import defaultdict

def parse_args():
    p = argparse.ArgumentParser(
        description="Keep the first N lines of each CSV file, and then keep the remaining lines that are unique to other files."
    )
    p.add_argument("indir", type=str)
    p.add_argument("--head", type=int, default=100)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--encoding", type=str, default="utf-8")
    return p.parse_args()

def is_zero_line(line: str) -> bool:

    s = line.rstrip()
    return s.endswith(";0.0") or s.endswith(",0.0") or s.split(";")[-1].strip().split(",")[-1].strip() == "0.0"

def extract_key(line: str) -> str:

    s = line.rstrip("\n\r")
    if ";" in s:
        return s.rsplit(";", 1)[0]
    if "," in s:
        return s.rsplit(",", 1)[0]
    return s

def read_lines_keep_order(fp: Path, encoding="utf-8"):
    lines = []
    with fp.open("r", encoding=encoding, errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n\r")
            if not line.strip():
                continue
            if is_zero_line(line):
                continue
            lines.append(line)
    return lines

def main():
    args = parse_args()
    indir = Path(args.indir).expanduser().resolve()
    assert indir.is_dir(), f"path error{indir}"

    outdir = Path(args.out).expanduser().resolve() if args.out else (indir / "filtered")
    outdir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(indir.glob("*.csv"))
    if not csv_files:
        print(f"can not find csv files{indir}")
        return

    filtered = {}  
    for fp in csv_files:
        lines = read_lines_keep_order(fp, encoding=args.encoding)
        head_lines = lines[: args.head]
        tail_lines = lines[args.head :]

        head_pairs = [(extract_key(l), l) for l in head_lines]
        tail_pairs = [(extract_key(l), l) for l in tail_lines]
        filtered[fp] = {"head": head_pairs, "tail": tail_pairs}

    key_presence_in_files = defaultdict(set) 
    for fp, parts in filtered.items():
        keys_in_this_file_tail = {k for k, _ in parts["tail"]}
        for k in keys_in_this_file_tail:
            key_presence_in_files[k].add(fp)

    for fp in csv_files:
        out_fp = outdir / fp.name
        head_pairs = filtered[fp]["head"]
        tail_pairs = filtered[fp]["tail"]

        unique_tail = [line for k, line in tail_pairs if len(key_presence_in_files[k]) == 1]

        with out_fp.open("w", encoding=args.encoding, newline="\n") as w:
            for _, line in head_pairs:
                w.write(line + "\n")
            for line in unique_tail:
                w.write(line + "\n")

        print(f"output:{out_fp}")

    print("Finish")

if __name__ == "__main__":
    main()
