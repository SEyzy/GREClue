from __future__ import annotations
import argparse
import csv
import json
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional

LOC_RE = re.compile(
    r"""
    ^
    (?P<pkg>[A-Za-z_][\w\.]*)
    \$
    (?P<cls>[A-Za-z_]\w*(?:\$(?:[A-Za-z_]\w*|\d+))*)  
    \#
    (?P<meth>[A-Za-z_]\w*|<init>|<clinit>)            
    (?:\([^)]*\))?                                     
    :
    (?P<line>\d+)
    (?:[;\t ,].*)?                                   
    $
    """,
    re.VERBOSE
)

def canonical_key(s: str) -> Optional[str]:
    s = s.strip()
    m = LOC_RE.match(s)
    if not m:
        return None
    pkg = m.group("pkg")
    cls = m.group("cls")
    meth = m.group("meth")
    line = m.group("line")
    return f"{pkg}${cls}#{meth}:{line}"

def load_sus_list(path: Path) -> List[Tuple[str, str]]:

    out: List[Tuple[str, str]] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        key = canonical_key(raw)
        if key:
            out.append((raw, key))
        else:
            parts = [p.strip() for p in re.split(r"[,\t]", raw)]
            if parts:
                key2 = canonical_key(parts[0])
                if key2:
                    out.append((raw, key2))
    return out

def sniff_delimiter(sample: str) -> str:
    if "\t" in sample and "," not in sample:
        return "\t"
    return ","

def load_label_map(path: Path) -> Dict[str, Tuple[str, int]]:

    txt = path.read_text(encoding="utf-8", errors="ignore")
    delim = sniff_delimiter(txt.splitlines()[0] if txt.splitlines() else ",")
    m: Dict[str, Tuple[str,int]] = {}
    reader = csv.reader(txt.splitlines(), delimiter=delim)
    for row in reader:
        if not row:
            continue
        loc = row[0].strip()
        key = canonical_key(loc)
        if not key:
            continue
        correct = row[1].strip() if len(row) > 1 else ""
        try:
            nlines = int(row[2].strip()) if len(row) > 2 and row[2].strip() else 1
        except Exception:
            nlines = 1
        m[key] = (correct, nlines)
    return m

def main():
    ap = argparse.ArgumentParser(description="pallel debugging cost")
    ap.add_argument("--sus", required=True, type=Path, help="sus.csv")
    ap.add_argument("--label", required=True, type=Path)
    ap.add_argument("--outfix", required=False, type=Path)
    args = ap.parse_args()

    sus_list = load_sus_list(args.sus)
    if not sus_list:
        print("error")
        return
    label_map = load_label_map(args.label)
    if not label_map:
        print("label error")
        return

    total = len(sus_list)
    first_idx: Optional[int] = None
    first_key: Optional[str] = None
    first_raw: Optional[str] = None
    fix_code = ""
    fix_lines = 0

    for i, (raw, key) in enumerate(sus_list, start=1): 
        if key in label_map:
            first_idx = i
            first_key = key
            first_raw = raw
            fix_code, fix_lines = label_map[key]
            break


    cost = first_idx / total if total > 0 else 0.0

    print("=== result ===")

    print(f"debugging cost: {cost:.6f}")


    if args.outfix:
        args.outfix.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "total_suspicious": total,
            "first_true_error_index": first_idx,
            "first_true_error_key": first_key,
            "first_true_error_raw": first_raw,
            "repair_cost": cost,
            "fix": {
                "correct_code": fix_code,
                "lines_to_modify": fix_lines
            }
        }
        args.outfix.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nfinish: {args.outfix}")

if __name__ == "__main__":
    main()
