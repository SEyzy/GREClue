from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import List, Set, Dict, Tuple

IDENT_RE = re.compile(r"\b[_A-Za-z]\w*\b")
C_KEYWORDS: Set[str] = {
    "auto","break","case","char","const","continue","default","do","double","else",
    "enum","extern","float","for","goto","if","inline","int","long","register",
    "restrict","return","short","signed","sizeof","static","struct","switch",
    "typedef","union","unsigned","void","volatile","while","_Bool","_Complex","_Atomic"
}
COMMON_TYPES: Set[str] = {
    "size_t","ssize_t","uint8_t","uint16_t","uint32_t","uint64_t",
    "int8_t","int16_t","int32_t","int64_t","bool","BOOL"
}

ASSIGN_OPS = [
    "<<=", ">>=", "+=", "-=", "*=", "/=", "%=", "&=", "^=", "|=",
    "="  
]

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def split_first_assignment(line: str) -> Tuple[str, str, str] | None:

    i = 0
    while i < len(line):
        if line[i] == '=':
            prev = line[i-1] if i-1 >= 0 else ''
            nxt = line[i+1] if i+1 < len(line) else ''
            if prev in ['<','>','!','=',] or nxt == '=':
                i += 1
                continue
        i += 1

    tmp = line.replace("==", "  ").replace("!=", "  ").replace("<=", "  ").replace(">=", "  ")
    pos = []
    for op in ASSIGN_OPS:
        j = tmp.find(op)
        if j != -1:
            pos.append((j, op))
    if not pos:
        return None
    j, op = sorted(pos, key=lambda x: x[0])[0]
    lhs = line[:j].strip()
    rhs = line[j+len(op):].strip().rstrip(';')
    return lhs, op, rhs

def extract_identifiers(s: str) -> List[str]:
    ids = []
    for m in IDENT_RE.finditer(s):
        w = m.group(0)
        if w in C_KEYWORDS or w in COMMON_TYPES:
            continue
        ids.append(w)
    return ids

def extract_declared_vars_and_type(line: str) -> Dict[str, str]:

    out: Dict[str, str] = {}
    s = line.strip().rstrip(';')
    if re.search(r"\)\s*\{", s) or re.search(r"\w+\s*\([^)]*\)\s*$", s):
        return out

    m = re.match(
        r"^(?P<type>(?:const\s+|unsigned\s+|signed\s+)*"
        r"(?:struct\s+\w+|\w+)"
        r"(?:\s*\*+)*)\s+(?P<rest>.+)$",
        s
    )
    if not m:
        return out
    typestr = norm(m.group("type"))
    rest = m.group("rest")
    parts = [p.strip() for p in rest.split(",")]
    for p in parts:

        m2 = IDENT_RE.search(p)
        if not m2:
            continue
        name = m2.group(0)
        if name in C_KEYWORDS:
            continue
        out[name] = typestr
    return out

def parse_ternary(rhs: str) -> Tuple[str, str, str] | None:
    m = re.search(r"(.+?)\?(.+?):(.+)", rhs)
    if not m:
        return None
    cond = m.group(1)
    a = m.group(2)
    b = m.group(3)
    return cond, a, b

def analyze_line(line: str) -> Tuple[List[Tuple[str,str]], List[Tuple[int,int]]]:

    line = norm(line)
    var_types = extract_declared_vars_and_type(line)  # 可能为空

    vars_order: List[str] = []   # 按首次出现
    def add_var(v: str):
        if v and v not in vars_order:
            vars_order.append(v)

    edges: List[Tuple[int,int]] = []

    asg = split_first_assignment(line)
    if asg:
        lhs, op, rhs = asg
        lhs_ids = extract_identifiers(lhs)
        lhs_vars = [lhs_ids[0]] if lhs_ids else []
        for v in lhs_vars:
            add_var(v)

        tern = parse_ternary(rhs)
        if tern:
            cond, a, b = tern
            rhs_vars = extract_identifiers(cond) + extract_identifiers(a) + extract_identifiers(b)
        else:
            rhs_vars = extract_identifiers(rhs)

        for v in rhs_vars:
            add_var(v)

        for r in rhs_vars:
            for lv in lhs_vars:
                if r != lv:
                    edges.append((vars_order.index(r)+1, vars_order.index(lv)+1))



    else:
        ids = extract_identifiers(line)
        for v in ids:
            add_var(v)
        for i in range(len(vars_order)-1):
            edges.append((i+1, i+2))

    nodes: List[Tuple[str,str]] = []
    for v in vars_order:
        vtype = var_types.get(v, "unknown")
        nodes.append((v, vtype))

    return nodes, edges

def write_one(out_path: Path, nodes: List[Tuple[str,str]], edges: List[Tuple[int,int]]):
    lines: List[str] = []
    lines.append("node:")
    for i, (v, t) in enumerate(nodes, start=1):
        lines.append(f"{i} {{content: {v}, type: {t}}}")
    lines.append("edge:")
    for (u,v) in edges:
        lines.append(f"{u}->{v}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def main():
    ap = argparse.ArgumentParser(description="Static Analysis")
    ap.add_argument("--in_dir", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for p in args.in_dir.glob("*.txt"):
        stem = p.stem
        if stem.isdigit():
            files.append(p)
    files.sort(key=lambda p: int(p.stem))

    for p in files:
        code = p.read_text(encoding="utf-8", errors="ignore").strip()
        nodes, edges = analyze_line(code)
        outp = args.out_dir / p.name
        write_one(outp, nodes, edges)

    print(f"finish {len(files)} files，outptu:{args.out_dir}")

if __name__ == "__main__":
    main()
