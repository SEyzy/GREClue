from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

WS_RE = re.compile(r"\s+")
def norm(s: str) -> str:
    return WS_RE.sub(" ", s.strip())

def parse_method_graph(text: str):
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    mode = None
    id_to_method: Dict[int, str] = {}
    method_edges_local: List[Tuple[int,int]] = []
    for ln in lines:
        if ln.lower().startswith("methodnodes"):
            mode = "nodes"; continue
        if ln == "edge:":
            mode = "edges"; continue
        if mode == "nodes":
            m = re.match(r"(\d+)\s+method:\s*\{content:\s*(.+?)\s*\}\s*$", ln)
            if m:
                idx = int(m.group(1))
                content = norm(m.group(2))
                id_to_method[idx] = content
        elif mode == "edges":
            m = re.match(r"(\d+)\s*->\s*(\d+)$", ln)
            if m:
                method_edges_local.append((int(m.group(1)), int(m.group(2))))
    method_edges_by_name: List[Tuple[str,str]] = []
    for u,v in method_edges_local:
        if u in id_to_method and v in id_to_method:
            method_edges_by_name.append((id_to_method[u], id_to_method[v]))
    return list(id_to_method.values()), method_edges_by_name

def parse_codeline_flow(text: str):
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    nodes_seq: List[Tuple[str, Dict[str,str]]] = []
    edges_local: List[Tuple[int,int]] = []
    codeline_local_ids: List[int] = []
    mode = "nodes"
    for ln in lines:
        if ln == "edge:":
            mode = "edges"; continue
        if mode == "nodes":
            m = re.match(r"(\d+)\s+(method|codeline):\s*\{(.+)\}\s*$", ln)
            if not m:
                continue
            idx = int(m.group(1))
            kind = m.group(2)
            body = m.group(3)
            payload: Dict[str,str] = {}
            for k,v in re.findall(r"(\w+)\s*:\s*([^,}]+)", body):
                payload[k] = norm(v)
            nodes_seq.append((kind, payload))
            if kind == "codeline":
                codeline_local_ids.append(idx)
        else:
            m = re.match(r"(\d+)\s*->\s*(\d+)$", ln)
            if m:
                edges_local.append((int(m.group(1)), int(m.group(2))))
    return nodes_seq, edges_local, codeline_local_ids

def parse_var_file(text: str):
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    mode = None
    nodes: List[Tuple[str,str]] = []
    edges: List[Tuple[int,int]] = []
    for ln in lines:
        low = ln.lower()
        if low == "node:":
            mode = "nodes"; continue
        if low == "edge:":
            mode = "edges"; continue
        if mode == "nodes":
            m = re.match(r"(\d+)\s*\{\s*content:\s*([^,}]+)\s*,\s*type:\s*([^}]+)\}\s*$", ln)
            if m:
                name = norm(m.group(2))
                vtype = norm(m.group(3))
                nodes.append((name, vtype))
        elif mode == "edges":
            m = re.match(r"(\d+)\s*->\s*(\d+)$", ln)
            if m:
                edges.append((int(m.group(1)), int(m.group(2))))
    return nodes, edges

COV_RE = re.compile(
    r"""
    ^
    (?P<pkg>[A-Za-z_][\w\.]*)
    \$
    (?P<classfull>[A-Za-z_]\w*(?:\$(?:[A-Za-z_]\w*|\d+))*)  
    \#
    (?P<meth>[A-Za-z_]\w*|<init>|<clinit>)
    \(\)
    :
    (?P<line>\d+)
    $
    """, re.VERBOSE
)

def parse_testfile(p: Path) -> Tuple[str, str, List[Tuple[str,int]]]:

    raw = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    name = norm(raw[0]) if len(raw) >= 1 else ""
    report = norm(raw[1]) if len(raw) >= 2 else ""
    cov: List[Tuple[str,int]] = []
    for ln in raw[2:]:
        s = ln.strip()
        if not s:
            continue
        m = COV_RE.match(s)
        if m:
            fn = m.group("meth")
            line = int(m.group("line"))
            cov.append((fn, line))
    return name, report, cov

class GlobalIndex:
    def __init__(self):
        self.nodes_out: List[str] = []
        self.key_to_gid: Dict[Tuple, int] = {}

    def add_method(self, content: str) -> int:
        key = ("method", content)
        if key in self.key_to_gid:
            return self.key_to_gid[key]
        gid = len(self.nodes_out) + 1
        self.key_to_gid[key] = gid
        self.nodes_out.append(f"{gid} method: {{content: {content}}}")
        return gid

    def add_codeline(self, content: str, linetype: str, sus: str) -> int:
        key = ("codeline", content, linetype, sus)
        if key in self.key_to_gid:
            return self.key_to_gid[key]
        gid = len(self.nodes_out) + 1
        self.key_to_gid[key] = gid
        self.nodes_out.append(f"{gid} codeline: {{content: {content}, type: {linetype}, sus: {sus}}}")
        return gid

    def add_var(self, name: str, vtype: str) -> int:
        key = ("var", name, vtype)
        if key in self.key_to_gid:
            return self.key_to_gid[key]
        gid = len(self.nodes_out) + 1
        self.key_to_gid[key] = gid
        self.nodes_out.append(f"{gid} var: {{content: {name}, type: {vtype}}}")
        return gid

    def add_test(self, content: str, report: str) -> int:
        key = ("test", content, report)
        if key in self.key_to_gid:
            return self.key_to_gid[key]
        gid = len(self.nodes_out) + 1
        self.key_to_gid[key] = gid
        self.nodes_out.append(f"{gid} [test] {{content: {content}, report: {report}}}")
        return gid

def main():
    ap = argparse.ArgumentParser(description="Static Analysis")
    ap.add_argument("--method_graph", type=Path, required=True)
    ap.add_argument("--codeline_flow", type=Path, required=True)
    ap.add_argument("--var_dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--testfile", type=Path, required=False)
    args = ap.parse_args()

    gi = GlobalIndex()
    edges_out: List[str] = []


    cl_text = args.codeline_flow.read_text(encoding="utf-8", errors="ignore")
    cl_nodes, cl_edges_local, cl_codeline_ids = parse_codeline_flow(cl_text)

    local_id_to_gid: Dict[int, int] = {}
    method_name_to_gid: Dict[str, int] = {}

    method_to_codeline_gids: Dict[str, List[int]] = {}
    shortname_to_methods: Dict[str, List[str]] = {}

    current_method_full: Optional[str] = None
    for local_idx, (kind, payload) in enumerate(cl_nodes, start=1):
        if kind == "method":
            mname = payload.get("content","")
            gid = gi.add_method(mname)
            local_id_to_gid[local_idx] = gid
            method_name_to_gid.setdefault(mname, gid)
            current_method_full = mname
            if "#" in mname:
                short = mname.split("#",1)[1]
                shortname_to_methods.setdefault(short, []).append(mname)
            method_to_codeline_gids.setdefault(mname, [])
        else:
            gid = gi.add_codeline(payload.get("content",""), payload.get("type",""), payload.get("sus",""))
            local_id_to_gid[local_idx] = gid
            if current_method_full:
                method_to_codeline_gids.setdefault(current_method_full, []).append(gid)

    for u,v in cl_edges_local:
        ug = local_id_to_gid.get(u)
        vg = local_id_to_gid.get(v)
        if ug and vg:
            edges_out.append(f"{ug}->{vg}")

    mg_text = args.method_graph.read_text(encoding="utf-8", errors="ignore")
    mg_methods, mg_edges_by_name = parse_method_graph(mg_text)

    for mname in mg_methods:
        if mname not in method_name_to_gid:
            gid = gi.add_method(mname)
            method_name_to_gid[mname] = gid
            if "#" in mname:
                short = mname.split("#",1)[1]
                shortname_to_methods.setdefault(short, []).append(mname)
            method_to_codeline_gids.setdefault(mname, [])

    for a_name, b_name in mg_edges_by_name:
        ag = method_name_to_gid.get(a_name)
        bg = method_name_to_gid.get(b_name)
        if ag and bg:
            edges_out.append(f"{ag}->{bg}")

    if not args.var_dir.is_dir():
        raise ValueError(f"--var_dir error：{args.var_dir}")

    for f in sorted(args.var_dir.glob("*.txt"), key=lambda p: (p.stem.isdigit(), int(p.stem) if p.stem.isdigit() else 10**12, p.stem)):
        stem = f.stem
        if not stem.isdigit():
            continue
        local_codeline_id = int(stem)
        codeline_gid = local_id_to_gid.get(local_codeline_id)
        if not codeline_gid:
            continue

        vtext = f.read_text(encoding="utf-8", errors="ignore")
        v_nodes, v_edges = parse_var_file(vtext)

        v_local_to_gid: Dict[int,int] = {}
        for i,(name,vtype) in enumerate(v_nodes, start=1):
            gid = gi.add_var(name, vtype)
            v_local_to_gid[i] = gid

        indeg: Dict[int,int] = {i:0 for i in v_local_to_gid.keys()}
        for u,v in v_edges:
            ug = v_local_to_gid.get(u)
            vg = v_local_to_gid.get(v)
            if ug and vg:
                edges_out.append(f"{ug}->{vg}")
                indeg[v] = indeg.get(v,0) + 1

        for i,deg in indeg.items():
            if deg == 0 and i in v_local_to_gid:
                edges_out.append(f"{codeline_gid}->{v_local_to_gid[i]}")

    if args.testfile and args.testfile.exists():
        t_name, t_report, cov = parse_testfile(args.testfile)
        test_gid = gi.add_test(t_name, t_report)

        for (fn_short, _line) in cov:
            methods = shortname_to_methods.get(fn_short, [])
            for mfull in methods:
                for cl_gid in method_to_codeline_gids.get(mfull, []):
                    edges_out.append(f"{test_gid}->{cl_gid}")


    out_lines: List[str] = []
    out_lines.append("nodes:")
    out_lines.extend(gi.nodes_out)
    out_lines.append("edge:")
    out_lines.extend(edges_out)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"fnish:{args.out}")

if __name__ == "__main__":
    main()
