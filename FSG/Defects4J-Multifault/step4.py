import re
import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Set

WS_RE = re.compile(r"\s+")

def norm(s: str) -> str:
    return WS_RE.sub(" ", s.strip())

def parse_graph1(text: str):
    lines = [l.strip() for l in text.splitlines()]
    method_nodes: List[str] = []
    edges: List[Tuple[str, str]] = []
    mode = None
    tmp_idx_to_name: Dict[str, str] = {}
    for ln in lines:
        if not ln:
            continue
        low = ln.lower()
        if low.startswith("fileline"):
            mode = None
            continue
        if low.startswith("methodnodes"):
            mode = "nodes"
            continue
        if ln == "edge:":
            mode = "edges"
            continue
        if mode == "nodes":
            m = re.match(r"(\d+)\s+(.+)$", ln)
            if m:
                idx, name = m.group(1), norm(m.group(2))
                tmp_idx_to_name[idx] = name
                method_nodes.append(name)
        elif mode == "edges":
            m = re.match(r"(\d+)\s*->\s*(\d+)$", ln)
            if m:
                a, b = m.group(1), m.group(2)
                if a in tmp_idx_to_name and b in tmp_idx_to_name:
                    edges.append((tmp_idx_to_name[a], tmp_idx_to_name[b]))
    return method_nodes, edges


def parse_graph2(text: str):
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    nodes: List[Tuple[str, dict]] = []
    edges: List[Tuple[int, int]] = []
    mode = "nodes"
    for ln in lines:
        if ln == "edge:":
            mode = "edges"
            continue
        if mode == "nodes":
            m = re.match(r"(\d+)\s+(method|codeline):\s*\{(.+)\}\s*$", ln)
            if not m:
                continue
            kind = m.group(2)
            body = m.group(3)
            payload: Dict[str, str] = {}
            kvs = re.findall(r"(\w+)\s*:\s*([^,}]+)", body)
            for k, v in kvs:
                payload[k] = norm(v)
            nodes.append((kind, payload))
        else:
            m = re.match(r"(\d+)\s*->\s*(\d+)$", ln)
            if m:
                edges.append((int(m.group(1)), int(m.group(2))))
    return nodes, edges


def parse_one_var_graph(text: str):
    node_lines = re.findall(r"Node\s+(\d+):\s*\[(.+?)\]", text)
    raw_nodes: Dict[int, Dict[str, str]] = {}
    for idx_str, inside in node_lines:
        idx = int(idx_str)
        fields = dict(re.findall(r"(\w+)\s*=\s*([^,\]]+)", inside))
        t = fields.get("Type", "").strip()
        name = fields.get("Name", "").strip()
        raw_nodes[idx] = {"Type": t, "Name": name}

    kept: Dict[int, Tuple[str, str]] = {}
    for nid, info in raw_nodes.items():
        t = info.get("Type", "")
        if t.upper() in ("BEGIN", "EXIT"):
            continue
        kept[nid] = (norm(info.get("Name", "")), norm(t))

    part = text.split("Graph Edges:")
    kept_edges: List[Tuple[int, int]] = []
    if len(part) > 1:
        var_edges_all = re.findall(r"(\d+)\s*->\s*(\d+)", part[1])
        for a_str, b_str in var_edges_all:
            a, b = int(a_str), int(b_str)
            if a in kept and b in kept:
                kept_edges.append((a, b))

    indeg: Dict[int, int] = {nid: 0 for nid in kept.keys()}
    for a, b in kept_edges:
        indeg[b] += 1
    heads = {nid for nid, d in indeg.items() if d == 0}

    return kept, kept_edges, heads


COV_RE = re.compile(
    r"""
    ^
    (?P<pkg>[a-zA-Z_][\w\.]*)     
    \$
    (?P<classfull>[A-Za-z_]\w*(?:\$(?:[A-Za-z_]\w*|\d+))*)  
    \#
    (?P<meth>[A-Za-z_][\w$]*|<clinit>|<init>)   
    \(\)
    :
    (?P<line>\d+)
    $
    """, re.VERBOSE
)

def normalize_cov_method(pkg: str, classfull: str, meth: str) -> str:

    return f"{pkg}.{classfull}#{meth}"


class GlobalIndexer:
    def __init__(self):
        self.key_to_gid: Dict[Tuple, int] = {}
        self.nodes_out: List[str] = []

    def get_or_add(self, kind: str, payload: Dict[str, str]) -> int:
        if kind == "method":
            key = (kind, payload.get("content", ""))
            line_repr = f"method: {{content: {payload.get('content','')}}}"
        elif kind == "codeline":
            key = (kind, payload.get("content",""), payload.get("type",""), payload.get("sus",""))
            line_repr = f"codeline: {{content: {payload.get('content','')}, type: {payload.get('type','')}, sus: {payload.get('sus','')}}}"
        elif kind == "var":
            key = (kind, payload.get("content",""), payload.get("type",""))
            line_repr = f"var: {{content: {payload.get('content','')}, type: {payload.get('type','')}}}"
        elif kind == "test":
            key = (kind, payload.get("content",""), payload.get("report",""))
            line_repr = f"[test] {{content: {payload.get('content','')}, report: {payload.get('report','')}}}"
        else:
            raise ValueError("unknown kind")

        if key in self.key_to_gid:
            return self.key_to_gid[key]
        gid = len(self.nodes_out) + 1
        self.key_to_gid[key] = gid
        self.nodes_out.append(f"{gid} {line_repr}")
        return gid

def main():
    ap = argparse.ArgumentParser(description="Static Analysis")
    ap.add_argument("--g1", type=Path, required=True)
    ap.add_argument("--g2", type=Path, required=True)
    ap.add_argument("--g3dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--testfile", type=Path, required=False)
    args = ap.parse_args()

    g1_text = args.g1.read_text(encoding="utf-8")
    g2_text = args.g2.read_text(encoding="utf-8")

    g1_methods, g1_edges = parse_graph1(g1_text)
    g2_nodes, g2_edges_local = parse_graph2(g2_text)


    gi = GlobalIndexer()

    name_to_gid: Dict[str, int] = {}
    for meth in g1_methods:
        gid = gi.get_or_add("method", {"content": norm(meth)})
        name_to_gid[meth] = gid
    file_edges_out: List[str] = []
    for a_name, b_name in g1_edges:
        a_id = name_to_gid.get(a_name)
        b_id = name_to_gid.get(b_name)
        if a_id and b_id:
            file_edges_out.append(f"{a_id}->{b_id}")


    g2_local_to_gid: Dict[int, int] = {}
    g2_local_kind: Dict[int, str] = {}

    method_to_codeline_gids: Dict[str, List[int]] = {}

    current_method_fullname: Optional[str] = None
    for local_idx, (kind, payload) in enumerate(g2_nodes, start=1):
        if kind == "method":
            mname = norm(payload.get("content",""))
            gid = gi.get_or_add("method", {"content": mname})
            current_method_fullname = mname

            method_to_codeline_gids.setdefault(mname, [])
        else:  # codeline
            gid = gi.get_or_add("codeline", {
                "content": norm(payload.get("content","")),
                "type": norm(payload.get("type","")),
                "sus": norm(payload.get("sus","")),
            })

            if current_method_fullname:
                method_to_codeline_gids.setdefault(current_method_fullname, []).append(gid)
        g2_local_to_gid[local_idx] = gid
        g2_local_kind[local_idx] = kind

    codeline_edges_out: List[str] = []
    for (u_loc, v_loc) in g2_edges_local:
        u_gid = g2_local_to_gid.get(u_loc)
        v_gid = g2_local_to_gid.get(v_loc)
        if u_gid and v_gid:
            codeline_edges_out.append(f"{u_gid}->{v_gid}")


    var_edges_out: List[str] = [] 
    if not args.g3dir.is_dir():
        raise ValueError(f"--g3dir path error")

    for txt_path in sorted(args.g3dir.glob("*.txt")):
        stem = txt_path.stem
        if not stem.isdigit():
            continue
        codeline_local_id = int(stem)

        cl_gid = g2_local_to_gid.get(codeline_local_id)
        if cl_gid is None or g2_local_kind.get(codeline_local_id) != "codeline":
            continue

        text = txt_path.read_text(encoding="utf-8", errors="ignore")
        kept_nodes, kept_edges, heads = parse_one_var_graph(text)

        var_local_to_gid: Dict[int, int] = {}
        for nid, (name, vtype) in kept_nodes.items():
            gid = gi.get_or_add("var", {"content": norm(name), "type": norm(vtype)})
            var_local_to_gid[nid] = gid

        for a, b in kept_edges:
            a_gid = var_local_to_gid.get(a)
            b_gid = var_local_to_gid.get(b)
            if a_gid and b_gid:
                var_edges_out.append(f"{a_gid}->{b_gid}")

        for h in heads:
            h_gid = var_local_to_gid.get(h)
            if h_gid:
                var_edges_out.append(f"{cl_gid}->{h_gid}")


    test_edges_out: List[str] = []
    if args.testfile and args.testfile.exists():
        raw = args.testfile.read_text(encoding="utf-8", errors="ignore").splitlines()
        if len(raw) >= 2:
            test_name = norm(raw[0])
            test_report = norm(raw[1])
            test_gid = gi.get_or_add("test", {"content": test_name, "report": test_report})


            for ln in raw[2:]:
                s = ln.strip()
                if not s:
                    continue
                m = COV_RE.match(s)
                if not m:
                    continue
                pkg = m.group("pkg")
                classfull = m.group("classfull")
                meth = m.group("meth")
                method_key = normalize_cov_method(pkg, classfull, meth)  # 与图2的 method content 对齐
                clist = method_to_codeline_gids.get(method_key, [])
                for gid in clist:
                    test_edges_out.append(f"{test_gid}->{gid}")


    out_lines: List[str] = []
    out_lines.append("nodes:")
    out_lines.extend(gi.nodes_out)
    out_lines.append("file_edge:")
    out_lines.extend(file_edges_out)
    out_lines.append("codeline_edge:")
    out_lines.extend(codeline_edges_out)
    out_lines.append("var_edge:")
    out_lines.extend(var_edges_out)
    out_lines.append("test_edge:")
    out_lines.extend(test_edges_out)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"finish{args.out}")

if __name__ == "__main__":
    main()
