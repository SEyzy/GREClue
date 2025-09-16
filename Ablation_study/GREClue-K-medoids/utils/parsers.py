
import re, json
from typing import List, Dict, Any

SUS_LINE_RE = re.compile(r"^(?P<sig>.+?):(?P<line>\d+);(?P<score>[0-9.]+)\s*$")

def parse_suspect_list(text: str) -> List[Dict[str, Any]]:
    items = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw: continue
        m = SUS_LINE_RE.match(raw)
        if not m:
            items.append({"signature": raw, "line": None, "score": 0.0})
            continue
        items.append({"signature": m.group("sig"),
                      "line": int(m.group("line")),
                      "score": float(m.group("score"))})
    return items

def parse_graph_json(text: str) -> Dict[str, Any]:
    return json.loads(text)

def parse_graph_block(text: str) -> Dict[str, Any]:
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    try: idx_nodes = lines.index("nodes:")
    except ValueError: idx_nodes = -1
    try: idx_edges = lines.index("file_edge:")
    except ValueError: idx_edges = -1

    nodes, edges = [], []

    if idx_nodes != -1:
        node_lines = lines[idx_nodes+1 : (idx_edges if idx_edges!=-1 else len(lines))]
        NODE_RE = re.compile(r"^\s*(\d+)\s+(\w+):\s*\{(.+)\}\s*$")
        for ln in node_lines:
            m = NODE_RE.match(ln)
            if not m: continue
            nid = int(m.group(1)); kind = m.group(2); attrs = m.group(3)
            content, ntype, sus = None, None, 0.0
            for part in attrs.split(","):
                if ":" not in part: continue
                k,v = part.split(":",1); k=k.strip(); v=v.strip()
                if k=="content": content=v
                elif k=="type": ntype=v
                elif k=="sus":
                    try: sus=float(v)
                    except: sus=0.0
            nodes.append({"id": nid, "kind": kind, "content": (content or "").strip(),
                          "type": (ntype or "").strip(), "sus": sus})

    if idx_edges != -1:
        edge_lines = lines[idx_edges+1:]
        E_RE = re.compile(r"^\s*(\d+)\s*->\s*(\d+)\s*$")
        for ln in edge_lines:
            m = E_RE.match(ln)
            if m: edges.append([int(m.group(1)), int(m.group(2))])

    return {"nodes": nodes, "edges": edges}
