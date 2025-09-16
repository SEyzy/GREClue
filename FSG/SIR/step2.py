from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

HAS_TS = True
try:
    from tree_sitter import Parser
    try:
        from tree_sitter_languages import get_language
        LANG_C = get_language("c")
    except Exception:
        HAS_TS = False
        LANG_C = None
except Exception:
    HAS_TS = False
    LANG_C = None

WS_RE = re.compile(r"\s+")

def norm_space(s: str) -> str:
    return WS_RE.sub(" ", (s or "").strip())

def read_lines(p: Path) -> List[str]:
    try:
        return p.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return p.read_text(encoding="latin-1", errors="ignore").splitlines()


LOC_RE = re.compile(
    r"""
    ^
    (?P<pkg>[A-Za-z_][\w\.]*)
    \$
    (?P<filebase>[A-Za-z_]\w*)
    \#
    (?P<func>[A-Za-z_]\w*)
    \(\)
    :
    (?P<line>\d+)
    ;
    (?P<sus>\d+(?:\.\d+)?)
    $
    """,
    re.VERBOSE
)

class Susp:
    def __init__(self, raw: str, pkg: str, filebase: str, func: str, line: int, sus: float):
        self.raw = raw
        self.pkg = pkg
        self.filebase = filebase
        self.func = func
        self.line = line
        self.sus = sus

    @property
    def rel_hint(self) -> Path:
        return Path(*self.pkg.split(".")) / f"{self.filebase}.c"

def parse_loc_line(s: str) -> Susp:
    m = LOC_RE.match(s.strip())
    if not m:
        raise ValueError(f"error：{s}")
    return Susp(
        raw=s.strip(),
        pkg=m.group("pkg"),
        filebase=m.group("filebase"),
        func=m.group("func"),
        line=int(m.group("line")),
        sus=float(m.group("sus")),
    )


def find_c_file(project_root: Path, rel_hint: Path, filebase: str) -> Optional[Path]:
    candidates = [
        project_root / rel_hint,
        project_root / "src" / rel_hint,
        project_root / "source" / rel_hint,
        project_root / "lib" / rel_hint,
        project_root / "src" / "c" / rel_hint,
        project_root / filebase / f"{filebase}.c",
    ]
    for p in candidates:
        if p.is_file():
            return p
    for p in project_root.rglob(f"{filebase}.c"):
        return p
    return None


def node_text(src: bytes, node) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")

def walk(node):
    yield node
    for i in range(getattr(node, "named_child_count", 0)):
        yield from walk(node.named_child(i))

def list_functions_ranges_ts(cfile: Path) -> Dict[str, Tuple[int,int]]:
    out: Dict[str, Tuple[int,int]] = {}
    if not HAS_TS or LANG_C is None:
        return out
    parser = Parser()
    parser.set_language(LANG_C)
    source = cfile.read_bytes()
    tree = parser.parse(source)
    root = tree.root_node

    for defn in walk(root):
        if defn.type != "function_definition":
            continue

        func_name = None
        decl = None
        for i in range(defn.named_child_count):
            ch = defn.named_child(i)
            if ch.type in ("function_declarator", "declarator"):
                decl = ch
                break
        if decl is not None:
            stack = [decl]
            while stack:
                n = stack.pop()
                if n.type == "identifier":
                    func_name = node_text(source, n).strip()
                    break
                for j in range(n.named_child_count):
                    stack.append(n.named_child(j))
        if not func_name:
            continue

        body = None
        for i in range(defn.named_child_count):
            ch = defn.named_child(i)
            if ch.type == "compound_statement":
                body = ch
                break
        if body is None:
            continue
        s = body.start_point[0] + 1
        e = body.end_point[0] + 1
        out[func_name] = (s, e)
    return out


FUNC_DEF_LINE_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*\([^;{]*\)\s*\{")

def rough_owner_func(lines: List[str], line_no: int) -> Optional[str]:
    i = min(max(line_no-1, 0), len(lines)-1)
    while i >= 0:
        m = FUNC_DEF_LINE_RE.match(lines[i])
        if m:
            return m.group(1)
        i -= 1
    return None


KEY_TYPES = [
    ("else if", "if"),
    ("if", "if"),
    ("else", "if"),
    ("return", "return"),
    ("for", "for"),
    ("while", "while"),
    ("do", "do"),
    ("switch", "switch"),
    ("break", "break"),
    ("continue", "continue"),
]

def classify_line(line: str) -> str:
    s = line.strip()
    low = s.lower()
    for kw, lab in KEY_TYPES:
        if low.startswith(kw):
            return lab
    if "=" in s and "==" not in s and "!=" not in s and not s.lstrip().startswith("#"):
        return "assignment"
    if "(" in s and s.rstrip().endswith(";"):
        return "call"
    return "other"


def main():
    ap = argparse.ArgumentParser(description="Static Analysis")
    ap.add_argument("--project", required=True, type=Path)
    ap.add_argument("--locs", required=True, type=Path)
    ap.add_argument("--txtout", required=True, type=Path)
    ap.add_argument("--linedir", required=False, type=Path)
    args = ap.parse_args()

    project_root = args.project.resolve()
    if args.linedir:
        args.linedir.mkdir(parents=True, exist_ok=True)

    loc_lines = [ln.strip() for ln in args.locs.read_text(encoding="utf-8").splitlines() if ln.strip()]
    susp_list: List[Susp] = [parse_loc_line(s) for s in loc_lines]


    method_to_items: Dict[str, List[Tuple[int, float, str]]] = defaultdict(list)  # mkey -> [(line_no, sus, code)]
    for sp in susp_list:
        cfile = find_c_file(project_root, sp.rel_hint, sp.filebase)
        rel = str(cfile.relative_to(project_root)) if cfile else ""
        lines = read_lines(cfile) if cfile and cfile.exists() else []
        code = lines[sp.line - 1] if 1 <= sp.line <= len(lines) else ""
        code = norm_space(code)


        owner_func = sp.func
        if cfile and cfile.exists():
            ranges = list_functions_ranges_ts(cfile) if HAS_TS and LANG_C else {}
            if ranges:
                for fn, (s, e) in ranges.items():
                    if s <= sp.line <= e:
                        owner_func = fn
                        break
            else:
                ofn = rough_owner_func(lines, sp.line)
                if ofn:
                    owner_func = ofn

        full_no_ext = str(Path(rel).with_suffix(""))
        mkey = f"{full_no_ext}#{owner_func}" if full_no_ext else f"<unknown>#{owner_func}"
        method_to_items[mkey].append((sp.line, sp.sus, code))


    nodes: List[str] = []
    edges: List[str] = []
    id_counter = 1
    id_map: Dict[Tuple[str, Optional[int]], int] = {}  # (method_key, line_no or None) -> id

    def add_method_node(method_key: str) -> int:
        k = (method_key, None)
        if k in id_map:
            return id_map[k]
        nonlocal id_counter
        nid = id_counter
        id_counter += 1
        id_map[k] = nid
        nodes.append(f"{nid} method: {{content: {method_key}}}")
        return nid

    def add_codeline_node(method_key: str, content: str, typ: str, sus: float, line_no: int) -> int:
        k = (method_key, line_no)
        if k in id_map:
            return id_map[k]
        nonlocal id_counter
        nid = id_counter
        id_counter += 1
        id_map[k] = nid
        nodes.append(f"{nid} codeline: {{content: {content}, type: {typ}, sus: {sus}}}")
        if args.linedir:
            (args.linedir / f"{nid}.txt").write_text(content + "\n", encoding="utf-8")
        return nid

    for mkey, items in method_to_items.items():
        mid = add_method_node(mkey)
        items_sorted = sorted(items, key=lambda x: x[0])

        first_line_node = None
        prev_node = None

        for (ln, sus, code) in items_sorted:
            typ = classify_line(code)
            cid = add_codeline_node(mkey, code, typ, sus, ln)
            if first_line_node is None:
                first_line_node = cid
                edges.append(f"{mid}->{cid}")
            if prev_node is not None:
                edges.append(f"{prev_node}->{cid}")
            prev_node = cid


    out_lines: List[str] = []
    out_lines.extend(nodes)
    out_lines.append("edge:")
    out_lines.extend(edges)

    args.txtout.parent.mkdir(parents=True, exist_ok=True)
    args.txtout.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"finish {args.txtout}")


if __name__ == "__main__":
    main()
