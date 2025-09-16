
from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Dict, Set, List, Tuple
from collections import defaultdict, deque
from tree_sitter import Parser

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


def load_c_language():
    try:
        from tree_sitter_languages import get_language
        return get_language("c")
    except Exception:
        raise RuntimeError("pip install tree_sitter tree_sitter_languages")

def node_text(src: bytes, node) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="ignore").strip()

def walk(node):
    yield node
    for i in range(node.named_child_count):
        yield from walk(node.named_child(i))

def extract_functions_and_calls_c(src_path: Path, parser: Parser) -> Dict[str, Tuple[int,int,Set[str]]]:

    source = src_path.read_bytes()
    tree = parser.parse(source)
    root = tree.root_node
    out: Dict[str, Tuple[int,int,Set[str]]] = {}

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
                    func_name = node_text(source, n)
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

        start_line = body.start_point[0] + 1
        end_line   = body.end_point[0] + 1


        callees: Set[str] = set()
        for inner in walk(body):
            if inner.type == "call_expression" and inner.named_child_count > 0:
                fn = inner.named_child(0)
                if fn.type == "identifier":
                    callee = node_text(source, fn)
                    if callee:
                        callees.add(callee)

        out[func_name] = (start_line, end_line, callees)

    return out


def read_lines(p: Path) -> List[str]:
    try:
        return p.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return p.read_text(encoding="latin-1", errors="ignore").splitlines()

def find_c_file(project_root: Path, rel_hint: Path, filebase: str) -> Path | None:
    cand = [
        project_root / rel_hint,
        project_root / "src" / rel_hint,
        project_root / "source" / rel_hint,
        project_root / "lib" / rel_hint,
        project_root / "src" / "c" / rel_hint,
    ]
    for p in cand:
        if p.is_file():
            return p
    for p in project_root.rglob(f"{filebase}.c"):
        return p
    return None

def index_project(project_root: Path) -> Tuple[Dict[str, Dict[str, Tuple[int,int,Set[str]]]], Dict[str,str]]:

    LANG_C = load_c_language()
    parser = Parser()
    parser.set_language(LANG_C)

    file_to_funcs: Dict[str, Dict[str, Tuple[int,int,Set[str]]]] = {}
    func_to_fullname: Dict[str, str] = {}

    for cfile in sorted(project_root.rglob("*.c")):
        try:
            fmap = extract_functions_and_calls_c(cfile, parser)
        except Exception:
            continue
        if not fmap:
            continue
        rel = str(cfile.relative_to(project_root))
        file_to_funcs[rel] = fmap
        base_no_ext = str(Path(rel).with_suffix(""))
        for fn in fmap.keys():
            func_to_fullname.setdefault(fn, f"{base_no_ext}#{fn}")
    return file_to_funcs, func_to_fullname

def build_direct_graph(file_to_funcs: Dict[str, Dict[str, Tuple[int,int,Set[str]]]]) -> Dict[str, Set[str]]:
    g: Dict[str, Set[str]] = defaultdict(set)
    for _rel, fmap in file_to_funcs.items():
        for fn, (_s,_e,callees) in fmap.items():
            g[fn].update(callees)
    return g


def locate_owner_func_for_line(cfile: Path, target_line: int, fmap: Dict[str, Tuple[int,int,Set[str]]]) -> str | None:
    for fn, (s, e, _c) in fmap.items():
        if s <= target_line <= e:
            return fn
    return None

def seeds_from_locs(project_root: Path,
                    file_to_funcs: Dict[str, Dict[str, Tuple[int,int,Set[str]]]],
                    suspicious: List[Susp]) -> Set[str]:
    seeds: Set[str] = set()

    for sp in suspicious:
        cpath = find_c_file(project_root, sp.rel_hint, sp.filebase)
        if not cpath:

            seeds.add(sp.func)
            continue
        rel = str(cpath.relative_to(project_root))
        fmap = file_to_funcs.get(rel)
        if not fmap:

            seeds.add(sp.func)
            continue
        owner = locate_owner_func_for_line(cpath, sp.line, fmap)
        seeds.add(owner if owner else sp.func)
    return seeds


def reachable_subgraph_from(seeds: Set[str],
                            g: Dict[str, Set[str]]) -> Set[str]:

    reach: Set[str] = set()
    q = deque(seeds)
    reach.update(seeds)
    while q:
        u = q.popleft()
        for v in g.get(u, ()):
            if v not in reach:
                reach.add(v)
                q.append(v)
    return reach


def main():
    ap = argparse.ArgumentParser(description="Static Analysis")
    ap.add_argument("--project", required=True, type=Path)
    ap.add_argument("--locs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    project_root = args.project.resolve()
    loc_lines = [ln.strip() for ln in args.locs.read_text(encoding="utf-8").splitlines() if ln.strip()]
    suspicious = [parse_loc_line(s) for s in loc_lines]


    file_to_funcs, func_to_fullname = index_project(project_root)
    g = build_direct_graph(file_to_funcs)


    seeds = seeds_from_locs(project_root, file_to_funcs, suspicious)


    keep_funcs = reachable_subgraph_from(seeds, g)


    fullname_to_id: Dict[str, int] = {}
    nodes_out: List[str] = []
    next_id = 1


    for rel in sorted(file_to_funcs.keys()):
        base_no_ext = str(Path(rel).with_suffix(""))
        for fn in sorted(file_to_funcs[rel].keys()):
            if fn not in keep_funcs:
                continue
            fullname = f"{base_no_ext}#{fn}"
            if fullname not in fullname_to_id:
                fullname_to_id[fullname] = next_id
                nodes_out.append(f"{next_id} method: {{content: {fullname}}}")
                next_id += 1


    edges_out: List[str] = []

    def fn_to_full(fn: str) -> str | None:

        return func_to_fullname.get(fn)

    def full_to_id(full: str) -> int | None:
        return fullname_to_id.get(full)

    for rel, fmap in file_to_funcs.items():
        base_no_ext = str(Path(rel).with_suffix(""))
        for fn, (_s,_e,callees) in fmap.items():
            if fn not in keep_funcs:
                continue
            src_full = f"{base_no_ext}#{fn}"
            src_id = full_to_id(src_full)
            if not src_id:
                continue
            for cal in sorted(callees):
                if cal not in keep_funcs:
                    continue
                dst_full = fn_to_full(cal)
                if not dst_full:
                    continue
                dst_id = full_to_id(dst_full)
                if dst_id:
                    edges_out.append(f"{src_id}->{dst_id}")


    out_lines: List[str] = []
    out_lines.append("methodnodes:")
    out_lines.extend(nodes_out)
    out_lines.append("edge:")
    out_lines.extend(edges_out)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"finish：{args.out}")

if __name__ == "__main__":
    main()
