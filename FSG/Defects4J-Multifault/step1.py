from __future__ import annotations
import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

import javalang

WS_RE = re.compile(r"\s+")

def normalize_content(s: str) -> str:
    if s is None:
        return ""
    return WS_RE.sub(" ", str(s)).strip()

LOC_RE = re.compile(
    r"""
    ^
    (?P<pkg>[a-zA-Z_][\w\.]*)          
    \$
    (?P<class_full>                    
        [A-Za-z_]\w*
        (?:\$(?:[A-Za-z_]\w*|\d+))*
    )
    \#
    (?P<meth>                          
        [A-Za-z_][\w$]*|<clinit>|<init>
    )
    \(
        (?P<params>[^)]*)              
    \)
    :
    (?P<line>\d+)
    ;
    (?P<score>\d+(?:\.\d+)?)
    $
    """,
    re.VERBOSE
)

@dataclass(frozen=True)
class SuspiciousLoc:
    package: str
    class_full: str
    method: str
    line: int
    score: float

    @property
    def top_class(self) -> str:
        return self.class_full.split("$", 1)[0]

    def rel_path(self) -> Path:
        return Path(*self.package.split(".")) / f"{self.top_class}.java"

    def method_key(self) -> str:
        return f"{self.package}.{self.class_full}#{self.method}"

def parse_suspicious_loc(s: str) -> SuspiciousLoc:
    m = LOC_RE.match(s.strip())
    if not m:
        raise ValueError(f"error：{s}")
    pkg = m.group("pkg")
    class_full = m.group("class_full")
    meth_raw = m.group("meth")
    meth = class_full.split("$")[-1] if meth_raw == class_full else meth_raw
    return SuspiciousLoc(
        package=pkg,
        class_full=class_full,
        method=meth,
        line=int(m.group("line")),
        score=float(m.group("score")),
    )

def find_java_file(project_root: Path, rel: Path) -> Optional[Path]:
    for p in [
        project_root / "src" / "main" / "java" / rel,
        project_root / "src" / "test" / "java" / rel,
        project_root / rel,
    ]:
        if p.is_file():
            return p
    name = rel.name
    for p in project_root.rglob(name):
        try:
            if tuple(p.parts[-len(rel.parts):]) == tuple(rel.parts):
                return p
        except Exception:
            pass
    return None

def read_lines(p: Path) -> List[str]:
    try:
        return p.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return p.read_text(encoding="latin-1", errors="ignore").splitlines()

@dataclass
class MethodAstView:
    key: str
    start_line: int
    end_line: int
    file_path: Path

def _iter_body_nodes(body):
    if body is None:
        return []
    if isinstance(body, list):
        return body
    stmts = getattr(body, "statements", None)
    return stmts or []

def _max_body_line(body):
    max_line = None
    for n in _iter_body_nodes(body):
        pos = getattr(n, "position", None)
        if pos and pos.line:
            max_line = pos.line if max_line is None else max(max_line, pos.line)
    return max_line

def list_all_methods_in_file(p: Path) -> List[MethodAstView]:
    try:
        src = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        src = p.read_text(encoding="latin-1", errors="ignore")
    try:
        cu = javalang.parse.parse(src)
    except Exception:
        return []

    pkg = cu.package.name if cu.package else ""
    methods: List[MethodAstView] = []

    InitDecl = getattr(javalang.tree, "InitializerDeclaration", None)
    InitAlias = getattr(javalang.tree, "Initializer", None)

    def _is_initializer(member) -> bool:
        if InitDecl and isinstance(member, InitDecl): return True
        if InitAlias and isinstance(member, InitAlias): return True
        return member.__class__.__name__ in ("InitializerDeclaration", "Initializer")

    def visit_type(t, stack: List[str]):
        if hasattr(t, "name") and isinstance(t.name, str):
            stack = stack + [t.name]
        full_class = "$".join(stack) if stack else ""
        simple = stack[-1] if stack else ""

        for decl in getattr(t, "methods", []) or []:
            pos = getattr(decl, "position", None)
            s = pos.line if pos else 1
            e = _max_body_line(getattr(decl, "body", None)) or s
            key = f"{pkg}.{full_class}#{decl.name}"
            methods.append(MethodAstView(key, s, e, p))

        for decl in getattr(t, "constructors", []) or []:
            pos = getattr(decl, "position", None)
            s = pos.line if pos else 1
            e = _max_body_line(getattr(decl, "body", None)) or s
            key = f"{pkg}.{full_class}#{simple}"
            methods.append(MethodAstView(key, s, e, p))

        for member in getattr(t, "body", []) or []:
            if _is_initializer(member):
                pos = getattr(member, "position", None)
                s = pos.line if pos else 1
                e = _max_body_line(getattr(member, "body", None)) or s
                name = "<clinit>" if bool(getattr(member, "static", False)) else "<init_block>"
                key = f"{pkg}.{full_class}#{name}"
                methods.append(MethodAstView(key, s, e, p))

        for inner in getattr(t, "types", []) or []:
            visit_type(inner, stack)

    for t in cu.types:
        visit_type(t, [])

    return methods

AST_TYPE_MAP = {
    "IfStatement": "if",
    "ReturnStatement": "return",
    "ThrowStatement": "throw",
    "TryStatement": "try",
    "SwitchStatement": "switch",
    "SynchronizedStatement": "synchronized",
    "WhileStatement": "while",
    "DoStatement": "do",
    "ForStatement": "for",
    "BreakStatement": "break",
    "ContinueStatement": "continue",
    "LocalVariableDeclaration": "var",
    "VariableDeclaration": "var",
    "Assignment": "assignment",
    "MethodInvocation": "call",
    "SuperMethodInvocation": "call",
    "ExplicitConstructorInvocation": "call",
    "ClassCreator": "call",
}

KEYWORD_GUESS = [
    ("return", "return"),
    ("throw", "throw"),
    ("if", "if"),
    ("else if", "if"),
    ("else", "if"),
    ("for", "for"),
    ("while", "while"),
    ("do", "do"),
    ("switch", "switch"),
    ("try", "try"),
    ("catch", "try"),
    ("finally", "try"),
    ("break", "break"),
    ("continue", "continue"),
]

def classify_line_by_ast_and_text(
    cu, file_lines: List[str],
    start_line: int, end_line: int,
    target_line: int
) -> str:

    if cu is None:
        line = file_lines[target_line - 1].strip() if 1 <= target_line <= len(file_lines) else ""
        low = line.lower()
        for kw, label in KEYWORD_GUESS:
            if low.startswith(kw):
                return label
        if "=" in line and not low.startswith(("if", "while", "for", "switch")):
            return "assignment"
        if "(" in line and line.rstrip().endswith(";"):
            return "call"
        return "other"

    best_depth, best_type = -1, None

    def node_span(n):
        pos = getattr(n, "position", None)
        if not (pos and pos.line):
            return None, None
        s = pos.line
        e = s
        me = _max_body_line(getattr(n, "body", None))
        if me and me >= s:
            e = me
        return s, e

    for path, node in cu:
        s, e = node_span(node)
        if s is None:
            continue
        if not (start_line <= s <= end_line):
            continue
        if s <= target_line <= max(e, s):
            tname = node.__class__.__name__
            label = AST_TYPE_MAP.get(tname)
            if label:
                depth = len(path)
                if depth > best_depth:
                    best_depth, best_type = depth, label

    if best_type:
        return best_type

    line = file_lines[target_line - 1].strip() if 1 <= target_line <= len(file_lines) else ""
    low = line.lower()
    for kw, label in KEYWORD_GUESS:
        if low.startswith(kw):
            return label
    if "=" in line and not low.startswith(("if", "while", "for", "switch")):
        return "assignment"
    if "(" in line and line.rstrip().endswith(";"):
        return "call"
    return "other"

JAVA_KEYWORDS: Set[str] = {
    "abstract","assert","boolean","break","byte","case","catch","char","class","const","continue",
    "default","do","double","else","enum","extends","final","finally","float","for","goto","if",
    "implements","import","instanceof","int","interface","long","native","new","package","private",
    "protected","public","return","short","static","strictfp","super","switch","synchronized","this",
    "throw","throws","transient","try","void","volatile","while","var","record","yield"
}
JAVA_TYPES_COMMON: Set[str] = {
    "String","Object","Integer","Long","Short","Byte","Boolean","Character","Double","Float",
    "List","ArrayList","Map","HashMap","Set","HashSet","Queue","Deque","Optional","Locale","TimeZone","Date"
}
IDENT_RE = re.compile(r"\b[_A-Za-z]\w*\b")

def extract_idents(line: str) -> Set[str]:
    ids = set(m.group(0) for m in IDENT_RE.finditer(line))
    return {x for x in ids if x not in JAVA_KEYWORDS and x not in JAVA_TYPES_COMMON}

def extract_def_use(line: str) -> Tuple[Set[str], Set[str]]:

    line = line.strip().rstrip(";")
    defs: Set[str] = set()
    uses: Set[str] = set()
    if "=" in line and "==" not in line and "!=" not in line:
        left, right = line.split("=", 1)
        left_ids = extract_idents(left)
        right_ids = extract_idents(right)
        defs |= left_ids
        uses |= right_ids
    else:
        uses |= extract_idents(line)
    return defs, uses

def main():
    ap = argparse.ArgumentParser(description="Static Analysis")
    ap.add_argument("--project", required=True, type=Path)
    ap.add_argument("--locs", required=True, type=Path)
    ap.add_argument("--txtout", required=True, type=Path)
    ap.add_argument("--linedir", required=False, type=Path)
    ap.add_argument("--maxgap", type=int, default=8)
    args = ap.parse_args()

    project_root: Path = args.project.resolve()
    loc_strs = [ln.strip() for ln in args.locs.read_text(encoding="utf-8").splitlines() if ln.strip()]

    method_to_items: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for s in loc_strs:
        loc = parse_suspicious_loc(s)
        method_to_items[loc.method_key()].append((loc.line, loc.score))

    nodes: List[Tuple[str, str, str, str, Optional[int]]] = []
    edges_raw: List[Tuple[int, int]] = []  # 暂存 nodes 索引间的边

    for method_key, items in method_to_items.items():
        nodes.append(("method", normalize_content(method_key), "method", "", None))
        method_node_idx = len(nodes) - 1

        _pkgclass, _meth_name = method_key.split("#", 1)
        pkg = ".".join(_pkgclass.split(".")[:-1])
        class_full = _pkgclass.split(".")[-1]
        top_class = class_full.split("$", 1)[0]
        rel = Path(*pkg.split(".")) / f"{top_class}.java"
        java_file = find_java_file(project_root, rel)
        file_lines = read_lines(java_file) if java_file and java_file.exists() else []

        method_view: Optional[MethodAstView] = None
        cu = None
        if java_file and java_file.exists():
            for mv in list_all_methods_in_file(java_file):
                if mv.key == method_key:
                    method_view = mv
                    break
            try:
                src = java_file.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                src = java_file.read_text(encoding="latin-1", errors="ignore")
            try:
                cu = javalang.parse.parse(src)
            except Exception:
                cu = None

        items_sorted = sorted(items, key=lambda x: x[0])

        first_line_node_idx = None
        prev_line_node_idx = None
        prev_line_no = None
        prev_defs: Set[str] = set()
        prev_uses: Set[str] = set()

        for line_no, sus in items_sorted:
            raw = file_lines[line_no - 1].rstrip("\n") if 1 <= line_no <= len(file_lines) else ""
            content = normalize_content(raw)

            if method_view and cu:
                tlabel = classify_line_by_ast_and_text(
                    cu, file_lines, method_view.start_line, method_view.end_line, line_no
                )
            else:
                tlabel = classify_line_by_ast_and_text(
                    None, file_lines, 1, len(file_lines), line_no
                )

            defs, uses = extract_def_use(content)

            
            nodes.append(("codeline", content, tlabel, f"{sus}", line_no))
            ln_idx = len(nodes) - 1

            if first_line_node_idx is None:
                first_line_node_idx = ln_idx
                edges_raw.append((method_node_idx, ln_idx))

            if prev_line_node_idx is not None and prev_line_no is not None:
                gap = line_no - prev_line_no
                shared = bool((prev_defs & uses) or
                              ((prev_defs | prev_uses) & (defs | uses)))
                if gap <= args.maxgap or shared:
                    edges_raw.append((prev_line_node_idx, ln_idx))

            prev_line_node_idx = ln_idx
            prev_line_no = line_no
            prev_defs, prev_uses = defs, uses

    id_map = {i: i + 1 for i in range(len(nodes))}

    if args.linedir:
        args.linedir.mkdir(parents=True, exist_ok=True)
        for i, (kind, content, tlabel, sus, line_no) in enumerate(nodes):
            if kind != "codeline":
                continue
            node_id = id_map[i]
            out_path = args.linedir / f"{node_id}.txt"
            out_path.write_text(normalize_content(content) + "\n", encoding="utf-8")

    lines_out: List[str] = []
    for i, (kind, content, tlabel, sus, line_no) in enumerate(nodes):
        node_id = id_map[i]
        content_out = normalize_content(content)
        if kind == "method":
            lines_out.append(f"{node_id} method: {{content: {content_out}}}")
        else:
            lines_out.append(f"{node_id} codeline: {{content: {content_out}, type: {tlabel}, sus: {sus}}}")

    lines_out.append("edge:")
    for u, v in edges_raw:
        lines_out.append(f"{id_map[u]}->{id_map[v]}")

    outp = args.txtout
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    print(f"finish{outp}")


if __name__ == "__main__":
    main()
