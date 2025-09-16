
import re
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
