#!/usr/bin/env python3
"""Re-apply project context enrichment on top of a freshly built graphify graph.

graphify's own pipeline produces bare AST nodes: label, source_file, location.
This script adds the two things that pipeline does not:

  1. `context` on EVERY node, derived deterministically from the source itself
     (docstrings, preceding comment blocks, declarations, file headers). No LLM,
     no API key.
  2. `doc_context` — the project's markdown prose duplicated into the nodes it
     describes, plus a document node per markdown file and `documents` edges.

Run it after ANY graphify rebuild (`graphify update .`, `/graphify`, or the
post-commit hook), because those regenerate graph.json from scratch and drop
everything below.

    python .claude/graphify-enrich/enrich.py

Idempotent: safe to re-run. Stages can be run individually with --only.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "graphify-out"
GRAPH = OUT / "graph.json"

MAX_CTX = 900
MAX_SYMBOL_NODES = 30
MAX_SCOPE_NODES = 12
MAX_SECTION_CHARS = 6000
MAX_DOC_PER_NODE = 8

CODE_EXT = (".py", ".c", ".h", ".cxx", ".cpp", ".hpp", ".hxx", ".cu", ".sh", ".lua")
C_LIKE = {".c", ".h", ".cxx", ".cpp", ".hpp", ".hxx", ".cu", ".cuh"}
LINE_COMMENT = {
    ".py": "#", ".sh": "#", ".yml": "#", ".yaml": "#", ".cmake": "#",
    ".c": "//", ".h": "//", ".cxx": "//", ".cpp": "//", ".hpp": "//",
    ".hxx": "//", ".cu": "//", ".cuh": "//", ".lua": "--",
}

# Markdown that is not this project's own documentation.
EXCLUDE_PARTS = (".venv", ".pytest_cache", "graphify-out", "build", ".git")
EXCLUDE_PREFIX = (".claude/skills/graphify/",)

STOP = {
    "the", "and", "for", "not", "python", "readme", "note", "notes", "true", "false",
    "none", "null", "int", "str", "float", "bool", "list", "dict", "set", "type",
    "main", "test", "tests", "data", "file", "files", "path", "paths", "name",
    "run", "make", "build", "src", "out", "output", "input", "value", "time",
    "step", "steps", "case", "cases", "all", "new", "old", "add", "get", "set",
    "print", "open", "close", "read", "write", "yes", "no", "ok", "id",
    "readme.md", "claude.md", "self", "args", "kwargs", "todo", "fixme",
}
GENERIC_TOKENS = {
    "simulaciones", "simulacion", "analisis", "readme", "notes", "note",
    "escalado", "refactor", "skill", "claude", "memory", "runbook", "estructura",
    "doc", "docs", "guide", "index", "md", "report", "plan", "draft", "local",
}

_SEP = re.compile(r"[-=*_~#]{4,}")
_cache: dict[str, list[str]] = {}


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", _SEP.sub(" ", text)).strip(" -=*_/").strip()


def lines_of(rel: str) -> list[str]:
    if rel not in _cache:
        try:
            _cache[rel] = (ROOT / rel).read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            _cache[rel] = []
    return _cache[rel]


# --------------------------------------------------------------------------
# Stage 1 — per-node context from source
# --------------------------------------------------------------------------
def preceding_comment(lines: list[str], idx: int, ext: str) -> str:
    tok = LINE_COMMENT.get(ext, "//")
    out: list[str] = []
    i = idx - 1
    while i >= 0 and (not lines[i].strip() or lines[i].lstrip().startswith("@")):
        i -= 1
    if i >= 0 and lines[i].rstrip().endswith("*/"):
        j = i
        while j >= 0 and "/*" not in lines[j]:
            j -= 1
        for b in lines[max(j, 0): i + 1]:
            b = b.strip().lstrip("/*").rstrip("*/").strip().lstrip("*").strip()
            if b:
                out.append(b)
        return " ".join(out)
    while i >= 0 and lines[i].lstrip().startswith(tok):
        out.append(lines[i].lstrip()[len(tok):].strip())
        i -= 1
    out.reverse()
    return " ".join(x for x in out if x)


def declaration(lines: list[str], idx: int, ext: str) -> str:
    if idx >= len(lines):
        return ""
    parts = [lines[idx].strip()]
    k = idx + 1
    # C-family often puts the return type on its own line; the node anchors there.
    if ext in C_LIKE and "(" not in parts[0] and not parts[0].endswith((";", "{", "}", ",")):
        while k < len(lines) and k < idx + 3 and "(" not in " ".join(parts):
            parts.append(lines[k].strip())
            k += 1
    joined = " ".join(parts)
    depth = joined.count("(") - joined.count(")")
    while depth > 0 and k < len(lines) and k < idx + 6:
        nxt = lines[k].strip()
        parts.append(nxt)
        depth += nxt.count("(") - nxt.count(")")
        k += 1
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def py_docstring(lines: list[str], idx: int) -> str:
    k = idx
    while k < len(lines) and k < idx + 6 and not lines[k].rstrip().endswith(":"):
        k += 1
    k += 1
    while k < len(lines) and not lines[k].strip():
        k += 1
    if k >= len(lines):
        return ""
    s = lines[k].strip()
    m = re.match(r'^(?:[rubf]{0,2})("""|\'\'\'|"|\')', s)
    if not m:
        return ""
    q = m.group(1)
    body = s[m.end():]
    if body.endswith(q) and len(body) >= len(q):
        return body[: -len(q)].strip()
    collected = [body]
    k += 1
    while k < len(lines) and q not in lines[k] and len(collected) < 25:
        collected.append(lines[k].strip())
        k += 1
    if k < len(lines):
        collected.append(lines[k].split(q)[0].strip())
    return " ".join(x for x in collected if x).strip()


def file_header(lines: list[str], ext: str) -> str:
    if ext == ".py":
        doc = py_docstring(["<mod>:"] + lines, 0)
        if doc:
            return doc
    # '#' starts a preprocessor directive in C, never documentation.
    starts = ("//", "/*", "*") if ext in C_LIKE else ("#", "--", "//")
    out = []
    for l in lines[:12]:
        s = l.strip()
        if not s:
            if out:
                break
            continue
        if s.startswith("#!"):
            continue
        if s.startswith(starts):
            out.append(s.lstrip("/*#-").strip().lstrip("*").rstrip("*/").strip())
        else:
            break
    return clean(" ".join(x for x in out if x))


def build_context(node: dict) -> tuple[str, str]:
    sf = node.get("source_file") or ""
    loc = node.get("source_location") or ""
    label = node.get("label") or node.get("id")
    if not sf:
        return "", "external"
    ext = Path(sf).suffix
    lines = lines_of(sf)
    if not lines:
        return f"{label} — file {sf} (unreadable)", "unreadable"
    m = re.match(r"^L(\d+)$", str(loc))
    if not m:
        return "", "noloc"
    idx = int(m.group(1)) - 1
    if idx < 0 or idx >= len(lines):
        return "", "outofrange"

    if node.get("label") == sf or (idx == 0 and str(label).endswith(ext)):
        hdr = file_header(lines, ext)
        ctx = f"{sf} — source file, {len([l for l in lines if l.strip()])} non-blank lines."
        if hdr:
            ctx += f" {hdr}"
        return ctx[:MAX_CTX], "file-header" if hdr else "file-stat"

    decl = declaration(lines, idx, ext)
    comment = clean(preceding_comment(lines, idx, ext))
    doc = clean(py_docstring(lines, idx)) if ext == ".py" else ""
    prov, pieces = "decl", []
    if decl:
        pieces.append(decl)
    if doc:
        pieces.append(doc)
        prov = "docstring"
    elif comment:
        pieces.append(comment)
        prov = "comment"
    if not pieces:
        return f"{label} — defined at {sf}:{loc}", "location-only"
    return f"{' — '.join(pieces)}  [{sf}:{loc}]"[:MAX_CTX], prov


def stage_context(graph: dict) -> Counter:
    nodes, links = graph["nodes"], graph["links"]
    byid = {n["id"]: n for n in nodes}
    prov_counts: Counter = Counter()
    for n in nodes:
        ctx, prov = build_context(n)
        if prov in ("external", "noloc", "outofrange") or not ctx:
            label = n.get("label") or n["id"]
            files: Counter = Counter()
            rel_n = 0
            for e in links:
                if e.get("source") == n["id"] or e.get("target") == n["id"]:
                    rel_n += 1
                    other = e.get("target") if e.get("source") == n["id"] else e.get("source")
                    o = byid.get(other) or {}
                    if o.get("source_file"):
                        files[o["source_file"]] += 1
            if files:
                top = ", ".join(f for f, _ in files.most_common(4))
                ctx = (f"{label} — external/unresolved symbol, no definition in this repo. "
                       f"Referenced from {len(files)} file(s): {top}.")
                prov = "external-neighbours"
            elif rel_n:
                ctx = f"{label} — symbol with {rel_n} graph relation(s)."
                prov = "external-relations"
            else:
                ctx = f"{label} — isolated node, no definition and no relations in the graph."
                prov = "isolated"
        n["context"] = ctx[:MAX_CTX]
        n["context_provenance"] = prov
        prov_counts[prov] += 1
    return prov_counts


# --------------------------------------------------------------------------
# Stage 2 — markdown inventory and section split
# --------------------------------------------------------------------------
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*$")
FENCE = re.compile(r"^\s*```")
BACKTICK = re.compile(r"`([^`\n]{2,120})`")
PATHLIKE = re.compile(r"\b((?:[\w.\-]+/)+[\w.\-]+\.(?:py|c|h|cxx|cpp|hpp|hxx|cu|sh|lua|json|yml|yaml))\b")
FUNCLIKE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]{2,})\s*\(\s*\)")
CAMEL = re.compile(r"\b([A-Z][a-z0-9]+(?:[A-Z][A-Za-z0-9]+)+)\b")
SNAKE_FILE = re.compile(r"\b([\w\-]+\.(?:py|c|h|cxx|cpp|hpp|hxx|cu|sh|lua))\b")


def relevant_md() -> list[str]:
    out = []
    for p in sorted(ROOT.rglob("*.md")):
        rel = p.relative_to(ROOT).as_posix()
        if any(part in EXCLUDE_PARTS for part in p.relative_to(ROOT).parts):
            continue
        if rel.startswith(EXCLUDE_PREFIX):
            continue
        out.append(rel)
    return out


def symbols_in(text: str) -> list[str]:
    found: set[str] = set()
    for m in BACKTICK.finditer(text):
        s = m.group(1).strip()
        found.add(s)
        for rx in (PATHLIKE, SNAKE_FILE, FUNCLIKE):
            found.update(x.group(1) for x in rx.finditer(s))
    for rx in (PATHLIKE, SNAKE_FILE, FUNCLIKE, CAMEL):
        found.update(m.group(1) for m in rx.finditer(text))
    return sorted(x for x in found if len(x) >= 3)


def split_sections(rel: str) -> list[dict]:
    lines = (ROOT / rel).read_text(encoding="utf-8", errors="ignore").splitlines()
    marks, in_fence = [], False
    for i, l in enumerate(lines):
        if FENCE.match(l):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = HEADING.match(l)
        if m:
            marks.append((i, len(m.group(1)), m.group(2).strip()))

    spans = []
    if not marks or marks[0][0] > 0:
        end = marks[0][0] if marks else len(lines)
        if any(l.strip() for l in lines[:end]):
            spans.append((0, end, 0, "(preamble)"))
    for k, (i, lvl, head) in enumerate(marks):
        spans.append((i, marks[k + 1][0] if k + 1 < len(marks) else len(lines), lvl, head))

    out = []
    for start, end, lvl, head in spans:
        text = "\n".join(lines[start:end]).strip()
        if text:
            out.append({"file": rel, "heading": head, "level": lvl,
                        "start": start + 1, "end": end, "text": text,
                        "symbols": symbols_in(text)})
    return out


# --------------------------------------------------------------------------
# Stage 3 — map markdown onto nodes
# --------------------------------------------------------------------------
def norm(s: str) -> str:
    s = s.strip().strip("`").strip()
    s = re.sub(r"\(\s*\)$", "", s)
    return re.sub(r"^[\.\*\-\s]+", "", s).strip()


def key(s: str) -> str:
    return norm(s).lower().lstrip(".")


def tokens_of(rel: str) -> list[str]:
    parts = re.split(r"[_\-.\s]+", Path(rel).stem)
    return [p.lower() for p in parts if p and p.lower() not in GENERIC_TOKENS and len(p) > 3]


def purge_previous(graph: dict) -> tuple[int, int]:
    """Drop prior markdown enrichment so a re-run rebuilds instead of duplicating."""
    nodes, links = graph["nodes"], graph["links"]
    doc_ids = {n["id"] for n in nodes if n.get("_origin") == "markdown"}
    kept_nodes = [n for n in nodes if n["id"] not in doc_ids]
    kept_links = [e for e in links
                  if e.get("relation") != "documents"
                  and e.get("source") not in doc_ids and e.get("target") not in doc_ids]
    dropped = (len(nodes) - len(kept_nodes), len(links) - len(kept_links))
    graph["nodes"], graph["links"] = kept_nodes, kept_links
    for n in kept_nodes:
        n.pop("doc_context", None)
        n.pop("doc_sources", None)
        n.pop("doc_complete", None)
    return dropped


def stage_markdown(graph: dict, sections: list[dict]) -> tuple[Counter, int, int]:
    nodes, links = graph["nodes"], graph["links"]
    by_label: dict[str, list[dict]] = defaultdict(list)
    by_file: dict[str, list[dict]] = defaultdict(list)
    file_node: dict[str, dict] = {}
    for n in nodes:
        by_label[key(n.get("label") or "")].append(n)
        nl = n.get("norm_label")
        if nl and key(nl) != key(n.get("label") or ""):
            by_label[key(nl)].append(n)
        sf = n.get("source_file")
        if sf:
            by_file[sf].append(n)
            if n.get("source_location") == "L1" and (
                    n.get("label") == sf or Path(sf).name == n.get("label")):
                file_node.setdefault(sf, n)
    known_files = sorted(by_file)
    for f in known_files:
        file_node.setdefault(f, by_file[f][0])

    basename_to, stem_to, dir_to = defaultdict(list), defaultdict(list), defaultdict(list)
    for f in known_files:
        p = Path(f)
        basename_to[p.name].append(f)
        stem_to[p.stem.lower()].append(f)
        dir_to[p.parent.as_posix()].append(f)

    def files_under(d: str) -> list[str]:
        d = d.rstrip("/")
        return [f for f in known_files if f == d or f.startswith(d + "/")]

    def rank(files: list[str], near: str) -> list[str]:
        return sorted(files, key=lambda f: (not f.startswith(near), f.count("/"), f))

    def resolve(sym: str, near: str) -> list[dict]:
        k = key(sym)
        if not k or k in STOP or len(k) < 3:
            return []
        raw = norm(sym)
        if raw in by_file:
            return [file_node[raw]]
        base = Path(raw).name
        if base in basename_to:
            return [file_node[rank(basename_to[base], near)[0]]]
        if k in stem_to:
            return [file_node[c] for c in rank(stem_to[k], near)[:3]]
        if raw.rstrip("/") in dir_to or any(f.startswith(raw.rstrip("/") + "/") for f in known_files):
            fs = rank([f for f in files_under(raw) if f.endswith(CODE_EXT)], near)
            return [file_node[f] for f in fs[:MAX_SCOPE_NODES]]
        return by_label.get(k, [])[:MAX_SYMBOL_NODES]

    def doc_id(f: str) -> str:
        return "doc_" + re.sub(r"[^a-z0-9]+", "_", f.lower()).strip("_")

    resolved: list[dict] = []
    stack: list[tuple[int, list[str]]] = []
    stats: Counter = Counter()
    current_file = None

    for sec in sections:
        # The heading stack is per-document; carrying it across files would let a
        # section inherit matches from an unrelated markdown — invented edges.
        if sec["file"] != current_file:
            stack, current_file = [], sec["file"]
        md_dir = Path(sec["file"]).parent.as_posix()
        near = "" if md_dir == "." else md_dir + "/"

        matched: dict[str, dict] = {}
        for sym in sec["symbols"]:
            for n in resolve(sym, near):
                matched[n["id"]] = n
        rule, ids = "symbol", list(matched)[:MAX_SYMBOL_NODES]

        if not ids:
            for lvl, parent_ids in reversed(stack):
                if lvl < sec["level"] and parent_ids:
                    ids, rule = parent_ids[:MAX_SCOPE_NODES], "inherited"
                    break
        if not ids and near:
            fs = rank([f for f in files_under(md_dir) if f.endswith(CODE_EXT)], near)
            ids, rule = [file_node[f]["id"] for f in fs[:MAX_SCOPE_NODES]], "dir-scope"
        if not ids:
            toks = tokens_of(sec["file"])
            hit = [f for f in known_files
                   if f.endswith(CODE_EXT) and any(t in f.lower() for t in toks)]
            if hit:
                ids = [file_node[f]["id"] for f in rank(hit, near)[:MAX_SCOPE_NODES]]
                rule = "token-scope"
        if not ids:
            # No code referent exists (project prose, personal notes). Anchor to the
            # document's own node rather than invent a link into the codebase.
            ids, rule = [doc_id(sec["file"])], "doc-self"

        while stack and stack[-1][0] >= sec["level"]:
            stack.pop()
        stack.append((sec["level"], [] if rule == "doc-self" else ids))
        stats[rule] += 1
        resolved.append({**sec, "node_ids": ids, "rule": rule})

    attach: dict[str, list[dict]] = defaultdict(list)
    for sec in resolved:
        entry = {"md_file": sec["file"], "heading": sec["heading"],
                 "lines": f"L{sec['start']}-L{sec['end']}", "match": sec["rule"],
                 "text": sec["text"][:MAX_SECTION_CHARS]}
        for i in sec["node_ids"]:
            attach[i].append(entry)

    existing = {n["id"] for n in nodes}
    md_files = sorted({s["file"] for s in resolved})
    added_nodes = added_links = 0
    for f in md_files:
        fid = doc_id(f)
        secs_of = [s for s in resolved if s["file"] == f]
        if fid not in existing:
            body = (ROOT / f).read_text(encoding="utf-8", errors="ignore")
            head = next((s["heading"] for s in secs_of if s["heading"] != "(preamble)"),
                        Path(f).stem)
            # The document node retains EVERY section. Code nodes keep only their
            # most relevant few, so without this the per-node cap would silently
            # drop sections out of the graph entirely.
            nodes.append({
                "id": fid, "label": Path(f).name, "norm_label": Path(f).name,
                "file_type": "document", "source_file": f, "source_location": "L1",
                "_origin": "markdown", "community": -1,
                "context": f"{f} — project markdown, {len(body.splitlines())} lines, "
                           f"{len(secs_of)} sections. Opening topic: {head}.",
                "context_provenance": "markdown-doc",
                "doc_context": [{"md_file": f, "heading": s["heading"],
                                 "lines": f"L{s['start']}-L{s['end']}", "match": "self",
                                 "text": s["text"][:MAX_SECTION_CHARS]} for s in secs_of],
                "doc_sources": [f], "doc_complete": True,
            })
            existing.add(fid)
            added_nodes += 1
        seen: set[tuple[str, str]] = set()
        for sec in secs_of:
            for nid in sec["node_ids"]:
                if nid == fid or (fid, nid) in seen:
                    continue
                seen.add((fid, nid))
                links.append({"source": fid, "target": nid, "relation": "documents",
                              "confidence": "EXTRACTED" if sec["rule"] == "symbol" else "INFERRED",
                              "context": "documentation",
                              "source_location": f"{f}:L{sec['start']}"})
                added_links += 1

    RULE_RANK = {"symbol": 0, "token-scope": 1, "dir-scope": 2,
                 "inherited": 3, "doc-self": 4, "self": 5}
    for n in nodes:
        entries = attach.get(n["id"])
        if not entries:
            continue
        if n.get("doc_complete"):
            have = {(e["md_file"], e["heading"]) for e in n["doc_context"]}
            n["doc_context"] += [e for e in entries
                                 if (e["md_file"], e["heading"]) not in have]
        else:
            merged = (n.get("doc_context") or []) + entries
            merged.sort(key=lambda e: RULE_RANK.get(e.get("match"), 9))
            n["doc_context"] = merged[:MAX_DOC_PER_NODE]
        n["doc_sources"] = sorted({e["md_file"] for e in n["doc_context"]})

    return stats, added_nodes, added_links


# --------------------------------------------------------------------------
# Stage 4 — place document nodes into communities
# --------------------------------------------------------------------------
def stage_communities(graph: dict) -> Counter:
    apath, lpath = OUT / ".graphify_analysis.json", OUT / ".graphify_labels.json"
    if not apath.exists():
        return Counter({"skipped (no .graphify_analysis.json)": 1})
    nodes, links = graph["nodes"], graph["links"]
    byid = {n["id"]: n for n in nodes}
    analysis = json.loads(apath.read_text(encoding="utf-8"))
    communities = {int(k): list(v) for k, v in analysis["communities"].items()}
    labels = {}
    if lpath.exists():
        labels = {int(k): v for k, v in json.loads(lpath.read_text(encoding="utf-8")).items()}

    targets = defaultdict(list)
    for e in links:
        if e.get("relation") == "documents":
            targets[e["source"]].append(e["target"])

    # Strip every document node from the community map, then drop the auto-created
    # "Project Notes" buckets that are now empty. Without this each re-run leaks a
    # fresh community per referent-less markdown and the count creeps upward.
    doc_ids = {n["id"] for n in nodes if n.get("_origin") == "markdown"}
    for cid in list(communities):
        communities[cid] = [i for i in communities[cid] if i not in doc_ids]
        if not communities[cid] and str(labels.get(cid, "")).startswith("Project Notes ("):
            del communities[cid]
            labels.pop(cid, None)

    next_cid = max(communities) + 1 if communities else 0
    placed: Counter = Counter()
    for n in nodes:
        if n.get("_origin") != "markdown":
            continue
        votes: Counter = Counter()
        for t in targets.get(n["id"], []):
            c = (byid.get(t) or {}).get("community")
            if isinstance(c, int) and c >= 0:
                votes[c] += 1
        if votes:
            cid = votes.most_common(1)[0][0]
            placed["joined documented community"] += 1
        else:
            cid = next_cid
            communities[cid] = []
            labels[cid] = f"Project Notes ({Path(n['source_file']).stem})"
            next_cid += 1
            placed["new community (no code referent)"] += 1
        n["community"] = cid
        if n["id"] not in communities[cid]:
            communities[cid].append(n["id"])

    analysis["communities"] = {str(k): v for k, v in communities.items()}
    apath.write_text(json.dumps(analysis, ensure_ascii=False), encoding="utf-8")
    if labels:
        lpath.write_text(json.dumps({str(k): v for k, v in labels.items()},
                                    ensure_ascii=False, indent=2), encoding="utf-8")
    return placed


def stage_report(graph: dict) -> str:
    """Regenerate GRAPH_REPORT.md, preserving the existing community labels."""
    import networkx as nx
    from graphify.analyze import god_nodes, suggest_questions, surprising_connections
    from graphify.cluster import score_all
    from graphify.report import generate

    apath, lpath = OUT / ".graphify_analysis.json", OUT / ".graphify_labels.json"
    analysis = json.loads(apath.read_text(encoding="utf-8"))
    labels = {int(k): v for k, v in
              json.loads(lpath.read_text(encoding="utf-8")).items()} if lpath.exists() else {}
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8")) \
        if (OUT / "manifest.json").exists() else {}

    G = nx.node_link_graph(graph, edges="links")
    communities = {int(k): v for k, v in analysis["communities"].items()}
    cohesion = score_all(G, communities)

    md_files = sorted({n["source_file"] for n in graph["nodes"]
                       if n.get("_origin") == "markdown"})
    corpus = sorted(set(manifest) | set(md_files))
    words = 0
    for rel in corpus:
        try:
            words += len((ROOT / rel).read_text(encoding="utf-8", errors="ignore").split())
        except OSError:
            pass
    detection = {"total_files": len(corpus), "total_words": words,
                 "files": {"code": list(manifest), "document": md_files},
                 "scan_root": str(ROOT)}

    gods = god_nodes(G)
    surprises = surprising_connections(G, communities)
    questions = suggest_questions(G, communities, labels)
    report = generate(G, communities, cohesion, labels, gods, surprises, detection,
                      analysis.get("tokens") or {"input": 0, "output": 0}, str(ROOT),
                      suggested_questions=questions,
                      built_at_commit=graph.get("built_at_commit"))
    (OUT / "GRAPH_REPORT.md").write_text(report, encoding="utf-8")
    analysis.update({"cohesion": {str(k): v for k, v in cohesion.items()},
                     "gods": gods, "surprises": surprises, "questions": questions})
    apath.write_text(json.dumps(analysis, ensure_ascii=False), encoding="utf-8")
    return (f"{len(corpus)} files ({len(md_files)} markdown) · {words:,} words · "
            f"{len(report.splitlines())} report lines")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["context", "markdown"],
                    help="run a single stage instead of the whole pipeline")
    ap.add_argument("--no-report", action="store_true",
                    help="skip regenerating GRAPH_REPORT.md")
    args = ap.parse_args()

    if not GRAPH.exists():
        raise SystemExit(f"no graph at {GRAPH} — run `graphify update .` first")
    graph = json.loads(GRAPH.read_text(encoding="utf-8"))
    before = len(graph["nodes"])

    # Purge first: the context stage must not describe stale document nodes that
    # the markdown stage is about to rebuild.
    if args.only != "context":
        pn, pl = purge_previous(graph)
        if pn or pl:
            print(f"[purge]    dropped {pn} stale document nodes, {pl} stale edges")
        before = min(before, len(graph["nodes"]))

    if args.only != "markdown":
        prov = stage_context(graph)
        missing = sum(1 for n in graph["nodes"] if not (n.get("context") or "").strip())
        print(f"[context]  {len(graph['nodes'])} nodes, {missing} without context")
        for k, v in prov.most_common():
            print(f"             {k:22s} {v:6d}")

    if args.only != "context":
        md = relevant_md()
        sections = [s for rel in md for s in split_sections(rel)]
        stats, dn, dl = stage_markdown(graph, sections)
        print(f"[markdown] {len(md)} files -> {len(sections)} sections")
        for k, v in stats.most_common():
            print(f"             {k:14s} {v:5d}")
        print(f"             +{dn} document nodes, +{dl} 'documents' edges")
        for k, v in stage_communities(graph).most_common():
            print(f"[community]  {k}: {v}")

    if len(graph["nodes"]) < before:
        raise SystemExit("refusing to write: node count shrank")
    GRAPH.write_text(json.dumps(graph, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {GRAPH.relative_to(ROOT)}  "
          f"({len(graph['nodes'])} nodes, {len(graph['links'])} edges)")

    if not args.no_report:
        try:
            print(f"[report]   {stage_report(graph)}")
        except Exception as exc:                      # noqa: BLE001 - report is optional
            print(f"[report]   skipped ({type(exc).__name__}: {exc})")
    print("Regenerate the interactive view with:  graphify export html")


if __name__ == "__main__":
    main()
