"""Graph views over the IR: collapse by module nesting depth, attach attribution scores.

The raw IR has one node per tensor op (ResNet-18: ~70, a 7B LLM: tens of thousands),
which is unreadable. ``graph_view`` folds every module deeper than ``depth`` into
a single node and wraps the rest in module *groups*, the dashed boxes in the UI.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

PRETTY_TYPES = {"iadd": "add", "__iadd__": "add", "__add__": "add", "iadd_": "add", "imul": "mul"}


def strip_call(call: str) -> str:
    """'layer1.0:1' -> 'layer1.0'"""
    head, sep, tail = call.rpartition(":")
    return head if sep and tail.isdigit() else call


def _short(address: str) -> str:
    return address.rsplit(".", 1)[-1] if address else address


def module_tree(ir: dict[str, Any]) -> list[dict[str, Any]]:
    """Modules in first-use order (a depth-first walk of the model), for the left rail."""
    seen: dict[str, None] = {}
    for op in ir["ops"]:
        for call in op["modules"]:
            seen.setdefault(strip_call(call), None)
    mods = ir.get("modules", {})
    out = []
    for addr in seen:
        info = mods.get(addr, {})
        out.append(
            {
                "address": addr,
                "name": _short(addr),
                "class_name": info.get("class_name", ""),
                "depth": addr.count(".") + 1,
                "num_params": info.get("num_params", 0),
            }
        )
    return out


def graph_view(ir: dict[str, Any], depth: int = 2, scores: dict[str, float] | None = None) -> dict[str, Any]:
    ops = {o["id"]: o for o in ir["ops"]}
    order = [o["id"] for o in ir["ops"]]
    mods = ir.get("modules", {})

    ops_in_call: dict[str, list[str]] = defaultdict(list)
    for o in ir["ops"]:
        for call in o["modules"]:
            ops_in_call[call].append(o["id"])

    # 1. assign every op to a visible node
    node_of: dict[str, str] = {}
    containers_of: dict[str, list[str]] = {}
    members: dict[str, list[str]] = defaultdict(list)
    for oid in order:
        o = ops[oid]
        stack = [] if (o["is_input"] or o["is_output"]) else o["modules"]
        if len(stack) > depth and len(ops_in_call[stack[depth]]) > 1:
            nid = "m:" + stack[depth]
            containers = stack[:depth]
        else:
            nid = oid
            # an op that is alone in its leaf module does not need a box around it
            containers = [c for c in stack if len(ops_in_call[c]) > 1]
            containers = containers[:depth] if len(stack) > depth else containers
        node_of[oid] = nid
        members[nid].append(oid)
        containers_of.setdefault(nid, containers)

    # 2. groups: only keep boxes that hold at least two visible nodes
    group_nodes: dict[str, set[str]] = defaultdict(set)
    for nid, containers in containers_of.items():
        for c in containers:
            group_nodes[c].add(nid)
    kept = {c for c, ns in group_nodes.items() if len(ns) >= 2}

    def innermost(containers: list[str]) -> str | None:
        for c in reversed(containers):
            if c in kept:
                return "g:" + c
        return None

    groups = []
    for c in sorted(kept, key=lambda c: order.index(ops_in_call[c][0])):
        addr = strip_call(c)
        # parent group = nearest kept ancestor in any member's stack
        some = next(iter(group_nodes[c]))
        stack = containers_of[some]
        parent = innermost(stack[: stack.index(c)]) if c in stack else None
        groups.append(
            {
                "id": "g:" + c,
                "address": addr,
                "class_name": mods.get(addr, {}).get("class_name", ""),
                "num_params": mods.get(addr, {}).get("num_params", 0),
                "parent": parent,
                "num_ops": len(ops_in_call[c]),
            }
        )

    # 3. nodes
    nodes = []
    for nid, mem in members.items():
        mem_set = set(mem)
        # the op whose tensor represents this node: last member with an edge leaving the node
        leaving = [
            m for m in mem if any(ch not in mem_set for ch in ops[m]["children"]) or ops[m]["is_output"]
        ]
        out_op = ops[(leaving or mem)[-1]]
        if nid.startswith("m:"):
            addr = strip_call(nid[2:])
            info = mods.get(addr, {})
            node = {
                "id": nid,
                "kind": "module",
                "label": _short(addr),
                "path": addr,
                "type": info.get("class_name", "module"),
                "num_ops": len(mem),
                "num_params": info.get("num_params", 0),
            }
        else:
            o = ops[nid]
            addr = o.get("module") or ""
            if not addr and o["modules"] and len(ops_in_call[o["modules"][-1]]) == 1:
                addr = strip_call(o["modules"][-1])  # e.g. BatchNorm, whose buffers hide its atomicity
            label = (
                "input"
                if o["is_input"]
                else "output"
                if o["is_output"]
                else (_short(addr) if addr else PRETTY_TYPES.get(o["type"], o["type"]))
            )
            node = {
                "id": nid,
                "kind": "op",
                "label": label,
                "path": addr or nid,
                "type": PRETTY_TYPES.get(o["type"], o["type"]),
                "num_ops": 1,
                "num_params": o["num_params"],
            }
        node.update(
            {
                "output_op": out_op["id"],
                "shape": out_op["shape"],
                "flops": sum(ops[m]["flops"] for m in mem),
                "stats": out_op["stats"],
                "group": innermost(containers_of[nid]),
                "is_input": nid in ops and ops[nid]["is_input"],
                "is_output": nid in ops and ops[nid]["is_output"],
            }
        )
        nodes.append(node)

    # 4. edges between visible nodes
    edges: dict[tuple[str, str], dict[str, Any]] = {}
    for oid in order:
        t = node_of[oid]
        for p in ops[oid]["parents"]:
            s = node_of.get(p)
            if s is None or s == t:
                continue
            edges.setdefault((s, t), {"source": s, "target": t, "shape": ops[p]["shape"]})

    view = {"depth": depth, "nodes": nodes, "edges": list(edges.values()), "groups": groups}
    if scores is not None:
        attach_scores(view, scores)
    return view


def attach_scores(view: dict[str, Any], scores: dict[str, float]) -> None:
    """Add attribution to a view: node score, per-edge share and the dominant backward path."""
    by_id = {n["id"]: n for n in view["nodes"]}
    for n in view["nodes"]:
        n["score"] = scores.get(n["output_op"])
    mags = [abs(n["score"]) for n in view["nodes"] if n["score"] is not None and not n["is_output"]]
    top = max(mags) if mags else 1.0
    for n in view["nodes"]:
        n["score_norm"] = (abs(n["score"]) / top) if (n["score"] is not None and top) else None

    parents: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in view["edges"]:
        parents[e["target"]].append(e)
    for es in parents.values():
        mags = [abs(by_id[e["source"]].get("score") or 0.0) for e in es]
        total = sum(mags)
        for e, m in zip(es, mags, strict=True):
            e["share"] = (m / total) if total else (1.0 / len(es))

    # greedy walk from the output back to the input along the strongest parent
    path = []
    cur = next((n["id"] for n in view["nodes"] if n["is_output"]), None)
    guard = 0
    while cur is not None and guard < len(by_id):
        path.append(cur)
        guard += 1
        es = parents.get(cur, [])
        cur = max(es, key=lambda e: e["share"])["source"] if es else None
    on_path = set(zip(path[1:], path[:-1], strict=True))
    for e in view["edges"]:
        e["on_path"] = (e["source"], e["target"]) in on_path
    view["path"] = path
