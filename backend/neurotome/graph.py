"""Graph views over the IR: collapse by module nesting depth, attach attribution scores.

The raw IR has one node per tensor op (ResNet-18: ~70, a 7B LLM: tens of thousands),
which is unreadable. ``graph_view`` folds every module deeper than ``depth`` into
a single node and wraps the rest in module *groups*, the dashed boxes in the UI.
``expand`` / ``collapse`` override the depth rule per module call.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
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


def _ops_in_call(ir: dict[str, Any]) -> dict[str, list[str]]:
    """Module call -> its op ids, in first-use order."""
    out: dict[str, list[str]] = defaultdict(list)
    for o in ir["ops"]:
        for call in o["modules"]:
            out[call].append(o["id"])
    return out


def _stack(o: dict[str, Any]) -> list[str]:
    return [] if (o["is_input"] or o["is_output"]) else o["modules"]


def _op_names(o: dict[str, Any], ops_in_call: dict[str, list[str]]) -> tuple[str, str, str]:
    """(label, path, type) of an op node."""
    addr = o.get("module") or ""
    if not addr and o["modules"] and len(ops_in_call[o["modules"][-1]]) == 1:
        addr = strip_call(o["modules"][-1])  # e.g. BatchNorm, whose buffers hide its atomicity
    kind: str = PRETTY_TYPES.get(o["type"]) or o["type"]
    label = "input" if o["is_input"] else "output" if o["is_output"] else (_short(addr) if addr else kind)
    return label, addr or o["id"], kind


def _meta(o: dict[str, Any]) -> tuple[str | None, str | None]:
    """(dtype, device) of an op's output; None when not recorded (pre-M1 runs have no device)."""
    return o.get("dtype") or None, o.get("device")


def unsaved(ir: dict[str, Any], op: dict[str, Any]) -> str | None:
    """Why ``op`` has no tensor on disk ("budget" | "modules" | "no_tensor"), None when saved.
    Runs before M4 lack the field: derived from ``saved`` and the run's ``truncated`` flag."""
    if "unsaved" in op:
        return op["unsaved"]
    if op.get("saved"):
        return None
    return "budget" if ir.get("model", {}).get("truncated") else "no_tensor"


def graph_view(
    ir: dict[str, Any],
    depth: int = 2,
    scores: dict[str, float] | None = None,
    expand: Iterable[str] = (),
    collapse: Iterable[str] = (),
    merges: dict[str, dict[str, float]] | None = None,
    start: str | None = None,
) -> dict[str, Any]:
    """``merges`` (op_gradients) turns merge shares into flow through the merge; ``start`` is the op the
    dominant path starts from (a unit objective), else the output."""
    ops = {o["id"]: o for o in ir["ops"]}
    order = [o["id"] for o in ir["ops"]]
    pos = {oid: i for i, oid in enumerate(order)}
    mods = ir.get("modules", {})
    expand, collapse = set(expand), set(collapse)
    ops_in_call = _ops_in_call(ir)

    def folds(c: str, i: int) -> bool:
        return (c in collapse or (i >= depth and c not in expand)) and len(ops_in_call[c]) > 1

    # 1. assign every op to a visible node: it folds into the first collapsed multi-op call of its stack
    node_of: dict[str, str] = {}
    containers_of: dict[str, list[str]] = {}
    members: dict[str, list[str]] = defaultdict(list)
    for oid in order:
        stack = _stack(ops[oid])
        fold = next((i for i, c in enumerate(stack) if folds(c, i)), None)
        if fold is not None:
            nid = "m:" + stack[fold]
            containers = stack[:fold]
        else:
            nid = oid
            # an op that is alone in its leaf module does not need a box around it
            containers = [c for c in stack if len(ops_in_call[c]) > 1]
        node_of[oid] = nid
        members[nid].append(oid)
        containers_of.setdefault(nid, containers)

    # 2. groups: only keep boxes that hold at least two visible nodes (or were expanded by hand)
    group_nodes: dict[str, set[str]] = defaultdict(set)
    for nid, containers in containers_of.items():
        for c in containers:
            group_nodes[c].add(nid)
    kept = {c for c, ns in group_nodes.items() if len(ns) >= 2 or c in expand}

    def innermost(containers: list[str]) -> str | None:
        for c in reversed(containers):
            if c in kept:
                return "g:" + c
        return None

    def first_use(c: str) -> tuple[int, int]:  # ties (same first op) -> outer call first
        first = ops_in_call[c][0]
        return pos[first], ops[first]["modules"].index(c)

    groups = []
    for c in sorted(kept, key=first_use):
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
            label, path, kind = _op_names(o, ops_in_call)
            node = {
                "id": nid,
                "kind": "op",
                "label": label,
                "path": path,
                "type": kind,
                "num_ops": 1,
                "num_params": o["num_params"],
            }
        node.update(
            {
                "output_op": out_op["id"],
                "shape": out_op["shape"],
                "flops": sum(ops[m]["flops"] for m in mem),
                "stats": out_op["stats"],
                "unsaved": unsaved(ir, out_op),
                "group": innermost(containers_of[nid]),
                "is_input": nid in ops and ops[nid]["is_input"],
                "is_output": nid in ops and ops[nid]["is_output"],
            }
        )
        dtype, device = _meta(out_op)
        counted = [st for m in mem if (st := ops[m].get("stats")) and "nan" in st and "inf" in st]
        node.update(
            {
                "dtype": dtype,
                "device": device,
                "nonfinite": sum(st["nan"] + st["inf"] for st in counted) if counted else None,
            }
        )
        nodes.append(node)

    # 4. edges between visible nodes
    edges: dict[tuple[str, str], dict[str, Any]] = {}
    pairs: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)  # every (parent, child) op pair
    for oid in order:
        t = node_of[oid]
        for p in ops[oid]["parents"]:
            s = node_of.get(p)
            if s is None or s == t:
                continue
            pairs[(s, t)].append((p, oid))
            if (s, t) in edges:
                continue
            dtype, device = _meta(ops[p])
            target_dtype, target_device = _meta(ops[oid])
            edges[(s, t)] = {
                "source": s,
                "target": t,
                "shape": ops[p]["shape"],
                "dtype": dtype,
                "device": device,
                "target_dtype": target_dtype,
                "target_device": target_device,
            }

    view = {"depth": depth, "nodes": nodes, "edges": list(edges.values()), "groups": groups}
    if scores is not None:
        attach_scores(view, scores, merges, pairs, start, node_of)
    return view


def search_index(ir: dict[str, Any]) -> list[dict[str, Any]]:
    """Every op (IR order), then every module call with more than one op (first-use order)."""
    # ponytail: the full index goes to the client; above ~20k ops switch to server-side ?q= filtering
    ops_in_call = _ops_in_call(ir)
    mods = ir.get("modules", {})
    ops = {o["id"]: o for o in ir["ops"]}
    out = []
    for o in ir["ops"]:
        label, path, kind = _op_names(o, ops_in_call)
        out.append(
            {"id": o["id"], "kind": "op", "label": label, "path": path, "type": kind,
             "shape": o["shape"], "calls": list(_stack(o))}
        )  # fmt: skip
    for call, members in ops_in_call.items():
        if len(members) < 2:
            continue
        addr = strip_call(call)
        stack = ops[members[0]]["modules"]
        out.append(
            {"id": "m:" + call, "kind": "module", "label": _short(addr), "path": addr,
             "type": mods.get(addr, {}).get("class_name", "module"),
             "shape": ops[members[-1]]["shape"], "calls": stack[: stack.index(call)]}
        )  # fmt: skip
    return out


def attach_scores(
    view: dict[str, Any],
    scores: dict[str, float],
    merges: dict[str, dict[str, float]] | None = None,
    pairs: dict[tuple[str, str], list[tuple[str, str]]] | None = None,
    start: str | None = None,
    node_of: dict[str, str] | None = None,
) -> None:
    """Add attribution to a view: node score, per-edge share and the dominant backward path.

    At a node with several incoming edges, an edge's share is its flow through the merge
    (Σ |merges[child][parent]| over its op pairs, ``share_method: merge``) when every incoming edge has
    such a pair; otherwise all of that node's edges fall back to |parent score| / Σ
    (``share_method: parent``).
    """
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
    merges, pairs = merges or {}, pairs or {}
    for t, es in parents.items():
        if len(es) == 1:
            es[0]["share"] = 1.0
            continue
        flows = [[abs(merges[c][p]) for p, c in pairs.get((e["source"], t), []) if p in merges.get(c, {})]
                 for e in es]  # fmt: skip
        method = "merge" if all(flows) and sum(map(sum, flows)) > 0 else "parent"
        if method == "merge":
            mags = [sum(f) for f in flows]
        else:
            mags = [abs(by_id[e["source"]].get("score") or 0.0) for e in es]
        total = sum(mags)
        for e, m in zip(es, mags, strict=True):
            e["share"] = (m / total) if total else (1.0 / len(es))
            if by_id[t].get("score"):  # no flow reaches t (after a unit objective): no split
                e["share_method"] = method

    # greedy walk from the output back to the input along the strongest parent
    path = []
    cur = next((n["id"] for n in view["nodes"] if n["is_output"]), None)
    if start is not None and node_of and start in node_of:
        cur = node_of[start]
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


def module_sites(ir: dict[str, Any], op_id: str) -> list[tuple[str, int]]:
    """Every (module address, call index) whose output is ``op_id``, innermost first: the calls in the op's
    stack for which it is the last op with a child outside the call. This covers leaf modules, BatchNorm
    (module is null) and collapsed module nodes, whose ``output_op`` is the block's last op."""
    ops = {o["id"]: o for o in ir["ops"]}
    if op_id not in ops:
        return []
    ops_in_call = _ops_in_call(ir)
    out = []
    for call in reversed(ops[op_id]["modules"]):
        members = set(ops_in_call[call])
        leaving = [m for m in ops_in_call[call] if any(ch not in members for ch in ops[m]["children"])]
        if leaving and leaving[-1] == op_id:
            addr, _, n = call.rpartition(":")
            out.append((addr, int(n)))
    return out


def module_site(ir: dict[str, Any], op_id: str) -> tuple[str, int] | None:
    """The innermost (module address, call index) whose output is ``op_id``, else None."""
    return next(iter(module_sites(ir, op_id)), None)


def module_calls(ir: dict[str, Any], address: str) -> int:
    """How many distinct calls of the module at ``address`` the IR holds."""
    return len({c for o in ir["ops"] for c in o["modules"] if strip_call(c) == address})
