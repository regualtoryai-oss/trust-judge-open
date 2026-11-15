"""Validate the explicit business evidence DAG and compute invalidation closure.

Edges point from a premise to the judgment depending on it. Shared evidence is
not itself a conclusion-to-conclusion dependency. Closure includes changed IDs.
"""

from collections import deque

NODE_KINDS = {"evidence", "judgment", "conclusion"}
RELATIONS = {"supports", "contradicts", "prerequisite"}


def _node_id(node):
    value = node.get("node_id", node.get("id"))
    if not isinstance(value, str) or not value:
        raise ValueError("node requires a nonempty node_id or id")
    if "node_id" in node and "id" in node and node["node_id"] != node["id"]:
        raise ValueError("conflicting node IDs")
    return value


def _edge(edge):
    if not isinstance(edge, dict):
        raise ValueError("edge must be an object")
    origin, target, relation = (edge.get(key) for key in ("from_node_id", "to_node_id", "relation"))
    if not all(isinstance(value, str) and value for value in (origin, target)):
        raise ValueError("edge requires both endpoint IDs")
    if not isinstance(relation, str) or relation not in RELATIONS:
        raise ValueError("unsupported business edge relation")
    if origin == target:
        raise ValueError("self-dependency is forbidden")
    return origin, target, relation


def validate_graph(nodes: list[dict], edges: list[dict]) -> None:
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("nodes and edges must be lists")
    by_id = {}
    for node in nodes:
        if not isinstance(node, dict):
            raise ValueError("node must be an object")
        identity = _node_id(node)
        if identity in by_id or not isinstance(node.get("kind"), str) or node["kind"] not in NODE_KINDS:
            raise ValueError("duplicate ID or unsupported node kind")
        if node.get("topic_id") is not None and (not isinstance(node["topic_id"], str) or not node["topic_id"]):
            raise ValueError("topic_id must be a nonempty string when supplied")
        by_id[identity] = node
    topics = {node["topic_id"] for node in nodes if node.get("topic_id") is not None}
    if len(topics) > 1:
        raise ValueError("graph cannot contain nodes from multiple topics")
    outgoing = {node_id: set() for node_id in by_id}
    indegree = dict.fromkeys(by_id, 0)
    seen = set()
    for edge in edges:
        origin, target, relation = _edge(edge)
        if origin not in by_id or target not in by_id:
            raise ValueError("graph edge endpoint does not exist")
        if (origin, target, relation) in seen:
            raise ValueError("duplicate graph edge")
        seen.add((origin, target, relation))
        origin_kind, target_kind = by_id[origin]["kind"], by_id[target]["kind"]
        if target_kind == "evidence":
            raise ValueError("evidence source cannot depend on a judgment")
        if relation == "prerequisite" and origin_kind == "evidence":
            raise ValueError("evidence must use supports/contradicts rather than prerequisite")
        if relation in {"supports", "contradicts"} and origin_kind != "evidence":
            raise ValueError("judgments and conclusions must declare prerequisite edges")
        if target not in outgoing[origin]:
            outgoing[origin].add(target)
            indegree[target] += 1
    ready = deque(node_id for node_id, degree in indegree.items() if degree == 0)
    processed = 0
    while ready:
        node_id = ready.popleft()
        processed += 1
        for target in outgoing[node_id]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    if processed != len(nodes):
        raise ValueError("business graph contains a dependency cycle")


def downstream_closure(edges: list[dict], changed_ids) -> set[str]:
    if not isinstance(edges, list) or isinstance(changed_ids, (str, bytes)):
        raise ValueError("edges must be a list and changed_ids an iterable of IDs")
    try:
        changed = set(changed_ids)
    except TypeError as error:
        raise ValueError("invalid changed node IDs") from error
    if not all(isinstance(node_id, str) and node_id for node_id in changed):
        raise ValueError("changed node IDs must be nonempty strings")
    outgoing = {}
    for edge in edges:
        origin, target, _ = _edge(edge)
        outgoing.setdefault(origin, set()).add(target)
    result, queue = set(changed), deque(changed)
    while queue:
        for target in outgoing.get(queue.popleft(), ()):
            if target not in result:
                result.add(target)
                queue.append(target)
    return result
