"""Conservative temporal links and disease/symptom-based retrieval groups."""
from collections import defaultdict


def temporal_relation(left, right):
    a, b = left.get("time"), right.get("time")
    if a and b and a["start_date"] and b["start_date"]:
        if a["end_date"] < b["start_date"]:
            return "before"
        if b["end_date"] < a["start_date"]:
            return "after"
        return "time_overlap"
    return "document_order"


def build_chains(events):
    groups = defaultdict(list)
    for event in events:
        groups[event["patient_id"]].append(event)
    chains = []
    for patient, items in sorted(groups.items()):
        # Fully dated histories can be sorted by time. Mixed unresolved histories
        # retain record/document order rather than invent a position for unknown dates.
        all_dated = all(e.get("time") and e["time"]["start_date"] for e in items)
        ordered = sorted(items, key=lambda e: (e["time"]["start_date"], e["record_id"], e["object"]["start"])) if all_dated else items
        links = [{"source": a["id"], "target": b["id"], "relation": temporal_relation(a, b), "provenance": "temporal_normalizer"}
                 for a, b in zip(ordered, ordered[1:])]
        chains.append({"id": f"patient:{patient}", "patient_id": patient, "ordering": "date" if all_dated else "document_order",
                       "event_ids": [e["id"] for e in ordered], "links": links})
    return chains


def combine_chains(chains, events):
    hits = defaultdict(lambda: defaultdict(list))
    for event in events:
        if event["type"] in ("Diagnosis", "Occurrence") and not event["negated"] and not event["uncertain"]:
            hits[event["object"]["text"]][event["patient_id"]].append(event["id"])
    return [{"id": f"concept:{concept}", "concept": concept,
             "chain_ids": [c["id"] for c in chains if c["patient_id"] in patients],
             "patient_ids": sorted(patients), "matching_event_ids": [eid for ids in patients.values() for eid in ids]}
            for concept, patients in sorted(hits.items()) if len(patients) >= 2]


def validate_kg(kg):
    if not isinstance(kg, dict) or not isinstance(kg.get("nodes"), list) or not isinstance(kg.get("edges"), list):
        raise ValueError("knowledge graph requires nodes and edges lists")
    ids = set()
    for node in kg["nodes"]:
        if not isinstance(node, dict) or not isinstance(node.get("id"), str) or not isinstance(node.get("label"), str):
            raise ValueError("KG nodes require string id and label")
        if node["id"] in ids:
            raise ValueError("Duplicate KG node ID")
        ids.add(node["id"])
    for edge in kg["edges"]:
        if not isinstance(edge, dict) or edge.get("source") not in ids or edge.get("target") not in ids:
            raise ValueError("KG edge has a missing endpoint")
        if not isinstance(edge.get("relation", "related_to"), str):
            raise ValueError("KG relation must be a string")


def fuse_graph(chains, combined, events, kg=None):
    nodes = [{"id": f"event:{e['id']}", "label": e["object"]["text"], "kind": "event", "event_id": e["id"], "patient_id": e["patient_id"]} for e in events]
    edges = [{**link, "source": f"event:{link['source']}", "target": f"event:{link['target']}"} for c in chains for link in c["links"]]
    for group in combined:
        nodes.append({"id": group["id"], "label": group["concept"], "kind": "shared_concept"})
        edges.extend({"source": group["id"], "target": f"event:{eid}", "relation": "mentioned_in", "provenance": "exact_concept_match"} for eid in group["matching_event_ids"])
    if kg is not None:
        validate_kg(kg)
        nodes.extend({**n, "id": f"kg:{n['id']}", "kind": "knowledge"} for n in kg["nodes"])
        edges.extend({**e, "source": f"kg:{e['source']}", "target": f"kg:{e['target']}", "relation": e.get("relation", "related_to"), "provenance": kg.get("source", "user_supplied")} for e in kg["edges"])
        for group in combined:
            for node in kg["nodes"]:
                if group["concept"] == node["label"] or group["concept"] in node.get("aliases", []):
                    edges.append({"source": group["id"], "target": f"kg:{node['id']}", "relation": "exact_label_match", "provenance": "user_supplied_kg"})
    return {"nodes": nodes, "edges": edges, "causality_inferred": False}
