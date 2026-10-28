"""Convert MLEE/BRAT standoff event triggers, retaining character offsets.

Nested event arguments are not trigger labels. Discontinuous triggers and BIO
overlaps cause an explicit error instead of silently changing annotations.
"""
import argparse
import json
from pathlib import Path


def convert_document(text_path):
    text_path = Path(text_path)
    text = text_path.read_bytes().decode("utf-8")  # preserve CRLF for standoff offsets
    annotations = []
    for suffix in (".a1", ".a2", ".ann"):
        path = text_path.with_suffix(suffix)
        if path.exists():
            annotations.extend(path.read_text(encoding="utf-8").splitlines())
    spans, trigger_ids = {}, set()
    for line in annotations:
        if line.startswith("T"):
            ident, spec, surface = line.split("\t", 2)
            kind, offset_text = spec.split(" ", 1)
            if ";" in offset_text:
                spans[ident] = {"discontinuous": True}
                continue
            start, end = map(int, offset_text.split())
            if text[start:end] != surface:
                raise ValueError(f"{text_path.name} {ident}: source offset mismatch")
            spans[ident] = {"start": start, "end": end, "text": surface, "type": kind}
        elif line.startswith("E"):
            event = line.split("\t", 1)[1].split()[0]
            trigger_ids.add(event.split(":", 1)[1])
    triggers = []
    seen = set()
    for ident in sorted(trigger_ids):
        if ident not in spans or spans[ident].get("discontinuous"):
            raise ValueError(f"{text_path.name}: missing or discontinuous trigger {ident}")
        span = spans[ident]
        key = (span["start"], span["end"], span["type"])
        if key not in seen:
            triggers.append(span)
            seen.add(key)
    triggers.sort(key=lambda x: (x["start"], x["end"]))
    for left, right in zip(triggers, triggers[1:]):
        if left["end"] > right["start"]:
            raise ValueError(f"{text_path.name}: overlapping triggers unsupported by BIO")
    return {"id": text_path.stem, "text": text, "triggers": triggers}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Directory containing .txt plus .a1/.a2 or .ann")
    parser.add_argument("output")
    args = parser.parse_args()
    paths = sorted(Path(args.input).glob("*.txt"))
    if not paths:
        parser.error("No .txt documents found")
    documents = [convert_document(path) for path in paths]
    Path(args.output).write_text("\n".join(json.dumps(doc, ensure_ascii=False) for doc in documents)+"\n", encoding="utf-8")
    print(json.dumps({"documents": len(documents), "triggers": sum(len(d["triggers"]) for d in documents)}))


if __name__ == "__main__":
    main()
