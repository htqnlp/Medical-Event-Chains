"""End-to-end construction with explicit provenance and no cross-patient causality."""
from dataclasses import asdict
from .backends import RuleBackend
from .schema import Record, TYPES
from .temporal import extract_times
from .text import associate_time, time_blocks
from .chains import build_chains, combine_chains, fuse_graph


class Pipeline:
    def __init__(self, backend=None):
        self.backend = backend or RuleBackend()

    def run(self, records, knowledge_graph=None):
        if not isinstance(records, list) or not 1 <= len(records) <= 100:
            raise ValueError("records must contain 1 to 100 records")
        parsed = [Record.parse(r) for r in records]
        ids = [r.record_id for r in parsed]
        if len(ids) != len(set(ids)):
            raise ValueError("record_id values must be unique")
        if sum(len(r.text) for r in parsed) > 500_000:
            raise ValueError("Total input exceeds 500000 characters")
        output, all_events = [], []
        for record in parsed:
            mentions, extra_times = self.backend.extract(record.text)
            times = extract_times(record.text, record.admission_date, extra_times)
            events = []
            for index, mention in enumerate(mentions):
                mention.trigger.validate(record.text)
                mention.object.validate(record.text)
                if mention.type not in TYPES:
                    raise ValueError(f"Unknown clinical event type: {mention.type}")
                time = associate_time(record.text, mention, times)
                event = {"id": f"{record.record_id}:e{index}", "record_id": record.record_id, "patient_id": record.patient_id,
                         **asdict(mention), "time_id": f"{record.record_id}:{time.id}" if time else None,
                         "time": asdict(time) if time else None}
                if event["time"]:
                    event["time"]["id"] = event["time_id"]
                events.append(event)
            blocks = time_blocks(record.text, times)
            for block in blocks:
                if block["time_id"]:
                    block["time_id"] = f"{record.record_id}:{block['time_id']}"
            serialized_times = [{**asdict(t), "id": f"{record.record_id}:{t.id}"} for t in times]
            output.append({**asdict(record), "times": serialized_times, "blocks": blocks, "events": events})
            all_events.extend(events)
        chains = build_chains(all_events)
        combined = combine_chains(chains, all_events)
        graph = fuse_graph(chains, combined, all_events, knowledge_graph)
        return {"schema_version": "1.0", "metadata": {"implementation": "reconstruction", "backend": self.backend.metadata,
                "offset_unit": "unicode_codepoint", "offset_end": "exclusive", "synthetic_demo": False},
                "records": output, "events": all_events, "chains": chains, "combined_chains": combined, "graph": graph}
