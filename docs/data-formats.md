# Data formats

All text files are UTF-8. JSON input also accepts a UTF-8 BOM. Offsets count Unicode code points, starting at zero, with an exclusive end. For each span, `source[start:end] == span.text` must hold.

## Pipeline input

```json
{
  "records": [
    {"record_id":"r1", "patient_id":"p1", "admission_date":"2026-10-20", "text":"今日出现皮疹。"},
    {"record_id":"r2", "patient_id":"p2", "text":"昨日出现皮疹。"}
  ],
  "knowledge_graph": {
    "source":"my-authorized-graph",
    "nodes":[{"id":"rash", "label":"皮疹", "aliases":["Rash"]}],
    "edges":[]
  }
}
```

`record_id` must be unique. Multiple records may share a patient ID; their events belong to the same patient chain. If a patient's events all have normalized intervals, they are date-sorted. Otherwise the input record order and within-record mention order are retained. Limits: 1–100 records, 100,000 characters per record, 500,000 total characters; HTTP body limit 2 MB.

`knowledge_graph` is optional. Node IDs must be unique and edges must refer to existing nodes. Edge shape: `{"source":"id1","target":"id2","relation":"relation_name"}`. The built-in graph uses synthetic example relations, not a downloaded medical knowledge base.

## Output schema 1.0

- `metadata`: implementation, actual backend name/description, offset convention and synthetic-demo flag.
- `records`: source records with time mentions, source-preserving blocks and events.
- `events`: flat index of `[Time, Trigger, Object, Type]` tuples plus record/patient IDs, exact trigger/object spans, negated/uncertain flags and provenance.
- `chains`: patient ID, ordered event IDs, ordering mode and explicit links. `before`/`after` mean disjoint normalized intervals; `time_overlap` does not assert simultaneity; `document_order` means only order of mentions.
- `combined_chains`: common concept, complete chain IDs, patients and matching event IDs. Only positive, certain `Occurrence`/`Diagnosis` mentions form a shared-concept group.
- `graph`: namespaced nodes/edges and provenance. Temporal relations connect only events from the same patient; group links retrieve histories and do not create cross-patient causal links.

Time status is `resolved`, `unresolved` or `invalid`, with optional inclusive start/end dates and precision. A failed normalization never substitutes the computer's current date.

## Trigger training JSONL

One complete document per line:

```json
{"id":"synthetic-001","text":"今日出现皮疹。","triggers":[{"start":2,"end":4,"text":"出现","type":"Occurrence"}]}
```

Labels are derived **only from training annotations**. Use independent document IDs in train/dev/test. Clinical chain labels are `Occurrence`, `Check`, `Diagnosis`, `Surgery`, `Treatment`. A trigger detector can also train on other labels for standalone experiments; the hybrid clinical pipeline requires clinical labels.

Overlapping triggers cannot be represented by this single BIO layer and raise errors. Cross-sentence spans and spans that do not align with fast-tokenizer offsets also raise errors. Long sentences are not silently truncated: split documents and correctly remap annotations first, or increase `--max-length` within BERT's positional limit. This implementation does not reproduce an undocumented sentence segmentation or data split from the paper.

## MLEE / BRAT conversion

`python -m medical_event_chains.mlee INPUT_DIR OUTPUT.jsonl` reads `.txt` plus `.a1/.a2` or `.ann` files. `E` lines identify trigger `T` IDs; protein entities and event arguments do not become triggers. Duplicate event references to the same trigger are deduplicated. Missing/discontinuous/overlapping triggers and source-text mismatches cause a descriptive error. The output retains original document offsets.

Conversion does not download data, split train/test, translate English biomedical labels into clinical chain types or claim the paper's original preprocessing. Keep corpus-provided train/test boundaries and use an appropriate English encoder for MLEE.
