# Validation record

Validated locally on Windows with Python 3.12.14, CPU PyTorch 2.15.0, Transformers 4.57.6, FastAPI 0.162.0, Uvicorn 0.54.0 and pytest 9.2.0. These are observed test-environment versions, not original-paper dependency versions.

## Automated software checks

`python -m pytest -q`: **42 passed**. One upstream Starlette/httpx deprecation warning did not affect results.

- Source-substring round trips, Unicode code-point offsets and exclusive ends.
- Calendar invalid dates, leap years, clamped month arithmetic, Chinese numerals, absolute/relative/coarse/unresolved time, unknown anchor handling.
- Chinese/English sentence boundaries without splitting decimal values.
- Clause-scoped negation and uncertainty; exclusion from shared positive concept grouping.
- No cross-patient temporal links; complete patient chains retrieved through common concepts.
- Multiple records per patient; mixed unknown-time order; interval overlap labels.
- Invalid record/KG inputs, API endpoints and request-size limits.
- Paddle adapter mocks: optional missing schema categories, time deduplication and trigger boundaries; no fabricated neural trigger fallback.
- Three-sentence window coverage and duplicate filler loss masking.
- Padding invariance, gradients through encoder/BiLSTM/BiGRU.
- Small randomly initialized **real BERT architecture** forward/backward.
- Local fast BERT tokenizer, full BERT training, checkpoint save/reload and prediction with no remote weights.
- Tiny-character encoder training loss reduction and deterministic checkpoint reload.
- Exact-span metrics, tolerant BIO decoding and MLEE standoff trigger conversion.
- HTML JSON escaping against script-breakout source text.

## End-to-end demo

`mec demo` generated `report.json` and a standalone `report.html` from two newly written synthetic narratives: **2 patient chains, 12 events, 1 combined group** around 皮疹, and 16 graph nodes after synthetic KG fusion.

The three-epoch tiny-model command in README executed successfully on 6 synthetic training documents and 2 synthetic development documents. Its loss decreased; the observed development F1 was **0**. This is an execution smoke test with insufficient artificial data, not a quality benchmark. No clinical or paper performance is claimed.

## Browser checks

Headless installed Chrome via Playwright:

- All eight live views loaded from the actual API output.
- Editing synthetic records performed new server extraction.
- Trigger selection and source inspector displayed actual character spans, including a non-BMP emoji prefix.
- JSON export contained the newly processed records.
- Exported standalone HTML opened from `file://` without a server and rendered the complete graph.
- Imported text containing HTML/SVG event-handler strings rendered as text and did not execute.
- 390 px mobile viewport had no page-level horizontal overflow (timelines/graphs scroll internally).
- No browser page errors. Screenshot: `docs/explorer-preview.png`.

## Not validated

- Original trained weights, original clinical datasets and original published experiments are unavailable.
- Paddle's remote UIE/WordTag assets were **not** downloaded/executed. Their adapter is mock-tested and documented as optional.
- No production, clinical, multilingual generalization or diagnostic validation is implied by these software checks.

GitHub Actions contains portable offline and neural test jobs. Local success does not imply a hosted CI run has finished; check the repository Actions page for its actual status.
