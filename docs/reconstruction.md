# Reconstruction record / 重建说明

## Available evidence

1. A 14 KB exploratory `final_test.py`: PaddleNLP UIE medical schema, WordTag scene/time items, dependency parsing, source-index manipulations, and event assembly. It did **not** contain the paper's neural trigger model or training code.
2. `requirement.txt`: only a human note that the script ran on the PaddlePaddle platform; no version information.
3. The four-page BIBM 2023 paper, DOI [10.1109/BIBM58861.2023.10385422](https://doi.org/10.1109/BIBM58861.2023.10385422).
4. `emr-event-atlas.html`: an eight-stage, hardcoded visual explanation using condensed English example histories. It is not an inference engine.

The original local files remain untouched. Source hashes and archive transformations are in `source-manifest.json`. The archived Python example's clinical narrative was replaced with synthetic text because its provenance cannot be established. That archive otherwise preserves the surviving logic and its known fragility.

## Mapping the paper to code

| Paper method | Code | Fidelity / choices |
| --- | --- | --- |
| UIE time extraction | `backends.PaddleBackend`, `temporal.extract_times` | Optional UIE spans; regex is an explicitly separate offline baseline |
| Establish timeline / time blocks | `text.time_blocks`, `text.associate_time` | Source-preserving boundaries; added date normalization and conservative sentence-aware association |
| BERT representations (Eq. 1) | `neural.model.TriggerModel.encoder` | Hugging Face AutoModel; no recovered pretrained task checkpoint |
| Sentence BiLSTM (Eq. 2–3) | `TriggerModel.sentence` | Bidirectional packed recurrence, preserving variable sentence lengths |
| Concatenate three sentences (Eq. 4) | `neural.data.windows`, `TriggerModel.forward` | Full sliding windows; duplicate last sentence when fewer than three |
| Context BiGRU (Eq. 5–7) | `TriggerModel.context` | Concatenate only valid tokens before context recurrence |
| Addition (Eq. 8) | `sentence + scattered` | Hidden dimensions match; no replacement by concatenation |
| Softmax / CE; Adam; dropout | `neural.model`, `neural.train` | CE on logits, argmax equivalent to softmax argmax; Adam optimizer |
| WordTag arguments | `PaddleBackend.extract` | WordTag medical label adapters plus UIE schema argument spans from surviving script |
| Medical event tuple | `schema.Mention`, `pipeline.Pipeline` | Time / trigger / object / type + provenance and assertion fields |
| Single and combined chains | `chains.build_chains`, `combine_chains` | Full patient histories retrieved by common positive disease/symptom |
| CMeKG fusion example | `chains.fuse_graph` | User-supplied KG interface; no claim of possessing the original CMeKG data |

## Details not specified in the paper

The paper does not fully specify token alignment, label encoding, batch padding, offset formats, exact hyperparameters, trigger/argument attachment, voting across overlapping windows or concept canonicalization. The following are **new, documented decisions**:

- BIO token classification. Triggers must align to tokenizer character boundaries; invalid annotations fail loudly. Exact span+type micro metrics are implemented rather than assuming the paper's scoring script.
- Windows predict all three sentences. Average logits over windows at inference. Repeated filler slots contribute context but receive no duplicate loss or prediction vote. For five sentence blocks, window coverage is 1/2/3/2/1 as described in the paper.
- The trainable model uses three sentence spans from deterministic Chinese/English punctuation splitting. Training documents should be prepared with appropriate sentence boundaries; abbreviations in English may need corpus-specific preprocessing. It is also possible to train on documents already separated by newlines.
- The chain pipeline segments time blocks for inspection but associates an extracted event's object with its nearest preceding time in the same sentence, or the first following time in that sentence, then the last preceding document time. This preserves objects before a postposed time. It is heuristic attachment, not a learned temporal relation model.
- Dates are inclusive intervals. A year remains a year interval; a month remains a month interval. Unknown and invalid expressions stay visible. Chinese N units “前/后” uses admission as the anchor; 次日/翌日 uses the previous exact day. Month/day uses admission year with no assumed rollover. “术后/出院后” remains unresolved. These conventions are **not** a complete Chinese temporal language parser.
- Every offset is a Python/Unicode code point index, end exclusive. The browser uses `Array.from` to handle non-BMP characters consistently.
- Local clause assertions are small transparent heuristics. Negation and uncertainty are retained; neither creates shared-concept groups. No accuracy claim is made for complex negation scope.
- `rule` is a demonstration baseline. `paddle` is a modernization of the surviving script. `hybrid` integrates a custom trained clinical trigger model with pretrained UIE/WordTag arguments. None is represented as the lost original implementation.
- The original dependency parser's fragile position-walking loops are replaced by same-clause nearest-trigger attachment. This is a reconstruction choice, not recovered original algorithm behavior.
- Combined chains use exact disease/symptom strings. Aliases can be supplied for matching KG concepts, but the extractor has no medical ontology normalization model.
- No causal edges, cross-patient temporal progression, clinical recommendations or paper-quality metrics are fabricated.

## Missing evidence and limits

The original clinical data, training splits, labels, random seeds, trained weights and CMeKG snapshot cannot be recovered from these files. The paper describes MLEE trigger evaluation and CCKS2020 chain construction, but neither dataset is included here. The published table has P=83.30, R=81.90, F=81.60; these are quoted paper values, not generated results (and F is not the harmonic mean of the displayed P/R). This repository does not silently repair or claim to reproduce that table.

New synthetic example annotations test software execution. Reproducing research results requires authorized datasets, a specified annotation/scoring protocol and full retraining/evaluation. A tiny smoke-test encoder is deliberately rejected for clinical hybrid inference.

## Implementation references

- [PaddleNLP Taskflow](https://paddlenlp.readthedocs.io/en/stable/source/paddlenlp.taskflow.taskflow.html)
- [Official WordTag-IE example](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/slm/examples/text_to_knowledge/wordtag-ie/README.md)
- [PyTorch packed sequences](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html)
- [Transformers token classification documentation](https://huggingface.co/docs/transformers/tasks/token_classification)

These describe dependency interfaces. The implementation and reconstruction decisions above should be checked against the paper and the available source, not treated as official upstream research code.
