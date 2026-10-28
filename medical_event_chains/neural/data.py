"""Strict JSONL span annotations, BIO labels and overlapping sentence windows."""
import json
from pathlib import Path
from ..text import split_spans


class CharacterTokenizer:
    """Only used with TinyEncoder for reproducible offline smoke tests."""
    def __init__(self, vocab):
        self.vocab = vocab

    @classmethod
    def build(cls, documents):
        chars = sorted({c for doc in documents for c in doc["text"]})
        return cls({char: index + 4 for index, char in enumerate(chars)})

    @property
    def vocab_size(self):
        return len(self.vocab) + 4

    def encode(self, text, max_length):
        if len(text) + 2 > max_length:
            raise ValueError("Sentence exceeds max_length; split it before training")
        return [2] + [self.vocab.get(c, 1) for c in text] + [3], [(0, 0)] + [(i, i+1) for i in range(len(text))] + [(0, 0)]


def load_documents(path):
    documents = []
    for index, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        doc = json.loads(line)
        if not isinstance(doc.get("id"), str) or not isinstance(doc.get("text"), str) or not doc["text"].strip():
            raise ValueError(f"Line {index}: id and nonempty text are required")
        if not isinstance(doc.get("triggers", []), list):
            raise ValueError(f"Line {index}: triggers must be a list")
        annotations = sorted(doc.get("triggers", []), key=lambda t: (t["start"], t["end"]))
        for position, ann in enumerate(annotations):
            if not isinstance(ann.get("type"), str) or not ann["type"] or not 0 <= ann["start"] < ann["end"] <= len(doc["text"]):
                raise ValueError(f"Line {index}: invalid trigger")
            if "text" in ann and doc["text"][ann["start"]:ann["end"]] != ann["text"]:
                raise ValueError(f"Line {index}: trigger text mismatch")
            if position and annotations[position-1]["end"] > ann["start"]:
                raise ValueError(f"Line {index}: overlapping triggers cannot be represented by BIO")
        doc["triggers"] = annotations
        documents.append(doc)
    if not documents or len({d["id"] for d in documents}) != len(documents):
        raise ValueError("Dataset is empty or document IDs are duplicated")
    return documents


def label_names(documents):
    kinds = sorted({a["type"] for d in documents for a in d.get("triggers", [])})
    if not kinds:
        raise ValueError("Training data contains no annotated triggers")
    return ["O"] + [prefix + kind for kind in kinds for prefix in ("B-", "I-")]


def windows(count):
    if count < 1:
        raise ValueError("A document needs at least one sentence")
    if count < 3:
        return [list(range(count)) + [count-1] * (3-count)]
    return [[i, i+1, i+2] for i in range(count-2)]


def encode_sentence(tokenizer, text, max_length):
    if isinstance(tokenizer, CharacterTokenizer):
        return tokenizer.encode(text, max_length)
    encoded = tokenizer(text, return_offsets_mapping=True, truncation=False, add_special_tokens=True)
    if len(encoded["input_ids"]) > max_length:
        raise ValueError("Sentence exceeds max_length; split the source and remap annotations (no silent truncation)")
    return encoded["input_ids"], [tuple(pair) for pair in encoded["offset_mapping"]]


def sentence_features(document, tokenizer, labels, max_length=256):
    spans = split_spans(document["text"])
    mapping = {label: index for index, label in enumerate(labels)}
    features, assigned = [], set()
    for sentence in spans:
        ids, offsets = encode_sentence(tokenizer, sentence.text, max_length)
        gold = [-100 if start == end else mapping["O"] for start, end in offsets]
        for ann_index, ann in enumerate(document.get("triggers", [])):
            if not sentence.start <= ann["start"] < sentence.end:
                continue
            if ann["end"] > sentence.end:
                raise ValueError("Trigger crosses a sentence boundary")
            tokens = [i for i, (a, b) in enumerate(offsets) if a != b and a + sentence.start < ann["end"] and b + sentence.start > ann["start"]]
            if not tokens or offsets[tokens[0]][0] + sentence.start != ann["start"] or offsets[tokens[-1]][1] + sentence.start != ann["end"]:
                raise ValueError("Trigger boundaries do not align with tokenizer offsets")
            for position, token in enumerate(tokens):
                label = ("B-" if position == 0 else "I-") + ann["type"]
                if label not in mapping:
                    raise ValueError(f"Unknown label: {label}")
                gold[token] = mapping[label]
            assigned.add(ann_index)
        features.append({"input_ids": ids, "labels": gold, "offsets": offsets, "sentence": sentence})
    if len(assigned) != len(document.get("triggers", [])):
        raise ValueError("One or more triggers lie outside sentence spans")
    return features


def training_samples(documents, tokenizer, labels, max_length):
    samples = []
    for document in documents:
        features = sentence_features(document, tokenizer, labels, max_length)
        for indices in windows(len(features)):
            slots, seen = [], set()
            for index in indices:
                feature = features[index]
                slots.append({"input_ids": feature["input_ids"], "labels": feature["labels"] if index not in seen else [-100] * len(feature["labels"])})
                seen.add(index)
            samples.append(slots)
    return samples


def collate(samples):
    import torch
    length = max(len(slot["input_ids"]) for sample in samples for slot in sample)
    ids = torch.zeros((len(samples), 3, length), dtype=torch.long)
    mask = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    for batch, sample in enumerate(samples):
        for index, slot in enumerate(sample):
            count = len(slot["input_ids"])
            ids[batch, index, :count] = torch.tensor(slot["input_ids"])
            mask[batch, index, :count] = 1
            labels[batch, index, :count] = torch.tensor(slot["labels"])
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}
