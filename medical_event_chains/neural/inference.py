import json
from pathlib import Path
import torch
from .model import TriggerModel, TinyEncoder
from .data import CharacterTokenizer, sentence_features, windows, collate


def decode_bio(labels, offsets, text, base=0):
    spans, active = [], None
    for label, (start, end) in zip(labels, offsets):
        if start == end:
            continue
        prefix, kind = (label.split("-", 1) if "-" in label else ("O", ""))
        if prefix == "I" and active and active["type"] == kind:
            active["end"] = base + end
        else:
            if active:
                active["text"] = text[active["start"]:active["end"]]
                spans.append(active)
                active = None
            # An illegal I starts a new span, a documented tolerant BIO policy.
            if prefix in ("B", "I"):
                active = {"start": base + start, "end": base + end, "type": kind}
    if active:
        active["text"] = text[active["start"]:active["end"]]
        spans.append(active)
    return spans


class NeuralTrigger:
    def __init__(self, checkpoint, device="cpu"):
        root = Path(checkpoint)
        self.config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        self.labels = self.config["labels"]
        self.device = torch.device(device)
        if self.config["encoder_kind"] == "tiny":
            self.tokenizer = CharacterTokenizer(json.loads((root / "vocab.json").read_text(encoding="utf-8")))
            encoder = TinyEncoder(self.tokenizer.vocab_size, self.config["embedding_size"])
        else:
            from transformers import AutoConfig, AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", use_fast=True, local_files_only=True)
            encoder = AutoModel.from_config(AutoConfig.from_pretrained(root / "encoder", local_files_only=True))
        self.model = TriggerModel(encoder, len(self.labels), self.config["hidden_size"], self.config["dropout"])
        self.model.load_state_dict(torch.load(root / "model.pt", map_location="cpu", weights_only=True))
        self.model.to(self.device).eval()

    @classmethod
    def from_components(cls, model, tokenizer, labels, max_length=256, device="cpu"):
        instance = cls.__new__(cls)
        instance.model, instance.tokenizer, instance.labels = model, tokenizer, labels
        instance.device = torch.device(device)
        instance.config = {"max_length": max_length}
        return instance

    @torch.no_grad()
    def predict(self, text):
        if not text.strip():
            return []
        self.model.eval()
        features = sentence_features({"text": text, "triggers": []}, self.tokenizer, self.labels, self.config["max_length"])
        totals = [torch.zeros(len(f["input_ids"]), len(self.labels)) for f in features]
        counts = [0] * len(features)
        for indices in windows(len(features)):
            batch = collate([[features[i] for i in indices]])
            outputs = self.model(**{key: value.to(self.device) for key, value in batch.items() if key != "labels"})["logits"][0].cpu()
            seen = set()
            for slot, index in enumerate(indices):
                if index in seen:
                    continue
                seen.add(index)
                totals[index] += outputs[slot, :len(features[index]["input_ids"])]
                counts[index] += 1
        results = []
        for feature, total, count in zip(features, totals, counts):
            predictions = [self.labels[i] for i in (total / count).argmax(-1).tolist()]
            results.extend(decode_bio(predictions, feature["offsets"], text, feature["sentence"].start))
        return results
