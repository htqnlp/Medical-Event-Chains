"""Train/evaluate a reconstructed detector, with no built-in performance claims."""
import argparse
import json
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .data import CharacterTokenizer, load_documents, label_names, training_samples, collate
from .model import TriggerModel, TinyEncoder
from .inference import NeuralTrigger


def evaluate(predictor, documents):
    gold, predicted = set(), set()
    for doc in documents:
        gold.update((doc["id"], a["start"], a["end"], a["type"]) for a in doc.get("triggers", []))
        predicted.update((doc["id"], a["start"], a["end"], a["type"]) for a in predictor.predict(doc["text"]))
    correct = len(gold & predicted)
    precision = correct / len(predicted) if predicted else 0.0
    recall = correct / len(gold) if gold else 0.0
    return {"precision": precision, "recall": recall, "f1": 2*precision*recall/(precision+recall) if precision+recall else 0.0,
            "gold": len(gold), "predicted": len(predicted), "correct": correct, "metric": "exact span + type micro"}


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    documents = load_documents(args.train)
    validation = load_documents(args.dev) if args.dev else None
    if validation and {d["id"] for d in documents} & {d["id"] for d in validation}:
        raise ValueError("Train/dev document IDs overlap")
    labels = label_names(documents)
    if args.encoder == "tiny":
        tokenizer = CharacterTokenizer.build(documents)
        encoder = TinyEncoder(tokenizer.vocab_size, 32)
    else:
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.bert, use_fast=True)
        if not tokenizer.is_fast:
            raise ValueError("A fast tokenizer with offsets is required")
        encoder = AutoModel.from_pretrained(args.bert)
        if args.max_length > encoder.config.max_position_embeddings:
            raise ValueError("max_length exceeds BERT position embeddings")
    samples = training_samples(documents, tokenizer, labels, args.max_length)
    loader = DataLoader(samples, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(args.seed))
    device = torch.device(args.device)
    model = TriggerModel(encoder, len(labels), args.hidden_size, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for batch in loader:
            optimizer.zero_grad()
            result = model(**{key: value.to(device) for key, value in batch.items()})
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += result["loss"].item()
        entry = {"epoch": epoch+1, "loss": total / len(loader)}
        if validation:
            predictor = NeuralTrigger.from_components(model, tokenizer, labels, args.max_length, args.device)
            entry["dev"] = evaluate(predictor, validation)
        history.append(entry)
        print(json.dumps(entry, ensure_ascii=False), flush=True)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    config = {"encoder_kind": args.encoder, "bert": args.bert if args.encoder == "bert" else None,
              "embedding_size": encoder.config.hidden_size, "hidden_size": args.hidden_size,
              "dropout": args.dropout, "labels": labels, "max_length": args.max_length,
              "seed": args.seed, "synthetic_smoke_model": args.encoder == "tiny"}
    (destination / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (destination / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if args.encoder == "tiny":
        (destination / "vocab.json").write_text(json.dumps(tokenizer.vocab, ensure_ascii=False), encoding="utf-8")
    else:
        tokenizer.save_pretrained(destination / "tokenizer")
        encoder.config.save_pretrained(destination / "encoder")
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, destination / "model.pt")
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("train")
    fit.add_argument("--train", required=True)
    fit.add_argument("--dev")
    fit.add_argument("--output", required=True)
    fit.add_argument("--encoder", choices=["tiny", "bert"], default="bert")
    fit.add_argument("--bert", default="bert-base-chinese")
    fit.add_argument("--epochs", type=int, default=10)
    fit.add_argument("--batch-size", type=int, default=4)
    fit.add_argument("--hidden-size", type=int, default=128)
    fit.add_argument("--dropout", type=float, default=0.2)
    fit.add_argument("--max-length", type=int, default=256)
    fit.add_argument("--lr", type=float, default=2e-5)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--threads", type=int, default=2)
    fit.add_argument("--device", default="cpu")
    for name in ("evaluate", "predict"):
        command = sub.add_parser(name)
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--device", default="cpu")
        if name == "evaluate":
            command.add_argument("--data", required=True)
        else:
            command.add_argument("--text", required=True)
    args = parser.parse_args()
    if args.command == "train":
        if args.epochs < 1 or args.batch_size < 1 or args.hidden_size < 1:
            parser.error("epochs, batch-size and hidden-size must be positive")
        train(args)
    else:
        predictor = NeuralTrigger(args.checkpoint, args.device)
        result = evaluate(predictor, load_documents(args.data)) if args.command == "evaluate" else predictor.predict(args.text)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
