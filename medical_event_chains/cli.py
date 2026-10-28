import argparse
import json
from pathlib import Path
from .pipeline import Pipeline
from .export import write_report

RESOURCES = Path(__file__).parent


def example_input():
    return json.loads((RESOURCES / "examples" / "records.json").read_text(encoding="utf-8"))


def make_backend(name, checkpoint=None, device="cpu"):
    if name == "rule":
        from .backends import RuleBackend
        return RuleBackend()
    from .backends import PaddleBackend
    predictor = None
    if name == "hybrid":
        if not checkpoint:
            raise ValueError("--backend hybrid requires --checkpoint")
        from .neural.inference import NeuralTrigger
        predictor = NeuralTrigger(checkpoint, device)
        if predictor.config["encoder_kind"] == "tiny":
            raise ValueError("A synthetic tiny checkpoint cannot be used as a clinical extraction backend")
        from .schema import TYPES
        if not any(label[2:] in TYPES for label in predictor.labels if label != "O"):
            raise ValueError("Hybrid extraction requires clinical trigger labels; MLEE event types are not interchangeable")
    return PaddleBackend(device_id=-1, trigger_predictor=predictor)


def main():
    parser = argparse.ArgumentParser(description="Time-guided medical event chains (reconstructed research implementation)")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("demo", "extract"):
        command = sub.add_parser(name)
        if name == "extract":
            command.add_argument("--input", required=True, help="JSON object containing records and optional knowledge_graph")
        command.add_argument("--output", default="artifacts/demo" if name == "demo" else "artifacts/extraction")
        command.add_argument("--backend", choices=["rule", "paddle", "hybrid"], default="rule")
        command.add_argument("--checkpoint")
        command.add_argument("--device", default="cpu")
    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--backend", choices=["rule", "paddle", "hybrid"], default="rule")
    serve.add_argument("--checkpoint")
    serve.add_argument("--device", default="cpu")
    args = parser.parse_args()
    try:
        backend = make_backend(args.backend, args.checkpoint, args.device)
        if args.command == "serve":
            import uvicorn
            from .api import create_app
            uvicorn.run(create_app(backend), host=args.host, port=args.port)
            return
        payload = example_input() if args.command == "demo" else json.loads(Path(args.input).read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict):
            raise ValueError("Input must be an object containing records")
        result = Pipeline(backend).run(payload.get("records"), payload.get("knowledge_graph"))
        result["metadata"]["synthetic_demo"] = args.command == "demo"
        paths = write_report(result, args.output)
        print(json.dumps({"records": len(result["records"]), "events": len(result["events"]), "chains": len(result["chains"]),
                          "combined_chains": len(result["combined_chains"]), "files": [str(p.resolve()) for p in paths]}, ensure_ascii=False, indent=2))
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"Error: {exc}\n")
