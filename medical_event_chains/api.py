"""Optional local-only-by-default web service. No patient text is logged."""
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from .pipeline import Pipeline
from .cli import example_input


def create_app(backend=None):
    app = FastAPI(title="Medical Event Chains", version="0.1.0")
    pipeline = Pipeline(backend)
    web = Path(__file__).parent / "web"

    @app.middleware("http")
    async def limit_request(request: Request, call_next):
        if request.method == "POST":
            chunks, total = [], 0
            async for chunk in request.stream():
                total += len(chunk)
                if total > 2_000_000:
                    return JSONResponse({"detail": "Request exceeds 2 MB"}, status_code=413)
                chunks.append(chunk)
            request._body = b"".join(chunks)
        return await call_next(request)

    @app.get("/api/health")
    def health():
        return {"status": "ok", "backend": pipeline.backend.metadata}

    @app.get("/api/example")
    def example():
        return example_input()

    @app.post("/api/extract")
    def extract(payload: dict):
        try:
            return pipeline.run(payload.get("records"), payload.get("knowledge_graph"))
        except (ValueError, TypeError, KeyError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/")
    def index():
        return FileResponse(web / "index.html")

    @app.get("/app.js")
    def javascript():
        return FileResponse(web / "app.js", media_type="text/javascript")

    @app.get("/style.css")
    def stylesheet():
        return FileResponse(web / "style.css", media_type="text/css")

    return app
