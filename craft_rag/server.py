"""Local web API and frontend for the SBC RAG prototype."""
from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .pipeline.common import load_config, write_json
from .pipeline.generate import GroundedGenerator
from .pipeline.index_build import build_indices
from .pipeline.prepared_data import load_prepared_corpus
from .pipeline.retrieve import HybridRetriever

ROOT = Path(__file__).resolve().parent
DEFAULT_HOST = "127.0.0.1"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
engine_lock = threading.Lock()
engine: dict | None = None
engine_error: str | None = None


def get_engine(config_path: str | Path | None = None) -> dict:
    global engine, engine_error
    with engine_lock:
        if engine is None:
            try:
                cfg = load_config(config_path)
                records, manifest = load_prepared_corpus(cfg)
                write_json(cfg["data"]["ingest_report"], manifest)
                index = build_indices(records, cfg)
                engine = {
                    "config": cfg,
                    "index": index,
                    "retriever": HybridRetriever(index, cfg),
                    "generator": GroundedGenerator(cfg),
                    "record_count": len(records),
                    "quality_issue_count": manifest["quality_issue_count"],
                }
                engine_error = None
            except Exception as exc:
                engine_error = f"{type(exc).__name__}: {exc}"
                logging.exception("Could not initialize RAG engine")
                raise
        return engine


class Handler(BaseHTTPRequestHandler):
    server_version = "SBC-RAG-Local/1.0"

    def log_message(self, fmt, *args):
        logging.info("%s - %s", self.address_string(), fmt % args)

    def respond(self, status: int, payload, content_type="application/json; charset=utf-8"):
        body = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/status":
            try:
                app = get_engine()
                return self.respond(200, {
                    "status": "ready", "records": app["record_count"],
                    "quality_issues": app["quality_issue_count"],
                    "backends": app["index"]["backend_info"],
                })
            except Exception:
                return self.respond(503, {"status": "error", "message": engine_error or "Engine initialization failed"})
        if path == "/api/filters":
            try:
                app = get_engine()
                chunks = app["index"]["chunks"]
                def values(key):
                    return sorted({str(row.get(key) or row.get("metadata", {}).get(key) or "").strip()
                                   for row in chunks} - {""})
                return self.respond(200, {"craft_clusters": values("craft_cluster"), "why_themes": values("why_theme")})
            except Exception as exc:
                return self.respond(503, {"error": f"Could not load filters: {type(exc).__name__}: {exc}"})
        if path in ("/", "/index.html"):
            return self.serve_file(ROOT / "web" / "index.html", "text/html; charset=utf-8")
        if path.startswith("/web/"):
            relative = path.removeprefix("/web/")
            candidate = (ROOT / "web" / relative).resolve()
            web_root = (ROOT / "web").resolve()
            if candidate != web_root and web_root not in candidate.parents:
                return self.respond(403, {"error": "Forbidden"})
            mime = {".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".svg": "image/svg+xml"}.get(candidate.suffix, "application/octet-stream")
            return self.serve_file(candidate, mime)
        return self.respond(404, {"error": "Not found"})

    def serve_file(self, path: Path, mime: str):
        try:
            return self.respond(200, path.read_bytes(), mime)
        except FileNotFoundError:
            return self.respond(404, {"error": "Not found"})

    def do_POST(self):
        if urlparse(self.path).path != "/api/query":
            return self.respond(404, {"error": "Not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 16_384:
                return self.respond(413, {"error": "Request body must be between 1 byte and 16 KB."})
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                return self.respond(400, {"error": "Request JSON must be an object."})
            question = str(data.get("question", "")).strip()
            if not question or len(question) > 1000:
                return self.respond(400, {"error": "Enter a question of 1–1000 characters."})
            app = get_engine()
            filters = {key: data[key] for key in ("craft_cluster", "why_theme") if data.get(key)}
            passages = app["retriever"].retrieve(question, filters=filters)
            answer = app["generator"].answer(question, passages)
            if answer.startswith("Insufficient evidence"):
                passages = []
            sources = []
            for passage in passages:
                meta = passage.get("metadata", {})
                sources.append({
                    "source_id": passage.get("source_id", "UNKNOWN"),
                    "title": meta.get("title", passage.get("title", "Untitled source")),
                    "evidence_level": passage.get("evidence_level", "E5"),
                    "grade": passage.get("grade", "C"),
                    "craft_cluster": passage.get("craft_cluster", meta.get("craft_cluster", "")),
                    "why_theme": passage.get("why_theme", meta.get("why_theme", "")),
                    "process_stage": passage.get("process_stage", meta.get("process_stage", "")),
                    "text": passage.get("text_chunk", ""),
                    "link": passage.get("link", ""),
                    "source_url": meta.get("source_url", ""),
                    "doi": meta.get("doi", ""),
                    "provenance": meta.get("provenance", {}),
                    "caution": passage.get("caution", ""),
                    "score": round(float(passage.get("retrieval_score", 0)), 5),
                })
            self.respond(200, {"question": question, "answer": answer, "sources": sources,
                               "backends": app["index"]["backend_info"]})
        except json.JSONDecodeError:
            self.respond(400, {"error": "Request must contain valid JSON."})
        except Exception as exc:
            logging.exception("Query failed")
            self.respond(500, {"error": f"Query failed: {type(exc).__name__}: {exc}"})


class LocalServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST, help="Bind address (defaults to loopback only)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", help="Configuration YAML path (defaults to craft_rag/config.yaml)")
    args = parser.parse_args()
    server = LocalServer((args.host, args.port), Handler)
    threading.Thread(
        target=get_engine,
        args=(args.config,),
        name="rag-index-init",
        daemon=True,
    ).start()
    print(f"SBC RAG local app: http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
