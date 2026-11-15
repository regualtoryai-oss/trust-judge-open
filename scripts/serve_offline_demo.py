"""Serve a clearly labelled, fixed-output engineering demo without a model.

Uses the project's original explicit test double to exercise import/worker/export
against original synthetic sources. Never reads .env or creates a real provider.
Requires dev dependencies and a completed `npm run build` in web/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient
import uvicorn

from app.config import Settings
from legacy_fixture import create_legacy_app
from test_import_scenarios import ScenarioProvider


class OfflineDemoProvider(ScenarioProvider):
    def account(self, operation):
        self.calls[operation] += 1
        # No model was called: do not populate synthetic provider token usage.


def wait(client, url):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        response = client.get(url)
        response.raise_for_status()
        row = response.json()
        if row["status"] in {"succeeded", "failed", "cancelled", "superseded"}:
            if row["status"] != "succeeded":
                raise RuntimeError("offline synthetic task failed")
            return row
        time.sleep(0.02)
    raise RuntimeError("offline synthetic task timed out")


def create_demo(settings):
    application = create_legacy_app(settings, OfflineDemoProvider(), embedded_worker=True)
    with TestClient(application) as client:
        if not client.get("/api/topics").json():
            for path in sorted((ROOT / "fixtures/import_samples").glob("*/input.json")):
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["title"] = "【离线合成演练·固定输出】" + payload["title"]
                preview = client.post("/api/imports/preview", json={"entries": [{"name": "input.json", "content": json.dumps(payload, ensure_ascii=False)}]})
                preview.raise_for_status()
                p = preview.json()
                imported = client.post("/api/imports/commit", json={"preview_id": p["preview_id"], "fingerprint": p["fingerprint"]})
                imported.raise_for_status()
                topic = imported.json()["topic_id"]
                questions = client.get(f"/api/topics/{topic}/questions").json()
                created = client.post(f"/api/topics/{topic}/runs", headers={"Idempotency-Key": "offline-synthetic-demo"},
                                      json={"question_ids": [q["id"] for q in questions], "base_revision_id": None, "include_baseline": False, "mode": "initial"})
                created.raise_for_status()
                run = wait(client, f"/api/runs/{created.json()['run_id']}")
                exported = client.post(f"/api/topics/{topic}/exports", headers={"Idempotency-Key": "offline-synthetic-export"},
                                       json={"revision_id": run["target_revision_id"]})
                exported.raise_for_status()
                wait(client, f"/api/exports/{exported.json()['export_id']}")

    application = create_legacy_app(settings, OfflineDemoProvider(), embedded_worker=True)

    @application.middleware("http")
    async def demo_boundary(request, call_next):
        if request.method not in {"GET", "HEAD", "OPTIONS"} and "/exports" not in request.url.path:
            return JSONResponse(status_code=405, content={"detail": {"code": "offline_demo_read_only", "message": "这是固定输出的离线合成演练，仅支持查看与导出；真实研判请启动正常工作台并配置自己的模型。"}})
        response = await call_next(request)
        response.headers["X-Trust-Demo"] = "synthetic-fixed-output-no-model"
        return response

    @application.get("/", response_class=HTMLResponse)
    def landing():
        html = (ROOT / "web/dist/index.html").read_text(encoding="utf-8")
        notice = '<aside role="note" style="position:fixed;inset:0 0 auto;z-index:99999;background:#fef3c7;color:#713f12;padding:7px 16px;font:600 13px/1.4 system-ui;text-align:center">离线合成演练 · 固定脚本输出 · 未调用模型 · 不代表模型效果</aside><style>body{padding-top:34px!important}</style>'
        return html.replace("<body>", "<body>" + notice)

    application.mount("/", StaticFiles(directory=ROOT / "web/dist", html=True), name="synthetic-demo-web")
    return application


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18790)
    parser.add_argument("--data-dir", type=Path, default=ROOT / ".runtime/public-demo")
    args = parser.parse_args()
    if not (ROOT / "web/dist/index.html").is_file():
        parser.error("build web first: cd web && npm ci && npm run build")
    settings = Settings(_env_file=None, data_dir=args.data_dir.resolve(), model_name="synthetic-offline-fixed-output", model_api_key="")
    application = create_demo(settings)
    print(f"Offline synthetic demo: http://127.0.0.1:{args.port} (fixed outputs; zero model calls)")
    uvicorn.run(application, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
