import argparse
import asyncio
import json
from pathlib import Path

from app.config import Settings
from app.data.ingest import import_topic
from app.errors import ServiceError
from app.repository import Repository
from app.worker import Worker


def main():
    parser = argparse.ArgumentParser(description="可信研判工作台 · 本地技术底座")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="初始化业务库")
    load = sub.add_parser("import", help="仅导入已投影的 input.json；拒绝gold字段")
    load.add_argument("--directory", type=Path, required=True)
    serve = sub.add_parser("serve", help="独立API进程，另需worker")
    serve.add_argument("--port", type=int, default=8790)
    serve.add_argument("--host", default="127.0.0.1")
    sub.add_parser("worker", help="独立持久化任务worker")
    recover = sub.add_parser("recover", help="停止worker后，显式复核并发布已完成检查点；不调用模型")
    recover.add_argument("--run-id", required=True)
    recover.add_argument("--idempotency-key", required=True)
    args = parser.parse_args()
    settings = Settings()
    repo = Repository(settings)
    if args.command == "init":
        print(json.dumps({"database": str(settings.database_path), "status": "ready"}, ensure_ascii=False))
    elif args.command == "import":
        paths = sorted(args.directory.resolve().rglob("input.json"))
        if not paths:
            parser.error("目录中未找到 input.json，请先运行 prepare_sample_inputs.py")
        for path in paths:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            if set(data) != {"title", "documents", "questions"}:
                parser.error("input.json 顶层仅允许 title/documents/questions")
            print(json.dumps(import_topic(settings.database_path, data["documents"], data["questions"], data["title"]), ensure_ascii=False))
    elif args.command == "serve":
        import uvicorn
        uvicorn.run("app.main:app_factory", factory=True, host=args.host, port=args.port)
    elif args.command == "recover":
        from app.recovery import recover_completed_run
        try:
            result = asyncio.run(recover_completed_run(repo, args.run_id, args.idempotency_key))
        except ServiceError as error:
            print(json.dumps({"code": error.code, "message": error.message}, ensure_ascii=False))
            raise SystemExit(1) from None
        print(json.dumps(result, ensure_ascii=False))
    else:
        try:
            asyncio.run(Worker(repo).run_forever())
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
