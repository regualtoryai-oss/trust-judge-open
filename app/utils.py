import hashlib
import json
from datetime import datetime, timezone


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(dumps(value).encode("utf-8")).hexdigest()


def loads(value, default=None):
    return json.loads(value) if value is not None else default
