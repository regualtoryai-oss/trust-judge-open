"""Prepare the one public encoding used by the default distributed test suite.

This setup command may fetch a public encoding table. The subsequent pytest
process still blocks external network access; no model or credential is used.
"""
from hashlib import sha1, sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import tempfile

import tiktoken

ENCODING = "cl100k_base"
SOURCE_URL = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
EXPECTED_SHA256 = "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"


def main():
    cache = os.environ.get("TIKTOKEN_CACHE_DIR", os.environ.get("DATA_GYM_CACHE_DIR",
        str(Path(tempfile.gettempdir()) / "data-gym-cache")))
    if not cache:
        raise SystemExit("Set a nonempty TIKTOKEN_CACHE_DIR to prepare a reusable encoding cache")
    encoding = tiktoken.get_encoding(ENCODING)
    asset = Path(cache) / sha1(SOURCE_URL.encode()).hexdigest()
    data = asset.read_bytes()
    if sha256(data).hexdigest() != EXPECTED_SHA256:
        raise SystemExit("Public encoding cache SHA-256 mismatch")
    probe = "合成资料与 Unicode 😀"
    if encoding.decode(encoding.encode(probe)) != probe:
        raise SystemExit("Public encoding round-trip check failed")
    print(json.dumps({"encoding": ENCODING, "package_version": version("tiktoken"),
                      "cache_directory": str(Path(cache).resolve()), "sha256": EXPECTED_SHA256,
                      "bytes": len(data), "model_calls": 0, "ready": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
