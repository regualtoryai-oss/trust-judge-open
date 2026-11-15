"""Bounded browser import boundary: preview, explicit projection, atomic commit.

Raw uploads are never persisted. A preview retains only canonical sources and
question_id/question in a short-lived process-local cache; no gold values cross
that boundary. Restarting the API requires another preview, not a blind commit.
"""

from collections import OrderedDict
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import secrets
import sqlite3
import threading
import time

from fastapi import APIRouter, Request
from starlette.concurrency import run_in_threadpool

from app.data.ingest import DOC_KEYS, PARA_KEYS, QUESTION_KEYS, canonical_json, import_topic, validate_inputs
from app.errors import ServiceError

MAX_REQUEST_BYTES = 8 * 1024 * 1024
MAX_FILE_BYTES = 6 * 1024 * 1024
MAX_DOCUMENTS = 1000
MAX_PARAGRAPHS = 10000
MAX_QUESTIONS = 200
PREVIEW_TTL_SECONDS = 15 * 60
MAX_PREVIEWS = 32
MAX_CACHE_BYTES = 32 * 1024 * 1024
# Paragraph text is optional on upload and derived from content during canonical
# validation. Bound its expanded UTF-8 JSON size before materializing that copy:
# many valid overlapping spans must not amplify a small upload without a limit.
MAX_EXPANDED_SOURCE_BYTES = 16 * 1024 * 1024


def fail(code, message, *, status=422):
    raise ServiceError(status, code, message)


def object_only(value, allowed, required, path):
    if not isinstance(value, dict):
        fail("import_invalid_shape", f"{path}：应为 JSON 对象。")
    if set(value) - allowed:
        fail("import_unsupported_fields", f"{path}：包含未支持的字段；原始资料只能包含来源字段，不能夹带证据标注或答案。")
    if required - set(value):
        fail("import_missing_field", f"{path}：缺少必填字段 {', '.join(sorted(required - set(value)))}。")


def string(value, path, *, maximum=None, empty=False):
    if not isinstance(value, str) or (not empty and not value.strip()):
        fail("import_invalid_string", f"{path}：应为{'可为空的' if empty else '非空'}字符串。")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        fail("import_invalid_unicode", f"{path}：包含无效 Unicode 字符，不能建立可核验的码点坐标。")
    if maximum is not None and len(value) > maximum:
        fail("import_limit_exceeded", f"{path}：超过 {maximum} 个字符的限制。")
    return value


def parse_json(content, path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                fail("import_duplicate_json_key", f"{path}：存在重复 JSON 键，请先修正文件，不能以最后一次出现的值覆盖。")
            result[key] = value
        return result

    def nonfinite(_):
        fail("import_invalid_json", f"{path}：JSON 不能包含 NaN 或 Infinity。")

    try:
        return json.loads(content.removeprefix("\ufeff"), object_pairs_hook=pairs, parse_constant=nonfinite)
    except (json.JSONDecodeError, RecursionError, ValueError) as error:
        if isinstance(error, json.JSONDecodeError):
            fail("import_invalid_json", f"{path}：JSON 语法错误（第 {error.lineno} 行，第 {error.colno} 列）。")
        fail("import_invalid_json", f"{path}：JSON 嵌套过深或格式无效。")


async def request_body(request):
    content = bytearray()
    async for chunk in request.stream():
        content.extend(chunk)
        if len(content) > MAX_REQUEST_BYTES:
            fail("import_request_too_large", "导入请求超过 8 MiB；请分专题整理输入文件。", status=413)
    try:
        text = bytes(content).decode("utf-8-sig")
    except UnicodeDecodeError:
        fail("import_invalid_utf8", "请求不是有效 UTF-8；请将 JSON 文件转换为 UTF-8 后重试。")
    return parse_json(text, "请求体")


def validate_sources(documents, questions):
    if not isinstance(documents, list) or not documents:
        fail("import_invalid_shape", "documents：应为非空数组。")
    if len(documents) > MAX_DOCUMENTS:
        fail("import_limit_exceeded", f"documents：最多接受 {MAX_DOCUMENTS} 份原始文档。")
    if not isinstance(questions, list) or not questions:
        fail("import_invalid_shape", "questions：应为非空数组。")
    if len(questions) > MAX_QUESTIONS:
        fail("import_limit_exceeded", f"questions：最多接受 {MAX_QUESTIONS} 道问题。")
    seen_docs, paragraph_count, expanded_source_bytes = set(), 0, 0

    def account_source_text(text, path):
        nonlocal expanded_source_bytes
        # JSON escapes can be larger than raw UTF-8 (e.g. embedded controls), so
        # count the serialized UTF-8 literal, not just character count. At most
        # one bounded source/paragraph literal is built here, before the full
        # payload is canonicalized or its paragraph texts are supplemented.
        expanded_source_bytes += len(canonical_json(text).encode("utf-8"))
        if expanded_source_bytes > MAX_EXPANDED_SOURCE_BYTES:
            fail("import_expanded_source_too_large", f"{path}：补齐段落原文后的 UTF-8 数据超过 16 MiB；请拆分专题或减少重复的长段落范围。合法重叠段落可以保留。", status=413)

    for index, doc in enumerate(documents):
        path = f"documents[{index}]"
        object_only(doc, DOC_KEYS, {"doc_id", "title", "content", "paragraphs"}, path)
        raw_id = string(doc["doc_id"], f"{path}.doc_id", maximum=200)
        if raw_id in seen_docs:
            fail("import_duplicate_document_id", f"{path}.doc_id：与本专题另一份文档的原始编号重复。")
        seen_docs.add(raw_id)
        content = string(doc["content"], f"{path}.content")
        account_source_text(content, f"{path}.content")
        string(doc["title"], f"{path}.title", maximum=1000, empty=True)
        for key, maximum in (("publish_time", 200), ("content_url", 2048)):
            if doc.get(key) is not None:
                string(doc[key], f"{path}.{key}", maximum=maximum, empty=True)
        credibility = doc.get("source_credibility")
        if credibility is not None and (type(credibility) not in (int, float) or not 0 <= credibility <= 1 or not math.isfinite(credibility)):
            fail("import_invalid_credibility", f"{path}.source_credibility：应为 0–1 的有限数值或 null。")
        paragraphs = doc["paragraphs"]
        if not isinstance(paragraphs, list) or not paragraphs:
            fail("import_invalid_shape", f"{path}.paragraphs：应为非空数组。")
        paragraph_count += len(paragraphs)
        if paragraph_count > MAX_PARAGRAPHS:
            fail("import_limit_exceeded", f"paragraphs：本专题最多接受 {MAX_PARAGRAPHS} 段原文。")
        seen_paras = set()
        for para_index, para in enumerate(paragraphs):
            location = f"{path}.paragraphs[{para_index}]"
            object_only(para, PARA_KEYS, {"para_id", "char_start", "char_end"}, location)
            para_id = string(para["para_id"], f"{location}.para_id", maximum=200)
            if para_id in seen_paras:
                fail("import_duplicate_paragraph_id", f"{location}.para_id：在同一文档内重复；不同文档可以复用段落编号。")
            seen_paras.add(para_id)
            start, end = para["char_start"], para["char_end"]
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(content):
                fail("import_invalid_span", f"{location}：char_start/char_end 必须是原文 Unicode 码点的左闭右开范围，且 0 ≤ start < end ≤ {len(content)}；不要使用 UTF-16 或 UTF-8 字节坐标。")
            account_source_text(content[start:end], location)
            if "text" in para:
                text = string(para["text"], f"{location}.text")
                if text != content[start:end]:
                    fail("import_span_text_mismatch", f"{location}.text：与 content[char_start:char_end] 不逐字一致；系统不会自动改写原文或模糊对齐。")
    seen_questions, projected, extras, excluded_count = set(), [], set(), 0
    for index, question in enumerate(questions):
        path = f"questions[{index}]"
        if not isinstance(question, dict):
            fail("import_invalid_shape", f"{path}：应为 JSON 对象。")
        if QUESTION_KEYS - set(question):
            fail("import_missing_field", f"{path}：必须包含 question_id 和 question。")
        question_id = string(question["question_id"], f"{path}.question_id", maximum=200)
        if question_id in seen_questions:
            fail("import_duplicate_question_id", f"{path}.question_id：在本专题内重复。")
        seen_questions.add(question_id)
        projected.append({"question_id": question_id, "question": string(question["question"], f"{path}.question", maximum=20000)})
        excluded = set(question) - QUESTION_KEYS
        # Field names only; excluded values are never copied to cache or errors.
        for key in excluded:
            string(key, f"{path} 的字段名称", maximum=100)
        extras.update(excluded)
        excluded_count += len(excluded)
    if len(extras) > 100:
        fail("import_limit_exceeded", "questions：非运行字段种类过多，请先导出只含 question_id/question 的问题文件。")
    docs, qs = validate_inputs(documents, projected)
    excluded_fields = [{"path": "questions[]", "fields": sorted(extras), "count": excluded_count}] if extras else []
    return docs, qs, excluded_fields


def prepare_import(body):
    object_only(body, {"title", "entries"}, {"entries"}, "请求体")
    entries = body["entries"]
    if not isinstance(entries, list) or not 1 <= len(entries) <= 2:
        fail("import_invalid_files", "entries：请选择一个 input.json，或原始文档 JSON 与问题 JSON 两个文件；不支持 ZIP。")
    parsed, sizes = {}, 0
    for index, entry in enumerate(entries):
        path = f"entries[{index}]"
        object_only(entry, {"name", "content"}, {"name", "content"}, path)
        name = string(entry["name"], f"{path}.name", maximum=200)
        if "/" in name or "\\" in name or ":" in name or name in {".", ".."}:
            fail("import_invalid_filename", f"{path}.name：仅接受文件名，不能包含目录或磁盘路径。")
        name = name.lower()
        if name not in {"input.json", "documents.json", "raw_documents.json", "questions.json", "conclusions.json"}:
            fail("import_unsupported_file", f"{path}：文件名或格式不受支持。请使用 input.json 或 documents.json/raw_documents.json 加 questions.json；证据答案文件不能导入。")
        if name in parsed:
            fail("import_duplicate_file", f"{path}：文件名重复。")
        content = string(entry["content"], f"{path}.content")
        size = len(content.encode("utf-8"))
        if size > MAX_FILE_BYTES:
            fail("import_file_too_large", f"{path}：单个文件超过 6 MiB。", status=413)
        sizes += size
        if sizes > MAX_REQUEST_BYTES:
            fail("import_request_too_large", "所有文件合计超过 8 MiB。", status=413)
        parsed[name] = parse_json(content, path)
    if set(parsed) == {"input.json"}:
        payload = parsed["input.json"]
        object_only(payload, {"title", "documents", "questions"}, {"documents", "questions"}, "input.json")
        documents, questions = payload["documents"], payload["questions"]
        title = body.get("title", payload.get("title", "新导入赛题"))
    elif len(parsed) == 2 and len(set(parsed) & {"documents.json", "raw_documents.json"}) == 1 and len(set(parsed) & {"questions.json", "conclusions.json"}) == 1:
        documents = next(parsed[key] for key in ("documents.json", "raw_documents.json") if key in parsed)
        questions = next(parsed[key] for key in ("questions.json", "conclusions.json") if key in parsed)
        title = body.get("title", "新导入赛题")
    else:
        fail("import_invalid_files", "文件组合无效：input.json 必须单独导入，或选择一份原始文档文件和一份问题文件。")
    title = string(title, "title", maximum=120).strip()
    docs, qs, excluded_fields = validate_sources(documents, questions)
    safe_payload = {"title": title, "documents": docs, "questions": qs}
    encoded = canonical_json(safe_payload).encode("utf-8")
    return safe_payload, sha256(encoded).hexdigest(), excluded_fields, len(encoded)


class ImportService:
    def __init__(self, repo):
        self.repo = repo
        self.previews = OrderedDict()
        self.lock = threading.Lock()

    def prune(self):
        now = time.monotonic()
        for key in list(self.previews):
            if self.previews[key]["deadline"] < now:
                self.previews.pop(key)

    def preview(self, body):
        payload, fingerprint, excluded_fields, byte_size = prepare_import(body)
        existing = self.repo.all("SELECT id FROM topics WHERE source_manifest_sha256=? LIMIT 1", (fingerprint,))
        counts = {"documents": len(payload["documents"]), "paragraphs": sum(len(doc["paragraphs"]) for doc in payload["documents"]), "questions": len(payload["questions"])}
        preview_id = secrets.token_urlsafe(24)
        expires_at = datetime.fromtimestamp(time.time() + PREVIEW_TTL_SECONDS, timezone.utc).isoformat()
        with self.lock:
            self.prune()
            if byte_size > MAX_CACHE_BYTES or sum(item["byte_size"] for item in self.previews.values()) + byte_size > MAX_CACHE_BYTES or len(self.previews) >= MAX_PREVIEWS:
                fail("import_preview_capacity", "预检缓存容量已满，请稍后重新预检；预检记录将在 15 分钟内自动过期。", status=429)
            self.previews[preview_id] = {"payload": payload, "fingerprint": fingerprint, "excluded_fields": excluded_fields, "counts": counts, "deadline": time.monotonic() + PREVIEW_TTL_SECONDS, "byte_size": byte_size}
        warnings = ["仅保存原始资料和 question_id/question；预检不会启动模型，也不会生成研判结果。"]
        if excluded_fields:
            warnings.append(f"检测到并排除了 {excluded_fields[0]['count']} 个题目非运行字段（可能含标准答案）；必须明确确认后导入，这些字段不会入库或进入模型。")
        if existing:
            warnings.append("相同标题和内容已经导入；确认后将返回已有专题，不会覆盖原始资料或已发布结果。")
        return {"preview_id": preview_id, "fingerprint": fingerprint, "title": payload["title"], "counts": counts,
                "excluded_fields": excluded_fields, "warnings": warnings, "existing_topic_id": existing[0]["id"] if existing else None,
                "expires_at": expires_at, "questions": payload["questions"][:10], "truncated_questions_count": max(0, len(payload["questions"]) - 10)}

    def commit(self, body):
        object_only(body, {"preview_id", "fingerprint", "acknowledge_excluded_fields"}, {"preview_id", "fingerprint"}, "请求体")
        preview_id = string(body["preview_id"], "preview_id", maximum=100)
        fingerprint = string(body["fingerprint"], "fingerprint", maximum=100)
        if len(fingerprint) != 64 or any(character not in "0123456789abcdef" for character in fingerprint):
            fail("import_fingerprint_mismatch", "内容指纹无效，请重新预检。", status=409)
        acknowledge = body.get("acknowledge_excluded_fields", False)
        if type(acknowledge) is not bool:
            fail("import_invalid_shape", "acknowledge_excluded_fields：应为布尔值。")
        with self.lock:
            self.prune()
            preview = self.previews.get(preview_id)
            if preview is None:
                fail("import_preview_expired", "预检已过期或服务已重启，请重新选择文件进行预检。", status=409)
            if not secrets.compare_digest(preview["fingerprint"], fingerprint):
                fail("import_fingerprint_mismatch", "确认内容与预检指纹不一致，请重新预检，不能修改预检后直接导入。", status=409)
            if preview["excluded_fields"] and not acknowledge:
                fail("import_projection_ack_required", "题目文件含非运行字段；请先明确确认仅保留 question_id/question 并排除答案字段。", status=409)
            payload = preview["payload"]
        try:
            result = import_topic(self.repo.settings.database_path, payload["documents"], payload["questions"], payload["title"])
        except sqlite3.Error:
            fail("import_storage_unavailable", "导入事务未完成，原有专题未被覆盖；请稍后重试。", status=503)
        return {**result, "title": payload["title"], "fingerprint": fingerprint}


def import_router(repo):
    router = APIRouter(prefix="/api/imports", tags=["原始赛题导入"])
    service = ImportService(repo)

    @router.post("/preview")
    async def preview(request: Request):
        body = await request_body(request)
        return await run_in_threadpool(service.preview, body)

    @router.post("/commit")
    async def commit(request: Request):
        body = await request_body(request)
        return await run_in_threadpool(service.commit, body)

    return router
