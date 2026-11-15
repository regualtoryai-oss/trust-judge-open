"""Bounded local search/read over one immutable, adapter-bound source catalog.

Dictionary consistency is checked here; raw-byte manifests and exact-message
budget preflight remain caller responsibilities. No model or evidence is created.
"""
from copy import deepcopy
from hashlib import sha256
import re

from app.domain.frozen_selection_inputs import canonical, _reject_references
from app.domain.paragraph_contributions import source_blocks
from app.domain.query_context import build_query_context
from app.domain.question_recall import recall_question_candidates


class SelectionReadTools:
    def __init__(self, frozen_case, *, max_search_calls, max_read_calls):
        for value in (max_search_calls, max_read_calls):
            if type(value) is not int or value < 0:
                raise ValueError("selection_read_nonnegative_call_caps_required")
        case = deepcopy(frozen_case); _reject_references(case)
        if case.get("format") != "frozen-unified-selection-input-v1":
            raise ValueError("selection_read_bound_adapter_case_required")
        question, catalog, metadata, binding = (case[k] for k in ("question", "catalog", "source_metadata", "binding"))
        if (not isinstance(question, dict) or set(question) != {"question_id", "question"}
                or not all(isinstance(v, str) and v.strip() for v in question.values()) or not isinstance(catalog, list)
                or not isinstance(metadata, dict) or not isinstance(binding, dict)):
            raise ValueError("selection_read_original_question_catalog_required")
        for value, key in ((question, "raw_question_sha256"), (catalog, "catalog_sha256"), (metadata, "source_metadata_sha256")):
            if sha256(canonical(value)).hexdigest() != binding.get(key):
                raise ValueError("selection_read_dictionary_binding_changed")
        if any(not isinstance(binding.get(k), str) or not re.fullmatch(r"[0-9a-f]{64}", binding[k])
               for k in ("raw_input_file_sha256", "pool_input_file_sha256")):
            raise ValueError("selection_read_manifest_hashes_required")
        if case.get("publishable") is not False or case.get("v19_formal_proof_created") is not False:
            raise ValueError("selection_read_nonpublication_case_required")
        if not isinstance(case.get("topic_id"), str) or not case["topic_id"] or case.get("question_id") != question["question_id"]:
            raise ValueError("selection_read_original_case_identity_required")
        sources, parents, identities = {}, [], set()
        for rank, row in enumerate(catalog, 1):
            if not isinstance(row, dict):
                raise ValueError("selection_read_catalog_row_required")
            ref = row.get("paragraph_ref")
            if not isinstance(ref, str):
                raise ValueError("selection_read_stable_alias_required")
            meta = metadata.get(ref, {})
            if not isinstance(meta, dict):
                raise ValueError("selection_read_source_metadata_required")
            if ref != f"P{rank}" or not isinstance(row.get("text"), str) or not isinstance(row.get("title"), str):
                raise ValueError("selection_read_stable_alias_original_text_required")
            pair = row.get("doc_id"), row.get("para_id")
            if any(not isinstance(v, str) or not v.strip() for v in pair) or pair in identities:
                raise ValueError("selection_read_unique_original_source_required")
            identities.add(pair); text = row["text"]; start, end = meta.get("document_start_cp"), meta.get("document_end_cp")
            if (not isinstance(meta, dict) or (meta.get("doc_id"), meta.get("para_id")) != pair or type(meta.get("candidate_rank")) is not int or meta["candidate_rank"] != rank
                    or meta.get("paragraph_id") != f'{case["topic_id"]}/{pair[0]}/{pair[1]}' 
                    or row.get("paragraph_complete") is not True or row.get("parent_complete") is not True
                    or meta.get("parent_complete") is not True or type(meta.get("parent_start_cp")) is not int
                    or type(meta.get("parent_end_cp")) is not int or meta["parent_start_cp"] != 0 or meta["parent_end_cp"] != len(text)
                    or type(start) is not int or type(end) is not int or start < 0 or end-start != len(text)
                    or meta.get("source_text_sha256_utf8") != sha256(text.encode()).hexdigest()
                    or meta.get("v19_text_json_digest") != sha256(canonical(text)).hexdigest()
                    or meta.get("semantic_verified") is not False or meta.get("field_or_support_conflict_verified") is not False):
                raise ValueError("selection_read_identity_text_or_coordinate_binding_changed")
            sources[ref] = row
            parents.append({"id": ref, "raw_doc_id": pair[0], "raw_para_id": pair[1], "text": text, "title": row["title"]})
        if set(sources) != set(metadata):
            raise ValueError("selection_read_exact_metadata_scope_required")
        self._question = question["question"]; self._sources = sources; self._parents = parents
        self._caps = {"search": max_search_calls, "read": max_read_calls}; self._calls = {"search": 0, "read": 0}
        self._exposed = set(); self._read = set(); self._trace = []

    def _begin(self, operation, arguments):
        record = {"call_index": len(self._trace)+1, "operation": operation, "arguments": deepcopy(arguments),
                  "status": "started", "returned_paragraph_refs": [], "local_only": True, "model_calls": 0}
        self._trace.append(record)
        if self._calls[operation] >= self._caps[operation]:
            self._fail(record, "call_limit_reached")
        self._calls[operation] += 1
        return record

    @staticmethod
    def _fail(record, code):
        record.update(status="rejected", code=code)
        raise ValueError("selection_read_"+code)

    def search(self, query, limit=5):
        record = self._begin("search", {"query": query if isinstance(query, str) else None, "limit": limit if type(limit) is int else None})
        if not isinstance(query, str) or (query != self._question and len(query) > 160) or type(limit) is not int or not 1 <= limit <= 10:
            self._fail(record, "invalid_search_arguments")
        results = []
        if query.strip():
            groups = None if query == self._question else [{"terms": [query]}]
            try:
                hits = recall_question_candidates(self._question, self._parents, limit=limit, query_groups=groups, variant="window_multi")["candidates"]
                byid = {row["id"]: row for row in self._parents}
                for hit in hits:
                    ref = hit["paragraph_id"]; row = self._sources[ref]
                    view = build_query_context(self._question, [byid[ref]], query_groups=groups)["contexts"][0]
                    complete = not view["omitted_before"] and not view["omitted_after"]
                    results.append({"paragraph_ref": ref, "title": row["title"], "doc_id": row["doc_id"], "para_id": row["para_id"],
                        "excerpt_text": view["excerpt_text"], "excerpt_start_cp": view["start_cp"], "excerpt_end_cp": view["end_cp"],
                        "parent_codepoints": len(row["text"]), "parent_complete_in_catalog": True, "paragraph_complete": complete,
                        "omitted_before": view["omitted_before"], "omitted_after": view["omitted_after"],
                        "unseen_context_state": "none_omitted" if complete else "unknown_not_irrelevant", "semantic_verified": False})
            except Exception as error:
                record.update(status="failed", code="local_search_failed", error_type=type(error).__name__)
                raise
            self._exposed.update(row["paragraph_ref"] for row in results)
        status = "completed" if query.strip() else "empty_query_no_search"
        record.update(status=status, returned_paragraph_refs=[row["paragraph_ref"] for row in results])
        return {"tool": "search", "status": status, "query": query, "original_question": self._question, "limit": limit,
                "result_count": len(results), "results": results, "no_match_proves_irrelevance": False}

    def read(self, paragraph_refs):
        args = list(paragraph_refs) if isinstance(paragraph_refs, (list, tuple)) else None
        record = self._begin("read", {"paragraph_refs": args})
        if args is None or any(not isinstance(ref, str) for ref in args) or len(set(args)) != len(args):
            self._fail(record, "unique_read_aliases_required")
        if any(ref not in self._sources or ref not in self._exposed for ref in args):
            self._fail(record, "unexposed_or_unknown_read_alias")
        if any(ref in self._read for ref in args):
            self._fail(record, "paragraph_already_read")
        paragraphs = []
        for ref in args:
            row = self._sources[ref]; text = row["text"]
            paragraphs.append({"paragraph_ref": ref, "title": row["title"], "doc_id": row["doc_id"], "para_id": row["para_id"],
                "blocks": source_blocks(text) if text else [], "parent_codepoints": len(text), "paragraph_complete": True,
                "semantic_verified": False})
        self._read.update(args); record.update(status="completed", returned_paragraph_refs=args)
        return {"tool": "read", "status": "completed", "result_count": len(paragraphs), "paragraphs": paragraphs,
                "source_metadata_remains_separate": True}

    def scope(self):
        return {"exposed_paragraph_refs": [ref for ref in self._sources if ref in self._exposed],
                "read_paragraph_refs": [ref for ref in self._sources if ref in self._read],
                "search_calls": self._calls["search"], "read_calls": self._calls["read"],
                "max_search_calls": self._caps["search"], "max_read_calls": self._caps["read"],
                "model_comprehension_verified": False, "parent_raw_bytes_reverified": False}

    def trace(self):
        return deepcopy(self._trace)
