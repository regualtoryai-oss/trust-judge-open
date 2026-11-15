"""Local navigation and complete source reads for an existing framework loop.

This backend owns no model/agent loop, token budget, final selection, or
publication. A successful read proves only that complete source bytes were
returned to its caller. The caller must additionally prove those bytes reached
the native model request without pruning, and admit its exact serialized wire.
"""
from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256

from app.domain.frozen_selection_inputs import canonical, REFERENCE_FIELDS
from app.domain.query_context import build_query_context
from app.domain.question_recall import _bm25, _features, question_query_plan
from app.domain.source_bundles import (
    SOURCE_BUNDLE_CORE_FIELDS, assemble_source_bundle, build_source_bundles,
    validate_source_bundle,
)


_RAW_FORBIDDEN = REFERENCE_FIELDS | {
    "reference", "reference_answer", "gold_answer", "fact_des", "argument",
    "verdict_label", "resolution_type", "expected_evidence", "reference_score",
}


def _raw_only(value):
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("source_bundle_tools_string_keys_required")
        if {key.lower() for key in value} & _RAW_FORBIDDEN:
            raise ValueError("source_bundle_tools_raw_only_inputs_required")
        for item in value.values():
            _raw_only(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _raw_only(item)


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError("source_bundle_tools_invalid_" + name)


class SourceBundleTools:
    """Bound one whole trusted raw corpus and provide search/read/cite tools.

    ``question`` is the original {question_id, question} object. Documents and
    paragraphs follow ``source_bundles``' complete trusted manifest contract.
    No preselected pool is accepted or inferred. Search excerpts never count as
    reads, including excerpts that happen to contain the entire short parent.

    Read limits are local engineering limits, not model-token estimates. Every
    requested bundle is returned in full or the whole burst is rejected.
    Citation ranges must fit inside one successful read bundle; separately
    returned ranges are never stitched into a fabricated delivery certificate.
    """

    def __init__(self, documents, paragraphs, question, *, max_search_calls=3,
                 max_read_calls=2, max_paragraphs=5, max_read_anchors=12,
                 max_read_codepoints=30000, max_read_payload_bytes=200000):
        for value, name in (
            (max_search_calls, "search_call_cap"), (max_read_calls, "read_call_cap"),
            (max_read_anchors, "read_anchor_cap"), (max_read_codepoints, "read_codepoint_cap"),
            (max_read_payload_bytes, "read_payload_byte_cap"),
        ):
            _positive_int(value, name)
        if type(max_paragraphs) is not int or not 1 <= max_paragraphs <= 5:
            raise ValueError("source_bundle_tools_invalid_paragraph_cap")
        if (not isinstance(question, dict) or set(question) != {"question_id", "question"}
                or any(not isinstance(v, str) or not v.strip() for v in question.values())):
            raise ValueError("source_bundle_tools_original_question_required")
        _raw_only((documents, paragraphs, question))
        self._documents, self._paragraphs, self._question = deepcopy((documents, paragraphs, question))
        # Empty anchors intentionally validate every supplied document and row.
        build_source_bundles(self._documents, self._paragraphs, [], max_paragraphs=max_paragraphs)
        if not self._documents or not self._paragraphs:
            raise ValueError("source_bundle_tools_nonempty_complete_corpus_required")
        documents_by_id = {row["id"]: row for row in self._documents}
        self._by_id = {row["id"]: row for row in self._paragraphs}
        self._parents = []
        for row in self._paragraphs:
            title = documents_by_id[row["document_id"]].get("title", "")
            if not isinstance(title, str):
                raise ValueError("source_bundle_tools_original_title_required")
            if "title" in row and row["title"] != title:
                raise ValueError("source_bundle_tools_paragraph_title_mismatch")
            self._parents.append({**row, "title": title})
        bound = {"documents": self._documents, "paragraphs": self._paragraphs, "question": self._question}
        self._binding = {
            "format": "trusted-whole-corpus-source-bundle-tools-v1",
            "input_sha256": sha256(canonical(bound)).hexdigest(),
            "question_sha256": sha256(canonical(self._question)).hexdigest(),
            "documents_sha256": sha256(canonical(self._documents)).hexdigest(),
            "paragraphs_sha256": sha256(canonical(self._paragraphs)).hexdigest(),
        }
        self._caps = {"search": max_search_calls, "read": max_read_calls}
        self._calls = {"search": 0, "read": 0}
        self._limits = {"max_paragraphs": max_paragraphs, "max_read_anchors": max_read_anchors,
                        "max_read_codepoints": max_read_codepoints,
                        "max_read_payload_bytes": max_read_payload_bytes}
        self._exposed, self._read_ids, self._trace, self._deliveries = set(), set(), [], []
        self._successful_reads = 0

    def _begin(self, operation, arguments, *, allow_read_reuse=False):
        record = {"call_index": len(self._trace) + 1, "operation": operation,
                  "arguments": deepcopy(arguments), "status": "started", "local_only": True,
                  "model_calls": 0, "returned_paragraph_ids": [], "returned_bundle_ids": []}
        self._trace.append(record)
        if operation in self._caps:
            if self._calls[operation] >= self._caps[operation]:
                if operation == "read" and allow_read_reuse:
                    record["read_quota_exhausted"] = True
                    return record
                self._fail(record, operation + "_call_limit_reached")
            self._calls[operation] += 1
        return record

    @staticmethod
    def _fail(record, code):
        record.update(status="rejected", code=code)
        raise ValueError("source_bundle_tools_" + code)

    def search(self, query, *, limit=10, exclude_paragraph_ids=()):
        """Rank the whole raw corpus; return navigation-only continuous previews."""
        excludes = list(exclude_paragraph_ids) if isinstance(exclude_paragraph_ids, (list, tuple)) else None
        record = self._begin("search", {"query": query if isinstance(query, str) else None,
            "limit": limit if type(limit) is int else None, "exclude_paragraph_ids": excludes})
        if (not isinstance(query, str) or (query != self._question["question"] and len(query) > 160)
                or type(limit) is not int or not 1 <= limit <= 60):
            self._fail(record, "invalid_search_arguments")
        if (excludes is None or any(not isinstance(pid, str) for pid in excludes)
                or len(set(excludes)) != len(excludes) or not set(excludes) <= set(self._by_id)):
            self._fail(record, "invalid_search_exclusions")
        rows = [row for row in self._parents if row["id"] not in excludes and row["text"].strip()]
        groups = None if query == self._question["question"] else [{"terms": [query]}]
        fusion = defaultdict(float)
        if query.strip():
            plan = question_query_plan(self._question["question"], groups)
            for facet in plan["facets"]:
                for kind, weight in (("word", 1.0), ("bigram", .65)):
                    # No intermediate top40/top60/top200 pool: every matching
                    # parent remains ranked and searchable through exclusions.
                    for rank, (_, index) in enumerate(_bm25(rows, facet["text"], kind, windowed=True), 1):
                        fusion[index] += weight / (60 + rank)
            question_terms = set(_features(self._question["question"], "word"))
            for index in fusion:
                overlap = question_terms & set(_features(rows[index]["title"], "word"))
                fusion[index] *= 1 + .08 * len(overlap) / max(1, len(question_terms))
        ranked = sorted(fusion, key=lambda index: (-fusion[index], index))
        results = []
        for index in ranked[:limit]:
            row = rows[index]
            view = build_query_context(self._question["question"], [row], query_groups=groups)["contexts"][0]
            results.append({"paragraph_id": row["id"], "raw_doc_id": row["raw_doc_id"],
                "raw_para_id": row["raw_para_id"], "title": row["title"], "navigation_score": fusion[index],
                "excerpt_text": view["excerpt_text"], "excerpt_start_cp": view["start_cp"],
                "excerpt_end_cp": view["end_cp"], "parent_codepoints": len(row["text"]),
                "omitted_before": view["omitted_before"], "omitted_after": view["omitted_after"],
                "navigation_only": True, "eligible_for_citation": False,
                "unread_context_state": "unknown_not_irrelevant", "semantic_verified": False})
        self._exposed.update(row["paragraph_id"] for row in results)
        record.update(status="completed", returned_paragraph_ids=[row["paragraph_id"] for row in results])
        return {"tool": "search_source_bundles", "status": "completed", "original_question": self._question["question"],
            "query": query, "corpus_paragraph_count": len(self._parents), "matching_parent_count": len(ranked),
            "more_matching_sources_available": len(ranked) > len(results), "results": results,
            "no_match_proves_irrelevance": False, "semantic_verified": False}

    def read(self, anchor_paragraph_ids, *, before=1, after=3):
        """Return complete source contexts atomically, without text truncation."""
        anchors = list(anchor_paragraph_ids) if isinstance(anchor_paragraph_ids, (list, tuple)) else None
        record = self._begin("read", {"anchor_paragraph_ids": anchors,
            "before": before if type(before) is int else None, "after": after if type(after) is int else None}, allow_read_reuse=True)
        exhausted = record.get("read_quota_exhausted", False)
        if (anchors is None or not anchors or len(anchors) > self._limits["max_read_anchors"]
                or any(not isinstance(pid, str) for pid in anchors) or len(set(anchors)) != len(anchors)):
            self._fail(record, "read_call_limit_reached" if exhausted else "unique_nonempty_read_anchors_required")
        if any(pid not in self._by_id or pid not in self._exposed for pid in anchors):
            self._fail(record, "read_call_limit_reached" if exhausted else "unexposed_or_unknown_read_anchor")
        try:
            bundles = build_source_bundles(self._documents, self._paragraphs, anchors,
                before=before, after=after, max_paragraphs=self._limits["max_paragraphs"])
        except ValueError as error:
            if exhausted:
                self._fail(record, "read_call_limit_reached")
            record.update(status="rejected", code="invalid_source_range")
            raise ValueError("source_bundle_tools_invalid_source_range") from error
        # Adjacent anchors naturally produce overlapping contexts. Preserve
        # every complete original bundle; overlap inside this atomic delivery
        # does not make a source invalid. Coverage counts unique original IDs,
        # while payload/CP limits still count all actually delivered bytes.
        requested_ids = list(dict.fromkeys(pid for bundle in bundles for pid in bundle["source_paragraph_ids"]))
        if exhausted:
            try:
                references = [{"requested_source": {key: deepcopy(bundle[key]) for key in SOURCE_BUNDLE_CORE_FIELDS if key != "quote"},
                    "source_delivery": self._delivery_for(bundle)} for bundle in bundles]
            except ValueError:
                self._fail(record, "read_call_limit_reached")
            # A locator for existing complete bytes is not another source
            # delivery. Never include quotes or combine earlier read ranges.
            record.update(status="reused", referenced_paragraph_ids=requested_ids,
                          referenced_bundle_ids=[bundle["bundle_id"] for bundle in bundles], new_delivery=False)
            return {"tool": "read_source_bundles", "status": "reused_existing_delivery",
                "new_delivery": False, "read_allowance_consumed": False, "complete_original_slices": False,
                "source_delivery_receipts": references, "semantic_verified": False,
                "native_model_delivery_verified": False, "adoption_or_publication_created": False}
        already_read = [pid for pid in requested_ids if pid in self._read_ids]
        new_ids = [pid for pid in requested_ids if pid not in self._read_ids]
        codepoints = sum(len(bundle["quote"]) for bundle in bundles)
        result = {"tool": "read_source_bundles", "status": "completed", "result_count": len(bundles),
            "bundles": bundles, "source_codepoints": codepoints,
            "source_utf8_bytes": sum(len(bundle["quote"].encode("utf-8")) for bundle in bundles),
            "source_bundle_set_sha256": sha256(canonical(bundles)).hexdigest(),
            "delivery": {"read_call_index": record["call_index"], "complete_delivery_index": self._successful_reads + 1,
                "new_original_paragraph_ids": new_ids, "already_read_original_paragraph_ids": already_read,
                "full_original_source_bytes_returned": True},
            "complete_original_slices": True, "semantic_verified": False,
            "native_model_delivery_verified": False, "adoption_or_publication_created": False}
        payload = canonical(result)
        if codepoints > self._limits["max_read_codepoints"]:
            self._fail(record, "complete_read_codepoint_budget_exceeded")
        if len(payload) > self._limits["max_read_payload_bytes"]:
            self._fail(record, "complete_read_payload_byte_budget_exceeded")
        receipt = {"read_call_index": record["call_index"], "read_result_sha256": sha256(payload).hexdigest(),
                   "read_result_bytes": len(payload), "read_result_serialization": "canonical-json-utf8"}
        for bundle in bundles:
            self._deliveries.append({**receipt, "bundle": deepcopy(bundle)})
        self._read_ids.update(requested_ids)
        self._successful_reads += 1
        record.update(status="completed", returned_paragraph_ids=requested_ids,
            returned_bundle_ids=[bundle["bundle_id"] for bundle in bundles], **receipt)
        return deepcopy(result)

    def plan(self, anchor_paragraph_ids, *, before=1, after=3):
        """Compile legal source ranges for program handles, never mark a read.

        This program-only result may be archived but is not a model delivery.
        Actual native source presence must still come from a later full read.
        """
        anchors = list(anchor_paragraph_ids) if isinstance(anchor_paragraph_ids, (list, tuple)) else None
        record = self._begin("plan", {"anchor_paragraph_ids": anchors, "before": before, "after": after})
        if (not anchors or any(not isinstance(pid, str) for pid in anchors) or len(set(anchors)) != len(anchors)
                or any(pid not in self._exposed for pid in anchors)):
            self._fail(record, "unique_exposed_plan_anchors_required")
        bundles = build_source_bundles(self._documents, self._paragraphs, anchors,
            before=before, after=after, max_paragraphs=self._limits["max_paragraphs"])
        record.update(status="planned", planned_bundle_ids=[bundle["bundle_id"] for bundle in bundles])
        return {"tool": "plan_source_bundles", "status": "planned", "bundles": bundles,
            "source_binding": deepcopy(self._binding), "complete_original_slices": False,
            "native_model_delivery_verified": False, "adoption_or_publication_created": False}

    def _delivery_for(self, bundle):
        ids = set(bundle["source_paragraph_ids"])
        for delivery in self._deliveries:
            read = delivery["bundle"]
            if (bundle["document_id"] == read["document_id"]
                    and ids <= set(read["source_paragraph_ids"])
                    and read["start_cp"] <= bundle["start_cp"] <= bundle["end_cp"] <= read["end_cp"]):
                return {key: value for key, value in delivery.items() if key != "bundle"} | {"read_bundle_id": read["bundle_id"]}
        raise ValueError("source_bundle_tools_citation_range_not_completely_returned")

    def cite(self, start_paragraph_id, end_paragraph_id):
        """Compile source proof for an already read subset; never adopt a fact."""
        record = self._begin("cite", {"start_paragraph_id": start_paragraph_id if isinstance(start_paragraph_id, str) else None,
            "end_paragraph_id": end_paragraph_id if isinstance(end_paragraph_id, str) else None})
        try:
            bundle = assemble_source_bundle(self._documents, self._paragraphs, start_paragraph_id,
                end_paragraph_id, max_paragraphs=self._limits["max_paragraphs"])
            delivery = self._delivery_for(bundle)
        except ValueError:
            record.update(status="rejected", code="invalid_or_unread_citation_range")
            raise
        record.update(status="completed", returned_paragraph_ids=bundle["source_paragraph_ids"],
                      returned_bundle_ids=[bundle["bundle_id"]])
        return {**bundle, "bundle_purpose": "citation_source", "source_delivery": delivery,
                "semantic_verified": False, "native_model_delivery_verified": False,
                "adoption_or_publication_created": False}

    def validate_citation(self, evidence):
        """Check exact source and successful local read proof, not annotations."""
        if not isinstance(evidence, Mapping) or evidence.get("bundle_purpose") not in (None, "citation_source"):
            raise ValueError("source_bundle_tools_read_context_is_not_citation")
        validated = validate_source_bundle(evidence, self._documents, self._paragraphs,
                                            max_paragraphs=self._limits["max_paragraphs"])
        delivery = self._delivery_for(validated)
        if "source_delivery" in validated and validated["source_delivery"] != delivery:
            raise ValueError("source_bundle_tools_delivery_proof_mismatch")
        return {**validated, "source_delivery": delivery, "semantic_verified": False,
                "native_model_delivery_verified": False, "adoption_or_publication_created": False}

    def scope(self):
        """Detached source/range receipts; no claim about native model presence."""
        return {"binding": deepcopy(self._binding), "original_question": deepcopy(self._question),
            "corpus_document_count": len(self._documents), "corpus_paragraph_count": len(self._paragraphs),
            "exposed_paragraph_ids": [row["id"] for row in self._paragraphs if row["id"] in self._exposed],
            "returned_complete_paragraph_ids": [row["id"] for row in self._paragraphs if row["id"] in self._read_ids],
            "complete_read_ranges": [{**{key: value for key, value in delivery.items() if key != "bundle"},
                **{key: deepcopy(delivery["bundle"][key]) for key in SOURCE_BUNDLE_CORE_FIELDS if key != "quote"}}
                for delivery in self._deliveries],
            "calls": dict(self._calls), "call_caps": dict(self._caps), "limits": dict(self._limits),
            "successful_read_calls": self._successful_reads,
            "unread_context_state": "unknown_not_irrelevant", "model_calls": 0,
            "native_model_delivery_verified": False, "semantic_verified": False,
            "adoption_or_publication_created": False}

    def trace(self):
        return deepcopy(self._trace)
