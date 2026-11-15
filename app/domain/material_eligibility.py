"""Literal-question publication eligibility, not truth assessment.

The parser/catalog use only the standard library. Workflow replay also calls
the pure exact-duplicate projection; no model/evaluator/settings/network is
loaded. Catalogs come from immutable repository rows, never model annotations.
Serialized decisions are replayed, not trusted by themselves.
"""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import re

VERSION = "literal-material-eligibility-v1"
PREFIX = r"(?:仅依据|仅基于|只依据|仅使用|仅采用|仅参考|限于)"
CUTOFF = r"(?:截至|截止至|截止到)"
PUBLICATION = re.compile(r"^\s*(?:请\s*)?(?P<policy>" + PREFIX + r"\s*(?P<operator>" + CUTOFF + r")" +
    r"\s*(?P<cutoff>[^，。；！？,;!?\n]{1,100}?)\s*(?:已经|已)(?:发布|公开|披露|公布|刊发|发表)的(?:材料|资料|文件|报道|信息|数据|证据))")
PUBLICATION_BEFORE = re.compile(r"^\s*(?:请\s*)?(?P<policy>" + PREFIX +
    r"\s*(?P<cutoff>[^，。；！？,;!?\n]{1,100}?)\s*(?P<operator>之前)(?:已经|已)?(?:发布|公开|披露|公布|刊发|发表)的(?:材料|资料|文件|报道|信息|数据|证据))")
EVENT = re.compile(r"^\s*(?:请\s*)?(?P<policy>(?:仅判断|仅考虑|仅依据|只判断)\s*(?P<operator>" + CUTOFF + r")" +
    r"\s*(?P<cutoff>[^，。；！？,;!?\n]{1,100}?)\s*(?:已经|已)(?:发生|完成)的(?:事件|事项|行为))")
RESTRICTIVE_REMAINDER = re.compile(r"截至|截止|不受|不限|不局限|发布时间|发布日期|(?:允许|可以|可)使用.*(?:材料|资料)")
CHINESE_DATE = r"(?P<year>\d{4})年(?P<month>\d{1,2})月(?P<day>\d{1,2})日"
ISO_DATE = r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})"
CLOCK = r"(?:T|\s)*(?P<hour>\d{1,2})[:：](?P<minute>\d{2})(?:[:：](?P<second>\d{2})(?:\.(?P<micro>\d{1,6}))?)?"


class Invalid(ValueError):
    def __init__(self, check):
        super().__init__(check)
        self.check = check


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def literal_span(question, start, end):
    return {"start_cp": start, "end_cp": end, "text": question[start:end]}


def parse_time_literal(value):
    """Only fully explicit year and zone can establish a comparable instant.

    HH:MM normalizes to HH:MM:00 as an ISO minute timestamp. A date alone never
    becomes end-of-day/midnight. No year/zone comes from other sources or clock.
    """
    if not isinstance(value, str) or not value.strip():
        return {"status": "missing", "instant_utc": None, "precision": None}
    text = value.strip()
    zone = None
    zone_match = re.search(r"(?:[（(](北京时间|UTC(?:[+-]\d{2}:\d{2})?|GMT(?:[+-]\d{2}:\d{2})?)[）)]|(北京时间)|(?<=\d)(Z|[+-]\d{2}:\d{2}))$", text)
    if zone_match:
        token = next(part for part in zone_match.groups() if part is not None)
        text = text[:zone_match.start()].strip()
        if token == "北京时间":
            zone = timezone(timedelta(hours=8))
        elif token in {"UTC", "GMT", "Z"}:
            zone = timezone.utc
        else:
            offset = token.removeprefix("UTC").removeprefix("GMT")
            sign = 1 if offset[0] == "+" else -1
            hour, minute = int(offset[1:3]), int(offset[4:6])
            if hour > 23 or minute > 59:
                return {"status": "invalid_timezone", "instant_utc": None, "precision": None}
            zone = timezone(sign * timedelta(hours=hour, minutes=minute))
    match = None
    for date_pattern, clock in ((CHINESE_DATE, CLOCK), (ISO_DATE, CLOCK.replace("(?:T|\\s)*", "(?:T| )"))):
        match = re.fullmatch(date_pattern + clock, text)
        if match:
            break
    if match:
        if zone is None:
            return {"status": "timezone_unknown", "instant_utc": None, "precision": "datetime"}
        try:
            parts = match.groupdict()
            instant = datetime(*(int(parts[key]) for key in ("year", "month", "day", "hour", "minute")),
                second=int(parts["second"] or 0), microsecond=int((parts["micro"] or "").ljust(6, "0") or 0), tzinfo=zone)
        except ValueError:
            return {"status": "invalid_calendar_time", "instant_utc": None, "precision": None}
        return {"status": "aware_exact", "instant_utc": instant.astimezone(timezone.utc).isoformat(),
            "precision": "second" if parts["second"] is not None else "minute"}
    if re.fullmatch(CHINESE_DATE, text) or re.fullmatch(ISO_DATE, text):
        return {"status": "date_only_uncertain", "instant_utc": None, "precision": "date"}
    if re.fullmatch(r"\d{1,2}月\d{1,2}日.*", text):
        return {"status": "year_unknown", "instant_utc": None, "precision": None}
    return {"status": "unsupported_literal", "instant_utc": None, "precision": None}


def extract_question_policy(question):
    if not isinstance(question, str) or not question.strip():
        raise Invalid("eligibility_question_required")
    match, mode = None, "unspecified"
    comparison = None
    for pattern, candidate_mode, candidate_comparison in ((PUBLICATION, "publication_cutoff", "le"),
            (PUBLICATION_BEFORE, "publication_cutoff", "lt"), (EVENT, "event_cutoff", "le")):
        match = pattern.search(question)
        if match:
            mode = candidate_mode
            comparison = candidate_comparison
            break
    base = {"version": VERSION, "question_sha256": digest(question), "mode": mode,
        "extraction_status": "no_supported_explicit_policy", "policy_span": None,
        "operator_span": None, "comparison": comparison,
        "cutoff_span": None, "cutoff_time": {"status": "missing", "instant_utc": None, "precision": None}}
    if match is None:
        return base
    base.update(policy_span=literal_span(question, match.start("policy"), match.end("policy")),
        operator_span=literal_span(question, match.start("operator"), match.end("operator")),
        cutoff_span=literal_span(question, match.start("cutoff"), match.end("cutoff")))
    remainder = question[match.end():]
    # Reuse the exact supported grammars without their initial ^ anchor. A
    # second constraint is unresolved, never silently collapsed into the first.
    repeated_policy = any(re.search(pattern.pattern[1:], remainder) for pattern in (PUBLICATION, PUBLICATION_BEFORE, EVENT))
    if RESTRICTIVE_REMAINDER.search(remainder) or repeated_policy:
        return {**base, "extraction_status": "multiple_or_overridden_policies_unresolved"}
    return {**base, "extraction_status": "supported_literal_policy",
        "cutoff_time": parse_time_literal(match.group("cutoff"))}


class MaterialEligibilityCatalog:
    """Bindings to stored metadata are not external authentication of that data."""
    def __init__(self, question, immutable_sources):
        self.__question = question
        self.__policy = extract_question_policy(question)
        self.__sources = {}
        seen_raw_docs = set()
        for source in immutable_sources:
            if (not isinstance(source, dict) or any(not isinstance(source.get(key), str) or not source[key]
                    for key in ("id", "raw_doc_id", "content_sha256")) or source["id"] in self.__sources):
                raise Invalid("eligibility_source_catalog_contract")
            if not re.fullmatch(r"[a-f0-9]{64}", source["content_sha256"]):
                raise Invalid("eligibility_source_hash_contract")
            if source["raw_doc_id"] in seen_raw_docs:
                raise Invalid("eligibility_duplicate_document_metadata")
            seen_raw_docs.add(source["raw_doc_id"])
            self.__sources[source["id"]] = deepcopy({key: source.get(key) for key in
                ("id", "raw_doc_id", "content_sha256", "raw_publish_time", "parsed_publish_time", "time_precision")})
        if not self.__sources:
            raise Invalid("eligibility_source_catalog_required")
        self.__catalog_sha = digest(self.__sources)

    @property
    def policy(self):
        return deepcopy(self.__policy)

    def assess(self, canonical_source_ref):
        # Deliberately ignore annotation/event_time/eligible/publish_time fields.
        source = self.__sources.get(canonical_source_ref.get("source_id"))
        if source is None or source["raw_doc_id"] != canonical_source_ref.get("raw_doc_id"):
            raise Invalid("eligibility_canonical_source_binding")
        policy = self.__policy
        output = {"version": VERSION, "question_sha256": policy["question_sha256"],
            "policy_sha256": digest(policy), "source_catalog_sha256": self.__catalog_sha,
            "source_id": source["id"], "raw_doc_id": source["raw_doc_id"],
            "stored_source_metadata_sha256": digest(source), "mode": policy["mode"],
            "status": "unknown", "reason": "question_policy_unspecified",
            "publication_requirement_applies": None, "publication_time": None,
            "source_authority_assessment": "not_performed", "event_truth_assessment": "not_performed",
            "event_time_assessment": "not_performed", "meaning": "material_qualification_only_not_truth_or_trust_grade"}
        if policy["extraction_status"] != "supported_literal_policy":
            output["reason"] = "question_policy_unresolved" if policy["mode"] != "unspecified" else "question_policy_unspecified"
        elif policy["mode"] == "event_cutoff":
            output.update(status="not_applicable", reason="event_cutoff_does_not_limit_material_publication", publication_requirement_applies=False)
        else:
            output["publication_requirement_applies"] = True
            cutoff = policy["cutoff_time"]
            published = parse_time_literal(source["raw_publish_time"])
            parsed_helper = parse_time_literal(source["parsed_publish_time"])
            if cutoff["status"] != "aware_exact":
                output["reason"] = "cutoff_" + cutoff["status"]
            elif published["status"] != "aware_exact":
                output["reason"] = "publication_" + published["status"]
            elif source["time_precision"] is not None and source["time_precision"] not in ("minute", "second"):
                output["reason"] = "publication_declared_precision_uncertain"
            elif parsed_helper["status"] == "aware_exact" and parsed_helper["instant_utc"] != published["instant_utc"]:
                output["reason"] = "publication_metadata_conflict"
            else:
                output["publication_time"] = published
                publication_value = datetime.fromisoformat(published["instant_utc"])
                cutoff_value = datetime.fromisoformat(cutoff["instant_utc"])
                if publication_value == cutoff_value and policy["comparison"] == "lt":
                    output.update(status="excluded_at_cutoff", reason="publication_equals_exclusive_cutoff")
                else:
                    late = publication_value > cutoff_value
                    output.update(status="after_cutoff" if late else "within_cutoff",
                        reason="published_after_material_cutoff" if late else "published_by_material_cutoff")
        return {**output, "decision_binding_sha256": digest(output)}

    def verify(self, decision, canonical_source_ref):
        if decision != self.assess(canonical_source_ref):
            raise Invalid("eligibility_decision_replay_binding")
        return deepcopy(decision)

    def partition(self, canonical_facts):
        buckets = {"within_cutoff": [], "after_cutoff": [], "excluded_at_cutoff": [], "unknown": [], "not_applicable": []}
        audit = []
        for fact in canonical_facts:
            decision = self.assess(fact["source"])
            row = deepcopy(fact)
            row["program_material_eligibility"] = decision
            buckets[decision["status"]].append(row)
            audit.append({"fact_id": fact["id"], "decision": decision})
        return {"policy": self.policy, "all_original_facts_retained": True,
            "classification_is_not_semantic_verification": True, "buckets": buckets, "audit": audit}


def _question_text(question):
    text = question.get("question") if isinstance(question, dict) else question
    if not isinstance(text, str) or not text.strip():
        raise Invalid("eligibility_question_required")
    return text


def _source_manifest(paragraphs):
    """Bind the actual repository text/metadata, never model date fields."""
    fields = ("id", "document_id", "raw_doc_id", "raw_para_id", "content_sha256",
              "raw_publish_time", "parsed_publish_time", "time_precision", "text")
    if not isinstance(paragraphs, list):
        raise Invalid("eligibility_repository_source_contract")
    rows = []
    for paragraph in paragraphs:
        if not isinstance(paragraph, dict) or not isinstance(paragraph.get("id"), str) or not isinstance(paragraph.get("text"), str):
            raise Invalid("eligibility_repository_source_contract")
        rows.append({key: deepcopy(paragraph.get(key)) for key in fields})
    if len({row["id"] for row in rows}) != len(rows):
        raise Invalid("eligibility_repository_source_contract")
    return sorted(rows, key=lambda row: row["id"])


def _repository_catalog(question, paragraphs):
    by_paragraph, documents = {}, {}
    for paragraph in paragraphs:
        raw_doc = paragraph.get("raw_doc_id")
        if not isinstance(raw_doc, str) or not raw_doc or not isinstance(paragraph.get("raw_para_id"), str):
            raise Invalid("eligibility_repository_source_contract")
        document_id = paragraph.get("document_id") or "raw:" + raw_doc
        source = {"id": document_id, "raw_doc_id": raw_doc,
            **{key: deepcopy(paragraph.get(key)) for key in
               ("content_sha256", "raw_publish_time", "parsed_publish_time", "time_precision")}}
        existing = documents.setdefault(document_id, {"metadata": source, "paragraphs": []})
        if existing["metadata"] != source:
            raise Invalid("eligibility_source_metadata_conflict")
        existing["paragraphs"].append({"id": paragraph["id"], "text": paragraph["text"]})
        by_paragraph[paragraph["id"]] = {"source_id": document_id, "raw_doc_id": raw_doc,
            "raw_para_id": paragraph["raw_para_id"], "text": paragraph["text"]}
    sources = []
    for document in documents.values():
        source = deepcopy(document["metadata"])
        if source["content_sha256"] is None:
            # Test/in-memory callers may lack a stored full-document hash.
            # This binds only supplied repository paragraph text, never claims
            # authentication of a document body or a publication timestamp.
            source["content_sha256"] = digest({"basis": "supplied_paragraph_manifest",
                "paragraphs": sorted(document["paragraphs"], key=lambda row: row["id"])})
        sources.append(source)
    return MaterialEligibilityCatalog(question, sources) if sources else None, by_paragraph


def apply_workflow_material_eligibility(question, paragraphs, evidence, *, source_documents=None, source_bundle_protocol=None):
    """Apply a supported explicit publication policy before representative choice.

    No policy/event-only/ambiguous policy does not impose a publication limit.
    With an explicit publication policy, unknown timestamps stay in the full
    audit but cannot enter the qualified projection. The input is untouched.
    """
    question = _question_text(question)
    policy = extract_question_policy(question)
    applies = policy["mode"] == "publication_cutoff" and policy["extraction_status"] == "supported_literal_policy"
    if not isinstance(evidence, list) or any(not isinstance(row, dict) or not isinstance(row.get("id"), str) for row in evidence):
        raise Invalid("eligibility_evidence_contract")
    if len({row["id"] for row in evidence}) != len(evidence):
        raise Invalid("eligibility_evidence_contract")
    original = deepcopy(evidence)
    for row in original:
        row.pop("program_material_eligibility", None)
    from app.domain.snapshot import validate_source_bundle_rows, SOURCE_DOCUMENT_PROTOCOLS
    try:
        validate_source_bundle_rows(original, protocol=source_bundle_protocol,
            source_documents=source_documents, source_paragraphs=paragraphs)
    except (KeyError, TypeError, ValueError):
        raise Invalid("eligibility_source_bundle_contract") from None
    manifest_sha = digest(_source_manifest(paragraphs))
    decisions, selected, excluded, unknown = [], [], [], []
    if applies:
        catalog, bindings = _repository_catalog(question, paragraphs)
        for row in original:
            binding = bindings.get(row.get("paragraph_id"))
            if (binding is None or row.get("raw_doc_id") != binding["raw_doc_id"]
                    or row.get("raw_para_id") != binding["raw_para_id"]):
                raise Invalid("eligibility_canonical_source_binding")
            if row.get("span_scope") == "document":
                # The authoritative bundle validator checks all original members
                # and the exact document slice. Every member must resolve to the
                # same repository document's metadata, not merely the first one.
                for pid in row["source_paragraph_ids"]:
                    member = bindings.get(pid)
                    if member is None or member["source_id"] != binding["source_id"] or member["raw_doc_id"] != binding["raw_doc_id"]:
                        raise Invalid("eligibility_source_bundle_contract")
            else:
                start, end, quote = row.get("start_cp"), row.get("end_cp"), row.get("quote")
                if (type(start) is not int or type(end) is not int or not isinstance(quote, str) or not quote
                        or not 0 <= start < end <= len(binding["text"]) or binding["text"][start:end] != quote):
                    raise Invalid("eligibility_canonical_source_span")
            decision = catalog.assess(binding)
            catalog.verify(decision, binding)
            decisions.append({"evidence_id": row["id"], "decision": decision})
            if decision["status"] == "within_cutoff":
                selected.append({**deepcopy(row), "program_material_eligibility": decision})
            elif decision["status"] == "unknown":
                unknown.append(row["id"])
            else:
                excluded.append(row["id"])
    else:
        selected = deepcopy(original)
    audit = {"version": VERSION, "format": "workflow-material-eligibility-v1",
        "applied": applies, "policy": policy, "source_manifest_sha256": manifest_sha,
        "input_evidence": original, "decisions": decisions,
        "qualified_evidence_ids": [row["id"] for row in selected],
        "excluded_evidence_ids": excluded, "unknown_evidence_ids": unknown,
        "counts": {"reviewed": len(original), "qualified": len(selected), "excluded": len(excluded), "unknown": len(unknown)},
        "all_original_evidence_retained": True,
        "meaning": "publication_material_qualification_only_not_truth_authority_or_semantic_precision"}
    if source_bundle_protocol in SOURCE_DOCUMENT_PROTOCOLS:
        audit.update(source_bundle_protocol=source_bundle_protocol, source_documents_sha256=digest(source_documents))
    audit["audit_sha256"] = digest(audit)
    return selected, audit


def validate_workflow_material_eligibility(result, question, paragraphs, *, source_documents=None):
    """Replay on freeze, publication and export against current raw metadata.

    Historical results without this audit remain usable only when the original
    question has no supported explicit publication requirement. Qualifying a
    source does not authenticate its statements or give it a trust grade.
    """
    from app.domain.snapshot import validate_source_bundle_rows, SOURCE_DOCUMENT_PROTOCOLS
    bundle_protocol = result.get("source_bundle_protocol")
    try:
        validate_source_bundle_rows(result.get("evidence", []), protocol=bundle_protocol,
            source_documents=source_documents, source_paragraphs=paragraphs)
    except (KeyError, TypeError, ValueError):
        raise Invalid("eligibility_source_bundle_contract") from None
    if bundle_protocol is None and (result.get("verification_protocol") == "paragraph-ranked-source-v19" or "paragraph_ranked_proof" in result):
        from app.domain.paragraph_ranked import validate_ranked_result
        try:
            validate_ranked_result(result, question, paragraphs)
        except (KeyError, TypeError, ValueError):
            raise Invalid("paragraph_ranked_material_projection_replay") from None
        return
    audit = result.get("material_eligibility")
    if bundle_protocol is None and (result.get("verification_protocol") == "quote-faithful-bank-independent-purpose-v18"
            or "quote_faithful_bank" in result or "purpose_review" in result
            or (isinstance(audit, dict) and "purpose_authorization" in audit)
            or any(isinstance(row, dict) and "purpose_authorization" in row for row in result.get("evidence", []))):
        from app.domain.premise_bank import validate_premise_projection, PremiseBankInvalid
        try:
            validate_premise_projection(result, question, paragraphs)
        except PremiseBankInvalid as error:
            raise Invalid(error.check) from None
        return
    policy = extract_question_policy(_question_text(question))
    applies = policy["mode"] == "publication_cutoff" and policy["extraction_status"] == "supported_literal_policy"
    if audit is None:
        if applies:
            raise Invalid("eligibility_missing_workflow_audit")
        return
    if not isinstance(audit, dict) or not isinstance(audit.get("input_evidence"), list):
        raise Invalid("eligibility_workflow_audit_contract")
    qualified, rebuilt = apply_workflow_material_eligibility(question, paragraphs, audit["input_evidence"],
        source_documents=source_documents, source_bundle_protocol=bundle_protocol)
    if audit != rebuilt:
        raise Invalid("eligibility_workflow_audit_replay")
    # Replay the existing exact same-paragraph duplicate policy after eligibility.
    # An ineligible former representative can never suppress an eligible row.
    from app.domain.scope_batches import deduplicate_verified_facts
    expected, duplicate_map = (qualified, []) if bundle_protocol in SOURCE_DOCUMENT_PROTOCOLS else deduplicate_verified_facts(qualified)
    if result.get("strict_scope_review", {}).get("exact_annotation_duplicate_map", []) != duplicate_map:
        raise Invalid("eligibility_deduplication_replay")
    actual = result.get("evidence")
    if (not isinstance(actual, list) or len(actual) != len(expected)
            or any(not isinstance(row, dict) for row in actual)):
        raise Invalid("eligibility_formal_projection_replay")
    by_id = {row.get("id"): row for row in actual}
    if len(by_id) != len(actual) or set(by_id) != {row["id"] for row in expected}:
        raise Invalid("eligibility_formal_projection_replay")
    for row in expected:
        # Freeze adds program source-lineage/inventory metadata afterwards.
        # Every existing annotation, quote and qualification value must remain.
        if any(key not in by_id[row["id"]] or by_id[row["id"]][key] != value for key, value in row.items()):
            raise Invalid("eligibility_formal_projection_replay")
