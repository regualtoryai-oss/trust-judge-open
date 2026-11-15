"""Bound paragraph dispositions with program-owned, whole-paragraph geometry.

This is an extraction wire contract, not a relevance or entailment guarantee.
Every supplied paragraph gets a decision. Annotations remain untrusted until
the ordinary independent scope review; no old decision or answer is consumed.
"""
from copy import deepcopy
from hashlib import sha256
import json
from typing import Annotated, Literal, Union

from pydantic import Field, ValidationError, create_model, field_validator

from app.agents.schemas import OutputModel
from app.domain.source_catalog import build_source_windows, resolve_source_window


PROTOCOL_VERSION = "paragraph-assertions-v17-whole-source-dispositions"
PROTOCOL = {
    "version": PROTOCOL_VERSION,
    "source_geometry": "one_complete_continuous_original_paragraph_per_alias",
    "source_validation": "build_source_windows(include_paragraphs=True)",
    "span_unit": "unicode_codepoint",
    "span_interval": "half_open",
    "dispositions": ["relevant", "no_target_fact"],
    "annotation_authority": "unverified_claims_not_published_evidence",
}
SOURCE_METADATA_FIELDS = (
    "document_id", "raw_doc_id", "raw_para_id", "title", "content_sha256",
    "content_url", "raw_publish_time", "parsed_publish_time", "time_precision",
    "start_cp", "end_cp", "document_start_cp", "document_end_cp",
    "source_credibility_json",
)


class ParagraphAssertionInvalid(ValueError):
    """Fixed diagnostic codes only; never interpolate source/model content."""

    def __init__(self, check):
        super().__init__(check)
        self.check = check
        self.details = {"check": check}


class ParagraphAssertionOutput(OutputModel):
    """Marker for strict raw-JSON handling at the provider boundary.

    The actual request schema replaces both fields with its fixed binding and
    mandatory alias map. The provider imports this marker only inside _request,
    so agents package initialization never re-enters this unfinished module.
    """

    source_catalog_id: str
    decisions: dict


def _digest(value):
    try:
        material = json.dumps(value, ensure_ascii=False, sort_keys=True,
                              separators=(",", ":"), allow_nan=False)
        return sha256(material.encode("utf-8")).hexdigest()
    except (TypeError, ValueError, UnicodeEncodeError):
        raise ParagraphAssertionInvalid("paragraph_assertion_binding_material") from None


def _schema(binding, aliases):
    from app.agents.schemas import EvidenceArguments

    class Annotation(OutputModel):
        fact_des: str = Field(min_length=1)
        argument: EvidenceArguments
        entities: list[str]

        @field_validator("fact_des")
        @classmethod
        def nonblank_fact(cls, value):
            if not value.strip():
                raise ValueError("paragraph_assertion_blank_fact")
            return value

    class Disposition(OutputModel):
        reason: str = Field(min_length=1, max_length=300)

        @field_validator("reason")
        @classmethod
        def nonblank_reason(cls, value):
            if not value.strip():
                raise ValueError("paragraph_assertion_blank_reason")
            return value

    class Relevant(Disposition):
        disposition: Literal["relevant"]
        annotations: list[Annotation] = Field(min_length=1)

    class NoTargetFact(Disposition):
        disposition: Literal["no_target_fact"]
        annotations: list[Annotation] = Field(max_length=0)

    decision = Annotated[Union[Relevant, NoTargetFact], Field(discriminator="disposition")]
    required = create_model("RequiredParagraphDispositions", __base__=OutputModel,
                            **{alias: (decision, Field(...)) for alias in aliases})
    return create_model("BoundParagraphAssertions", __base__=ParagraphAssertionOutput,
                        source_catalog_id=(Literal[binding], Field(...)),
                        decisions=(required, Field(...)))


def build_paragraph_assertion_request(question, paragraphs):
    """Return (payload, bound Pydantic schema, program-only alias catalog).

    All original paragraph metadata is hash-bound. Only explicit source
    metadata fields are sent to the model; arbitrary extra keys are never
    forwarded. Question answer-bearing/unknown keys are removed first.
    """
    from app.domain.strict_verification import project_question

    projected = project_question(question)
    # Retain the existing exact source limits. A long paragraph gets its whole
    # fixed choice, rather than a silently truncated prompt or altered quote.
    windows = build_source_windows(paragraphs, include_paragraphs=True)
    whole = {(row["paragraph_id"], row["start_cp"], row["end_cp"]): (index, row)
             for index, row in enumerate(windows)}
    binding = "PC" + _digest({"protocol": PROTOCOL, "question": projected,
                               "paragraphs": paragraphs})
    catalog, visible = [], []
    for index, paragraph in enumerate(paragraphs, 1):
        alias = f"P{index}"
        window_index, window = whole[(paragraph["id"], 0, len(paragraph["text"]))]
        metadata = {key: deepcopy(paragraph[key]) for key in SOURCE_METADATA_FIELDS if key in paragraph}
        catalog.append({"id": alias, "paragraph_id": paragraph["id"],
                        "source_window_id": f"W{window_index + 1}",
                        "paragraph_sha256": window["paragraph_sha256"],
                        "source_record_sha256": _digest(paragraph),
                        "source_metadata_sha256": _digest({key: value for key, value in paragraph.items()
                                                          if key != "text"})})
        visible.append({"id": alias, "paragraph_id": paragraph["id"],
                        "text": paragraph["text"], "source_metadata": metadata})
    payload = {"paragraph_assertion_protocol": PROTOCOL_VERSION,
               "source_catalog_id": binding, "question": projected,
               "source_paragraphs": visible}
    return payload, _schema(binding, [row["id"] for row in catalog]), catalog


def _parse_output(output):
    if not isinstance(output, str):
        return output

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ParagraphAssertionInvalid("paragraph_assertion_duplicate_json_key")
            result[key] = value
        return result

    def invalid_constant(_value):
        raise ParagraphAssertionInvalid("paragraph_assertion_contract")

    try:
        return json.loads(output, object_pairs_hook=unique, parse_constant=invalid_constant)
    except (TypeError, ValueError) as error:
        if isinstance(error, ParagraphAssertionInvalid):
            raise
        raise ParagraphAssertionInvalid("paragraph_assertion_contract") from None


def resolve_paragraph_assertions(output, question, paragraphs):
    """Return (unverified candidates, complete safe disposition audit).

    Recompile the request against current sources before validating the reply.
    The model has no quote, coordinate, paragraph-ID or window-ID field with
    which to move an annotation to another paragraph. Distinct assertions in
    one paragraph stay distinct; only exact repeated annotations are rejected.
    """
    payload, schema, catalog = build_paragraph_assertion_request(question, paragraphs)
    try:
        typed = schema.model_validate(_parse_output(output)).model_dump()
    except (ValidationError, TypeError, ValueError) as error:
        if isinstance(error, ParagraphAssertionInvalid):
            raise
        raise ParagraphAssertionInvalid("paragraph_assertion_contract") from None
    windows = build_source_windows(paragraphs, include_paragraphs=True)
    candidates, decisions = [], []
    for entry in catalog:
        decision = typed["decisions"][entry["id"]]
        seen = set()
        for annotation in decision["annotations"]:
            annotation_hash = _digest(annotation)
            if annotation_hash in seen:
                raise ParagraphAssertionInvalid("paragraph_assertion_duplicate_annotation")
            seen.add(annotation_hash)
            # All text and geometry come from the validated existing primitive.
            # Its expected catalog is rebuilt, so no caller alias map is trusted.
            candidates.append(resolve_source_window(
                {"source_window_id": entry["source_window_id"], **deepcopy(annotation)},
                windows, paragraphs, include_paragraphs=True))
        decisions.append({"id": entry["id"], "paragraph_id": entry["paragraph_id"],
                          "disposition": decision["disposition"],
                          "annotation_count": len(decision["annotations"]),
                          "reason": decision["reason"],
                          "paragraph_sha256": entry["paragraph_sha256"],
                          "source_record_sha256": entry["source_record_sha256"],
                          "source_metadata_sha256": entry["source_metadata_sha256"]})
    audit = {"protocol_version": PROTOCOL_VERSION,
             "source_catalog_id": payload["source_catalog_id"],
             "question_sha256": _digest(payload["question"]),
             "source_manifest_sha256": _digest(catalog),
             "paragraph_count": len(catalog), "candidate_count": len(candidates),
             "complete": True, "semantically_verified": False, "decisions": decisions}
    return candidates, audit


def safe_paragraph_diagnostics(diagnostics, *, paragraph_count=0):
    """Repair sees fixed codes/paths only, never prior bodies or source hints.

    The current request recompiles all sources and question independently.
    Paragraph aliases may become zero-based locations only when they belong
    to that current bounded catalog; unknown locations are redacted.
    """
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    checks = {"schema", "paragraph_assertion_contract", "paragraph_assertion_duplicate_json_key",
              "paragraph_assertion_duplicate_annotation"}
    check = diagnostics.get("check")
    safe = {"check": check if isinstance(check, str) and check in checks else "paragraph_assertion_contract"}
    count = diagnostics.get("duplicate_key_count")
    if type(count) is int and count >= 0:
        safe["duplicate_key_count"] = count
    fields = {"source_catalog_id", "decisions", "disposition", "annotations", "reason",
              "fact_des", "argument", "entities", "时间", "主体", "地点", "内容", "影响", "数值口径", "<field>"}
    kinds = {"missing", "extra_forbidden", "literal_error", "string_type", "string_too_short",
             "string_too_long", "list_type", "dict_type", "model_type", "too_short", "too_long",
             "union_tag_invalid", "union_tag_not_found", "value_error", "json_invalid", "json_type"}
    # Complete source catalogs cannot exceed 512 nonempty paragraphs. Invalid
    # caller counts grant no alias authority instead of allocating an arbitrary
    # attacker-controlled map. This is not a truncation of a valid catalog.
    aliases = {f"P{i + 1}": i for i in range(paragraph_count)} if (
        type(paragraph_count) is int and 0 <= paragraph_count <= 512) else {}
    errors = diagnostics.get("schema_errors")
    if isinstance(errors, list):
        safe["schema_errors"] = []
        for row in errors:
            if not isinstance(row, dict):
                continue
            location = row.get("loc")
            clean = []
            if isinstance(location, list):
                for index, part in enumerate(location):
                    if (index == 1 and location[0] == "decisions"
                            and isinstance(part, str) and part in aliases):
                        clean.append(aliases[part])
                    elif type(part) is int and part >= 0:
                        clean.append(part)
                    elif isinstance(part, str) and part in fields:
                        clean.append(part)
                    else:
                        clean.append("<field>")
            kind = row.get("type")
            safe["schema_errors"].append({"loc": clean,
                "type": kind if isinstance(kind, str) and kind in kinds else "value_error"})
    return safe
