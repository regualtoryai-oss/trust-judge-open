"""Typed model output before service-owned identifiers and reference validation."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class OutputModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class EvidenceArguments(OutputModel):
    时间: str | None = Field(default=None, description="所选原文事实适用的时点、截至时间或区间；未知为null，不用发布日期代替。")
    主体: str | None = Field(default=None, description="原文中的事实主体；涉及转述时保留谁声称/谁报道，不把主张当已核实事实。")
    地点: str | None = Field(default=None, description="原文明确陈述的地点；未知为null。")
    内容: str | None = Field(default=None, description="原文直接支持的同一主题断言及必要限定，保留否定、条件及已核实/未核实边界。")
    影响: str | None = Field(default=None, description="原文明确陈述的影响；不从主题或常识自行推导。")
    数值口径: str | None = Field(default=None, description="原文明确的数值、单位、统计主体、范围、含/不含群体及适用时点；未知部分不补全。")


class EvidenceFields(OutputModel):
    paragraph_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)
    fact_des: str = Field(min_length=1)
    argument: EvidenceArguments
    entities: list[str]


class ExtractionCandidate(EvidenceFields):
    """Untrusted coordinate hints; the source verifier owns their geometry.

    A provider can emit zero, reversed or one-sided integer hints even for a
    unique exact quote. Preserve these hints until deterministic location;
    never mistake the wire representation for accepted source evidence.
    """
    start_cp: int | None = Field(default=None, description="Optional untrusted coordinate hint; prefer null")
    end_cp: int | None = Field(default=None, description="Optional untrusted coordinate hint; prefer null")


class EvidenceCandidate(EvidenceFields):
    start_cp: int | None = Field(default=None, ge=0)
    end_cp: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def paired_offsets(self):
        if (self.start_cp is None) != (self.end_cp is None):
            raise ValueError("Both paragraph-relative offsets must be supplied together")
        if self.start_cp is not None and self.end_cp <= self.start_cp:
            raise ValueError("Evidence must have a nonempty half-open span")
        return self


class ExtractionOutput(OutputModel):
    evidence: list[ExtractionCandidate] = Field(
        description="可逐字定位于给定段落的候选事实；没有合格原文时返回空数组，不编造证据。",
        json_schema_extra={"maxItems": 12},
    )


class SourceSelectionCandidate(OutputModel):
    """Model selects exactly one predefined source window, never geometry."""
    source_window_id: str = Field(min_length=1, description="本次source_catalog_id绑定目录中的一个窗口短ID，例如W1；围绕同一主题的事实与必要限定，可选固定的完整原段窗口。引用几何由程序固定，不输出起止字段。")
    fact_des: str = Field(min_length=1, description="所选原文直接支持的连贯断言簇；保留说话主体、时点、统计范围和未核实等限定，不编造归因，不强制切成极短单原子。")
    argument: EvidenceArguments
    entities: list[str]


class SourceSelectionOutput(OutputModel):
    source_catalog_id: str = Field(min_length=1, description="逐字回显本次窗口目录的绑定ID；窗口短ID只在此目录内有效，不能沿用其他请求的绑定。")
    evidence: list[SourceSelectionCandidate] = Field(description="完整保留本批原文中直接回答、限定或反驳问题的连贯断言及各自出处，包括支持、更正和反证；不设凑数配额，仅主题相关不算支撑。没有合格证据可返回空数组。")


class EvidenceAssessment(OutputModel):
    evidence_id: str = Field(min_length=1)
    relation: Literal["support", "conflict", "qualifier", "background", "duplicate"]
    reason: str = Field(min_length=1, description="根据原文说明与问题具体断言的关系，不能只写相关或相似。")
    equivalent_to: str | None = Field(default=None, description="仅duplicate填写：被保留且完整覆盖本条所有有效事实与限定的证据ID；这不表示两来源独立性已核实。")


class EvidenceSelectionOutput(OutputModel):
    selected_evidence_ids: list[str] = Field(description="保留直接回答、反驳或必要限定的证据；不设固定条数，不能为精简而丢弃独立反证、时点或不同口径。")
    assessments: list[EvidenceAssessment] = Field(description="逐条覆盖全部候选，各ID恰好一次；support/conflict/qualifier必须保留，background/duplicate不进入本题有效证据。")


class PrecisionDecision(OutputModel):
    """One wire decision; candidate identity is the required map key."""
    relation: Literal["support", "conflict", "qualifier", "background", "duplicate"]
    reason: str = Field(min_length=1, description="根据候选原文说明与本题具体断言的关系，不得只写相关、相似或来源等级。")
    equivalent_to: str | None = Field(default=None, description="仅duplicate填写本批保留候选的C短ID；其他关系必须为null。")


class PrecisionDecisionsOutput(OutputModel):
    """The adapter replaces decisions with a complete required-key model."""
    candidate_catalog_id: str = Field(min_length=1, description="逐字回显本次候选目录的绑定ID；C短ID只对这一问题和候选目录有效。")
    decisions: dict[str, PrecisionDecision] = Field(description="每个本批C短ID必须恰好有一个决定，不输出额外候选ID或selected列表。")


class JudgmentOutput(OutputModel):
    # Advertise the business contract to the model without moving its authority
    # into Pydantic. The workflow reports an indexed grounding diagnostic and
    # validates the full dependency DAG against the frozen snapshot.
    model_config = ConfigDict(extra="forbid", strict=True, json_schema_extra={
        "anyOf": [
            {"properties": {"evidence_ids": {"minItems": 1}}, "required": ["evidence_ids"]},
            {"properties": {"prerequisite_judgment_ids": {"minItems": 1}}, "required": ["prerequisite_judgment_ids"]},
        ],
    })
    id: str = Field(min_length=1, description="Local identifier such as J1; never a resource UUID")
    statement: str = Field(min_length=1, description="有证据依据的简短判断；无引用的资料缺口不能单列为判断。")
    evidence_ids: list[str] = Field(
        description="直接支持本判断的当前冻结evidence.id，不能用paragraph_id；与前提列表不可同时为空。",
        json_schema_extra={"uniqueItems": True},
    )
    prerequisite_judgment_ids: list[str] = Field(
        description="本次judgments中局部ID的前提，不能自引或成环；前提链最终必须落到证据。",
        json_schema_extra={"uniqueItems": True},
    )


class AnalysisOutput(OutputModel):
    model_config = ConfigDict(extra="forbid", strict=True, json_schema_extra={
        "allOf": [{
            "if": {"properties": {"decision": {"const": "asserted"}}, "required": ["decision"]},
            "then": {"properties": {"supporting_evidence_ids": {"minItems": 1}}},
        }],
    })
    decision: Literal["asserted", "undetermined"]
    conclusion: str = Field(min_length=1, description="逐项回答问题并保留主体、时点、范围、已核实程度等关键限定；可陈述有依据的部分事实并说明未决部分，不以笼统缺证代替全部分析。")
    judgments: list[JudgmentOutput] = Field(description="仅含有依据判断；未决且无合格判断时允许[]。")
    supporting_evidence_ids: list[str] = Field(
        description="实质支持本次结论已述事实的当前evidence.id；asserted至少一项，undetermined允许[]。未决但有依据的有限事实仍应保留引用，不能因未决清空已有依据。",
        json_schema_extra={"uniqueItems": True},
    )
    conflicting_evidence_ids: list[str] = Field(
        description="与本次结论实际相反的当前evidence.id；保留真正反证和未消解的相反材料，不把同主题背景或单纯缺证当反证，不能为通过校验而省略。",
        json_schema_extra={"uniqueItems": True},
    )


class ConflictOutput(OutputModel):
    id: str = Field(min_length=1, description="Local conflict identifier such as C1")
    statement: str = Field(min_length=1)
    status: Literal["resolved", "coexisting", "unresolved"]
    supporting_evidence_ids: list[str] = Field(min_length=1)
    conflicting_evidence_ids: list[str] = Field(min_length=1)
    resolution_type: str = Field(min_length=1)
    basis: str = Field(min_length=1)
    accepted_reason: str = Field(min_length=1)
    rejected_reason: str = Field(min_length=1)
    minority_opinion: str = Field(min_length=1)
    reopen_conditions: str = Field(min_length=1)


class ArbitrationOutput(AnalysisOutput):
    conflicts: list[ConflictOutput]


FieldEntailment = Literal["entailed", "not_present", "needs_narrowing", "unsupported", "contradicted"]


class ScopeArgumentChecks(OutputModel):
    时间: FieldEntailment
    主体: FieldEntailment
    地点: FieldEntailment
    内容: FieldEntailment
    影响: FieldEntailment
    数值口径: FieldEntailment


class ScopeFieldChecks(OutputModel):
    fact_des: FieldEntailment
    argument: ScopeArgumentChecks
    entities: FieldEntailment


class ScopeVerificationDecision(OutputModel):
    fact_entailment: Literal["entailed", "needs_narrowing", "unsupported", "contradicted"]
    target_relation: Literal["direct_answer", "explicit_counterclaim", "necessary_qualifier", "outside_target", "uncertain"]
    target_question_text: str | None = Field(description="直接复制本题中所对应的连续文字；目标内关系必须非空，不新增问题前提。")
    qualifier_dimension: Literal["subject", "time", "scope", "value", "attribution", "certainty", "causality"] | None
    requires_unstated_bridge: bool = Field(description="原句到该事实或该事实到题目所问属性是否需要原文未给出的桥接推断。")
    # Optional only when replaying historical ledgers or explicitly injected
    # adapters. The live provider's bound schema requires all three v15 fields.
    quote_requires_unstated_bridge: bool | None = Field(default=None, description="仅判断quote到当前事实解释是否加入原文未述前提；不判断事实能否独自证明整道题。")
    target_requires_unstated_bridge: bool | None = Field(default=None, description="判断所声称的问题关系是否需额外前提。必要限定只需明确改变某一所问属性的边界，不要求单独证明该属性或整个复合问题。")
    field_checks: ScopeFieldChecks | None = Field(default=None, description="逐项核对fact_des、六个argument字段及entities；非空字段须entailed，空字段可not_present。不能用整体通过掩盖某字段强化。")
    target_fact_present_but_annotation_incomplete: bool = Field(description="当前注释即使正确但偏离目标，固定quote中是否另有本题直接事实或关键限定尚未被该注释完整表达；true进入同引文定向重抽。")
    reason: str = Field(min_length=1, description="分别说明原句是否支撑当前事实解释、该事实与题目具体请求的直接关系。")


class ScopeVerificationOutput(OutputModel):
    candidate_catalog_id: str = Field(min_length=1)
    decisions: dict[str, ScopeVerificationDecision]


class NarrowedAnnotation(OutputModel):
    fact_des: str = Field(min_length=1)
    argument: EvidenceArguments
    entities: list[str]


class AnnotationCorrection(OutputModel):
    model_config = ConfigDict(extra="forbid", strict=True, json_schema_extra={"allOf": [
        {"if": {"properties": {"status": {"const": "corrected"}}, "required": ["status"]},
         "then": {"properties": {"annotations": {"minItems": 1}}}},
        {"if": {"properties": {"status": {"const": "no_supported_annotation"}}, "required": ["status"]},
         "then": {"properties": {"annotations": {"maxItems": 0}}}},
    ]})
    status: Literal["corrected", "no_supported_annotation"]
    annotations: list[NarrowedAnnotation] = Field(description="仅重抽固定原句直接支持的目标断言及必要限定；同一事件连贯断言可保持一条。corrected非空，no_supported_annotation为空，不输出source/quote/坐标。")


class AnnotationNarrowingOutput(OutputModel):
    candidate_catalog_id: str = Field(min_length=1)
    corrections: dict[str, AnnotationCorrection]
