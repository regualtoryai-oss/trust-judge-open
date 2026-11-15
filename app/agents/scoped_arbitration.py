"""Strict keyed arbitration relations with an explicit model-owned claim target.

This codec does not decide truth or infer conflicts from retrieval/selection
labels. LangChain can use its returned Pydantic model as an ordinary strict
tool schema. The caller retains the real independent analyses and arbitration.
"""
from copy import deepcopy
from hashlib import sha256
import json
import re
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, create_model

from app.agents.schemas import ArbitrationOutput


VERSION = "principal-claim-evidence-relations-v1"
Relation = Literal["support_principal_claim", "oppose_principal_claim", "qualification", "unused"]
SUPPORT_ROLES = {"support_principal_claim", "qualification"}
CONFLICT_ROLE = "oppose_principal_claim"
REMOVED_FIELDS = {"supporting_evidence_ids", "conflicting_evidence_ids"}
TARGET = "model_principal_claim_answering_main_question_with_attribution_and_limits"

ISSUE_FRAME_VERSION = "issue-frame-evidence-relations-v2"
ISSUE_FRAME_TARGET = "model_issue_frame_answering_original_question_with_directional_qualifiers"
CODEBOOK_VERSION = "directional-issue-role-codes-v1"
IssueRelation = Literal["S", "O", "QS", "QO", "U"]
ClaimKind = Literal["fact", "official_finding", "causal", "normative", "attribution"]
ISSUE_CODEBOOK = {"S": "support_target_proposition", "O": "oppose_target_proposition",
    "QS": "qualify_support_target_proposition", "QO": "qualify_oppose_target_proposition",
    "U": "unused_for_target_proposition"}

CONCLUSION_FRAME_VERSION = "conclusion-aligned-evidence-relations-v3"
CONCLUSION_TARGET = "exact_final_conclusion_not_an_independent_proposition"
CONCLUSION_CODEBOOK_VERSION = "conclusion-relation-with-independent-alternative-side-v3"
CONCLUSION_CODEBOOK = {"S": "support_final_conclusion", "O": "oppose_final_conclusion",
    "QS": "qualify_support_final_conclusion", "QO": "qualify_oppose_final_conclusion", "U": "unused_for_final_conclusion"}
ConclusionAssessment = Literal[
    "S_fact", "S_attribution_only", "S_supporting_limit", "S_procedure",
    "QS_fact", "QS_attribution_only", "QS_supporting_limit", "QS_procedure",
    "O_direct_contradiction", "O_exclusive_alternative", "O_limiting_counterclaim", "O_attribution_contradiction",
    "QO_direct_contradiction", "QO_exclusive_alternative", "QO_limiting_counterclaim", "QO_attribution_contradiction",
    "U_fact", "U_attribution_only", "U_supporting_limit", "U_procedure", "U_background", "U_gap"]
# Closed, prevalidated combinations avoid a schema-valid role/basis pair being
# rejected only after the paid response. This is decoding, not a semantic vote.
CONCLUSION_ASSESSMENT_CODEBOOK = {code: dict(zip(("role", "basis"), code.split("_", 1)))
    for code in get_args(ConclusionAssessment)}
AlternativeRelation = Literal["exclusive", "compatible", "unknown", "not_binary"]
AlternativeSide = Literal["X", "Y", "both", "neither"]
ConclusionBasis = Literal["fact", "attribution_only", "supporting_limit", "procedure", "background", "gap",
    "direct_contradiction", "exclusive_alternative", "limiting_counterclaim", "attribution_contradiction"]
OPPOSITION_BASES = {"direct_contradiction", "exclusive_alternative", "limiting_counterclaim", "attribution_contradiction"}


class AnswerAlternatives(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    X: str = Field(description="从完整原题提炼的第一种实质说法，保留主体/时间/归属条件；非二选一题填空串。")
    Y: str = Field(description="第二种实质说法；不能仅因原题使用还是就假设与X互斥；非二选一题填空串。")


class ConclusionFrameOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    question_anchor: str = Field(min_length=1, description="逐字复制完整原题中的实质争议连续文本。")
    claim_kind: ClaimKind
    scope: str = Field(min_length=1, description="最终conclusion的主体、时点、统计口径、归属/责任层级与边界。")
    alternatives: AnswerAlternatives
    alternatives_relation: AlternativeRelation


class ConclusionEvidenceRole(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    assessment: ConclusionAssessment = Field(description="联合编码：下划线前为相对于最终conclusion的S/O/QS/QO/U，后为模型声明的依据；不以X/Y单方为目标。")
    side: AlternativeSide = Field(description="此证据涉及X/Y/both/neither，独立记录，绝不自动转换为支持或冲突。")


CONCLUSION_FRAME_INSTRUCTIONS = """本轮唯一关系目标是你最终输出的conclusion全文，不另写proposition或改变目标极性。conclusion先说清原题的实质判断和已有依据，再说明相反说法与未决边界；不得只写笼统不可判定，也不得为凑支持/冲突强行选边。decision=undetermined表示仍有未决部分，不把所有互相对立的事实都改标为支持“缺证”。
issue_frame只记录原题逐字锚点、最终结论的层级/范围、原题两种说法X/Y及其exclusive/compatible/unknown/not_binary关系；非二选一时X/Y空串。两种功能、责任、渠道或不同统计时点可并存，不能仅由“还是”推出互斥。
每个E输出一个短对象{assessment,side}，不重复事实。assessment从schema枚举选择“角色_依据”联合编码，如S_fact、QS_procedure、O_direct_contradiction、QO_limiting_counterclaim、U_background、U_gap。角色仅以最终conclusion为目标：S直接支持，O实质反对，QS顺向保留结论边界，QO给出使结论不能成立或需收窄的反向依据，U未用于结论。side独立标X/Y/both/neither，另一侧说法不自动等于反证。
联合编码中O/QO的依据只取direct_contradiction/exclusive_alternative/limiting_counterclaim/attribution_contradiction；exclusive_alternative只适用于已声明exclusive且同主体同时间同范围的真正互斥说法。background/gap只能U；单纯提问、报道中没有答案或未找到证据，不是证明相反事实。原文明确“尚未调查/认定”的程序记录可用procedure，不能偷换成原因不存在。
attribution_only只证明某人说过或提出过，不直接证明所说事实、因果或责任；只有最终结论本身在判断转述归属时才可直接S，其他情况保留归属限定而不跨层级。普通事实、归属、程序状态和范围限定分开判断，程序不替你判语义。
conflicts只记录同一可比较命题的真实不相容断言；不同层面的并列安排不因同题就成为冲突。不存在合格反证可全部无O/QO，也可没有conflicts；不要凑配额。保留judgments完整依据及前提链，保留真正相反意见和未决理由，引用仅本轮E，H1和检索stance不是最终关系。顶层支持/冲突列表不输出，程序仅固定解码assessment，再按S/QS、O/QO映射，不根据side或词语替你改判。跨层级或两方关系的可疑组合只作审计警告，不代表程序已验证语义。"""


class IssueFrameOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    question_anchor: str = Field(min_length=1, description="逐字复制原题中对应实质争议的连续文本，不改写或只写标点。")
    proposition: str = Field(min_length=1, description="模型选择的有主体及必要限定的实质目标论断；可肯定或否定，不以笼统资料不足替代。")
    claim_kind: ClaimKind = Field(description="区分事实、官方认定、因果、规范责任与转述归属，不把不同层级互换。")
    scope: str = Field(min_length=1, description="明确目标的主体、时点、统计口径、归属或责任层级等适用边界；未给出的限定不补成事实。")


ISSUE_FRAME_INSTRUCTIONS = """本次仲裁先给issue_frame。question_anchor逐字取原题中的连续实质文本；proposition由你选择，须回答该原题实质争议并保留必要限定，可肯定或否定，不以笼统“资料不足/不可判定”替代。claim_kind区分事实、官方认定、因果、规范责任和转述；scope说明目标主体、时点、统计范围、归属或责任层级。目标和关系均为模型判断，程序不替你认定语义正确。
decision/conclusion单独表达是否成立及未决程度；conclusion应自然说明同一实质目标及其相反说法和适用边界，不把目标只藏在内部字段。不要求强行选边，也不因undetermined把所有材料改成“支持缺证”。
relations对每个已给E恰好一个短码，不缺key、不新增key，唯一目标是issue_frame.proposition：S=直接支持，O=实质反对，QS=顺向限定（保留目标成立的边界），QO=反向限定（限定该目标的成立范围、归属或证据效力），U=未用于该目标。QS/QO的方向必须相对于该目标，不能把无方向的限定一律归支持，也不能为增加反证数量制造QO。顶层supporting_evidence_ids/conflicting_evidence_ids不输出，由程序分别按S/QS与O/QO映射。
未确认某原因不等于已证实该原因未发生；否认已有官方认定不等于否认事实因果。政府监管与企业责任可能不同层级并可并存，不自动互斥；不同时点、统计对象或范围的数字也不自动冲突。相对目标确无反向材料时，允许没有O/QO，不设支持或冲突数量配额。
H1及选证stance/use_type只是未验证线索，不能直接抄为最终关系。完整保留judgments的statement、evidence_ids、前提链及conflicts全部字段和真正未消解的两侧材料；只引用本轮E，不借A/B的判断地址冒充本轮judgment前提。不重写或缩短原文来适配方向。无可用证据允许undetermined、全部U、judgments/conflicts为空，并保留具体而有范围的目标及缺口；不得伪造来源、判断或冲突。"""

INSTRUCTIONS = """本次仲裁先声明principal_claim：直接回答原题主要争议、保留主体归属与时间/口径/核实程度的实质主张；不能只写“不可判定/证据不足”。conclusion应自然说明同一实质主张，再交代相反说法和未决边界，不把主张只藏在内部字段；保留不确定性，decision仍可为undetermined，不要求强行选边。
relations必须逐个填写每个已给E，不缺key、不新增key，关系目标仅为本次principal_claim：support_principal_claim=直接支持该主张；oppose_principal_claim=实质反对该主张；qualification=支持该主张成立边界的关键限定；unused=未用于该主张。限定不是反证，不把同主题背景或单纯缺证当反证。
顶层supporting_evidence_ids/conflicting_evidence_ids由程序按relations生成，本次工具不输出这两个列表；conflicts内部的双方证据字段仍须完整保留。
不要因为总判定为undetermined就把相互对立的说法都当作“支持不可判定”。明确存在两方对立主张时，应保留conflicts及各侧证据，并相对于principal_claim分别分配关系，逐项解释未决/采纳原因；没有合格反证时不能为了凑数制造oppose。
H1或选证阶段的stance只是待核线索，既不决定真伪，也不一定与principal_claim同向，不能直接复制成最终支持/冲突角色。程序仅按本次relations映射S/C，不替你改判。
保留原judgments、前提链和conflicts全部必需字段，引用只用已给E，判断必须落到证据或有效前提链。若确无可用证据，允许undetermined、全部unused、judgments/conflicts为空，并在principal_claim/conclusion清楚说明具体资料缺口；不得编造证据、字段或冲突。"""


class ScopedArbitrationError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _fail(code):
    raise ScopedArbitrationError(code)


def _aliases(evidence, *, wire=True):
    if not isinstance(evidence, (list, tuple)):
        _fail("scoped_evidence_alias_list_required")
    aliases = []
    for row in evidence:
        alias = row.get("id") if isinstance(row, dict) else row
        if (not isinstance(alias, str) or not alias or alias in aliases
                or (wire and re.fullmatch(r"E[1-9]\d*", alias) is None)):
            _fail("scoped_unknown_or_duplicate_evidence_alias")
        aliases.append(alias)
    return aliases


def prepare_scoped_arbitration(base_schema, payload_evidence_Ealiases, *, question=None, issue_frame=False,
                               conclusion_aligned=True):
    """Return an ordinary strict Pydantic schema and compact target instructions.

    Evidence is a list of E aliases or projected evidence dictionaries with an
    ``id`` E alias. No source text, selection stance or hypothesis is rewritten.
    All original nested fields keep their original Pydantic definitions.
    """
    if (not isinstance(base_schema, type) or not issubclass(base_schema, ArbitrationOutput)
            or not REMOVED_FIELDS.issubset(base_schema.model_fields)):
        _fail("scoped_arbitration_base_schema_required")
    if type(issue_frame) is not bool or type(conclusion_aligned) is not bool:
        _fail("scoped_issue_frame_boolean_required")
    if issue_frame and conclusion_aligned:
        return _prepare_conclusion_arbitration(base_schema, payload_evidence_Ealiases, question=question)
    aliases = _aliases(payload_evidence_Ealiases)
    question_text = _question_text(question) if issue_frame else None
    relation_fields = {alias: (IssueRelation if issue_frame else Relation,
                              Field(description="仅针对issue_frame.proposition的短码。" if issue_frame
                                    else "仅针对principal_claim的模型关系编码。"))
                       for alias in aliases}
    relations = create_model("ScopedEvidenceRelations", __config__=ConfigDict(extra="forbid", strict=True),
                             **relation_fields)
    fields = {name: (field.annotation, deepcopy(field)) for name, field in base_schema.model_fields.items()
              if name not in REMOVED_FIELDS}
    if issue_frame:
        fields.update(issue_frame=(IssueFrameOutput, Field(description="原题锚定的模型目标及边界，不是程序已核实事实。")),
            relations=(relations, Field(description="每个已给E恰好一个S/O/QS/QO/U，唯一目标是issue_frame.proposition。")))
    else:
        fields.update(principal_claim=(str, Field(min_length=1,
            description="回答原题主要争议的有归属/限定实质主张；不得仅写不可判定或证据不足。")),
            relations=(relations, Field(description="每个已给E恰好一个关系，目标仅为principal_claim。")))
    # Original conditional schema refers to the two removed wire arrays. Their
    # complete business checks run again after canonical S/C reconstruction.
    config = dict(base_schema.model_config)
    config.pop("json_schema_extra", None)
    config.update(extra="forbid", strict=True)
    schema = create_model("ScopedArbitrationWireOutput", __config__=ConfigDict(**config), **fields)
    schema.__scoped_base_schema__ = base_schema
    schema.__scoped_issue_frame__ = issue_frame
    schema.__scoped_conclusion_aligned__ = False
    if issue_frame:
        schema.__scoped_question_text__ = question_text
        schema.__scoped_question_sha256__ = sha256(question_text.encode("utf8")).hexdigest()
        schema.__scoped_evidence_order__ = tuple(aliases)
    return schema, ISSUE_FRAME_INSTRUCTIONS if issue_frame else INSTRUCTIONS


def _refs(refs, allowed):
    if (not isinstance(refs, list) or any(not isinstance(ref, str) for ref in refs)
            or len(refs) != len(set(refs)) or any(ref not in allowed for ref in refs)):
        _fail("scoped_invalid_or_duplicate_evidence_reference")


def _principal_claim(claim):
    if not isinstance(claim, str):
        _fail("scoped_substantive_principal_claim_required")
    bare = re.sub(r"[\s，。！？；：,.!?;:]", "", claim).lower()
    if not bare or bare in {"不可判定", "无法判定", "暂不可判定", "公开信息层面不可判定", "证据不足", "无法确定", "不能确定", "undetermined", "unknown"}:
        _fail("scoped_substantive_principal_claim_required")
    return claim


def _digest(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf8")).hexdigest()


def _question_text(question):
    text = question.get("question") if isinstance(question, dict) else question
    if not isinstance(text, str) or not text.strip():
        _fail("scoped_original_question_required")
    return text


def _issue_frame(value, question=None):
    frame = IssueFrameOutput.model_validate(value).model_dump()
    for field in ("question_anchor", "proposition", "scope"):
        if not any(char.isalnum() for char in frame[field]):
            _fail("scoped_issue_frame_substantive_text_required")
    _principal_claim(frame["proposition"])
    if question is not None and frame["question_anchor"] not in _question_text(question):
        _fail("scoped_question_anchor_not_literal")
    return frame


def validate_scoped_result(scope_audit, base_output, evidence_aliases, *, question=None):
    """Replay scope against final S/C, before publish or export, without repair.

    Unlike the wire codec, this accepts already-remapped real evidence IDs.
    The final workflow may also have remapped judgment IDs; those unrelated
    service identifiers are not rewritten or interpreted here.
    V2 requires the trusted original question; a self-declared digest alone
    cannot establish its anchor. Final evidence row order may differ from the
    separately bound model ledger order.
    """
    aliases = _aliases(evidence_aliases, wire=False)
    if isinstance(scope_audit, dict) and scope_audit.get("format") == ISSUE_FRAME_VERSION:
        return _validate_issue_result(scope_audit, base_output, aliases, question=question)
    if isinstance(scope_audit, dict) and scope_audit.get("format") == CONCLUSION_FRAME_VERSION:
        return _validate_conclusion_result(scope_audit, base_output, aliases, question=question)
    if not isinstance(scope_audit, dict) or scope_audit.get("format") != VERSION:
        _fail("scoped_relation_scope_version_required")
    _principal_claim(scope_audit.get("principal_claim"))
    if scope_audit.get("target") != TARGET:
        _fail("scoped_principal_claim_target_required")
    roles = scope_audit.get("evidence_roles")
    if (not isinstance(roles, dict) or set(roles) != set(aliases)
            or any(not isinstance(role, str) or role not in get_args(Relation) for role in roles.values())):
        _fail("scoped_complete_explicit_evidence_roles_required")
    if not isinstance(base_output, dict):
        _fail("scoped_final_output_required")
    allowed = set(aliases)
    support = base_output.get("supporting_evidence_ids")
    conflict = base_output.get("conflicting_evidence_ids")
    _refs(support, allowed); _refs(conflict, allowed)
    if set(support) & set(conflict):
        _fail("scoped_overlapping_evidence_relations")
    if base_output.get("decision") == "asserted" and not support:
        _fail("scoped_assertion_without_support")
    expected_support = {ref for ref in aliases if roles[ref] in SUPPORT_ROLES}
    expected_conflict = {ref for ref in aliases if roles[ref] == CONFLICT_ROLE}
    if set(support) != expected_support or set(conflict) != expected_conflict:
        _fail("scoped_final_relations_do_not_match_model_roles")
    for key, role in (("qualification_evidence_ids", "qualification"),
                      ("unused_for_principal_claim_evidence_ids", "unused")):
        refs = scope_audit.get(key)
        _refs(refs, allowed)
        if set(refs) != {ref for ref in aliases if roles[ref] == role}:
            _fail("scoped_audit_role_inventory_mismatch")
    return deepcopy(scope_audit)


def _validate_issue_result(scope, base_output, aliases, *, question=None):
    text = _question_text(question)
    required = {"format", "principal_claim", "target", "issue_frame", "issue_frame_sha256", "question_sha256",
        "codebook_version", "codebook", "evidence_order", "evidence_order_sha256", "evidence_roles", "support_role_codes", "conflict_role_codes",
        "qualification_evidence_ids", "qualification_support_evidence_ids", "qualification_oppose_evidence_ids",
        "unused_for_principal_claim_evidence_ids", "h1_role", "selection_stances_used_for_mapping", "role_origin",
        "semantic_target_verified", "missing_fields_fabricated"}
    if set(scope) != required or scope.get("target") != ISSUE_FRAME_TARGET:
        _fail("scoped_issue_frame_scope_contract")
    frame = _issue_frame(scope.get("issue_frame"), question=text)
    if scope["principal_claim"] != frame["proposition"] or scope["issue_frame_sha256"] != _digest(frame):
        _fail("scoped_issue_frame_binding_changed")
    qsha = scope.get("question_sha256")
    if (not isinstance(qsha, str) or re.fullmatch(r"[0-9a-f]{64}", qsha) is None
            or qsha != sha256(text.encode("utf8")).hexdigest()):
        _fail("scoped_question_binding_changed")
    if (scope.get("codebook_version") != CODEBOOK_VERSION or scope.get("codebook") != ISSUE_CODEBOOK
            or scope.get("support_role_codes") != ["S", "QS"] or scope.get("conflict_role_codes") != ["O", "QO"]):
        _fail("scoped_issue_codebook_changed")
    order = scope.get("evidence_order")
    _refs(order, set(aliases))
    if set(order) != set(aliases) or scope.get("evidence_order_sha256") != _digest(order):
        _fail("scoped_issue_evidence_order_changed")
    roles = scope.get("evidence_roles")
    if (not isinstance(roles, dict) or set(roles) != set(aliases)
            or any(not isinstance(role, str) or role not in ISSUE_CODEBOOK for role in roles.values())):
        _fail("scoped_complete_explicit_evidence_roles_required")
    if (scope.get("h1_role") != "provisional_search_hypothesis_not_truth_or_final_relation"
            or scope.get("selection_stances_used_for_mapping") is not False
            or scope.get("role_origin") != "model_explicit_issue_frame_relations"
            or scope.get("semantic_target_verified") is not False or scope.get("missing_fields_fabricated") is not False):
        _fail("scoped_issue_role_origin_changed")
    if not isinstance(base_output, dict):
        _fail("scoped_final_output_required")
    # Check the unmodified base structure/DAG as well as the generated S/C.
    core = ArbitrationOutput.model_validate({key: base_output[key] for key in ArbitrationOutput.model_fields
        if key in base_output}).model_dump()
    _check_grounding(core, aliases)
    expected_support = {ref for ref in aliases if roles[ref] in {"S", "QS"}}
    expected_conflict = {ref for ref in aliases if roles[ref] in {"O", "QO"}}
    if set(core["supporting_evidence_ids"]) != expected_support or set(core["conflicting_evidence_ids"]) != expected_conflict:
        _fail("scoped_final_relations_do_not_match_model_roles")
    inventories = {"qualification_evidence_ids": {"QS", "QO"}, "qualification_support_evidence_ids": {"QS"},
        "qualification_oppose_evidence_ids": {"QO"}, "unused_for_principal_claim_evidence_ids": {"U"}}
    for key, codes in inventories.items():
        refs = scope.get(key)
        _refs(refs, set(aliases))
        if set(refs) != {ref for ref in aliases if roles[ref] in codes}:
            _fail("scoped_audit_role_inventory_mismatch")
    return deepcopy(scope)


def _check_grounding(output, aliases):
    allowed = set(aliases)
    support, conflict = output["supporting_evidence_ids"], output["conflicting_evidence_ids"]
    _refs(support, allowed); _refs(conflict, allowed)
    if set(support) & set(conflict):
        _fail("scoped_overlapping_evidence_relations")
    if output["decision"] == "asserted" and not support:
        _fail("scoped_assertion_without_support")
    judgments = output["judgments"]
    local_ids = [row["id"] for row in judgments]
    if len(local_ids) != len(set(local_ids)):
        _fail("scoped_duplicate_judgment_id")
    dependencies = {}
    for row in judgments:
        _refs(row["evidence_ids"], allowed)
        prerequisites = row["prerequisite_judgment_ids"]
        if not row["evidence_ids"] and not prerequisites:
            _fail("scoped_ungrounded_judgment")
        if len(prerequisites) != len(set(prerequisites)) or any(ref not in local_ids for ref in prerequisites):
            _fail("scoped_unknown_or_duplicate_prerequisite")
        dependencies[row["id"]] = prerequisites
    visiting, visited = set(), set()
    def visit(node):
        if node in visiting:
            _fail("scoped_prerequisite_cycle")
        if node in visited:
            return
        visiting.add(node)
        for previous in dependencies[node]:
            visit(previous)
        visiting.remove(node); visited.add(node)
    for node in dependencies:
        visit(node)
    conflict_ids = [row["id"] for row in output["conflicts"]]
    if len(conflict_ids) != len(set(conflict_ids)):
        _fail("scoped_duplicate_conflict_id")
    for row in output["conflicts"]:
        left, right = row["supporting_evidence_ids"], row["conflicting_evidence_ids"]
        _refs(left, allowed); _refs(right, allowed)
        if set(left) & set(right):
            _fail("scoped_overlapping_conflict_sides")
        if row["status"] == "resolved" and bool(set(support) & set(left)) == bool(set(support) & set(right)):
            _fail("scoped_ambiguous_conflict_resolution")


def canonical_scoped_arbitration(value, evidence_aliases, *, question=None, issue_frame=False,
                                 conclusion_aligned=True):
    """Decode model roles relative to its claim and validate the original output.

    Plain wire dictionaries use production ArbitrationOutput. A typed dynamic
    wire instance retains the base schema supplied to ``prepare_*``. No missing
    field, reference, judgment or conflict is fabricated or silently repaired.
    """
    aliases = _aliases(evidence_aliases)
    if type(issue_frame) is not bool or type(conclusion_aligned) is not bool:
        _fail("scoped_issue_frame_boolean_required")
    bound_mode = getattr(type(value), "__scoped_issue_frame__", None)
    if bound_mode is False and issue_frame:
        _fail("scoped_typed_protocol_binding_changed")
    use_issue_frame = issue_frame or bound_mode is True
    if use_issue_frame:
        alignment = getattr(type(value), "__scoped_conclusion_aligned__", conclusion_aligned)
        if alignment:
            return _canonical_conclusion_arbitration(value, aliases, question=question)
        return _canonical_issue_arbitration(value, aliases, question=question)
    base_schema = getattr(type(value), "__scoped_base_schema__", ArbitrationOutput)
    schema, _ = prepare_scoped_arbitration(base_schema, aliases)
    raw = value.model_dump(by_alias=True) if isinstance(value, BaseModel) else deepcopy(value)
    wire = schema.model_validate(raw).model_dump(by_alias=True)
    claim = _principal_claim(wire["principal_claim"])
    relations = wire["relations"]
    canonical = {name: item for name, item in wire.items() if name not in {"principal_claim", "relations"}}
    canonical["supporting_evidence_ids"] = [alias for alias in aliases if relations[alias] in SUPPORT_ROLES]
    canonical["conflicting_evidence_ids"] = [alias for alias in aliases if relations[alias] == CONFLICT_ROLE]
    output = base_schema.model_validate(canonical).model_dump(by_alias=True)
    _check_grounding(output, aliases)
    audit = {"format": VERSION, "principal_claim": claim,
        "target": TARGET,
        "evidence_roles": {alias: relations[alias] for alias in aliases},
        "support_role_codes": ["support_principal_claim", "qualification"],
        "conflict_role_codes": [CONFLICT_ROLE],
        "qualification_evidence_ids": [alias for alias in aliases if relations[alias] == "qualification"],
        "unused_for_principal_claim_evidence_ids": [alias for alias in aliases if relations[alias] == "unused"],
        "h1_role": "provisional_search_hypothesis_not_truth_or_final_relation",
        "selection_stances_used_for_mapping": False, "role_origin": "model_explicit_scoped_relations",
        "semantic_target_verified": False, "missing_fields_fabricated": False}
    return output, validate_scoped_result(audit, output, aliases)


def _canonical_issue_arbitration(value, aliases, *, question):
    base_schema = getattr(type(value), "__scoped_base_schema__", ArbitrationOutput)
    bound_question = getattr(type(value), "__scoped_question_text__", None)
    if bound_question is not None:
        if question is not None and _question_text(question) != bound_question:
            _fail("scoped_typed_question_binding_changed")
        if tuple(aliases) != getattr(type(value), "__scoped_evidence_order__", ()):
            _fail("scoped_typed_evidence_order_changed")
        if sha256(bound_question.encode("utf8")).hexdigest() != getattr(type(value), "__scoped_question_sha256__", None):
            _fail("scoped_typed_question_binding_changed")
        question = bound_question
    text = _question_text(question)
    schema, _ = prepare_scoped_arbitration(base_schema, aliases, question=text, issue_frame=True, conclusion_aligned=False)
    raw = value.model_dump(by_alias=True) if isinstance(value, BaseModel) else deepcopy(value)
    wire = schema.model_validate(raw).model_dump(by_alias=True)
    frame = _issue_frame(wire["issue_frame"], question=text)
    relations = wire["relations"]
    canonical = {name: item for name, item in wire.items() if name not in {"issue_frame", "relations"}}
    canonical["supporting_evidence_ids"] = [alias for alias in aliases if relations[alias] in {"S", "QS"}]
    canonical["conflicting_evidence_ids"] = [alias for alias in aliases if relations[alias] in {"O", "QO"}]
    output = base_schema.model_validate(canonical).model_dump(by_alias=True)
    _check_grounding(output, aliases)
    audit = {"format": ISSUE_FRAME_VERSION, "principal_claim": frame["proposition"], "target": ISSUE_FRAME_TARGET,
        "issue_frame": frame, "issue_frame_sha256": _digest(frame), "question_sha256": sha256(text.encode("utf8")).hexdigest(),
        "codebook_version": CODEBOOK_VERSION, "codebook": deepcopy(ISSUE_CODEBOOK), "evidence_order": list(aliases),
        "evidence_order_sha256": _digest(list(aliases)),
        "evidence_roles": {alias: relations[alias] for alias in aliases}, "support_role_codes": ["S", "QS"],
        "conflict_role_codes": ["O", "QO"], "qualification_evidence_ids": [a for a in aliases if relations[a] in {"QS", "QO"}],
        "qualification_support_evidence_ids": [a for a in aliases if relations[a] == "QS"],
        "qualification_oppose_evidence_ids": [a for a in aliases if relations[a] == "QO"],
        "unused_for_principal_claim_evidence_ids": [a for a in aliases if relations[a] == "U"],
        "h1_role": "provisional_search_hypothesis_not_truth_or_final_relation", "selection_stances_used_for_mapping": False,
        "role_origin": "model_explicit_issue_frame_relations", "semantic_target_verified": False, "missing_fields_fabricated": False}
    return output, validate_scoped_result(audit, output, aliases, question=text)


def _prepare_conclusion_arbitration(base_schema, evidence, *, question):
    aliases, text = _aliases(evidence), _question_text(question)
    relations = create_model("ConclusionEvidenceRelationsV3", __config__=ConfigDict(extra="forbid", strict=True),
        **{alias: (ConclusionEvidenceRole, ...) for alias in aliases})
    fields = {name: (field.annotation, deepcopy(field)) for name, field in base_schema.model_fields.items()
              if name not in REMOVED_FIELDS}
    fields.update(issue_frame=(ConclusionFrameOutput, Field(description="只约束原题、结论层级和两方说法；唯一目标来自最终conclusion，不另写proposition。")),
        relations=(relations, Field(description="每个E恰好一个assessment/side短对象，联合编码以最终conclusion为唯一目标。")))
    config = dict(base_schema.model_config)
    config.pop("json_schema_extra", None)
    config.update(extra="forbid", strict=True)
    schema = create_model("ScopedArbitrationWireOutput", __config__=ConfigDict(**config), **fields)
    schema.__scoped_base_schema__ = base_schema
    schema.__scoped_issue_frame__ = True
    schema.__scoped_conclusion_aligned__ = True
    schema.__scoped_question_text__ = text
    schema.__scoped_question_sha256__ = sha256(text.encode("utf8")).hexdigest()
    schema.__scoped_evidence_order__ = tuple(aliases)
    return schema, CONCLUSION_FRAME_INSTRUCTIONS


def _conclusion_frame(value, question):
    frame = ConclusionFrameOutput.model_validate(value).model_dump()
    for field in ("question_anchor", "scope"):
        if not any(char.isalnum() for char in frame[field]):
            _fail("scoped_conclusion_frame_substantive_text_required")
    if frame["question_anchor"] not in _question_text(question):
        _fail("scoped_question_anchor_not_literal")
    return frame


def _conclusion_frame_warnings(frame):
    warnings = []
    x, y = frame["alternatives"]["X"], frame["alternatives"]["Y"]
    if frame["alternatives_relation"] == "not_binary":
        if x or y:
            warnings.append("nonbinary_alternatives_not_empty")
    elif not x.strip() or not y.strip() or x == y:
        warnings.append("binary_alternatives_not_distinct_and_substantive")
    return warnings


def _conclusion_assessments(value, aliases, frame, *, expanded=False):
    if not isinstance(value, dict) or set(value) != set(aliases):
        _fail("scoped_complete_explicit_evidence_roles_required")
    result = {}
    for alias in aliases:
        source = value[alias]
        if expanded:
            if not isinstance(source, dict):
                _fail("scoped_conclusion_expanded_assessment_required")
            source = {"assessment": source.get("assessment"), "side": source.get("side")}
        wire = ConclusionEvidenceRole.model_validate(source).model_dump()
        row = {**wire, **CONCLUSION_ASSESSMENT_CODEBOOK[wire["assessment"]]}
        role, basis, side = row["role"], row["basis"], row["side"]
        warnings = []
        if basis == "exclusive_alternative" and (frame["alternatives_relation"] != "exclusive" or side not in {"X", "Y"}):
            warnings.append("exclusive_alternative_not_established_by_frame_and_side")
        if frame["alternatives_relation"] == "not_binary" and side != "neither":
            warnings.append("nonbinary_alternative_side_not_neither")
        if basis == "attribution_only" and role == "S" and frame["claim_kind"] != "attribution":
            warnings.append("attribution_only_direct_support_of_other_claim_layer")
        row["semantic_warnings"] = warnings
        if expanded and value[alias] != row:
            _fail("scoped_conclusion_expanded_assessment_changed")
        result[alias] = row
    return result


def _conclusion_audit(output, frame, assessments, aliases, question):
    roles = {alias: assessments[alias]["role"] for alias in aliases}
    return {"format": CONCLUSION_FRAME_VERSION, "principal_claim": output["conclusion"],
        "target": CONCLUSION_TARGET, "conclusion_sha256": _digest(output["conclusion"]),
        "target_binding_verified": True, "issue_frame": deepcopy(frame), "issue_frame_sha256": _digest(frame),
        "issue_frame_semantic_warnings": _conclusion_frame_warnings(frame),
        "question_sha256": sha256(_question_text(question).encode("utf8")).hexdigest(),
        "codebook_version": CONCLUSION_CODEBOOK_VERSION, "codebook": deepcopy(CONCLUSION_CODEBOOK),
        "assessment_codebook": deepcopy(CONCLUSION_ASSESSMENT_CODEBOOK),
        "evidence_order": list(aliases), "evidence_order_sha256": _digest(list(aliases)),
        "evidence_roles": roles, "evidence_assessments": deepcopy(assessments),
        "support_role_codes": ["S", "QS"], "conflict_role_codes": ["O", "QO"],
        "qualification_evidence_ids": [alias for alias in aliases if roles[alias] in {"QS", "QO"}],
        "qualification_support_evidence_ids": [alias for alias in aliases if roles[alias] == "QS"],
        "qualification_oppose_evidence_ids": [alias for alias in aliases if roles[alias] == "QO"],
        "unused_for_principal_claim_evidence_ids": [alias for alias in aliases if roles[alias] == "U"],
        "h1_role": "provisional_search_hypothesis_not_truth_or_final_relation",
        "selection_stances_used_for_mapping": False, "alternative_side_used_for_mapping": False,
        "role_origin": "model_explicit_conclusion_aligned_relations", "semantic_target_verified": False,
        "missing_fields_fabricated": False}


def _validate_conclusion_result(scope, base_output, aliases, *, question):
    if not isinstance(base_output, dict):
        _fail("scoped_final_output_required")
    core = ArbitrationOutput.model_validate({key: base_output[key] for key in ArbitrationOutput.model_fields
        if key in base_output}).model_dump()
    _principal_claim(core["conclusion"])
    _check_grounding(core, aliases)
    frame = _conclusion_frame(scope.get("issue_frame"), question)
    order = scope.get("evidence_order")
    _refs(order, set(aliases))
    if set(order) != set(aliases):
        _fail("scoped_issue_evidence_order_changed")
    assessments = _conclusion_assessments(scope.get("evidence_assessments"), order, frame, expanded=True)
    expected = _conclusion_audit(core, frame, assessments, order, question)
    if scope != expected:
        _fail("scoped_final_conclusion_relation_binding_changed")
    roles = expected["evidence_roles"]
    if (set(core["supporting_evidence_ids"]) != {ref for ref in aliases if roles[ref] in {"S", "QS"}}
            or set(core["conflicting_evidence_ids"]) != {ref for ref in aliases if roles[ref] in {"O", "QO"}}):
        _fail("scoped_final_relations_do_not_match_model_roles")
    return deepcopy(scope)


def _canonical_conclusion_arbitration(value, aliases, *, question):
    base_schema = getattr(type(value), "__scoped_base_schema__", ArbitrationOutput)
    bound_question = getattr(type(value), "__scoped_question_text__", None)
    if bound_question is not None:
        if (question is not None and _question_text(question) != bound_question
                or tuple(aliases) != getattr(type(value), "__scoped_evidence_order__", ())
                or sha256(bound_question.encode("utf8")).hexdigest() != getattr(type(value), "__scoped_question_sha256__", None)):
            _fail("scoped_typed_conclusion_protocol_binding_changed")
        question = bound_question
    text = _question_text(question)
    schema, _ = _prepare_conclusion_arbitration(base_schema, aliases, question=text)
    raw = value.model_dump(by_alias=True) if isinstance(value, BaseModel) else deepcopy(value)
    wire = schema.model_validate(raw).model_dump(by_alias=True)
    _principal_claim(wire["conclusion"])
    frame = _conclusion_frame(wire["issue_frame"], text)
    assessments = _conclusion_assessments(wire["relations"], aliases, frame)
    canonical = {name: item for name, item in wire.items() if name not in {"issue_frame", "relations"}}
    canonical["supporting_evidence_ids"] = [alias for alias in aliases if assessments[alias]["role"] in {"S", "QS"}]
    canonical["conflicting_evidence_ids"] = [alias for alias in aliases if assessments[alias]["role"] in {"O", "QO"}]
    output = base_schema.model_validate(canonical).model_dump(by_alias=True)
    _check_grounding(output, aliases)
    audit = _conclusion_audit(output, frame, assessments, aliases, text)
    return output, validate_scoped_result(audit, output, aliases, question=text)
