"""Pure paragraph portfolio planning; no retrieval, model calls, or gold access.

The output is a *selection plan*, never a published fact or truth probability.
Question aspects and contribution labels must come from the runtime question /
rank response; literal anchors bind labels to source text but do not establish
entailment. Provenance groups diversify context, and never create source votes.
"""
from dataclasses import dataclass
from math import isfinite
from typing import Literal


Role = Literal["direct_answer", "refutation", "premise", "qualifier", "context",
               "official_record", "limiting_condition", "competing_claim", "background"]
Stance = Literal["supports_X", "supports_Y", "official", "qualifier", "neutral"]
ROLES = {"direct_answer": 3.0, "refutation": 3.0, "premise": 1.5,
         "qualifier": 2.5, "context": 0.0, "official_record": 3.0,
         "limiting_condition": 2.5, "competing_claim": 2.0, "background": 0.0}
STANCES = {"supports_X", "supports_Y", "official", "qualifier", "neutral"}


@dataclass(frozen=True)
class Contribution:
    aspect_id: str
    role: Role
    stance: Stance
    anchor: str
    # A bounded model label is a redundancy hint, not proven fact equivalence.
    claim_key: str = ""


@dataclass(frozen=True)
class PortfolioCandidate:
    paragraph_ref: str
    doc_id: str
    para_id: str
    source_text: str
    score: int
    contributions: tuple[Contribution, ...]
    source_credibility: float | None = None
    source_group: str | None = None


@dataclass(frozen=True)
class PortfolioPolicy:
    # These are operational safety bounds, not expected answer counts.
    maximum_paragraphs: int = 30
    maximum_quote_characters: int = 16000
    corroborating_groups_per_claim: int = 2
    preserve_exact_mirror_locations: bool = True


def _nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def _validate(candidates, aspect_ids, policy):
    if (not isinstance(policy, PortfolioPolicy)
            or type(policy.preserve_exact_mirror_locations) is not bool
            or any(type(value) is not int or value < 1 for value in (
                policy.maximum_paragraphs, policy.maximum_quote_characters,
                policy.corroborating_groups_per_claim))):
        raise ValueError("portfolio_policy_invalid")
    if (not aspect_ids or any(not _nonempty(a) for a in aspect_ids)
            or len(set(aspect_ids)) != len(aspect_ids)):
        raise ValueError("portfolio_aspects_invalid")
    refs, keys = set(), set()
    for item in candidates:
        if not isinstance(item, PortfolioCandidate):
            raise ValueError("portfolio_candidate_invalid")
        if (any(not _nonempty(v) for v in (item.paragraph_ref, item.doc_id,
                                          item.para_id, item.source_text))
                or item.paragraph_ref in refs or (item.doc_id, item.para_id) in keys
                or type(item.score) is not int or not 0 <= item.score <= 3
                or not isinstance(item.contributions, tuple)
                or (item.source_group is not None and not _nonempty(item.source_group))):
            raise ValueError("portfolio_candidate_invalid")
        credibility = item.source_credibility
        if credibility is not None and (type(credibility) not in (int, float)
                or not isfinite(credibility) or not 0 <= credibility <= 1):
            raise ValueError("portfolio_source_credibility_invalid")
        refs.add(item.paragraph_ref); keys.add((item.doc_id, item.para_id))
        for contribution in item.contributions:
            if (not isinstance(contribution, Contribution)
                    or contribution.aspect_id not in aspect_ids
                    or contribution.role not in ROLES or contribution.stance not in STANCES
                    or not _nonempty(contribution.anchor)
                    or contribution.anchor not in item.source_text
                    or not isinstance(contribution.claim_key, str)
                    or len(contribution.claim_key) > 80):
                raise ValueError("portfolio_contribution_not_source_bound")


def _contributions(item):
    # Repeated model labels / anchors for one claim cannot multiply utility.
    return {(c.aspect_id, c.role, c.stance, c.claim_key) for c in item.contributions if ROLES[c.role]}


def _quality(item):
    # Credibility is an original-source ordering prior, not factual correctness.
    credibility = .5 if item.source_credibility is None else item.source_credibility
    return (.7 + .3 * credibility) * (.85 if item.score == 2 else 1.0)


def plan_portfolio(candidates, *, aspect_ids, policy=None):
    """Cover question-aspect / role / stance / claim units before repeating one claim.

    Every unit admits its strongest source first, then bounded corroboration from
    another provenance group. Extra same-group paragraphs still qualify when
    they add a different unit. Nothing is removed from the candidate catalogue;
    each unselected paragraph gets a transparent deferred / rejection reason.
    Character budgets are Unicode codepoints, explicitly *not token estimates*.
    """
    candidates, aspect_ids = tuple(candidates), tuple(aspect_ids)
    policy = PortfolioPolicy() if policy is None else policy
    _validate(candidates, aspect_ids, policy)
    eligible = [c for c in candidates if c.score >= 2 and _contributions(c)]
    chosen, coverage, used_chars, trace = [], {}, 0, []
    remaining = {c.paragraph_ref: c for c in eligible}

    def marginal(item):
        gain = 0.0; new_units, corroborated = [], []
        group = item.source_group or "unproven-document:" + item.doc_id
        for unit in sorted(_contributions(item)):
            groups = coverage.get(unit, {})
            old = sorted(groups.values(), reverse=True)[:policy.corroborating_groups_per_claim]
            updated = dict(groups); updated[group] = max(updated.get(group, 0), _quality(item))
            new = sorted(updated.values(), reverse=True)[:policy.corroborating_groups_per_claim]
            # First representation has full weight; further source groups offer
            # corroboration context at diminishing weight, never extra votes.
            weights = [1.0] + [.2 / i for i in range(1, policy.corroborating_groups_per_claim)]
            difference = sum(v * w for v, w in zip(new, weights)) - sum(v * w for v, w in zip(old, weights))
            if difference > 1e-12:
                gain += ROLES[unit[1]] * difference
                (new_units if not groups else corroborated).append(list(unit))
        return gain, new_units, corroborated

    while remaining and len(chosen) < policy.maximum_paragraphs:
        feasible = [c for c in remaining.values()
                    if used_chars + len(c.source_text) <= policy.maximum_quote_characters]
        if not feasible:
            break
        # Full marginal coverage first; source quality settles comparable units.
        # Last tie is stable identity, never the previous retrieval order.
        item = min(feasible, key=lambda c: (-marginal(c)[0], -_quality(c),
                                           len(c.source_text), c.doc_id, c.para_id, c.paragraph_ref))
        gain, new_units, corroborated = marginal(item)
        if gain <= 1e-12:
            break
        group = item.source_group or "unproven-document:" + item.doc_id
        for unit in _contributions(item):
            groups = coverage.setdefault(unit, {})
            groups[group] = max(groups.get(group, 0), _quality(item))
        chosen.append(item.paragraph_ref); used_chars += len(item.source_text)
        remaining.pop(item.paragraph_ref)
        trace.append({"paragraph_ref": item.paragraph_ref, "marginal_coverage_gain": round(gain, 8),
                      "new_coverage_units": new_units, "additional_source_context": corroborated})

    core_refs = list(chosen)
    # A compact reasoning context must not silently delete already-ranked exact
    # mirror source locations from the evidence plan. No fuzzy fact equivalence
    # is inferred, and source-group voting remains wholly outside this module.
    if policy.preserve_exact_mirror_locations:
        originals = [item for item in eligible if item.paragraph_ref in core_refs]
        for item in sorted(remaining.values(), key=lambda c: (c.doc_id, c.para_id, c.paragraph_ref)):
            if (len(chosen) >= policy.maximum_paragraphs
                    or used_chars + len(item.source_text) > policy.maximum_quote_characters):
                continue
            if any(item.source_text == original.source_text
                   and _contributions(item) == _contributions(original) for original in originals):
                chosen.append(item.paragraph_ref); used_chars += len(item.source_text)
                trace.append({"paragraph_ref": item.paragraph_ref, "marginal_coverage_gain": 0.0,
                    "new_coverage_units": [], "additional_source_context": [],
                    "reason": "preserve_rank_qualified_exact_mirror_source_location"})
    decisions = []
    for item in candidates:
        gain, units, _ = marginal(item)
        if item.paragraph_ref in chosen:
            reason = "selected_for_question_coverage"
        elif item.score < 2:
            reason = "rank_below_evidence_threshold"
        elif not _contributions(item):
            reason = "no_nonbackground_question_contribution"
        elif gain <= 1e-12:
            reason = "already_covered_claim_context_retained_as_candidate"
        elif used_chars + len(item.source_text) > policy.maximum_quote_characters:
            reason = "deferred_source_character_budget"
        else:
            reason = "deferred_operational_paragraph_bound"
        decisions.append({"paragraph_ref": item.paragraph_ref, "reason": reason,
                          "remaining_gain": round(gain, 8), "uncovered_units": units})
    represented = {unit[0] for unit in coverage}
    return {"version": "question-evidence-portfolio-v1", "selected_refs": chosen,
            "analysis_core_refs": core_refs,
            "selected_quote_characters": used_chars, "character_count_is_not_tokens": True,
            "represented_aspect_ids": sorted(represented),
            "unrepresented_aspect_ids": sorted(set(aspect_ids) - represented),
            "selection_trace": trace, "decisions": decisions,
            "semantic_coverage_verified": False, "independent_source_votes_created": 0,
            "publication_or_formal_evidence_created": False}


def legacy_contributions(*, source_text, score, stance, critical_qualifier=False,
                         aspect_id="whole_question"):
    """Explicitly coarse adapter for old closed 0--3 rank outputs.

    It does not infer missing question subparts or semantic labels from reasons.
    Its results are useful for offline selector comparisons only; rich runtime
    contribution labels are needed to avoid collapsing distinct same-role facts.
    """
    if (not _nonempty(source_text) or type(score) is not int or not 0 <= score <= 3
            or stance not in STANCES or type(critical_qualifier) is not bool):
        raise ValueError("portfolio_legacy_rank_invalid")
    role = "context" if score < 2 else "qualifier" if critical_qualifier else "direct_answer" if score == 3 else "premise"
    return (Contribution(aspect_id, role, stance, source_text),)
