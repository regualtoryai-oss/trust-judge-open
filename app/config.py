from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRUST_", env_file=PROJECT_ROOT / ".env", extra="ignore")

    data_dir: Path = PROJECT_ROOT / ".runtime"
    model_name: str = ""
    model_provider: Literal["openai_compatible", "deepseek"] = "openai_compatible"
    model_api_key: SecretStr = SecretStr("")
    model_base_url: str = "https://api.openai.com/v1"
    model_timeout_seconds: float = 90
    model_max_retries: int = 1
    model_max_output_tokens: int = Field(default=8192, ge=1, le=393216)
    model_extraction_max_output_tokens: int = Field(default=32768, ge=1, le=393216)
    model_analysis_max_output_tokens: int = Field(default=32768, ge=1, le=393216)
    model_arbitration_max_output_tokens: int = Field(default=65536, ge=1, le=393216)
    model_capacity_max_output_tokens: int = Field(default=131072, ge=1, le=393216)
    model_capacity_retries: int = Field(default=2, ge=0, le=3)
    model_generation_timeout_seconds: float = Field(default=600, gt=0, allow_inf_nan=False)
    model_thinking: Literal["enabled", "disabled"] | None = None
    model_reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] | None = None
    # Experimental v18 did not improve the fixed pilot's precision/recall.
    # Keep the v17 paragraph pipeline unless explicitly enabled.
    model_premise_bank_enabled: bool = False
    model_paragraph_ranked_enabled: bool = False
    # Experimental workbench front half. Default keeps the established pipeline.
    model_strict_evidence_enabled: bool = False
    model_source_bundle_enabled: bool = False
    source_bundle_front_framework: Literal["dsh", "langchain"] = "dsh"
    source_bundle_compact_handles: bool = False
    source_bundle_local_rank_enabled: bool = False
    source_bundle_selection_only: bool = False
    source_bundle_graded_selection: bool = False
    source_bundle_compact_analyses_enabled: bool = False
    source_bundle_issue_frame_enabled: bool = False
    source_bundle_context_passages_enabled: bool = False
    source_bundle_complete_grading_enabled: bool = False
    source_bundle_hybrid_retrieval_enabled: bool = False
    source_bundle_paragraph_quality_enabled: bool = False
    source_bundle_paragraph_quality_compact_enabled: bool = False
    source_bundle_parent_token_pack_enabled: bool = False
    source_bundle_recover_front_run_id: str | None = None
    source_bundle_recover_front_response: Path | None = None
    source_bundle_recover_front_response_sha256: str | None = None
    source_bundle_local_rank_manifest: Path | None = None
    source_bundle_local_rank_manifest_sha256: str | None = None
    source_bundle_front_token_stop: int = Field(default=12000, ge=1, le=40000)
    source_bundle_tail_reserved_tokens: int = Field(default=18000, ge=1, le=60000)
    source_bundle_native_max_requests: int = Field(default=6, ge=1, le=8)
    source_bundle_native_max_output_tokens: int = Field(default=4096, ge=512, le=8192)
    source_bundle_navigation_max_output_tokens: int = Field(default=1024, ge=256, le=8192)
    source_bundle_submit_max_output_tokens: int = Field(default=4096, ge=512, le=8192)
    source_bundle_timeout_seconds: float = Field(default=300, gt=0, le=1800, allow_inf_nan=False)
    strict_evidence_balanced_enabled: bool = False
    strict_evidence_candidate_limit: int = Field(default=60, ge=1, le=60)
    strict_evidence_max_requests: int = Field(default=5, ge=1, le=10)
    strict_evidence_token_stop: int = Field(default=40000, ge=1, le=120000)
    # Explicit cost-only lineage across changed prompts; never checkpoint reuse.
    strict_evidence_cost_parent_run_ids: list[str] = Field(default_factory=list)
    strict_evidence_response_reuse_closures: list[Path] = Field(default_factory=list)
    strict_evidence_response_reuse_operations: list[str] = Field(default_factory=lambda: ["strict_selection", "strict_annotation"])
    paragraph_candidate_limit: int = Field(default=60, ge=40, le=60)
    paragraph_minimum_score: int = Field(default=2, ge=2, le=3)
    paragraph_selection_cap: int = Field(default=10, ge=1, le=30)
    paragraph_rank_batch_size: int = Field(default=12, ge=1, le=20)
    paragraph_mirror_enabled: bool = False
    paragraph_mirror_candidate_limit: int = Field(default=60, ge=0, le=120)
    local_actor: str = "local-operator"
    retrieval_limit: int = 16
    # BM25 orders source batches; it no longer discards lower-ranked sources.
    source_batch_codepoints: int = Field(default=20000, ge=1, le=64000)
    source_batch_spans: int = Field(default=128, ge=1, le=512)
    model_concurrency: int = 2
    max_questions_per_run: int = 20

    @property
    def model_prompt_version(self) -> str:
        if self.model_source_bundle_enabled:
            return ("trusted-analysis-langchain-source-bundles-v1" if self.source_bundle_front_framework == "langchain"
                    else "trusted-analysis-dsh-source-bundles-v1")
        if self.model_strict_evidence_enabled:
            return "trusted-analysis-strict-workbench-v1"
        if self.model_paragraph_ranked_enabled:
            return "trusted-analysis-json-v19-paragraph-ranked"
        return ("trusted-analysis-json-v18-quote-bank-purpose" if self.model_premise_bank_enabled
                else "trusted-analysis-json-v17-paragraph-recall-recovery")

    @model_validator(mode="after")
    def exclusive_strategy(self):
        recovery = (self.source_bundle_recover_front_run_id, self.source_bundle_recover_front_response,
                    self.source_bundle_recover_front_response_sha256)
        if any(value is not None for value in recovery):
            if (not all(value is not None for value in recovery) or not self.source_bundle_paragraph_quality_compact_enabled
                    or not self.source_bundle_recover_front_run_id.strip()
                    or not self.source_bundle_recover_front_response.is_absolute()
                    or len(self.source_bundle_recover_front_response_sha256) != 64
                    or any(c not in "0123456789abcdef" for c in self.source_bundle_recover_front_response_sha256)):
                raise ValueError("sealed front recovery requires an explicit run, absolute response and SHA256 with compact paragraph quality")
        if self.source_bundle_compact_handles and not self.model_source_bundle_enabled:
            raise ValueError("compact source handles require the explicit source bundle workflow")
        if self.source_bundle_front_framework != "dsh" and not self.model_source_bundle_enabled:
            raise ValueError("source bundle framework requires the explicit source bundle workflow")
        if self.source_bundle_front_framework == "langchain" and self.source_bundle_compact_handles:
            raise ValueError("DSH short handles cannot be combined with the LangChain source book")
        if self.source_bundle_local_rank_enabled:
            if not self.model_source_bundle_enabled or self.source_bundle_front_framework != "langchain":
                raise ValueError("local complete-parent ranking requires the LangChain source book")
            digest = self.source_bundle_local_rank_manifest_sha256
            if (self.source_bundle_local_rank_manifest is None or not isinstance(digest, str)
                    or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
                raise ValueError("local ranking requires a frozen model manifest SHA256")
        if self.source_bundle_selection_only and (not self.model_source_bundle_enabled or self.source_bundle_front_framework != "langchain"):
            raise ValueError("source-ID-only selection requires the LangChain source book")
        if self.source_bundle_graded_selection and not self.source_bundle_selection_only:
            raise ValueError("graded source-ID selection requires explicit source-ID-only selection")
        if self.source_bundle_compact_analyses_enabled and (not self.model_source_bundle_enabled or self.source_bundle_front_framework != "langchain"):
            raise ValueError("compact independent analyses require the LangChain source book")
        if self.source_bundle_issue_frame_enabled and (not self.model_source_bundle_enabled or self.source_bundle_front_framework != "langchain"):
            raise ValueError("explicit issue frames require the LangChain source book")
        if self.source_bundle_context_passages_enabled and not (
                self.model_source_bundle_enabled and self.source_bundle_front_framework == "langchain"
                and self.source_bundle_local_rank_enabled and self.source_bundle_graded_selection
                and self.source_bundle_compact_analyses_enabled):
            raise ValueError("sentence contexts require local-rank graded compact LangChain source books")
        if self.source_bundle_complete_grading_enabled and not self.source_bundle_context_passages_enabled:
            raise ValueError("complete context grading requires the explicit sentence-context workflow")
        if self.source_bundle_hybrid_retrieval_enabled and not self.source_bundle_context_passages_enabled:
            raise ValueError("hybrid context retrieval requires the explicit sentence-context workflow")
        if self.source_bundle_paragraph_quality_compact_enabled and not self.source_bundle_paragraph_quality_enabled:
            raise ValueError("compact paragraph fields require the explicit paragraph quality workflow")
        if self.source_bundle_parent_token_pack_enabled:
            if not self.source_bundle_paragraph_quality_compact_enabled or self.source_bundle_local_rank_enabled:
                raise ValueError("parent token packing requires compact paragraph quality and excludes local rank")
            if self.source_bundle_recover_front_run_id or self.strict_evidence_response_reuse_closures:
                cache_stages = set(self.strict_evidence_response_reuse_operations)
                if not (self.source_bundle_recover_front_run_id and self.strict_evidence_response_reuse_closures
                        and self.source_bundle_recover_front_run_id in self.strict_evidence_cost_parent_run_ids
                        and cache_stages in ({"authority_analyst", "cross_validation_analyst"},
                            {"authority_analyst", "cross_validation_analyst", "arbitrate"})):
                    raise ValueError("parent token packing recovery requires explicit sealed source, cost parent and exact A/B caches")
        if self.source_bundle_paragraph_quality_enabled:
            if not (self.model_source_bundle_enabled and self.source_bundle_front_framework == "langchain"
                    and self.source_bundle_compact_analyses_enabled):
                raise ValueError("paragraph quality requires compact LangChain source books")
            if (self.source_bundle_context_passages_enabled or self.source_bundle_complete_grading_enabled
                    or self.source_bundle_graded_selection or self.source_bundle_selection_only):
                raise ValueError("paragraph quality and legacy selection/context protocols are mutually exclusive")
        if sum((self.model_premise_bank_enabled, self.model_paragraph_ranked_enabled, self.model_strict_evidence_enabled, self.model_source_bundle_enabled)) > 1:
            raise ValueError("evidence strategies are mutually exclusive")
        if any(not identity.strip() for identity in self.strict_evidence_cost_parent_run_ids):
            raise ValueError("cost parent run IDs must be nonempty")
        if not self.model_source_bundle_enabled and self.strict_evidence_max_requests > 6:
            raise ValueError("legacy strict request limit cannot exceed six")
        if self.model_source_bundle_enabled:
            front_calls = 1 if self.source_bundle_front_framework == "langchain" else self.source_bundle_native_max_requests
            if self.strict_evidence_max_requests < front_calls + 3:
                raise ValueError("source bundle call limit must reserve both independent analysts and arbitration")
            if self.source_bundle_front_token_stop + self.source_bundle_tail_reserved_tokens > self.strict_evidence_token_stop:
                raise ValueError("source bundle front and tail reservations exceed the frozen whole-question stop")
            if max(self.source_bundle_navigation_max_output_tokens, self.source_bundle_submit_max_output_tokens) > self.source_bundle_native_max_output_tokens:
                raise ValueError("source bundle phase output capacity exceeds the native guard capacity")
        return self

    @property
    def source_bundle_config(self):
        graded = {}
        if self.source_bundle_graded_selection:
            if self.source_bundle_context_passages_enabled:
                from app.domain.lc_context_case import POLICY as GRADED_POLICY
            else:
                from app.domain.lc_bundle_case import GRADED_POLICY
            graded = {"graded_selection": True, "graded_selection_policy": dict(GRADED_POLICY)}
        planning = {}
        if (self.source_bundle_graded_selection or self.source_bundle_paragraph_quality_enabled) and self.source_bundle_compact_analyses_enabled:
            from copy import deepcopy
            from app.agents.episode_budget import POLICY
            if self.source_bundle_paragraph_quality_enabled:
                from app.agents.episode_budget import PARAGRAPH_QUALITY_POLICY as POLICY
                if self.source_bundle_paragraph_quality_compact_enabled:
                    from app.domain.paragraph_quality_compact_case import VERSION
                    if VERSION == "lc-paragraph-quality-v4":
                        from app.agents.episode_budget import PARAGRAPH_QUALITY_COMPACT_V4_POLICY as POLICY
                if self.source_bundle_parent_token_pack_enabled:
                    from app.agents.episode_budget import PARAGRAPH_QUALITY_BUDGET_V5_POLICY as POLICY
            planning = {"whole_chain_planning": deepcopy(POLICY)}
        context = {}
        if self.source_bundle_context_passages_enabled:
            from copy import deepcopy
            from app.domain.context_candidates import POLICY
            if self.source_bundle_hybrid_retrieval_enabled:
                from app.domain.context_candidates import HYBRID_POLICY as POLICY
            context = {"context_passages": True, "context_policy": deepcopy(POLICY)}
        complete = {}
        if self.source_bundle_complete_grading_enabled:
            from copy import deepcopy
            from app.domain.complete_context_grading import POLICY
            complete = {"complete_grading": True, "complete_grading_policy": deepcopy(POLICY)}
        hybrid = {}
        if self.source_bundle_hybrid_retrieval_enabled:
            from copy import deepcopy
            from app.domain.context_lexical_rank import POLICY
            hybrid = {"hybrid_retrieval": True, "hybrid_retrieval_policy": deepcopy(POLICY)}
        paragraph = {}
        if self.source_bundle_paragraph_quality_enabled:
            from copy import deepcopy
            from app.domain.paragraph_quality_case import POLICY
            if self.source_bundle_paragraph_quality_compact_enabled:
                from app.domain.paragraph_quality_compact_case import POLICY
            if self.source_bundle_parent_token_pack_enabled:
                from app.domain.paragraph_quality_budget_case import POLICY
            paragraph = {"paragraph_quality": True, "paragraph_quality_policy": deepcopy(POLICY)}
            if self.source_bundle_paragraph_quality_compact_enabled:
                paragraph.update(paragraph_quality_compact=True, paragraph_quality_compact_policy=deepcopy(POLICY))
            if self.source_bundle_parent_token_pack_enabled:
                paragraph.update(parent_token_pack=True, parent_token_pack_policy=deepcopy(POLICY))
        return {**({"front_framework": "langchain"} if self.source_bundle_front_framework == "langchain" else {}),
                **({"compact_handles": True} if self.source_bundle_compact_handles else {}),
                **({"selection_only": True, "selection_output_tokens": 2048 if self.source_bundle_graded_selection else 1024} if self.source_bundle_selection_only else {}),
                **graded,
                **({"compact_analyses": True} if self.source_bundle_compact_analyses_enabled else {}),
                **({"issue_frame": True, "issue_frame_relation_version": "conclusion-aligned-evidence-relations-v3"} if self.source_bundle_issue_frame_enabled else {}),
                **context,
                **complete,
                **hybrid,
                **paragraph,
                **({"sealed_front_recovery": {"origin_run_id": self.source_bundle_recover_front_run_id,
                    "response": {"file": str(self.source_bundle_recover_front_response),
                        "sha256": self.source_bundle_recover_front_response_sha256},
                    "source_model_calls": 0, "original_response_modified": False}}
                   if self.source_bundle_recover_front_run_id else {}),
                **({"selection_output_tokens": 8192} if self.source_bundle_paragraph_quality_enabled else {}),
                **planning,
                **({"local_parent_rank": {"enabled": True,
                    "model_manifest_sha256": self.source_bundle_local_rank_manifest_sha256,
                    "candidate_limit": 32, "per_document_soft_cap": 3,
                    "policy": "full-question-literal-facets-body-rrf60-v1"}}
                    if self.source_bundle_local_rank_enabled else {}),
                "front_token_stop": self.source_bundle_front_token_stop,
                "tail_reserved_tokens": self.source_bundle_tail_reserved_tokens,
                "native_max_requests": self.source_bundle_native_max_requests,
                "native_max_output_tokens": self.source_bundle_native_max_output_tokens,
                "navigation_max_output_tokens": self.source_bundle_navigation_max_output_tokens,
                "submit_max_output_tokens": self.source_bundle_submit_max_output_tokens,
                "timeout_seconds": self.source_bundle_timeout_seconds,
                "scoped_arbitration": True}

    @property
    def paragraph_ranked_config(self):
        return {"candidate_limit": self.paragraph_candidate_limit, "minimum_score": self.paragraph_minimum_score,
                "selection_cap": self.paragraph_selection_cap, "rank_batch_size": self.paragraph_rank_batch_size,
                "mirror_enabled": self.paragraph_mirror_enabled, "mirror_candidate_limit": self.paragraph_mirror_candidate_limit}

    @property
    def model_configured(self) -> bool:
        return bool(self.model_name.strip() and self.model_api_key.get_secret_value())

    @property
    def database_path(self) -> Path:
        return self.data_dir / "business.sqlite3"

    @property
    def checkpoint_path(self) -> Path:
        return self.data_dir / "checkpoints.sqlite3"

    @property
    def export_dir(self) -> Path:
        return self.data_dir / "exports"
