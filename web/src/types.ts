export interface Health {
  status: string;
  model_configured: boolean;
  workflow_available: boolean;
  version: string;
  model_name?: string;
  model_provider?: string;
  max_questions_per_run?: number;
  source_bundle_enabled?: boolean;
  source_bundle_front_framework?: "dsh" | "langchain" | null;
}
export interface Topic {
  id: string;
  title: string;
  document_count: number;
  paragraph_count: number;
  question_count: number;
  head_revision_id: string | null;
  published_revision_id: string | null;
}
export interface Question {
  id: string;
  raw_question_id: string;
  question: string;
}
export interface RunError {
  code: string;
  stage?: string;
  message: string;
  details?: Record<string, unknown>;
}
export interface Run {
  mode?: string;
  strict_scope_enabled?: boolean;
  premise_bank_enabled?: boolean;
  paragraph_ranked_enabled?: boolean;
  strict_evidence_enabled?: boolean;
  source_bundle_enabled?: boolean;
  source_bundle_front_framework?: "dsh" | "langchain" | null;
  derived_inputs?: { question_id: string; artifact_sha256: string; source_revision_id: string; source_run_id: string; candidate_count: number; historical_source_usage: unknown; historical_source_coverage?: unknown }[];
  run_id: string;
  status: string;
  view_status?: string;
  resumable?: boolean;
  resume_blocked_reason?: string | null;
  resumed_from_run_id?: string | null;
  resume_mode?: string | null;
  question_ids?: string[];
  target_revision_id?: string;
  error?: RunError | string | null;
  created_at?: string;
  completed_at?: string | null;
  usage?: unknown;
}
export interface RunEvent {
  run_id: string;
  seq: number;
  type: string;
  data: Record<string, unknown>;
  created_at: string;
}
export interface ToolCall {
  call_id: string;
  name: string;
  question_id?: string | null;
  status: string;
  started_at: string;
  completed_at?: string | null;
  elapsed_ms?: number | null;
  summary?: string | Record<string, unknown>;
  kind?: "local" | "model";
  engine?: string | null;
  stage?: string | null;
  code?: string;
  message?: string;
  details?: Record<string, unknown>;
}
export interface Activity {
  run: Run;
  events: RunEvent[];
  progress: {
    completed_stages: { question_id: string | null; stage: string }[];
    active_stages: { question_id: string | null; stage: string }[];
    failed_stages?: { question_id: string | null; stage: string; code?: string; message?: string; details?: Record<string, unknown> }[];
    paused_stages?: { question_id: string | null; stage: string }[];
    current_stage: string | null;
    completed_count: number;
    total_count: number;
    elapsed_seconds: number;
  };
  tool_calls: ToolCall[];
  usage: unknown;
}
export interface BusinessNode {
  id?: string;
  node_id?: string;
  kind: string;
  content?: string | Record<string, unknown>;
  status?: string;
  evidence_id?: string;
  question_id?: string;
  locked?: boolean;
  [key: string]: unknown;
}
export interface BusinessEdge {
  from_node_id: string;
  to_node_id: string;
  relation: string;
  [key: string]: unknown;
}
export interface Graph {
  nodes: BusinessNode[];
  edges: BusinessEdge[];
  revision_id: string;
}
export interface Artifacts {
  conclusions?: unknown[];
  conflicts?: unknown[];
  confidence?: unknown[];
  agent_outputs?: unknown[];
  [key: string]: unknown;
}
export interface Evidence {
  id?: string;
  evidence_id?: string;
  quote: string;
  paragraph_id: string;
  start_cp: number;
  end_cp: number;
  span_scope?: "paragraph" | "document" | null;
  raw_doc_id?: string;
  raw_para_id?: string;
  fact_des?: string;
}
export interface Paragraph {
  text: string;
  raw_doc_id: string;
  raw_para_id: string;
  start_cp?: number;
  end_cp?: number;
}
export interface SourceContextMember {
  paragraph_id: string;
  raw_para_id: string;
  start_cp: number;
  end_cp: number;
}
export interface SourceContextBundle {
  span_scope: "document";
  paragraph_id: string;
  raw_doc_id: string;
  raw_para_id: string;
  start_cp: number;
  end_cp: number;
  quote: string;
  source_paragraph_ids: string[];
  document_id: string;
  document_content_sha256: string;
  paragraph_manifest_sha256: string;
  bundle_id: string;
  bundle_purpose: "read_context";
  anchor_paragraph_ids: string[];
  anchor_paragraph_id: string;
  anchor_raw_para_id: string;
  anchor_start_cp: number;
  anchor_end_cp: number;
  paragraph_count: number;
  members: SourceContextMember[];
}
export interface ChangePreview {
  preview_id: string;
  content_hash: string;
  affected_question_ids: string[];
  affected_node_ids: string[];
  must_rerun_agents?: string[];
  [key: string]: unknown;
}
export interface ExportJob {
  export_id: string;
  topic_id: string;
  revision_id: string;
  status: string;
  error?: unknown;
  download_url?: string | null;
}
