-- 可信研判工作台技术底座迁移 v1，源自 v0.2 SQLite 结构设计。
-- UUID、逐字引用、跨专题一致性、图无环、状态转移和发布CAS由服务层校验。
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE topics (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  source_manifest_sha256 TEXT NOT NULL,
  head_revision_id TEXT,
  published_revision_id TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY (head_revision_id) REFERENCES graph_revisions(id) DEFERRABLE INITIALLY DEFERRED,
  FOREIGN KEY (published_revision_id) REFERENCES graph_revisions(id) DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE documents (
  id TEXT PRIMARY KEY,
  topic_id TEXT NOT NULL REFERENCES topics(id),
  raw_doc_id TEXT NOT NULL,
  title TEXT NOT NULL,
  content TEXT NOT NULL,
  content_sha256 TEXT NOT NULL,
  source_credibility_json TEXT CHECK (source_credibility_json IS NULL OR json_valid(source_credibility_json)),
  content_url TEXT,
  raw_publish_time TEXT,
  parsed_publish_time TEXT,
  time_precision TEXT NOT NULL DEFAULT 'unknown',
  time_parse_status TEXT NOT NULL DEFAULT 'unparsed',
  UNIQUE (topic_id, raw_doc_id)
);

CREATE TABLE paragraphs (
  id TEXT PRIMARY KEY,
  document_id TEXT NOT NULL REFERENCES documents(id),
  raw_para_id TEXT NOT NULL,
  start_cp INTEGER NOT NULL CHECK (start_cp >= 0),
  end_cp INTEGER NOT NULL CHECK (end_cp >= start_cp),
  text TEXT NOT NULL,
  text_sha256 TEXT NOT NULL,
  UNIQUE (document_id, raw_para_id)
);

CREATE TABLE questions (
  id TEXT PRIMARY KEY,
  topic_id TEXT NOT NULL REFERENCES topics(id),
  raw_question_id TEXT NOT NULL,
  question TEXT NOT NULL,
  UNIQUE (topic_id, raw_question_id)
);

-- 引用身份不可变；修订引用须新建evidence，再变更图中的采用关系。
CREATE TABLE evidence (
  id TEXT PRIMARY KEY,
  topic_id TEXT NOT NULL REFERENCES topics(id),
  paragraph_id TEXT NOT NULL REFERENCES paragraphs(id),
  start_cp INTEGER NOT NULL CHECK (start_cp >= 0),
  end_cp INTEGER NOT NULL CHECK (end_cp > start_cp),
  quote TEXT NOT NULL CHECK (length(quote) > 0),
  quote_sha256 TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE node_identities (
  id TEXT PRIMARY KEY,
  topic_id TEXT NOT NULL REFERENCES topics(id),
  question_id TEXT REFERENCES questions(id),
  evidence_id TEXT REFERENCES evidence(id),
  kind TEXT NOT NULL CHECK (kind IN ('evidence','judgment','conclusion')),
  CHECK ((kind = 'evidence' AND evidence_id IS NOT NULL)
      OR (kind <> 'evidence' AND evidence_id IS NULL AND question_id IS NOT NULL))
);

CREATE TABLE graph_revisions (
  id TEXT PRIMARY KEY,
  topic_id TEXT NOT NULL REFERENCES topics(id),
  revision_no INTEGER NOT NULL CHECK (revision_no >= 0),
  parent_revision_id TEXT REFERENCES graph_revisions(id),
  state TEXT NOT NULL CHECK (state IN ('building','awaiting_review','ready','failed','cancelled','superseded')),
  input_manifest_sha256 TEXT NOT NULL,
  config_sha256 TEXT NOT NULL,
  created_at TEXT NOT NULL,
  completed_at TEXT,
  UNIQUE (topic_id, revision_no)
);

CREATE TABLE runs (
  id TEXT PRIMARY KEY,
  topic_id TEXT NOT NULL REFERENCES topics(id),
  base_revision_id TEXT REFERENCES graph_revisions(id),
  target_revision_id TEXT NOT NULL UNIQUE REFERENCES graph_revisions(id),
  mode TEXT NOT NULL CHECK (mode IN ('initial','recompute','retry')),
  status TEXT NOT NULL CHECK (status IN ('queued','running','awaiting_review','succeeded','failed','cancelled','superseded')),
  cancel_requested INTEGER NOT NULL DEFAULT 0 CHECK (cancel_requested IN (0,1)),
  question_ids_json TEXT NOT NULL CHECK (json_valid(question_ids_json)),
  model_config_json TEXT NOT NULL CHECK (json_valid(model_config_json)),
  prompt_bundle_sha256 TEXT NOT NULL,
  code_version TEXT NOT NULL,
  actor_id TEXT NOT NULL DEFAULT 'local-operator',
  idempotency_key TEXT NOT NULL,
  request_sha256 TEXT NOT NULL,
  usage_json TEXT CHECK (usage_json IS NULL OR json_valid(usage_json)),
  error_json TEXT CHECK (error_json IS NULL OR json_valid(error_json)),
  created_at TEXT NOT NULL,
  completed_at TEXT,
  UNIQUE (topic_id, idempotency_key)
);
CREATE UNIQUE INDEX one_active_run_per_topic ON runs(topic_id)
  WHERE status IN ('queued','running','awaiting_review');

CREATE TABLE node_states (
  revision_id TEXT NOT NULL REFERENCES graph_revisions(id),
  node_id TEXT NOT NULL REFERENCES node_identities(id),
  status TEXT NOT NULL CHECK (status IN ('valid','invalid','stale','needs_review')),
  authority TEXT NOT NULL CHECK (authority IN ('machine','human')),
  locked INTEGER NOT NULL DEFAULT 0 CHECK (locked IN (0,1)),
  adopted INTEGER NOT NULL DEFAULT 1 CHECK (adopted IN (0,1)),
  content_json TEXT NOT NULL CHECK (json_valid(content_json)),
  provenance_json TEXT NOT NULL CHECK (json_valid(provenance_json)),
  machine_candidate_json TEXT CHECK (machine_candidate_json IS NULL OR json_valid(machine_candidate_json)),
  node_version INTEGER NOT NULL DEFAULT 1 CHECK (node_version >= 1),
  PRIMARY KEY (revision_id, node_id),
  CHECK (locked = 0 OR authority = 'human')
);

CREATE TABLE graph_edges (
  revision_id TEXT NOT NULL,
  from_node_id TEXT NOT NULL,
  to_node_id TEXT NOT NULL,
  relation TEXT NOT NULL CHECK (relation IN ('supports','contradicts','prerequisite')),
  basis_json TEXT NOT NULL CHECK (json_valid(basis_json)),
  PRIMARY KEY (revision_id, from_node_id, to_node_id, relation),
  FOREIGN KEY (revision_id, from_node_id) REFERENCES node_states(revision_id,node_id),
  FOREIGN KEY (revision_id, to_node_id) REFERENCES node_states(revision_id,node_id),
  CHECK (from_node_id <> to_node_id)
);

CREATE TABLE source_groups (
  revision_id TEXT NOT NULL REFERENCES graph_revisions(id),
  group_id TEXT NOT NULL,
  basis_json TEXT NOT NULL CHECK (json_valid(basis_json)),
  PRIMARY KEY (revision_id, group_id)
);
CREATE TABLE source_group_members (
  revision_id TEXT NOT NULL,
  group_id TEXT NOT NULL,
  evidence_id TEXT NOT NULL REFERENCES evidence(id),
  PRIMARY KEY (revision_id, evidence_id),
  FOREIGN KEY (revision_id,group_id) REFERENCES source_groups(revision_id,group_id)
);

CREATE TABLE evidence_snapshots (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES runs(id),
  snapshot_sha256 TEXT NOT NULL,
  snapshot_json TEXT NOT NULL CHECK (json_valid(snapshot_json)),
  created_at TEXT NOT NULL
);

CREATE TABLE agent_executions (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES runs(id),
  question_id TEXT NOT NULL REFERENCES questions(id),
  role TEXT NOT NULL CHECK (role IN ('retrieve','verify','authority_analyst','cross_validation_analyst','arbitrate','baseline')),
  round INTEGER NOT NULL CHECK (round >= 0),
  attempt INTEGER NOT NULL DEFAULT 1 CHECK (attempt >= 1),
  evidence_snapshot_id TEXT NOT NULL REFERENCES evidence_snapshots(id),
  input_sha256 TEXT NOT NULL,
  prompt_version TEXT NOT NULL,
  model_name TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('running','succeeded','failed','cancelled')),
  input_json TEXT NOT NULL CHECK (json_valid(input_json)),
  output_json TEXT CHECK (output_json IS NULL OR json_valid(output_json)),
  usage_json TEXT CHECK (usage_json IS NULL OR json_valid(usage_json)),
  UNIQUE (run_id, question_id, role, round, attempt)
);

CREATE TABLE confidence_scores (
  revision_id TEXT NOT NULL,
  node_id TEXT NOT NULL,
  measurement TEXT NOT NULL DEFAULT 'quality_index' CHECK (measurement = 'quality_index'),
  score REAL NOT NULL CHECK (score >= 0 AND score <= 1),
  tier TEXT NOT NULL CHECK (tier IN ('high','medium','low')),
  formula_version TEXT NOT NULL,
  parameter_version TEXT NOT NULL,
  components_json TEXT NOT NULL CHECK (json_valid(components_json)),
  inputs_json TEXT NOT NULL CHECK (json_valid(inputs_json)),
  calculation TEXT NOT NULL,
  PRIMARY KEY (revision_id, node_id),
  FOREIGN KEY (revision_id,node_id) REFERENCES node_states(revision_id,node_id)
);

CREATE TABLE conflicts (
  revision_id TEXT NOT NULL REFERENCES graph_revisions(id),
  id TEXT NOT NULL,
  question_id TEXT NOT NULL REFERENCES questions(id),
  status TEXT NOT NULL CHECK (status IN ('resolved','coexisting','unresolved')),
  details_json TEXT NOT NULL CHECK (json_valid(details_json)),
  PRIMARY KEY (revision_id,id)
);

CREATE TABLE run_events (
  run_id TEXT NOT NULL REFERENCES runs(id),
  seq INTEGER NOT NULL CHECK (seq >= 1),
  type TEXT NOT NULL,
  data_json TEXT NOT NULL CHECK (json_valid(data_json)),
  created_at TEXT NOT NULL,
  PRIMARY KEY (run_id,seq)
);

CREATE TABLE change_previews (
  id TEXT PRIMARY KEY,
  topic_id TEXT NOT NULL REFERENCES topics(id),
  base_revision_id TEXT NOT NULL REFERENCES graph_revisions(id),
  operations_json TEXT NOT NULL CHECK (json_valid(operations_json)),
  operations_sha256 TEXT NOT NULL,
  config_sha256 TEXT NOT NULL,
  impact_json TEXT NOT NULL CHECK (json_valid(impact_json)),
  reason TEXT NOT NULL CHECK (length(reason) > 0),
  expires_at TEXT NOT NULL,
  consumed_by_run_id TEXT UNIQUE REFERENCES runs(id)
);
CREATE TABLE changesets (
  id TEXT PRIMARY KEY,
  preview_id TEXT NOT NULL UNIQUE REFERENCES change_previews(id),
  run_id TEXT NOT NULL UNIQUE REFERENCES runs(id),
  actor_id TEXT NOT NULL,
  reason TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE review_requests (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES runs(id),
  target_revision_id TEXT NOT NULL REFERENCES graph_revisions(id),
  actor_id TEXT NOT NULL,
  idempotency_key TEXT NOT NULL,
  request_sha256 TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE (run_id,idempotency_key)
);
CREATE TABLE review_decisions (
  id TEXT PRIMARY KEY,
  review_request_id TEXT NOT NULL REFERENCES review_requests(id),
  node_id TEXT NOT NULL REFERENCES node_identities(id),
  expected_node_version INTEGER NOT NULL CHECK (expected_node_version >= 1),
  action TEXT NOT NULL CHECK (action IN ('keep_manual','rewrite_manual','release_override')),
  decision_json TEXT NOT NULL CHECK (json_valid(decision_json)),
  reason TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE (review_request_id,node_id)
);

CREATE TABLE export_jobs (
  id TEXT PRIMARY KEY,
  topic_id TEXT NOT NULL REFERENCES topics(id),
  revision_id TEXT NOT NULL REFERENCES graph_revisions(id),
  status TEXT NOT NULL CHECK (status IN ('queued','running','succeeded','failed')),
  adapter_version TEXT NOT NULL,
  manifest_json TEXT CHECK (manifest_json IS NULL OR json_valid(manifest_json)),
  artifact_path TEXT,
  error_json TEXT CHECK (error_json IS NULL OR json_valid(error_json)),
  idempotency_key TEXT NOT NULL,
  request_sha256 TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE (topic_id,idempotency_key)
);

-- 原始资料只读。更换原始资料须新建数据快照，不绕过旧证据引用。
CREATE TRIGGER documents_immutable_update BEFORE UPDATE ON documents BEGIN SELECT RAISE(ABORT,'immutable_document'); END;
CREATE TRIGGER documents_immutable_delete BEFORE DELETE ON documents BEGIN SELECT RAISE(ABORT,'immutable_document'); END;
CREATE TRIGGER paragraphs_immutable_update BEFORE UPDATE ON paragraphs BEGIN SELECT RAISE(ABORT,'immutable_paragraph'); END;
CREATE TRIGGER paragraphs_immutable_delete BEFORE DELETE ON paragraphs BEGIN SELECT RAISE(ABORT,'immutable_paragraph'); END;
CREATE TRIGGER evidence_immutable_update BEFORE UPDATE ON evidence BEGIN SELECT RAISE(ABORT,'immutable_evidence'); END;
CREATE TRIGGER evidence_immutable_delete BEFORE DELETE ON evidence BEGIN SELECT RAISE(ABORT,'immutable_evidence'); END;
CREATE TRIGGER node_identities_immutable_update BEFORE UPDATE ON node_identities BEGIN SELECT RAISE(ABORT,'immutable_node_identity'); END;
CREATE TRIGGER node_identities_immutable_delete BEFORE DELETE ON node_identities BEGIN SELECT RAISE(ABORT,'immutable_node_identity'); END;
CREATE TRIGGER questions_immutable_update BEFORE UPDATE ON questions BEGIN SELECT RAISE(ABORT,'immutable_question'); END;
CREATE TRIGGER questions_immutable_delete BEFORE DELETE ON questions BEGIN SELECT RAISE(ABORT,'immutable_question'); END;
CREATE TRIGGER evidence_snapshots_immutable_update BEFORE UPDATE ON evidence_snapshots BEGIN SELECT RAISE(ABORT,'immutable_evidence_snapshot'); END;
CREATE TRIGGER evidence_snapshots_immutable_delete BEFORE DELETE ON evidence_snapshots BEGIN SELECT RAISE(ABORT,'immutable_evidence_snapshot'); END;
CREATE TRIGGER run_events_append_only_update BEFORE UPDATE ON run_events BEGIN SELECT RAISE(ABORT,'append_only_event'); END;
CREATE TRIGGER run_events_append_only_delete BEFORE DELETE ON run_events BEGIN SELECT RAISE(ABORT,'append_only_event'); END;
CREATE TRIGGER changesets_immutable_update BEFORE UPDATE ON changesets BEGIN SELECT RAISE(ABORT,'immutable_changeset'); END;
CREATE TRIGGER changesets_immutable_delete BEFORE DELETE ON changesets BEGIN SELECT RAISE(ABORT,'immutable_changeset'); END;
CREATE TRIGGER review_requests_immutable_update BEFORE UPDATE ON review_requests BEGIN SELECT RAISE(ABORT,'immutable_review_request'); END;
CREATE TRIGGER review_requests_immutable_delete BEFORE DELETE ON review_requests BEGIN SELECT RAISE(ABORT,'immutable_review_request'); END;
CREATE TRIGGER review_decisions_immutable_update BEFORE UPDATE ON review_decisions BEGIN SELECT RAISE(ABORT,'immutable_review_decision'); END;
CREATE TRIGGER review_decisions_immutable_delete BEFORE DELETE ON review_decisions BEGIN SELECT RAISE(ABORT,'immutable_review_decision'); END;

CREATE TRIGGER agent_executions_ready_insert BEFORE INSERT ON agent_executions
WHEN (SELECT g.state FROM runs r JOIN graph_revisions g ON g.id=r.target_revision_id WHERE r.id=NEW.run_id)='ready'
BEGIN SELECT RAISE(ABORT,'immutable_ready_revision'); END;
CREATE TRIGGER agent_executions_ready_update BEFORE UPDATE ON agent_executions
WHEN (SELECT g.state FROM runs r JOIN graph_revisions g ON g.id=r.target_revision_id WHERE r.id=OLD.run_id)='ready'
  OR (SELECT g.state FROM runs r JOIN graph_revisions g ON g.id=r.target_revision_id WHERE r.id=NEW.run_id)='ready'
BEGIN SELECT RAISE(ABORT,'immutable_ready_revision'); END;
CREATE TRIGGER agent_executions_ready_delete BEFORE DELETE ON agent_executions
WHEN (SELECT g.state FROM runs r JOIN graph_revisions g ON g.id=r.target_revision_id WHERE r.id=OLD.run_id)='ready'
BEGIN SELECT RAISE(ABORT,'immutable_ready_revision'); END;

-- 已发布图内容不可变。重算须生成新revision；LangGraph checkpoint另库存储。
CREATE TRIGGER node_states_ready_insert BEFORE INSERT ON node_states
WHEN (SELECT state FROM graph_revisions WHERE id=NEW.revision_id)='ready'
BEGIN SELECT RAISE(ABORT,'immutable_ready_revision'); END;
CREATE TRIGGER node_states_ready_update BEFORE UPDATE ON node_states
WHEN (SELECT state FROM graph_revisions WHERE id=OLD.revision_id)='ready'
  OR (SELECT state FROM graph_revisions WHERE id=NEW.revision_id)='ready'
BEGIN SELECT RAISE(ABORT,'immutable_ready_revision'); END;
CREATE TRIGGER node_states_ready_delete BEFORE DELETE ON node_states
WHEN (SELECT state FROM graph_revisions WHERE id=OLD.revision_id)='ready'
BEGIN SELECT RAISE(ABORT,'immutable_ready_revision'); END;
CREATE TRIGGER graph_edges_ready_insert BEFORE INSERT ON graph_edges
WHEN (SELECT state FROM graph_revisions WHERE id=NEW.revision_id)='ready'
BEGIN SELECT RAISE(ABORT,'immutable_ready_revision'); END;
CREATE TRIGGER graph_edges_ready_update BEFORE UPDATE ON graph_edges
WHEN (SELECT state FROM graph_revisions WHERE id=OLD.revision_id)='ready'
  OR (SELECT state FROM graph_revisions WHERE id=NEW.revision_id)='ready'
BEGIN SELECT RAISE(ABORT,'immutable_ready_revision'); END;

CREATE TRIGGER revisions_ready_update BEFORE UPDATE ON graph_revisions
WHEN OLD.state='ready' BEGIN SELECT RAISE(ABORT,'immutable_ready_revision'); END;
CREATE TRIGGER revisions_ready_delete BEFORE DELETE ON graph_revisions
WHEN OLD.state='ready' BEGIN SELECT RAISE(ABORT,'immutable_ready_revision'); END;
CREATE TRIGGER graph_edges_ready_delete BEFORE DELETE ON graph_edges
WHEN (SELECT state FROM graph_revisions WHERE id=OLD.revision_id)='ready'
BEGIN SELECT RAISE(ABORT,'immutable_ready_revision'); END;

CREATE INDEX documents_by_topic ON documents(topic_id);
CREATE INDEX questions_by_topic ON questions(topic_id);
CREATE INDEX evidence_by_paragraph ON evidence(paragraph_id);
CREATE INDEX outgoing_dependencies ON graph_edges(revision_id,from_node_id);
CREATE INDEX incoming_dependencies ON graph_edges(revision_id,to_node_id);

-- GENERATED READY SNAPSHOT GUARDS v0.2
CREATE TRIGGER source_groups_ready_insert BEFORE INSERT ON source_groups WHEN (SELECT state FROM graph_revisions WHERE id=NEW.revision_id)='ready' BEGIN SELECT RAISE(ABORT,'immutable_ready_revision'); END;
CREATE TRIGGER source_groups_ready_update BEFORE UPDATE ON source_groups WHEN (SELECT state FROM graph_revisions WHERE id=OLD.revision_id)='ready' OR (SELECT state FROM graph_revisions WHERE id=NEW.revision_id)='ready' BEGIN SELECT RAISE(ABORT,'immutable_ready_revision'); END;
CREATE TRIGGER source_groups_ready_delete BEFORE DELETE ON source_groups WHEN (SELECT state FROM graph_revisions WHERE id=OLD.revision_id)='ready' BEGIN SELECT RAISE(ABORT,'immutable_ready_revision'); END;
CREATE TRIGGER source_group_members_ready_insert BEFORE INSERT ON source_group_members WHEN (SELECT state FROM graph_revisions WHERE id=NEW.revision_id)='ready' BEGIN SELECT RAISE(ABORT,'immutable_ready_revision'); END;
CREATE TRIGGER source_group_members_ready_update BEFORE UPDATE ON source_group_members WHEN (SELECT state FROM graph_revisions WHERE id=OLD.revision_id)='ready' OR (SELECT state FROM graph_revisions WHERE id=NEW.revision_id)='ready' BEGIN SELECT RAISE(ABORT,'immutable_ready_revision'); END;
CREATE TRIGGER source_group_members_ready_delete BEFORE DELETE ON source_group_members WHEN (SELECT state FROM graph_revisions WHERE id=OLD.revision_id)='ready' BEGIN SELECT RAISE(ABORT,'immutable_ready_revision'); END;
CREATE TRIGGER confidence_scores_ready_insert BEFORE INSERT ON confidence_scores WHEN (SELECT state FROM graph_revisions WHERE id=NEW.revision_id)='ready' BEGIN SELECT RAISE(ABORT,'immutable_ready_revision'); END;
CREATE TRIGGER confidence_scores_ready_update BEFORE UPDATE ON confidence_scores WHEN (SELECT state FROM graph_revisions WHERE id=OLD.revision_id)='ready' OR (SELECT state FROM graph_revisions WHERE id=NEW.revision_id)='ready' BEGIN SELECT RAISE(ABORT,'immutable_ready_revision'); END;
CREATE TRIGGER confidence_scores_ready_delete BEFORE DELETE ON confidence_scores WHEN (SELECT state FROM graph_revisions WHERE id=OLD.revision_id)='ready' BEGIN SELECT RAISE(ABORT,'immutable_ready_revision'); END;
CREATE TRIGGER conflicts_ready_insert BEFORE INSERT ON conflicts WHEN (SELECT state FROM graph_revisions WHERE id=NEW.revision_id)='ready' BEGIN SELECT RAISE(ABORT,'immutable_ready_revision'); END;
CREATE TRIGGER conflicts_ready_update BEFORE UPDATE ON conflicts WHEN (SELECT state FROM graph_revisions WHERE id=OLD.revision_id)='ready' OR (SELECT state FROM graph_revisions WHERE id=NEW.revision_id)='ready' BEGIN SELECT RAISE(ABORT,'immutable_ready_revision'); END;
CREATE TRIGGER conflicts_ready_delete BEFORE DELETE ON conflicts WHEN (SELECT state FROM graph_revisions WHERE id=OLD.revision_id)='ready' BEGIN SELECT RAISE(ABORT,'immutable_ready_revision'); END;
