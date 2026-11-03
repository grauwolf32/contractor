-- One mutable identity points at immutable assertion revisions. Accepted Task
-- snapshots are independently frozen; no latest lookup participates in a Run.
ALTER TABLE audits
 ADD COLUMN initial_hypothesis_offset integer NOT NULL DEFAULT 0 CHECK(initial_hypothesis_offset BETWEEN 0 AND 10000),
 ADD COLUMN hypothesis_count integer NOT NULL DEFAULT 0 CHECK(hypothesis_count BETWEEN 0 AND 10000),
 ADD COLUMN hypothesis_revision_count integer NOT NULL DEFAULT 0 CHECK(hypothesis_revision_count BETWEEN 0 AND 50000),
 ADD COLUMN hypothesis_bytes bigint NOT NULL DEFAULT 0 CHECK(hypothesis_bytes BETWEEN 0 AND 268435456);

CREATE TABLE audit_hypotheses (
 hypothesis_id text PRIMARY KEY CHECK(hypothesis_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'),
 audit_id text NOT NULL REFERENCES audits(audit_id) ON DELETE CASCADE,
 source_key text NOT NULL CHECK(octet_length(source_key) BETWEEN 1 AND 256),
 fixed boolean NOT NULL,
 ordinal integer NOT NULL CHECK(ordinal BETWEEN 1 AND 10000),
 current_bytes integer NOT NULL CHECK(current_bytes BETWEEN 1 AND 32768),
 current_revision bigint NOT NULL CHECK(current_revision BETWEEN 1 AND 9007199254740991),
 UNIQUE(audit_id,fixed,source_key), UNIQUE(audit_id,ordinal), UNIQUE(hypothesis_id,audit_id)
);
CREATE TABLE audit_hypothesis_revisions (
 hypothesis_id text NOT NULL,
 audit_id text NOT NULL,
 revision bigint NOT NULL CHECK(revision BETWEEN 1 AND 9007199254740991),
 document json NOT NULL CHECK(json_typeof(document)='object' AND octet_length(document::text)<=32768),
 PRIMARY KEY(hypothesis_id,revision),
 FOREIGN KEY(hypothesis_id,audit_id) REFERENCES audit_hypotheses(hypothesis_id,audit_id) ON DELETE CASCADE,
 CHECK(document->>'schema'='contractor.audit.hypothesis.v1'),
 CHECK(document->>'id'=hypothesis_id AND (document->>'revision')::numeric=revision AND document->'origin'->>'audit_id'=audit_id)
);
CREATE INDEX audit_hypothesis_revisions_purge_idx ON audit_hypothesis_revisions(audit_id,hypothesis_id,revision);
CREATE TRIGGER audit_hypothesis_revisions_immutable BEFORE UPDATE ON audit_hypothesis_revisions
 FOR EACH ROW EXECUTE FUNCTION contractor_protect_immutable_history();
CREATE TRIGGER audit_hypotheses_identity_immutable
 BEFORE UPDATE OF hypothesis_id,audit_id,source_key,fixed,ordinal ON audit_hypotheses
 FOR EACH ROW EXECUTE FUNCTION contractor_protect_immutable_history();

ALTER TABLE audit_tasks ADD COLUMN attention boolean NOT NULL DEFAULT false;
ALTER TABLE audit_tasks ADD COLUMN hypothesis_snapshot json CHECK(hypothesis_snapshot IS NULL OR
 (json_typeof(hypothesis_snapshot)='object' AND octet_length(hypothesis_snapshot::text)<=32768));
ALTER TABLE audit_tasks DROP COLUMN metadata_bytes;
ALTER TABLE audit_tasks ADD COLUMN metadata_bytes integer GENERATED ALWAYS AS
 (8192+octet_length(document::text)+octet_length(source::text)+COALESCE(octet_length(hypothesis_snapshot::text),0)) STORED;
CREATE INDEX audit_tasks_ready_workflow_idx ON audit_tasks(audit_id,workflow_alias,attention DESC,ordinal)
 INCLUDE(task_id,metadata_bytes) WHERE state='ready';
CREATE TRIGGER audit_tasks_hypothesis_immutable BEFORE UPDATE OF hypothesis_snapshot,attention ON audit_tasks
 FOR EACH ROW EXECUTE FUNCTION contractor_protect_immutable_history();

-- The source assertion is ordinary independently owned Project material.
ALTER TABLE finding_proposal_receipts ADD COLUMN audit_hypothesis_snapshot jsonb
 CHECK(audit_hypothesis_snapshot IS NULL OR (audit_hypothesis IS NOT NULL AND project_proposal IS NOT NULL AND jsonb_typeof(audit_hypothesis_snapshot)='object' AND octet_length(audit_hypothesis_snapshot::text)<=4096));
CREATE TRIGGER finding_receipt_hypothesis_immutable BEFORE UPDATE OF audit_hypothesis_snapshot ON finding_proposal_receipts
 FOR EACH ROW EXECUTE FUNCTION contractor_protect_immutable_history();

ALTER TABLE audit_tasks ADD CONSTRAINT audit_tasks_hypothesis_shape CHECK (
 (document::jsonb ? 'hypothesis')=(hypothesis_snapshot IS NOT NULL) AND
 (hypothesis_snapshot IS NULL OR
  (document->'hypothesis'->>'id'=hypothesis_snapshot->>'id' AND
   (document->'hypothesis'->>'revision')::numeric=(hypothesis_snapshot->>'revision')::numeric)));

CREATE FUNCTION contractor_advance_hypothesis_revision() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF NEW.current_revision<>OLD.current_revision+1 THEN
  RAISE EXCEPTION 'hypothesis revisions advance exactly once' USING ERRCODE='23514';
 END IF;
 RETURN NEW;
END;
$$;
CREATE TRIGGER audit_hypothesis_revision_advance BEFORE UPDATE OF current_revision,current_bytes ON audit_hypotheses
 FOR EACH ROW EXECUTE FUNCTION contractor_advance_hypothesis_revision();

CREATE INDEX audit_tasks_hypothesis_idx ON audit_tasks(audit_id,(document->'hypothesis'->>'id'),ordinal)
 WHERE document->'hypothesis' IS NOT NULL;
