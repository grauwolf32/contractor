-- Correlation belongs to existing findings and Projects. No observation is
-- merged, deleted or moved to another Audit's analyst state.
ALTER TABLE projects ADD COLUMN finding_correlation_revision bigint NOT NULL DEFAULT 1
 CHECK (finding_correlation_revision BETWEEN 1 AND 9007199254740991);

CREATE FUNCTION contractor_protect_finding_correlation_revision() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF NEW.finding_correlation_revision NOT IN (OLD.finding_correlation_revision, OLD.finding_correlation_revision + 1) THEN
  RAISE EXCEPTION 'finding correlation revision must advance monotonically' USING ERRCODE='23514';
 END IF;
 RETURN NEW;
END;
$$;
CREATE TRIGGER projects_protect_correlation_revision BEFORE UPDATE OF finding_correlation_revision ON projects
 FOR EACH ROW EXECUTE FUNCTION contractor_protect_finding_correlation_revision();

-- Inputs are bounded ASCII identities/hashes. to_json also quotes each string
-- correctly; the array contains no separator whitespace, matching RFC 8785.
CREATE FUNCTION contractor_finding_group_key(project_id text, algorithm text, value text)
 RETURNS text LANGUAGE sql IMMUTABLE STRICT AS $$
 SELECT encode(sha256(convert_to('[' || to_json('contractor/finding-group/v1'::text)::text || ',' ||
  to_json(project_id)::text || ',' || to_json(algorithm)::text || ',' || to_json(value)::text || ']', 'UTF8')), 'hex');
$$;

-- The receipt's projections were strictly validated at intake. Only qualified
-- native keys or a single distinct snapshot issue key may become primary.
CREATE FUNCTION contractor_finding_primary_key(selected_receipt text)
 RETURNS TABLE(algorithm text, value text, reason text) LANGUAGE plpgsql AS $$
DECLARE
 fingerprint jsonb;
 anchors jsonb;
 issue_keys text[];
BEGIN
 SELECT fingerprint_projection::jsonb, anchor_projection::jsonb INTO fingerprint, anchors
 FROM finding_proposal_receipts WHERE receipt_id = selected_receipt;
 IF NOT FOUND THEN
  RAISE EXCEPTION 'finding first receipt is missing' USING ERRCODE='23503';
 END IF;
 IF fingerprint->>'availability' = 'available' THEN
  IF fingerprint->>'algorithm' IS DISTINCT FROM 'contractor/producer-issue/v1'
   OR fingerprint->>'adapter' IS DISTINCT FROM 'pinned-sarif-native/v1' THEN
   RAISE EXCEPTION 'unsupported primary producer policy' USING ERRCODE='23514';
  END IF;
  RETURN QUERY SELECT 'contractor/producer-issue/v1'::text, fingerprint->>'value', ''::text;
  RETURN;
 END IF;
 IF anchors->>'availability' = 'available' THEN
  SELECT array_agg(key) INTO issue_keys FROM (
   SELECT DISTINCT item->>'issueKey' AS key FROM jsonb_array_elements(anchors->'keys') AS item
   WHERE item ? 'issueKey' LIMIT 2
  ) AS qualified;
 END IF;
 IF cardinality(issue_keys) = 1 THEN
  RETURN QUERY SELECT 'contractor/snapshot-issue-anchor/v1'::text, issue_keys[1], ''::text;
 ELSIF cardinality(issue_keys) > 1 THEN
  RETURN QUERY SELECT ''::text, ''::text, 'primary_key_ambiguous'::text;
 ELSIF fingerprint->>'reason' = 'not_recorded' AND anchors->>'reason' = 'not_recorded' THEN
  RETURN QUERY SELECT ''::text, ''::text, 'not_recorded'::text;
 ELSE
  RETURN QUERY SELECT ''::text, ''::text, 'no_qualified_key'::text;
 END IF;
END;
$$;

ALTER TABLE audit_findings
 ADD COLUMN correlation_project_id text,
 ADD COLUMN correlation_algorithm text NOT NULL DEFAULT '',
 ADD COLUMN correlation_value text NOT NULL DEFAULT '',
 ADD COLUMN correlation_group_key text NOT NULL DEFAULT '',
 ADD COLUMN correlation_reason text NOT NULL DEFAULT 'not_recorded';

-- Freeze known retained keys without reparsing or modifying any receipt. Older
-- receipts whose projections are not_recorded remain explicitly unqualified.
WITH selected AS (
 SELECT finding.finding_id, audit.project_id, primary_key.*
 FROM audit_findings AS finding JOIN audits AS audit USING(audit_id)
 CROSS JOIN LATERAL contractor_finding_primary_key(finding.first_receipt_id) AS primary_key
)
UPDATE audit_findings AS finding SET correlation_project_id = selected.project_id,
 correlation_algorithm = selected.algorithm, correlation_value = selected.value,
 correlation_reason = selected.reason, correlation_group_key = CASE WHEN selected.algorithm = '' THEN ''
 ELSE contractor_finding_group_key(selected.project_id, selected.algorithm, selected.value) END
FROM selected WHERE selected.finding_id = finding.finding_id;

ALTER TABLE audit_findings ALTER COLUMN correlation_project_id SET NOT NULL;
ALTER TABLE audits ADD CONSTRAINT audits_correlation_scope UNIQUE(audit_id, project_id);
ALTER TABLE audit_findings ADD CONSTRAINT audit_findings_correlation_scope
 FOREIGN KEY(audit_id, correlation_project_id) REFERENCES audits(audit_id, project_id) ON DELETE CASCADE;
CREATE INDEX audit_findings_correlation_scope_idx ON audit_findings(audit_id, correlation_project_id);
CREATE INDEX audit_findings_correlation_group_idx ON audit_findings(correlation_project_id, correlation_group_key, finding_id COLLATE "C")
 INCLUDE(audit_id) WHERE correlation_group_key <> '';
ALTER TABLE audit_findings ADD CONSTRAINT audit_findings_correlation_shape CHECK (
 (correlation_algorithm = '' AND correlation_value = '' AND correlation_group_key = ''
  AND correlation_reason IN ('not_recorded','no_qualified_key','primary_key_ambiguous'))
 OR (correlation_algorithm IN ('contractor/producer-issue/v1','contractor/snapshot-issue-anchor/v1')
  AND correlation_value ~ '^[0-9a-f]{64}$' AND correlation_reason = ''
  AND correlation_group_key = contractor_finding_group_key(correlation_project_id, correlation_algorithm, correlation_value))
);

CREATE FUNCTION contractor_initialize_finding_correlation() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE
 actual_project text;
 primary_key record;
 actual_group text;
BEGIN
 SELECT project_id INTO actual_project FROM audits WHERE audit_id = NEW.audit_id;
 IF NOT FOUND THEN
  RAISE EXCEPTION 'finding Audit is missing' USING ERRCODE='23503';
 END IF;
 SELECT * INTO STRICT primary_key FROM contractor_finding_primary_key(NEW.first_receipt_id);
 actual_group := CASE WHEN primary_key.algorithm = '' THEN ''
  ELSE contractor_finding_group_key(actual_project, primary_key.algorithm, primary_key.value) END;
 IF NEW.correlation_algorithm IS NULL OR NEW.correlation_value IS NULL OR NEW.correlation_group_key IS NULL OR NEW.correlation_reason IS NULL
  OR NEW.correlation_project_id IS NOT NULL AND NEW.correlation_project_id <> actual_project
  OR NEW.correlation_algorithm <> '' AND NEW.correlation_algorithm <> primary_key.algorithm
  OR NEW.correlation_value <> '' AND NEW.correlation_value <> primary_key.value
  OR NEW.correlation_group_key <> '' AND NEW.correlation_group_key <> actual_group
  OR NEW.correlation_reason <> 'not_recorded' AND NEW.correlation_reason <> primary_key.reason THEN
  RAISE EXCEPTION 'finding correlation does not match its first receipt or Audit' USING ERRCODE='23514';
 END IF;
 NEW.correlation_project_id := actual_project;
 NEW.correlation_algorithm := primary_key.algorithm;
 NEW.correlation_value := primary_key.value;
 NEW.correlation_group_key := actual_group;
 NEW.correlation_reason := primary_key.reason;
 RETURN NEW;
END;
$$;
CREATE TRIGGER audit_findings_initialize_correlation BEFORE INSERT ON audit_findings
 FOR EACH ROW EXECUTE FUNCTION contractor_initialize_finding_correlation();

CREATE FUNCTION contractor_protect_finding_correlation() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF ROW(NEW.audit_id,NEW.first_receipt_id,NEW.correlation_project_id,NEW.correlation_algorithm,NEW.correlation_value,NEW.correlation_group_key,NEW.correlation_reason)
  IS DISTINCT FROM ROW(OLD.audit_id,OLD.first_receipt_id,OLD.correlation_project_id,OLD.correlation_algorithm,OLD.correlation_value,OLD.correlation_group_key,OLD.correlation_reason) THEN
  RAISE EXCEPTION 'first-receipt finding correlation is immutable' USING ERRCODE='23514';
 END IF;
 RETURN NEW;
END;
$$;
CREATE TRIGGER audit_findings_protect_correlation BEFORE UPDATE OF audit_id,first_receipt_id,correlation_project_id,correlation_algorithm,correlation_value,correlation_group_key,correlation_reason ON audit_findings
 FOR EACH ROW EXECUTE FUNCTION contractor_protect_finding_correlation();

-- Child writers already hold Run/Audit authority. They must never wait for a
-- Project deletion's parent lock: NOWAIT aborts the whole transaction as the
-- existing retryable serialization class, rather than introducing a lock cycle.
CREATE FUNCTION contractor_lock_finding_correlation_project(selected_project text) RETURNS void LANGUAGE plpgsql AS $$
BEGIN
 BEGIN
  PERFORM project_id FROM projects WHERE project_id = selected_project FOR NO KEY UPDATE NOWAIT;
 EXCEPTION WHEN lock_not_available THEN
  RAISE EXCEPTION 'Project correlation is concurrently changing' USING ERRCODE='40001';
 END;
END;
$$;

CREATE FUNCTION contractor_advance_finding_correlation(affected_projects text[]) RETURNS void LANGUAGE plpgsql AS $$
DECLARE
 selected_project text;
BEGIN
 FOR selected_project IN SELECT DISTINCT unnest(affected_projects) ORDER BY 1 LOOP
  PERFORM contractor_lock_finding_correlation_project(selected_project);
  UPDATE projects SET finding_correlation_revision = finding_correlation_revision + 1 WHERE project_id = selected_project;
 END LOOP;
END;
$$;

CREATE FUNCTION contractor_finding_correlation_inserted() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 PERFORM contractor_advance_finding_correlation(ARRAY(SELECT DISTINCT correlation_project_id FROM inserted_findings));
 RETURN NULL;
END;
$$;
CREATE TRIGGER audit_findings_correlation_inserted AFTER INSERT ON audit_findings
 REFERENCING NEW TABLE AS inserted_findings FOR EACH STATEMENT EXECUTE FUNCTION contractor_finding_correlation_inserted();

CREATE FUNCTION contractor_finding_correlation_deleted() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 -- Purge changes the selected Audit set as well as Project membership. No
 -- synthetic analyst decision/event is appended and no surviving peer is written.
 UPDATE audits SET revision = revision + 1,
  updated_at = GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
 WHERE audit_id IN (SELECT DISTINCT audit_id FROM deleted_findings);
 PERFORM contractor_advance_finding_correlation(ARRAY(SELECT DISTINCT correlation_project_id FROM deleted_findings));
 RETURN NULL;
END;
$$;
CREATE TRIGGER audit_findings_correlation_deleted AFTER DELETE ON audit_findings
 REFERENCING OLD TABLE AS deleted_findings FOR EACH STATEMENT EXECUTE FUNCTION contractor_finding_correlation_deleted();

CREATE FUNCTION contractor_finding_correlation_updated() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE
 affected_projects text[];
 affected_audits text[];
BEGIN
 SELECT array_agg(DISTINCT n.correlation_project_id), array_agg(DISTINCT n.audit_id)
 INTO affected_projects, affected_audits FROM updated_findings AS n JOIN previous_findings AS o USING(finding_id)
 WHERE ROW(n.state,n.rejection_reason,n.duplicate_target_id,n.current_assessment_id,n.current_decision_id,n.revision,n.updated_at)
  IS DISTINCT FROM ROW(o.state,o.rejection_reason,o.duplicate_target_id,o.current_assessment_id,o.current_decision_id,o.revision,o.updated_at);
 -- Normal result writes already advance Audit revisions through their events.
 -- Purge pointer resets need a fence as well, including before row removal.
 UPDATE audits SET revision = revision + 1,
  updated_at = GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
 WHERE audit_id = ANY(affected_audits) AND state = 'deleting';
 PERFORM contractor_advance_finding_correlation(affected_projects);
 RETURN NULL;
END;
$$;
CREATE TRIGGER audit_findings_correlation_updated AFTER UPDATE ON audit_findings
 REFERENCING OLD TABLE AS previous_findings NEW TABLE AS updated_findings
 FOR EACH STATEMENT EXECUTE FUNCTION contractor_finding_correlation_updated();

CREATE FUNCTION contractor_finding_correlation_source_deleted() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 -- Admission requires a Project-bound source Run in the same Project. Its
 -- immutable receipt scope invalidates provenance without enumerating peers.
 PERFORM contractor_advance_finding_correlation(ARRAY(
  SELECT DISTINCT n.project_id FROM updated_receipts AS n JOIN previous_receipts AS o USING(receipt_id)
  WHERE n.project_id IS NOT NULL AND n.source_run_deleted_at IS DISTINCT FROM o.source_run_deleted_at
 ));
 RETURN NULL;
END;
$$;
CREATE TRIGGER finding_receipts_correlation_source_deleted AFTER UPDATE ON finding_proposal_receipts
 REFERENCING OLD TABLE AS previous_receipts NEW TABLE AS updated_receipts
 FOR EACH STATEMENT EXECUTE FUNCTION contractor_finding_correlation_source_deleted();

-- Additional contributors can change candidate evidence without changing the
-- frozen primary group. The first contribution is covered by admission.
CREATE FUNCTION contractor_finding_correlation_contributed() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 UPDATE audits SET revision = revision + 1,
  updated_at = GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
 WHERE audit_id IN (SELECT DISTINCT audit_id FROM inserted_contributions WHERE relation <> 'first');
 PERFORM contractor_advance_finding_correlation(ARRAY(
  SELECT DISTINCT audit.project_id FROM inserted_contributions AS contribution
  JOIN audits AS audit USING(audit_id) WHERE contribution.relation <> 'first'
 ));
 RETURN NULL;
END;
$$;
CREATE TRIGGER finding_contributions_correlation_inserted AFTER INSERT ON audit_finding_contributions
 REFERENCING NEW TABLE AS inserted_contributions FOR EACH STATEMENT EXECUTE FUNCTION contractor_finding_correlation_contributed();

CREATE FUNCTION contractor_finding_correlation_contribution_deleted() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 UPDATE audits SET revision = revision + 1,
  updated_at = GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
 WHERE audit_id IN (SELECT DISTINCT audit_id FROM deleted_contributions WHERE relation <> 'first');
 PERFORM contractor_advance_finding_correlation(ARRAY(
  SELECT DISTINCT audit.project_id FROM deleted_contributions AS contribution
  JOIN audits AS audit USING(audit_id) WHERE contribution.relation <> 'first'
 ));
 RETURN NULL;
END;
$$;
CREATE TRIGGER finding_contributions_correlation_deleted AFTER DELETE ON audit_finding_contributions
 REFERENCING OLD TABLE AS deleted_contributions FOR EACH STATEMENT EXECUTE FUNCTION contractor_finding_correlation_contribution_deleted();
