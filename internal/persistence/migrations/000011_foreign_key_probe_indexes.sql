-- Index the foreign keys that deletions probe (V370-023).
--
-- For every deleted or re-keyed referenced row PostgreSQL checks or cascades
-- a foreign key by querying the referencing table on the key columns. Without
-- an index led by those columns each probe reads the whole table, or the whole
-- of an index that holds the key in a later column: an Audit purge read every
-- entry of an audit_executions index once for the Audit and once per Round,
-- whichever Audit it belonged to, and scanned audit_findings once per deleted
-- assessment; audit_review_requests and audit_review_decisions had the same
-- gap for each deleted finding and Round. Every foreign key now has such an
-- index or an allowlist entry in TestForeignKeysHaveProbeIndexes that names
-- the index or trigger that keeps its probe cheap. Partial indexes cover only
-- rows whose key is set, the only rows a probe can match.

CREATE INDEX audit_executions_audit_round_idx ON audit_executions USING btree (audit_id, round_id);

CREATE INDEX audit_finding_assessments_finding_idx ON audit_finding_assessments USING btree (finding_id, audit_id);

CREATE INDEX audit_findings_current_assessment_idx ON audit_findings USING btree (current_assessment_id)
    WHERE (current_assessment_id IS NOT NULL);

CREATE INDEX audit_findings_current_decision_idx ON audit_findings USING btree (current_decision_id)
    WHERE (current_decision_id IS NOT NULL);

CREATE INDEX audit_findings_duplicate_target_idx ON audit_findings USING btree (duplicate_target_id, audit_id)
    WHERE (duplicate_target_id IS NOT NULL);

CREATE INDEX audit_review_decisions_duplicate_target_idx ON audit_review_decisions USING btree (duplicate_target_id, audit_id)
    WHERE (duplicate_target_id IS NOT NULL);

CREATE INDEX audit_review_requests_finding_idx ON audit_review_requests USING btree (finding_id, audit_id)
    WHERE (finding_id IS NOT NULL);

CREATE INDEX audit_review_requests_report_round_idx ON audit_review_requests USING btree (report_round_id, audit_id)
    WHERE (report_round_id IS NOT NULL);

CREATE INDEX eval_commands_experiment_idx ON eval_commands USING btree (experiment_id);

CREATE INDEX eval_experiments_dataset_revision_idx ON eval_experiments USING btree (project_id, dataset_id, dataset_revision)
    WHERE (dataset_id IS NOT NULL);

CREATE INDEX gateway_allocation_routes_route_idx ON gateway_allocation_routes USING btree (route_key);

CREATE INDEX gateway_run_routes_route_idx ON gateway_run_routes USING btree (route_key);
