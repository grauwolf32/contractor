-- Drop indexes and duplicate UNIQUE constraints that no reader needs, and
-- align the Project deletion claim index with the claim's order (V409-007).
--
-- Every change below was measured before and after on disposable fixtures,
-- in the custom and the generic plan, for each production statement that
-- reads the table with a matching predicate; the decisions of task V409-007
-- record the fixtures and the row counts. Afterwards each of those statements
-- reads the same rows through another index, or one more, the Run or Stage it
-- names, and the deletion claim reads far fewer. The other reviewed
-- candidates stay because a reader still needs them. Nothing here changes a
-- foreign key, its referenced unique index or its actions, and no statement
-- uses CASCADE: a DROP that a dependent object still needed would fail
-- instead.

-- No statement reads Run events by occurred_at. The event page walks the
-- primary key (run_id, sequence_number), which also bounds the probe of the
-- event-to-Run foreign key. The UNIQUE event_id constraint stays.
DROP INDEX workflow_run_events_time_idx;

-- No statement applies an operator to runtime_labels; the column is only
-- stored, returned and checked.
DROP INDEX workflow_runs_runtime_labels_idx;

-- Every statement that tests publication_mode also selects one Run by its ID,
-- so the primary key or another Run identity index bounds it.
DROP INDEX workflow_runs_publication_mode_idx;

-- Every active-Stage statement names its Run and reads
-- stage_executions_run_active_idx, which holds the same active states.
DROP INDEX stage_executions_recovery_idx;

-- (owner_id, experiment_id): every statement that names both reads the one
-- experiment through the primary key. The Project, state, dataset and
-- control-mode page indexes stay, because their filters read far fewer rows
-- with them.
DROP INDEX eval_experiments_owner_page_idx;

-- (project_id, owner_id) duplicates projects_owner_identity (owner_id,
-- project_id), the unique index the four Project-owner foreign keys
-- reference, and project_id alone is the primary key.
ALTER TABLE projects DROP CONSTRAINT projects_project_owner_key;

-- (audit_id, project_id, owner_id) is unique because audit_id is the primary
-- key; no foreign key references it.
ALTER TABLE audits DROP CONSTRAINT audits_project_owner_identity;

-- The deletion claim takes the oldest claimable deleting Project in
-- (deletion_requested_at, project_id) order. The index led by the claim
-- expiry could serve neither that order nor the claim predicate, so each claim
-- read and sorted every deleting Project. In claim order a claim reads the
-- held claims ahead of the first claimable Project and that Project. Without
-- the claim columns, a claim, its renewal and its release no longer change
-- an indexed column of the row.
DROP INDEX projects_deletion_claim_idx;
CREATE INDEX projects_deletion_claim_idx ON projects USING btree (deletion_requested_at, project_id)
    WHERE (lifecycle_state = 'deleting'::text);
