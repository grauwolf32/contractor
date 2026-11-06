-- Fold the one-to-one private plan and current selection into their owners.
-- Exact bytes and attribution survive; operational readers use only the new schema.
ALTER TABLE eval_experiments
    ADD COLUMN plan_kind text,
    ADD COLUMN plan_document bytea,
    ADD COLUMN plan_document_sha256 text GENERATED ALWAYS AS
        (CASE WHEN plan_document IS NOT NULL THEN 'sha256:' || encode(sha256(plan_document), 'hex') END) STORED,
    ADD COLUMN plan_sha256 text,
    ADD COLUMN plan_setup bytea,
    ADD COLUMN plan_created_at timestamptz,
    ADD COLUMN plan_readiness bytea;

-- Backfill is not an experiment observation or authority edit.
ALTER TABLE eval_experiments DISABLE TRIGGER eval_experiments_protect;
UPDATE eval_experiments e
SET plan_kind=p.document_kind, plan_document=p.document, plan_sha256=p.plan_sha256,
    plan_setup=p.setup, plan_created_at=p.created_at, plan_readiness=p.readiness
FROM eval_frozen_plans p WHERE p.experiment_id=e.experiment_id;
ALTER TABLE eval_experiments ENABLE TRIGGER eval_experiments_protect;

ALTER TABLE eval_experiments
    ADD CONSTRAINT eval_experiments_plan_shape CHECK (
        (plan_document IS NULL AND plan_document_sha256 IS NULL AND plan_kind IS NULL AND plan_sha256 IS NULL
         AND plan_setup IS NULL AND plan_created_at IS NULL AND plan_readiness IS NULL)
        OR (plan_document IS NOT NULL AND plan_document_sha256 IS NOT NULL AND plan_kind IS NOT NULL AND plan_sha256 IS NOT NULL
            AND plan_setup IS NOT NULL AND plan_created_at IS NOT NULL
            AND ((plan_kind='playground.plan/v1') = (plan_readiness IS NOT NULL)))),
    ADD CONSTRAINT eval_experiments_plan_kind CHECK (plan_kind IN ('playground.plan/v1','ExternalRegistration')),
    ADD CONSTRAINT eval_experiments_plan_sha256 CHECK (plan_sha256 ~ '^sha256:[0-9a-f]{64}$'),
    ADD CONSTRAINT eval_experiments_plan_document_check CHECK (octet_length(plan_document) BETWEEN 2 AND 1048576),
    ADD CONSTRAINT eval_experiments_plan_setup_check CHECK (octet_length(plan_setup) BETWEEN 2 AND 1048576),
    ADD CONSTRAINT eval_experiments_plan_readiness_check CHECK (octet_length(plan_readiness) BETWEEN 2 AND 1048576);

ALTER TABLE eval_members
    ADD COLUMN selected_result_kind text GENERATED ALWAYS AS ('result'::text) STORED,
    ADD COLUMN selected_result_sha256 text,
    ADD COLUMN selected_assessment_kind text GENERATED ALWAYS AS ('assessment'::text) STORED,
    ADD COLUMN selected_assessment_sha256 text,
    ADD COLUMN selection_revision bigint NOT NULL DEFAULT 0,
    ADD COLUMN selected_by text;
ALTER TABLE eval_members DISABLE TRIGGER eval_members_immutable;
UPDATE eval_members m
SET selected_result_sha256=s.result_sha256, selected_assessment_sha256=s.assessment_sha256,
    selection_revision=s.revision, selected_by=s.actor_id
FROM eval_selections s WHERE s.experiment_id=m.experiment_id AND s.member_id=m.member_id;
ALTER TABLE eval_members ENABLE TRIGGER eval_members_immutable;
ALTER TABLE eval_members
    ADD CONSTRAINT eval_members_selection_shape CHECK (
        (selected_result_sha256 IS NULL AND selected_assessment_sha256 IS NULL AND selection_revision=0 AND selected_by IS NULL)
        OR (selected_result_sha256 IS NOT NULL AND selection_revision>0 AND selected_by IS NOT NULL)),
    ADD CONSTRAINT eval_members_selected_result_fk
        FOREIGN KEY (experiment_id,member_id,selected_result_kind,selected_result_sha256)
        REFERENCES eval_records(experiment_id,member_id,kind,record_sha256),
    ADD CONSTRAINT eval_members_selected_assessment_fk
        FOREIGN KEY (experiment_id,member_id,selected_assessment_kind,selected_assessment_sha256)
        REFERENCES eval_records(experiment_id,member_id,kind,record_sha256);
CREATE INDEX eval_members_selected_idx ON eval_members(experiment_id,member_id)
    WHERE selected_result_sha256 IS NOT NULL;

-- Retain immutable matrix protection without blocking current-selection updates.
DROP TRIGGER eval_members_immutable ON eval_members;
CREATE TRIGGER eval_members_immutable BEFORE DELETE OR UPDATE OF experiment_id, member_id, pair_id, ordinal, suite_id, case_id, sample, variant_id, case_sha256, binding_sha256, eligibility, execution_kind, recipe, submission_key, eligibility_reason
ON eval_members FOR EACH ROW EXECUTE FUNCTION contractor_eval_immutable();
CREATE TRIGGER eval_members_selection_projection AFTER UPDATE OF selected_result_sha256, selected_assessment_sha256, selection_revision, selected_by
ON eval_members FOR EACH ROW WHEN (NEW.selected_result_sha256 IS NOT NULL)
EXECUTE FUNCTION contractor_eval_member_changed();

-- Members and resources still require a frozen plan, now on the parent row.
ALTER TABLE eval_members DROP CONSTRAINT eval_members_experiment_id_fkey;
ALTER TABLE eval_members ADD CONSTRAINT eval_members_experiment_id_fkey
    FOREIGN KEY (experiment_id) REFERENCES eval_experiments(experiment_id) ON DELETE CASCADE;
ALTER TABLE eval_plan_resources DROP CONSTRAINT eval_plan_resources_experiment_id_fkey;
ALTER TABLE eval_plan_resources ADD CONSTRAINT eval_plan_resources_experiment_id_fkey
    FOREIGN KEY (experiment_id) REFERENCES eval_experiments(experiment_id) ON DELETE CASCADE;
CREATE FUNCTION contractor_eval_require_frozen_plan() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM eval_experiments WHERE experiment_id=NEW.experiment_id AND plan_document IS NOT NULL) THEN
        RAISE EXCEPTION 'Evaluation plan is not frozen' USING ERRCODE='23514';
    END IF;
    RETURN NEW;
END; $$;
-- Combine the existing matrix-bound check with frozen-parent authority, so
-- member insertion keeps one parent lookup per row.
CREATE OR REPLACE FUNCTION contractor_eval_member_bound() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM eval_experiments WHERE experiment_id=NEW.experiment_id
                   AND plan_document IS NOT NULL AND NEW.ordinal<expected_count) THEN
        RAISE EXCEPTION 'Evaluation member is outside frozen matrix' USING ERRCODE='23514';
    END IF;
    RETURN NEW;
END; $$;
CREATE TRIGGER eval_resources_frozen_plan BEFORE INSERT ON eval_plan_resources
FOR EACH ROW EXECUTE FUNCTION contractor_eval_require_frozen_plan();

-- Both guards only check and raise: AFTER UPDATE still rejects the whole
-- statement transactionally, without BEFORE UPDATE triggers forcing the stored
-- generated digest to rehash an unchanged document on every progress update.
DROP TRIGGER eval_experiments_protect ON eval_experiments;
CREATE TRIGGER eval_experiments_protect AFTER UPDATE ON eval_experiments
FOR EACH ROW EXECUTE FUNCTION contractor_eval_protect_experiment();

CREATE TRIGGER eval_plans_immutable AFTER UPDATE OF plan_kind,plan_document,plan_document_sha256,plan_sha256,plan_setup,plan_created_at,plan_readiness
ON eval_experiments FOR EACH ROW WHEN (OLD.plan_document IS NOT NULL)
EXECUTE FUNCTION contractor_eval_immutable();
CREATE CONSTRAINT TRIGGER eval_plans_complete_matrix AFTER INSERT OR UPDATE OF plan_document ON eval_experiments
DEFERRABLE INITIALLY DEFERRED FOR EACH ROW WHEN (NEW.plan_document IS NOT NULL)
EXECUTE FUNCTION contractor_eval_complete_matrix();

CREATE OR REPLACE FUNCTION contractor_eval_protect_experiment() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF ROW(NEW.experiment_id,NEW.owner_id,NEW.project_id,NEW.portable_id,NEW.control_mode,NEW.created_at)
       IS DISTINCT FROM ROW(OLD.experiment_id,OLD.owner_id,OLD.project_id,OLD.portable_id,OLD.control_mode,OLD.created_at)
       OR NEW.revision NOT IN (OLD.revision, OLD.revision + 1)
       OR (NEW.revision = OLD.revision
           AND ROW(NEW.name,NEW.draft,NEW.dataset_id,NEW.dataset_revision,NEW.max_in_flight,
                   NEW.wall_ms,NEW.token_limit,NEW.deletion_requested_at)
               IS DISTINCT FROM
               ROW(OLD.name,OLD.draft,OLD.dataset_id,OLD.dataset_revision,OLD.max_in_flight,
                   OLD.wall_ms,OLD.token_limit,OLD.deletion_requested_at))
       OR NEW.updated_at <= OLD.updated_at
       OR NEW.observed_tokens < OLD.observed_tokens
       OR (OLD.started_at IS NOT NULL AND ROW(NEW.started_at,NEW.deadline_at) IS DISTINCT FROM ROW(OLD.started_at,OLD.deadline_at))
       OR (OLD.deletion_requested_at IS NOT NULL AND NEW.deletion_requested_at IS DISTINCT FROM OLD.deletion_requested_at)
       OR (OLD.plan_document IS NOT NULL
           AND ROW(NEW.draft,NEW.dataset_id,NEW.dataset_revision,NEW.expected_count,NEW.max_in_flight,NEW.wall_ms,NEW.token_limit)
               IS DISTINCT FROM ROW(OLD.draft,OLD.dataset_id,OLD.dataset_revision,OLD.expected_count,OLD.max_in_flight,OLD.wall_ms,OLD.token_limit))
    THEN RAISE EXCEPTION 'Evaluation revision or immutable fields changed' USING ERRCODE = '23514'; END IF;
    RETURN NEW;
END; $$;


DROP TABLE eval_selections;
DROP TABLE eval_frozen_plans;
