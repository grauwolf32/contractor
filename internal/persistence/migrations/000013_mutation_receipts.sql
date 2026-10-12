-- One receipt table for every keyed mutation (V370-007, PER-R24).
--
-- A keyed mutation is a request its caller may repeat under an idempotency
-- key. Three tables recorded the same thing under different names:
-- audit_idempotency for the owner's Audit mutations, eval_mutation_receipts
-- for managed Eval mutations and runtime_management_operations for Runtime
-- label binding and Agent label set operations. They fold into
-- mutation_receipts, written only through persistencepostgres.KeyedMutation:
-- the key's advisory lock, the replay of a stored receipt, then the effect and
-- the claim of the key in the same transaction.
--
-- key_scope is what a key is unique within, led by the family:
-- ['audit', owner, operation], ['eval', owner, Project, resource, operation]
-- or ['runtime-management']. Each family keeps its key space, its replay
-- and its retention:
--
-- * An Audit receipt leaves with its Audit (ON DELETE CASCADE); the Eval
--   tombstone trigger still finds an Audit's create receipt before the Audit
--   row goes.
-- * An Eval receipt stays replayable after its experiment's purge until its
--   Project's purge (LIFE-EVM-7), which deletes it under
--   contractor.eval_purge.
-- * A Runtime management receipt holds the digest of the operator's key and
--   is never deleted.
--
-- No receipt changes after its claim. The response is stored as the bytes
-- the replay returns. Two columns lose their place: the Eval receipt's
-- expected revision, which no reader used and the request digest already
-- covers, and the Runtime management actor and time, which no reader used;
-- the binding and the principal keep their own actor and time.

CREATE TABLE mutation_receipts (
    key_scope text[] NOT NULL,
    idempotency_key text NOT NULL,
    operation text NOT NULL,
    request_digest text NOT NULL,
    resource_id text NOT NULL,
    response bytea NOT NULL,
    owner_id text,
    project_id text,
    audit_id text,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    CONSTRAINT mutation_receipts_family_check CHECK (
        CASE key_scope[1]
            WHEN 'audit' THEN
                owner_id IS NOT NULL AND audit_id IS NOT NULL AND project_id IS NULL
                AND key_scope = ARRAY['audit', owner_id, operation]
                AND idempotency_key ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'
            WHEN 'eval' THEN
                owner_id IS NOT NULL AND project_id IS NOT NULL AND audit_id IS NULL
                AND key_scope = ARRAY['eval', owner_id, project_id, resource_id, operation]
            WHEN 'runtime-management' THEN
                owner_id IS NULL AND project_id IS NULL AND audit_id IS NULL
                AND key_scope = ARRAY['runtime-management']
                AND idempotency_key ~ '^sha256:[0-9a-f]{64}$'
            ELSE false
        END
    ),
    CONSTRAINT mutation_receipts_idempotency_key_check CHECK (
        octet_length(idempotency_key) BETWEEN 1 AND 128
    ),
    CONSTRAINT mutation_receipts_operation_check CHECK (
        operation ~ '^[a-z][a-z0-9_.-]{0,127}$'
    ),
    CONSTRAINT mutation_receipts_request_digest_check CHECK (
        request_digest ~ '^sha256:[0-9a-f]{64}$'
    ),
    CONSTRAINT mutation_receipts_resource_id_check CHECK (
        btrim(resource_id) <> '' AND octet_length(resource_id) <= 1024
    ),
    CONSTRAINT mutation_receipts_response_check CHECK (
        octet_length(response) BETWEEN 2 AND 1048576
    ),
    CONSTRAINT mutation_receipts_owner_id_check CHECK (
        owner_id IS NULL OR (btrim(owner_id) <> '' AND octet_length(owner_id) <= 256)
    )
);

ALTER TABLE mutation_receipts
    ADD CONSTRAINT mutation_receipts_pkey PRIMARY KEY (key_scope, idempotency_key);

ALTER TABLE mutation_receipts
    ADD CONSTRAINT mutation_receipts_project_id_fkey FOREIGN KEY (project_id)
        REFERENCES projects(project_id) ON DELETE RESTRICT;

ALTER TABLE mutation_receipts
    ADD CONSTRAINT mutation_receipts_audit_id_fkey FOREIGN KEY (audit_id)
        REFERENCES audits(audit_id) ON DELETE CASCADE;

CREATE INDEX mutation_receipts_project_idx ON mutation_receipts USING btree (project_id)
    WHERE project_id IS NOT NULL;

CREATE INDEX mutation_receipts_audit_idx ON mutation_receipts USING btree (audit_id)
    WHERE audit_id IS NOT NULL;

-- An Eval mutation addressed by its resource finds its Project through its
-- receipt after the experiment's purge (EVM-R226).
CREATE INDEX mutation_receipts_owner_resource_idx ON mutation_receipts
    USING btree (owner_id, resource_id, operation, idempotency_key)
    WHERE owner_id IS NOT NULL;

INSERT INTO mutation_receipts (
    key_scope, idempotency_key, operation, request_digest, resource_id, response,
    owner_id, project_id, audit_id, created_at
)
SELECT ARRAY['audit', owner_id, operation], idempotency_key, operation, request_digest,
       resource_id, convert_to(response_snapshot::text, 'UTF8'),
       owner_id, NULL, audit_id, created_at
  FROM audit_idempotency;

INSERT INTO mutation_receipts (
    key_scope, idempotency_key, operation, request_digest, resource_id, response,
    owner_id, project_id, audit_id, created_at
)
SELECT ARRAY['eval', owner_id, project_id, resource_id, operation], operation_key, operation,
       request_sha256, resource_id, response, owner_id, project_id, NULL, created_at
  FROM eval_mutation_receipts;

INSERT INTO mutation_receipts (
    key_scope, idempotency_key, operation, request_digest, resource_id, response,
    owner_id, project_id, audit_id, created_at
)
SELECT ARRAY['runtime-management'], idempotency_key_digest, operation_kind, request_digest,
       resource_id, convert_to(result::text, 'UTF8'), NULL, NULL, NULL, performed_at
  FROM runtime_management_operations;

CREATE FUNCTION contractor_protect_mutation_receipt() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    -- An Audit receipt leaves with its Audit; an Eval receipt only in its
    -- Project's purge. No receipt changes.
    IF TG_OP = 'DELETE' AND (
        OLD.key_scope[1] = 'audit'
        OR (OLD.key_scope[1] = 'eval' AND current_setting('contractor.eval_purge', true) = 'on')
    ) THEN
        RETURN OLD;
    END IF;
    RAISE EXCEPTION 'mutation receipt is immutable' USING ERRCODE = '23514';
END; $$;

CREATE TRIGGER mutation_receipts_protect_immutable
    BEFORE DELETE OR UPDATE ON mutation_receipts
    FOR EACH ROW EXECUTE FUNCTION contractor_protect_mutation_receipt();

CREATE OR REPLACE FUNCTION contractor_eval_retain_deleted_execution() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
DECLARE bound_experiment text; bound_member text; kind text; execution text; outcome text; never_started boolean;
BEGIN
    IF TG_TABLE_NAME='workflow_runs' THEN
        SELECT op.experiment_id,op.member_id INTO bound_experiment,bound_member
        FROM eval_suboperations op JOIN eval_experiments e USING(experiment_id)
        WHERE op.kind='run-create' AND op.operation_key=OLD.request_idempotency_key
          AND e.owner_id=OLD.owner_id AND e.project_id=OLD.project_id AND op.request_sha256=OLD.request_digest;
        kind:='run';execution:=OLD.run_id;outcome:=OLD.state;never_started:=false;
    ELSE
        SELECT op.experiment_id,op.member_id INTO bound_experiment,bound_member
        FROM mutation_receipts i JOIN eval_suboperations op ON op.operation_key=i.idempotency_key
        JOIN eval_experiments e ON e.experiment_id=op.experiment_id
        JOIN eval_submissions s ON s.experiment_id=op.experiment_id AND s.member_id=op.member_id
        WHERE i.audit_id=OLD.audit_id AND i.operation='audit.create' AND op.kind='audit-create'
          AND e.owner_id=OLD.owner_id AND i.owner_id=OLD.owner_id AND s.execution_project_id=OLD.project_id AND op.request_sha256=i.request_digest;
        kind:='audit';execution:=OLD.audit_id;never_started:=(OLD.started_at IS NULL);
        -- Audit deletion can replace its old terminal state with deleting.
        -- Drain is confirmed by ordinary purge, but quality remains unknown.
        outcome:=CASE WHEN OLD.state IN ('completed','cancelled','failed') THEN OLD.state ELSE 'unknown' END;
    END IF;
    IF bound_experiment IS NOT NULL THEN
        INSERT INTO eval_execution_tombstones(experiment_id,member_id,execution_kind,execution_id,terminal_state,never_started)
        VALUES(bound_experiment,bound_member,kind,execution,outcome,never_started);
        UPDATE eval_experiments SET view_generation=view_generation+1,
            updated_at=GREATEST(clock_timestamp(),updated_at+interval '1 microsecond')
        WHERE experiment_id=bound_experiment;
    END IF;
    RETURN OLD;
END; $$;

DROP TABLE audit_idempotency;
DROP TABLE eval_mutation_receipts;
DROP TABLE runtime_management_operations;
