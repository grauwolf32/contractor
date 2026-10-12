-- Bound the binding revisions of one RunScope (V370-024, LIM-ART-6).
--
-- Run deletion purges a RunScope in one transaction (DEL-R22), so the size of
-- the Run is what bounds that transaction. Every binding, candidate version
-- and owned lineage edge of a Run hangs off one of its binding revisions, so
-- at most 50,000 revisions outside the Scheduler-managed outputs Namespace
-- bound all of them. outputs revisions are bound by the Workflow's declared
-- outputs and Stage attempts instead, so a Run that exhausted its limit can
-- still bind its outputs.
--
-- artifact_scopes.revision_count counts those revisions of a RunScope.
-- Revisions are never deleted one by one: the purge deletes the scope row
-- together with them, so the counter never goes down. The statement trigger
-- adds a statement's revisions to their scope; that UPDATE holds the scope
-- row lock until commit, so concurrent writers of one Run serialize there and
-- the limit is exact. The lifecycle purge's FOR UPDATE on the same row keeps
-- writers and purge apart, as before. Other scopes are not counted and stay 0.

ALTER TABLE artifact_scopes
    ADD COLUMN revision_count bigint DEFAULT 0 NOT NULL,
    ADD CONSTRAINT artifact_scopes_revision_count_check CHECK (
        revision_count >= 0 AND (scope_kind = 'run' OR revision_count = 0)
    );

UPDATE artifact_scopes AS scope
   SET revision_count = counted.revisions
  FROM (
      SELECT scope_id, count(*) AS revisions
        FROM artifact_binding_revisions
       WHERE scope_kind = 'run' AND namespace <> 'outputs'
       GROUP BY scope_id
  ) AS counted
 WHERE scope.scope_kind = 'run' AND scope.scope_id = counted.scope_id;

CREATE FUNCTION contractor_count_run_artifact_revisions() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
DECLARE
    exceeded boolean;
BEGIN
    WITH added AS (
        SELECT scope_id, count(*) AS revisions
          FROM inserted_revisions
         WHERE scope_kind = 'run' AND namespace <> 'outputs'
         GROUP BY scope_id
    ), counted AS (
        UPDATE artifact_scopes AS scope
           SET revision_count = scope.revision_count + added.revisions
          FROM added
         WHERE scope.scope_kind = 'run' AND scope.scope_id = added.scope_id
        RETURNING scope.revision_count
    )
    SELECT COALESCE(bool_or(revision_count > 50000), false) INTO exceeded FROM counted;
    IF exceeded THEN
        RAISE EXCEPTION 'Run artifact revision limit exceeded'
            USING ERRCODE = '54000', CONSTRAINT = 'artifact_scopes_run_revision_limit';
    END IF;
    RETURN NULL;
END;
$$;

CREATE TRIGGER artifact_binding_revisions_run_limit
    AFTER INSERT ON artifact_binding_revisions
    REFERENCING NEW TABLE AS inserted_revisions
    FOR EACH STATEMENT EXECUTE FUNCTION contractor_count_run_artifact_revisions();
