-- Automatic bounded history retention of every logical Artifact (ART-R123,
-- ART-R124, DEL-R62 to DEL-R65, LIM-ART-9). The first part adds and
-- backfills the retained-history counters of each binding; the second adds
-- the exact-reference holds of each revision and the trusted retention
-- delete exception.

-- Retained-history accounting of every logical Artifact binding (ART-R123,
-- LIM-ART-9). Each binding records how many revisions it retains and their
-- full encoded size: every retained revision is charged the size of its blob,
-- also when the version or blob is shared. ArtifactStore writes and forks keep
-- both counters in the statement that inserts the revision; this migration
-- backfills them from the stored revisions. Nothing here removes a revision.

ALTER TABLE artifact_bindings
    ADD COLUMN retained_revisions bigint,
    ADD COLUMN retained_bytes bigint;

-- The admission guard of Project bindings rejects every update of a binding
-- whose Project is deleting; the backfill changes no admitted content, so it
-- runs without that guard. No other row trigger of artifact_bindings fires for
-- these columns.
ALTER TABLE artifact_bindings DISABLE TRIGGER artifact_bindings_project_admission;

UPDATE artifact_bindings AS binding
SET retained_revisions = retained.revisions,
    retained_bytes = retained.bytes
FROM (
    SELECT revision.scope_kind, revision.scope_id, revision.namespace, revision.name,
           count(*) AS revisions, sum(blob.size_bytes) AS bytes
    FROM artifact_binding_revisions AS revision
    JOIN artifact_versions AS version ON version.version_id = revision.version_id
    JOIN artifact_blobs AS blob ON blob.sha256 = version.blob_sha256
    GROUP BY revision.scope_kind, revision.scope_id, revision.namespace, revision.name
) AS retained
WHERE binding.scope_kind = retained.scope_kind AND binding.scope_id = retained.scope_id
  AND binding.namespace = retained.namespace AND binding.name = retained.name;

ALTER TABLE artifact_bindings ENABLE TRIGGER artifact_bindings_project_admission;

-- A binding always retains at least its current revision, so a binding the
-- backfill left without counters has no stored revision: corrupt data that
-- must stop the migration rather than be accounted as empty.
ALTER TABLE artifact_bindings
    ALTER COLUMN retained_revisions SET NOT NULL,
    ALTER COLUMN retained_bytes SET NOT NULL,
    ADD CONSTRAINT artifact_bindings_retained_revisions_check CHECK (retained_revisions >= 1),
    ADD CONSTRAINT artifact_bindings_retained_bytes_check CHECK (retained_bytes >= 0);

-- Exact-reference holds of Artifact revisions and the trusted retention
-- delete exception (DEL-R62 to DEL-R65, ART-R124). Every holder that protects
-- an exact revision changes artifact_binding_revisions.hold_count in the
-- transaction that records or releases its reference, so automatic history
-- retention finds the revisions it may remove through one partial index
-- instead of rescanning holder documents. A revision is removed only by the
-- ArtifactStore while contractor.artifact_retention is set to 'evict' for
-- the transaction, only when no holder protects it, and never while it is the
-- current revision of its binding.
-- Controlled scope purge keeps its own exception and ignores holds.

ALTER TABLE artifact_binding_revisions
    ADD COLUMN hold_count integer NOT NULL DEFAULT 0
    CONSTRAINT artifact_binding_revisions_hold_count_check CHECK (hold_count >= 0);

-- A revision key is the JSON array [scope kind, scope ID, namespace, name,
-- revision]; a key set is a JSON array of distinct keys in key order.

-- contractor_artifact_retention_enabled reports whether the transaction runs
-- the ArtifactStore's automatic history retention.
CREATE FUNCTION contractor_artifact_retention_enabled() RETURNS boolean
    LANGUAGE sql STABLE AS $$
SELECT COALESCE(current_setting('contractor.artifact_retention', true), '') = 'evict'
$$;

-- contractor_artifact_ref_key returns the revision key of the exact ArtifactRef
-- object ref in scope (scope_kind, scope_id), or NULL when the scope is
-- unknown or ref is not an object with a string namespace, name and revision.
CREATE FUNCTION contractor_artifact_ref_key(scope_kind text, scope_id text, ref jsonb)
    RETURNS jsonb
    LANGUAGE sql IMMUTABLE AS $$
SELECT CASE
    WHEN scope_kind IS NOT NULL AND scope_id IS NOT NULL
     AND jsonb_typeof(ref -> 'namespace') = 'string' AND jsonb_typeof(ref -> 'name') = 'string'
     AND jsonb_typeof(ref -> 'revision') = 'string'
    THEN jsonb_build_array(scope_kind, scope_id, ref -> 'namespace', ref -> 'name', ref -> 'revision')
END
$$;

-- contractor_jsonb_values returns the member values of an object and nothing
-- for any other value.
CREATE FUNCTION contractor_jsonb_values(value jsonb) RETURNS SETOF jsonb
    LANGUAGE sql IMMUTABLE AS $$
SELECT member.value
FROM jsonb_each(CASE WHEN jsonb_typeof(value) = 'object' THEN value ELSE '{}'::jsonb END) AS member
$$;

-- contractor_jsonb_elements returns the elements of an array and nothing for
-- any other value.
CREATE FUNCTION contractor_jsonb_elements(value jsonb) RETURNS SETOF jsonb
    LANGUAGE sql IMMUTABLE AS $$
SELECT element
FROM jsonb_array_elements(CASE WHEN jsonb_typeof(value) = 'array' THEN value ELSE '[]'::jsonb END)
    AS element
$$;

-- contractor_artifact_key_set returns the distinct complete keys of keys in
-- key order. A holder protects each revision it names once.
CREATE FUNCTION contractor_artifact_key_set(keys jsonb[])
    RETURNS jsonb
    LANGUAGE sql IMMUTABLE AS $$
SELECT COALESCE(jsonb_agg(DISTINCT key ORDER BY key), '[]'::jsonb)
FROM unnest(keys) AS key
WHERE key IS NOT NULL
$$;

-- contractor_artifact_hold_apply adds one hold to every revision of acquired
-- and removes one from every revision of released, counting a key as often
-- as it occurs. It locks the changed revisions in key order first, so
-- concurrent holders and retention, which locks its victims in the same
-- order, cannot deadlock on them. A strict acquisition of a revision that
-- does not exist in a scope that does, for example one that retention
-- removed after the holder read it, fails with a foreign key violation of
-- artifact_exact_hold (DEL-R63). A reference into a scope that no longer
-- exists holds nothing, because that scope's purge removed every revision it
-- could name; the backfill below treats a stored one the same way. A lenient
-- acquisition, for a holder whose contract records a reference without
-- proving it (a Run's or an Audit's Skill sources or a Planner session),
-- protects nothing when the revision is missing. A release
-- of a revision that a scope purge already removed changes nothing.
CREATE FUNCTION contractor_artifact_hold_apply(acquired jsonb, released jsonb, strict boolean)
    RETURNS void
    LANGUAGE plpgsql AS $$
DECLARE
    kinds text[];
    ids text[];
    namespaces text[];
    names text[];
    revisions text[];
    deltas integer[];
    missing bigint;
BEGIN
    SELECT COALESCE(array_agg(change.scope_kind ORDER BY change.scope_kind, change.scope_id, change.namespace,
                                  change.name, change.revision), '{}'),
           COALESCE(array_agg(change.scope_id ORDER BY change.scope_kind, change.scope_id, change.namespace,
                                  change.name, change.revision), '{}'),
           COALESCE(array_agg(change.namespace ORDER BY change.scope_kind, change.scope_id, change.namespace,
                                  change.name, change.revision), '{}'),
           COALESCE(array_agg(change.name ORDER BY change.scope_kind, change.scope_id, change.namespace,
                                  change.name, change.revision), '{}'),
           COALESCE(array_agg(change.revision ORDER BY change.scope_kind, change.scope_id, change.namespace,
                                  change.name, change.revision), '{}'),
           COALESCE(array_agg(change.delta ORDER BY change.scope_kind, change.scope_id, change.namespace,
                                  change.name, change.revision), '{}')
    INTO kinds, ids, namespaces, names, revisions, deltas
    FROM (
        SELECT key.scope_kind, key.scope_id, key.namespace, key.name, key.revision,
               sum(key.delta)::integer AS delta
        FROM (
            SELECT held ->> 0 AS scope_kind, held ->> 1 AS scope_id, held ->> 2 AS namespace,
                   held ->> 3 AS name, held ->> 4 AS revision, 1 AS delta
            FROM jsonb_array_elements(acquired) AS held
            UNION ALL
            SELECT freed ->> 0, freed ->> 1, freed ->> 2, freed ->> 3, freed ->> 4, -1
            FROM jsonb_array_elements(released) AS freed
        ) AS key
        WHERE key.scope_kind IS NOT NULL AND key.scope_id IS NOT NULL AND key.namespace IS NOT NULL
          AND key.name IS NOT NULL AND key.revision IS NOT NULL
        GROUP BY key.scope_kind, key.scope_id, key.namespace, key.name, key.revision
        HAVING sum(key.delta) <> 0
    ) AS change;
    IF cardinality(deltas) = 0 THEN
        RETURN;
    END IF;
    PERFORM 1
    FROM artifact_binding_revisions AS revision
    JOIN unnest(kinds, ids, namespaces, names, revisions)
        AS change(scope_kind, scope_id, namespace, name, revision)
      ON (revision.scope_kind, revision.scope_id, revision.namespace, revision.name, revision.revision)
       = (change.scope_kind, change.scope_id, change.namespace, change.name, change.revision)
    ORDER BY revision.scope_kind, revision.scope_id, revision.namespace, revision.name, revision.revision
    FOR NO KEY UPDATE OF revision;
    UPDATE artifact_binding_revisions AS revision
    SET hold_count = revision.hold_count + change.delta
    FROM unnest(kinds, ids, namespaces, names, revisions, deltas)
        AS change(scope_kind, scope_id, namespace, name, revision, delta)
    WHERE (revision.scope_kind, revision.scope_id, revision.namespace, revision.name, revision.revision)
        = (change.scope_kind, change.scope_id, change.namespace, change.name, change.revision);
    IF strict THEN
        SELECT count(*) INTO missing
        FROM unnest(kinds, ids, namespaces, names, revisions, deltas)
            AS change(scope_kind, scope_id, namespace, name, revision, delta)
        WHERE change.delta > 0
          AND EXISTS (
              SELECT 1 FROM artifact_scopes AS scope
              WHERE scope.scope_kind = change.scope_kind AND scope.scope_id = change.scope_id
          )
          AND NOT EXISTS (
              SELECT 1 FROM artifact_binding_revisions AS revision
              WHERE (revision.scope_kind, revision.scope_id, revision.namespace, revision.name,
                     revision.revision)
                  = (change.scope_kind, change.scope_id, change.namespace, change.name, change.revision)
          );
        IF missing > 0 THEN
            RAISE EXCEPTION 'exact Artifact revision to hold does not exist'
                USING ERRCODE = '23503', CONSTRAINT = 'artifact_exact_hold';
        END IF;
    END IF;
END;
$$;

-- Holder documents. Each function returns the revisions one stored row
-- protects, as a key set; the JSON members are those the Go types encode
-- (internal/persistence/snapshot_json_paths_test.go).

-- A StageExecution protects, in its Run's scope, the artifacts its
-- StageContext pins, the continuation's summary and artifacts, and the
-- artifacts and summary of its candidate and accepted StageResults.
CREATE FUNCTION contractor_stage_execution_artifact_holds(
    run_id text, context jsonb, candidate jsonb, accepted jsonb
) RETURNS jsonb
    LANGUAGE sql IMMUTABLE AS $$
SELECT contractor_artifact_key_set(ARRAY(
    SELECT contractor_artifact_ref_key('run', run_id, pin -> 'artifact')
    FROM contractor_jsonb_values(context -> 'artifacts') AS pin
    UNION ALL
    SELECT contractor_artifact_ref_key('run', run_id, context -> 'continuation' -> 'summaryArtifact')
    UNION ALL
    SELECT contractor_artifact_ref_key('run', run_id, slot)
    FROM contractor_jsonb_values(context -> 'continuation' -> 'artifacts') AS slot
    UNION ALL
    SELECT contractor_artifact_ref_key('run', run_id, result.value -> 'summaryArtifact')
    FROM (VALUES (candidate), (accepted)) AS result(value)
    UNION ALL
    SELECT contractor_artifact_ref_key('run', run_id, slot)
    FROM (VALUES (candidate), (accepted)) AS result(value),
         contractor_jsonb_values(result.value -> 'artifacts') AS slot
))
$$;

-- A Planner session's state names, in its Run's scope, the artifacts and
-- summary of its completed result and its scan plan, job inputs and job
-- reports. The scope comes from the session's StageExecution.
-- The derived references are revision keys whose scope ID is empty.
CREATE FUNCTION contractor_planner_session_artifact_refs(state jsonb)
    RETURNS jsonb
    LANGUAGE sql IMMUTABLE AS $$
SELECT contractor_artifact_key_set(ARRAY(
    SELECT contractor_artifact_ref_key('run', '', state -> 'completion' -> 'result' -> 'summaryArtifact')
    UNION ALL
    SELECT contractor_artifact_ref_key('run', '', slot)
    FROM contractor_jsonb_values(state -> 'completion' -> 'result' -> 'artifacts') AS slot
    UNION ALL
    SELECT contractor_artifact_ref_key('run', '', state -> 'scan' -> 'state' -> 'plan')
    UNION ALL
    SELECT contractor_artifact_ref_key('run', '', job -> 'report')
    FROM contractor_jsonb_elements(state -> 'scan' -> 'state' -> 'jobs') AS job
    UNION ALL
    SELECT contractor_artifact_ref_key('run', '', input)
    FROM contractor_jsonb_elements(state -> 'scan' -> 'state' -> 'jobs') AS job,
         contractor_jsonb_values(job -> 'inputArtifacts') AS input
))
$$;

-- contractor_run_artifact_keys returns a Planner session's references as
-- keys of the scope of run_id, and nothing without a Run.
CREATE FUNCTION contractor_run_artifact_keys(run_id text, refs jsonb)
    RETURNS jsonb
    LANGUAGE sql IMMUTABLE AS $$
SELECT COALESCE(jsonb_agg(jsonb_set(ref, '{1}', to_jsonb(run_id)) ORDER BY ref), '[]'::jsonb)
FROM jsonb_array_elements(refs) AS ref
WHERE run_id IS NOT NULL
$$;

-- A WorkflowRun, and an Audit baseline, protect the exact owner Skill source
-- each selected Skill names in the owner's UserScope.
CREATE FUNCTION contractor_skill_source_artifact_holds(owner_id text, skills jsonb)
    RETURNS jsonb
    LANGUAGE sql IMMUTABLE AS $$
SELECT contractor_artifact_key_set(ARRAY(
    SELECT contractor_artifact_ref_key('user', owner_id, skill -> 'source')
    FROM contractor_jsonb_elements(skills) AS skill
))
$$;

-- An Audit protects, in its Project's scope, every exact input of its draft
-- selection and of its baseline.
CREATE FUNCTION contractor_audit_input_artifact_holds(
    project_id text, selection jsonb, baseline jsonb
) RETURNS jsonb
    LANGUAGE sql IMMUTABLE AS $$
SELECT contractor_artifact_key_set(ARRAY(
    SELECT contractor_artifact_ref_key('project', project_id, input -> 'ref')
    FROM contractor_jsonb_values(selection -> 'inputs') AS input
    UNION ALL
    SELECT contractor_artifact_ref_key('project', project_id, input -> 'ref')
    FROM contractor_jsonb_values(baseline -> 'inputs') AS input
))
$$;

-- A finding proposal receipt protects, in its source Run's scope, the
-- proposal and every exact evidence revision it pinned, until that Run's
-- purge removes them (FND-R83).
CREATE FUNCTION contractor_finding_receipt_artifact_holds(
    run_id text, proposal_ref jsonb, evidence jsonb
) RETURNS jsonb
    LANGUAGE sql IMMUTABLE AS $$
SELECT contractor_artifact_key_set(ARRAY(
    SELECT contractor_artifact_ref_key('run', run_id, proposal_ref)
    UNION ALL
    SELECT contractor_artifact_ref_key('run', run_id, item -> 'ref')
    FROM contractor_jsonb_elements(evidence) AS item
))
$$;

-- The Planner session's references, derived from each stored state, so a
-- state update compares two small arrays instead of reading the previous
-- state document.
ALTER TABLE planner_sessions
    ADD COLUMN artifact_refs jsonb
    GENERATED ALWAYS AS (contractor_planner_session_artifact_refs(state)) STORED;

-- Revisions stay immutable except for their hold count. Deletion stays the
-- controlled scope purge's, which ignores holds, plus the ArtifactStore's
-- automatic retention of an unprotected revision.
DROP TRIGGER artifact_binding_revisions_immutable ON artifact_binding_revisions;
CREATE TRIGGER artifact_binding_revisions_immutable
    BEFORE UPDATE OF scope_kind, scope_id, namespace, name, revision, version_id, created_at
    ON artifact_binding_revisions
    FOR EACH ROW EXECUTE FUNCTION contractor_protect_artifact_immutable();

-- Backfill every stored hold before any trigger maintains them. Each holder
-- row is read once.
CREATE TEMPORARY TABLE artifact_history_hold_backfill ON COMMIT DROP AS
SELECT hold.key ->> 0 AS scope_kind, hold.key ->> 1 AS scope_id, hold.key ->> 2 AS namespace,
       hold.key ->> 3 AS name, hold.key ->> 4 AS revision, hold.validated
FROM (
    SELECT jsonb_build_array(lineage.target_scope_kind, lineage.target_scope_id, lineage.target_namespace,
                             lineage.target_name, lineage.target_revision) AS key,
           false AS validated
    FROM artifact_lineage AS lineage
    UNION ALL
    SELECT jsonb_build_array(lineage.source_scope_kind, lineage.source_scope_id, lineage.source_namespace,
                             lineage.source_name, lineage.source_revision), false
    FROM artifact_lineage AS lineage
    UNION ALL
    SELECT jsonb_build_array('run', publication.run_id, publication.source_namespace, publication.source_name,
                             publication.source_revision), false
    FROM workflow_run_output_publications AS publication
    UNION ALL
    SELECT jsonb_build_array('project', publication.project_id, publication.target_namespace,
                             publication.target_name, publication.target_revision), false
    FROM workflow_run_output_publications AS publication
    WHERE publication.target_revision IS NOT NULL
    UNION ALL
    SELECT held, true
    FROM stage_executions AS stage,
         jsonb_array_elements(contractor_stage_execution_artifact_holds(stage.run_id,
             stage.stage_context_snapshot, stage.candidate_stage_result, stage.accepted_stage_result)) AS held
    UNION ALL
    SELECT held, false
    FROM planner_sessions AS session
    JOIN stage_executions AS stage ON stage.stage_execution_id = session.stage_execution_id,
         jsonb_array_elements(contractor_run_artifact_keys(stage.run_id, session.artifact_refs)) AS held
    UNION ALL
    SELECT held, false
    FROM workflow_runs AS run,
         jsonb_array_elements(contractor_skill_source_artifact_holds(run.owner_id, run.skill_snapshot)) AS held
    UNION ALL
    SELECT held, false
    FROM audits AS audit,
         jsonb_array_elements(contractor_audit_input_artifact_holds(audit.project_id, audit.input_selection,
             audit.baseline_snapshot)) AS held
    UNION ALL
    SELECT held, false
    FROM audits AS audit,
         jsonb_array_elements(contractor_skill_source_artifact_holds(audit.owner_id,
             audit.baseline_snapshot -> 'skills')) AS held
    UNION ALL
    SELECT held, true
    FROM finding_proposal_receipts AS receipt,
         jsonb_array_elements(contractor_finding_receipt_artifact_holds(receipt.run_id, receipt.proposal_ref,
             receipt.evidence)) AS held

) AS hold;

-- A StageExecution or a finding proposal receipt recorded only references its
-- writer proved in the same transaction, and nothing but the purge of the
-- Run's scope removes a revision of that scope. A missing revision of a scope
-- that still exists is therefore corrupt data, which stops the migration
-- rather than becoming an unprotected reference. Lineage and output
-- publications have foreign keys; Audit inputs may name revisions that the
-- purge of another Audit's namespace removed, and lenient holders record
-- references they never proved.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM artifact_history_hold_backfill AS hold
        WHERE hold.validated
          AND EXISTS (
              SELECT 1 FROM artifact_scopes AS scope
              WHERE scope.scope_kind = hold.scope_kind AND scope.scope_id = hold.scope_id
          )
          AND NOT EXISTS (
              SELECT 1 FROM artifact_binding_revisions AS revision
              WHERE (revision.scope_kind, revision.scope_id, revision.namespace, revision.name,
                     revision.revision)
                  = (hold.scope_kind, hold.scope_id, hold.namespace, hold.name, hold.revision)
          )
    ) THEN
        RAISE EXCEPTION 'a stored exact Artifact reference names a missing revision of a live scope'
            USING ERRCODE = '23503', CONSTRAINT = 'artifact_exact_hold';
    END IF;
END;
$$;

UPDATE artifact_binding_revisions AS revision
SET hold_count = counted.holds
FROM (
    SELECT hold.scope_kind, hold.scope_id, hold.namespace, hold.name, hold.revision,
           count(*)::integer AS holds
    FROM artifact_history_hold_backfill AS hold
    GROUP BY hold.scope_kind, hold.scope_id, hold.namespace, hold.name, hold.revision
) AS counted
WHERE (revision.scope_kind, revision.scope_id, revision.namespace, revision.name, revision.revision)
    = (counted.scope_kind, counted.scope_id, counted.namespace, counted.name, counted.revision);

DROP TABLE artifact_history_hold_backfill;

CREATE FUNCTION contractor_protect_artifact_revision_delete() RETURNS trigger
    LANGUAGE plpgsql AS $$
BEGIN
    IF contractor_lifecycle_purge_enabled() THEN
        RETURN OLD;
    END IF;
    -- The binding's foreign key to its current revision is deferred to the
    -- commit, so retention refuses the current revision here.
    IF contractor_artifact_retention_enabled() AND OLD.hold_count = 0 AND NOT EXISTS (
        SELECT 1 FROM artifact_bindings AS binding
        WHERE binding.scope_kind = OLD.scope_kind AND binding.scope_id = OLD.scope_id
          AND binding.namespace = OLD.namespace AND binding.name = OLD.name
          AND binding.current_revision = OLD.revision
    ) THEN
        RETURN OLD;
    END IF;
    RAISE EXCEPTION 'immutable ArtifactStore row cannot be changed' USING ERRCODE = '23514';
END;
$$;

CREATE TRIGGER artifact_binding_revisions_delete
    BEFORE DELETE ON artifact_binding_revisions
    FOR EACH ROW EXECUTE FUNCTION contractor_protect_artifact_revision_delete();

-- Retention collects the versions and blobs its removed revisions leave
-- unreferenced, under the same reference checks as the scope purge.
CREATE OR REPLACE FUNCTION contractor_protect_artifact_version() RETURNS trigger
    LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP = 'DELETE' AND (contractor_lifecycle_purge_enabled() OR contractor_artifact_retention_enabled()) THEN
        RETURN OLD;
    END IF;
    IF TG_OP = 'UPDATE' AND OLD.git_resolved_commit IS NULL AND NEW.git_resolved_commit IS NOT NULL
        AND ROW(NEW.version_id, NEW.blob_sha256, NEW.media_type, NEW.created_at)
            IS NOT DISTINCT FROM ROW(OLD.version_id, OLD.blob_sha256, OLD.media_type, OLD.created_at)
    THEN
        RETURN NEW;
    END IF;
    RAISE EXCEPTION 'immutable ArtifactStore row cannot be changed' USING ERRCODE = '23514';
END;
$$;

CREATE OR REPLACE FUNCTION contractor_protect_artifact_blob_immutable() RETURNS trigger
    LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        IF contractor_lifecycle_purge_enabled() OR contractor_artifact_retention_enabled() THEN
            RETURN OLD;
        END IF;
        RAISE EXCEPTION 'immutable Artifact blob cannot be changed' USING ERRCODE = '23514';
    END IF;
    IF ROW(NEW.sha256, NEW.payload, NEW.size_bytes, NEW.created_at)
       IS DISTINCT FROM
       ROW(OLD.sha256, OLD.payload, OLD.size_bytes, OLD.created_at)
    THEN
        RAISE EXCEPTION 'immutable Artifact blob cannot be changed' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

-- The history retention may examine: unprotected revisions of each binding,
-- oldest first. The current revision is the only unprotected row a scan
-- skips. Releasing a hold puts its revision back in key order, so the next
-- scan of the binding finds it before any younger revision (ART-R125).
CREATE INDEX artifact_binding_revisions_eligible_idx
    ON artifact_binding_revisions (scope_kind, scope_id, namespace, name, created_at, revision)
    WHERE hold_count = 0;

-- Lineage edges and output publications hold both revisions they name.
CREATE FUNCTION contractor_hold_lineage_artifacts() RETURNS trigger
    LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        PERFORM contractor_artifact_hold_apply((
            SELECT COALESCE(jsonb_agg(key), '[]'::jsonb)
            FROM inserted_lineage AS edge,
                 LATERAL (VALUES
                     (jsonb_build_array(edge.target_scope_kind, edge.target_scope_id, edge.target_namespace,
                                        edge.target_name, edge.target_revision)),
                     (jsonb_build_array(edge.source_scope_kind, edge.source_scope_id, edge.source_namespace,
                                        edge.source_name, edge.source_revision))) AS end_key(key)
        ), '[]'::jsonb, true);
    ELSE
        PERFORM contractor_artifact_hold_apply('[]'::jsonb, (
            SELECT COALESCE(jsonb_agg(key), '[]'::jsonb)
            FROM deleted_lineage AS edge,
                 LATERAL (VALUES
                     (jsonb_build_array(edge.target_scope_kind, edge.target_scope_id, edge.target_namespace,
                                        edge.target_name, edge.target_revision)),
                     (jsonb_build_array(edge.source_scope_kind, edge.source_scope_id, edge.source_namespace,
                                        edge.source_name, edge.source_revision))) AS end_key(key)
        ), false);
    END IF;
    RETURN NULL;
END;
$$;

CREATE TRIGGER artifact_lineage_hold
    AFTER INSERT ON artifact_lineage
    REFERENCING NEW TABLE AS inserted_lineage
    FOR EACH STATEMENT EXECUTE FUNCTION contractor_hold_lineage_artifacts();

CREATE TRIGGER artifact_lineage_release
    AFTER DELETE ON artifact_lineage
    REFERENCING OLD TABLE AS deleted_lineage
    FOR EACH STATEMENT EXECUTE FUNCTION contractor_hold_lineage_artifacts();

CREATE FUNCTION contractor_hold_output_publication_artifacts() RETURNS trigger
    LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        PERFORM contractor_artifact_hold_apply((
            SELECT COALESCE(jsonb_agg(key), '[]'::jsonb)
            FROM inserted_publications AS publication,
                 LATERAL (VALUES
                     (jsonb_build_array('run', publication.run_id, publication.source_namespace,
                                        publication.source_name, publication.source_revision)),
                     (CASE WHEN publication.target_revision IS NOT NULL THEN
                          jsonb_build_array('project', publication.project_id, publication.target_namespace,
                                            publication.target_name, publication.target_revision) END)
                 ) AS end_key(key)
            WHERE key IS NOT NULL
        ), '[]'::jsonb, true);
    ELSE
        PERFORM contractor_artifact_hold_apply('[]'::jsonb, (
            SELECT COALESCE(jsonb_agg(key), '[]'::jsonb)
            FROM deleted_publications AS publication,
                 LATERAL (VALUES
                     (jsonb_build_array('run', publication.run_id, publication.source_namespace,
                                        publication.source_name, publication.source_revision)),
                     (CASE WHEN publication.target_revision IS NOT NULL THEN
                          jsonb_build_array('project', publication.project_id, publication.target_namespace,
                                            publication.target_name, publication.target_revision) END)
                 ) AS end_key(key)
            WHERE key IS NOT NULL
        ), false);
    END IF;
    RETURN NULL;
END;
$$;

CREATE TRIGGER workflow_run_output_publications_hold
    AFTER INSERT ON workflow_run_output_publications
    REFERENCING NEW TABLE AS inserted_publications
    FOR EACH STATEMENT EXECUTE FUNCTION contractor_hold_output_publication_artifacts();

CREATE TRIGGER workflow_run_output_publications_release
    AFTER DELETE ON workflow_run_output_publications
    REFERENCING OLD TABLE AS deleted_publications
    FOR EACH STATEMENT EXECUTE FUNCTION contractor_hold_output_publication_artifacts();

-- A StageExecution holds its references for the life of its Run; the Run's
-- purge removes the revisions with the scope, so no delete trigger releases
-- them. Its writer proved every reference in the same transaction.
CREATE FUNCTION contractor_hold_stage_execution_artifacts() RETURNS trigger
    LANGUAGE plpgsql AS $$
DECLARE
    previous jsonb := '[]';
BEGIN
    IF TG_OP = 'UPDATE' THEN
        previous := contractor_stage_execution_artifact_holds(OLD.run_id, OLD.stage_context_snapshot,
            OLD.candidate_stage_result, OLD.accepted_stage_result);
    END IF;
    PERFORM contractor_artifact_hold_apply(
        contractor_stage_execution_artifact_holds(NEW.run_id, NEW.stage_context_snapshot,
            NEW.candidate_stage_result, NEW.accepted_stage_result),
        previous, true);
    RETURN NULL;
END;
$$;

CREATE TRIGGER stage_executions_hold_artifacts
    AFTER INSERT ON stage_executions
    FOR EACH ROW EXECUTE FUNCTION contractor_hold_stage_execution_artifacts();

CREATE TRIGGER stage_executions_hold_artifacts_on_update
    AFTER UPDATE OF stage_context_snapshot, candidate_stage_result, accepted_stage_result ON stage_executions
    FOR EACH ROW
    WHEN (OLD.stage_context_snapshot IS DISTINCT FROM NEW.stage_context_snapshot
          OR OLD.candidate_stage_result IS DISTINCT FROM NEW.candidate_stage_result
          OR OLD.accepted_stage_result IS DISTINCT FROM NEW.accepted_stage_result)
    EXECUTE FUNCTION contractor_hold_stage_execution_artifacts();

-- A Planner session holds its references for the life of its Run, like its
-- StageExecution. Its state records references the Planner has not proved,
-- so the hold is lenient; the StageExecution's candidate proves them.
CREATE FUNCTION contractor_hold_planner_session_artifacts() RETURNS trigger
    LANGUAGE plpgsql AS $$
DECLARE
    session_run_id text;
    previous jsonb := '[]';
BEGIN
    IF TG_OP = 'UPDATE' THEN
        previous := OLD.artifact_refs;
    END IF;
    SELECT stage.run_id INTO session_run_id
    FROM stage_executions AS stage
    WHERE stage.stage_execution_id = NEW.stage_execution_id;
    PERFORM contractor_artifact_hold_apply(
        contractor_run_artifact_keys(session_run_id, NEW.artifact_refs),
        contractor_run_artifact_keys(session_run_id, previous),
        false);
    RETURN NULL;
END;
$$;

CREATE TRIGGER planner_sessions_hold_artifacts
    AFTER INSERT ON planner_sessions
    FOR EACH ROW WHEN (NEW.artifact_refs <> '[]'::jsonb)
    EXECUTE FUNCTION contractor_hold_planner_session_artifacts();

CREATE TRIGGER planner_sessions_hold_artifacts_on_update
    AFTER UPDATE OF state ON planner_sessions
    FOR EACH ROW WHEN (OLD.artifact_refs IS DISTINCT FROM NEW.artifact_refs)
    EXECUTE FUNCTION contractor_hold_planner_session_artifacts();

-- A WorkflowRun holds its selected Skill sources until it is deleted. The
-- Run recorded them from the owner's current Skills before any lock, so the
-- hold is lenient: initialization reports a source it no longer finds
-- (RUN-R27).
CREATE FUNCTION contractor_hold_run_skill_artifacts() RETURNS trigger
    LANGUAGE plpgsql AS $$
DECLARE
    acquired jsonb := '[]';
    released jsonb := '[]';
BEGIN
    IF TG_OP <> 'DELETE' THEN
        acquired := contractor_skill_source_artifact_holds(NEW.owner_id, NEW.skill_snapshot);
    END IF;
    IF TG_OP <> 'INSERT' THEN
        released := contractor_skill_source_artifact_holds(OLD.owner_id, OLD.skill_snapshot);
    END IF;
    PERFORM contractor_artifact_hold_apply(acquired, released, false);
    RETURN NULL;
END;
$$;

CREATE TRIGGER workflow_runs_hold_skill_sources
    AFTER INSERT ON workflow_runs
    FOR EACH ROW WHEN (NEW.skill_snapshot <> '[]'::jsonb)
    EXECUTE FUNCTION contractor_hold_run_skill_artifacts();

CREATE TRIGGER workflow_runs_hold_skill_sources_on_update
    AFTER UPDATE OF skill_snapshot ON workflow_runs
    FOR EACH ROW WHEN (OLD.skill_snapshot IS DISTINCT FROM NEW.skill_snapshot)
    EXECUTE FUNCTION contractor_hold_run_skill_artifacts();

CREATE TRIGGER workflow_runs_release_skill_sources
    AFTER DELETE ON workflow_runs
    FOR EACH ROW WHEN (OLD.skill_snapshot <> '[]'::jsonb)
    EXECUTE FUNCTION contractor_hold_run_skill_artifacts();

-- An Audit holds its selected and baseline inputs, which its writer read as
-- exact Project revisions, and its baseline Skill sources until it is
-- deleted. A missing input fails the write; a missing Skill source protects
-- nothing, as for a Run.
CREATE FUNCTION contractor_hold_audit_artifacts() RETURNS trigger
    LANGUAGE plpgsql AS $$
DECLARE
    inputs_acquired jsonb := '[]';
    inputs_released jsonb := '[]';
    skills_acquired jsonb := '[]';
    skills_released jsonb := '[]';
BEGIN
    IF TG_OP <> 'DELETE' THEN
        inputs_acquired := contractor_audit_input_artifact_holds(NEW.project_id, NEW.input_selection,
            NEW.baseline_snapshot);
        skills_acquired := contractor_skill_source_artifact_holds(NEW.owner_id, NEW.baseline_snapshot -> 'skills');
    END IF;
    IF TG_OP <> 'INSERT' THEN
        inputs_released := contractor_audit_input_artifact_holds(OLD.project_id, OLD.input_selection,
            OLD.baseline_snapshot);
        skills_released := contractor_skill_source_artifact_holds(OLD.owner_id, OLD.baseline_snapshot -> 'skills');
    END IF;
    PERFORM contractor_artifact_hold_apply(inputs_acquired, inputs_released, true);
    PERFORM contractor_artifact_hold_apply(skills_acquired, skills_released, false);
    RETURN NULL;
END;
$$;

CREATE TRIGGER audits_hold_artifacts
    AFTER INSERT ON audits
    FOR EACH ROW EXECUTE FUNCTION contractor_hold_audit_artifacts();

CREATE TRIGGER audits_hold_artifacts_on_update
    AFTER UPDATE OF input_selection, baseline_snapshot ON audits
    FOR EACH ROW
    WHEN (OLD.input_selection IS DISTINCT FROM NEW.input_selection
          OR OLD.baseline_snapshot IS DISTINCT FROM NEW.baseline_snapshot)
    EXECUTE FUNCTION contractor_hold_audit_artifacts();

CREATE TRIGGER audits_release_artifacts
    AFTER DELETE ON audits
    FOR EACH ROW EXECUTE FUNCTION contractor_hold_audit_artifacts();

-- A finding proposal receipt holds its proposal and evidence until its source
-- Run's purge removes them; the receipt itself outlives them as provenance.
CREATE FUNCTION contractor_hold_finding_receipt_artifacts() RETURNS trigger
    LANGUAGE plpgsql AS $$
BEGIN
    PERFORM contractor_artifact_hold_apply(
        contractor_finding_receipt_artifact_holds(NEW.run_id, NEW.proposal_ref, NEW.evidence), '[]', true);
    RETURN NULL;
END;
$$;

CREATE TRIGGER finding_proposal_receipts_hold_artifacts
    AFTER INSERT ON finding_proposal_receipts
    FOR EACH ROW EXECUTE FUNCTION contractor_hold_finding_receipt_artifacts();

