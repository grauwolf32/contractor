-- The Project-wide finding correlation revision (FND-R177) moves from the
-- shared projects row to a fence row that only finding writers lock and
-- write. Run creation, Audit start, Project artifact admission, Git import and
-- Eval checks hold the Project row FOR SHARE; the exclusive lock that finding
-- writers took on that row aborted every one of them that overlapped such a
-- holder, and the triggers of another owner's tables wrote projects.
CREATE TABLE finding_correlation_fences (
 project_id text PRIMARY KEY REFERENCES projects(project_id) ON DELETE CASCADE,
 revision bigint NOT NULL DEFAULT 1 CHECK (revision BETWEEN 1 AND 9007199254740991)
);

-- Every existing Project keeps its exact revision, with or without an Audit.
INSERT INTO finding_correlation_fences(project_id, revision)
SELECT project_id, finding_correlation_revision FROM projects;

DROP TRIGGER projects_protect_correlation_revision ON projects;
DROP FUNCTION contractor_protect_finding_correlation_revision();
ALTER TABLE projects DROP COLUMN finding_correlation_revision;

-- The revision steps by one or stays, so a reader's fence never sees it
-- decrease or skip.
CREATE FUNCTION contractor_protect_finding_correlation_fence() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF NEW.project_id <> OLD.project_id OR NEW.revision NOT IN (OLD.revision, OLD.revision + 1) THEN
  RAISE EXCEPTION 'finding correlation revision must advance monotonically' USING ERRCODE='23514';
 END IF;
 RETURN NEW;
END;
$$;
CREATE TRIGGER finding_correlation_fences_protect_revision BEFORE UPDATE ON finding_correlation_fences
 FOR EACH ROW EXECUTE FUNCTION contractor_protect_finding_correlation_fence();

-- The row goes only with its Project, through the cascade of the Project row
-- deletion, so the revision never restarts while the Project exists.
CREATE FUNCTION contractor_retain_finding_correlation_fence() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF EXISTS (SELECT 1 FROM projects WHERE project_id = OLD.project_id) THEN
  RAISE EXCEPTION 'finding correlation fence is kept while its Project exists' USING ERRCODE='23514';
 END IF;
 RETURN OLD;
END;
$$;
CREATE TRIGGER finding_correlation_fences_retain BEFORE DELETE ON finding_correlation_fences
 FOR EACH ROW EXECUTE FUNCTION contractor_retain_finding_correlation_fence();

-- Finding writers already hold Run/Audit authority and never wait here. The
-- shared Project lock serializes them with Project deletion and lifecycle
-- changes, which lock the row exclusively, and stays compatible with every
-- admission that shares it. The fence row, locked only by finding writers
-- after that shared lock, serializes them with each other. Contention is the
-- existing retryable 40001. The row exists for every Project with an Audit,
-- because Audit creation inserts it, so its absence is an invariant failure.
DROP FUNCTION contractor_lock_finding_correlation_project(text);
CREATE FUNCTION contractor_lock_finding_correlation_fence(selected_project text) RETURNS void LANGUAGE plpgsql AS $$
DECLARE
 fenced boolean;
BEGIN
 BEGIN
  PERFORM 1 FROM projects WHERE project_id = selected_project FOR SHARE NOWAIT;
  SELECT true INTO fenced FROM finding_correlation_fences WHERE project_id = selected_project
   FOR NO KEY UPDATE NOWAIT;
 EXCEPTION WHEN lock_not_available THEN
  RAISE EXCEPTION 'Project finding correlation is concurrently changing' USING ERRCODE='40001';
 END;
 IF fenced IS NULL THEN
  RAISE EXCEPTION 'Project % has no finding correlation fence', selected_project USING ERRCODE='P0004';
 END IF;
END;
$$;

-- The six finding triggers keep calling this function: one step per affected
-- Project per statement, the Projects locked in sorted order.
CREATE OR REPLACE FUNCTION contractor_advance_finding_correlation(affected_projects text[]) RETURNS void LANGUAGE plpgsql AS $$
DECLARE
 selected_project text;
BEGIN
 FOR selected_project IN SELECT DISTINCT unnest(affected_projects) ORDER BY 1 LOOP
  PERFORM contractor_lock_finding_correlation_fence(selected_project);
  UPDATE finding_correlation_fences SET revision = revision + 1 WHERE project_id = selected_project;
 END LOOP;
END;
$$;

-- Admission requires a Project-bound source Run in the same Project. Its
-- immutable receipt scope invalidates provenance without enumerating peers.
-- A Project-bound Run's receipts exist whether or not its Project has an
-- Audit. A Project without a fence row never had an Audit, so no finding,
-- contribution or hold cites the receipt and its source deletion fences
-- nothing.
CREATE OR REPLACE FUNCTION contractor_finding_correlation_source_deleted() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 PERFORM contractor_advance_finding_correlation(ARRAY(
  SELECT DISTINCT n.project_id FROM updated_receipts AS n JOIN previous_receipts AS o USING(receipt_id)
  JOIN finding_correlation_fences AS fence ON fence.project_id = n.project_id
  WHERE n.source_run_deleted_at IS DISTINCT FROM o.source_run_deleted_at
 ));
 RETURN NULL;
END;
$$;
