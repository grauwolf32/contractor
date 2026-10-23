-- Durable scan sweeps retain one closure and one immutable request per Project.
-- Claims use the existing ControllerClaims shape on the parent row. Project
-- and Run identifiers deliberately have no FK: their independent purge must
-- preserve captured membership and admission facts.
CREATE TABLE scan_sweeps (
 sweep_id text PRIMARY KEY CHECK (sweep_id <> '' AND octet_length(sweep_id) <= 128),
 owner_id text NOT NULL CHECK (owner_id <> '' AND octet_length(owner_id) <= 256),
 workflow text NOT NULL,
 snapshot bytea NOT NULL CHECK (octet_length(snapshot) BETWEEN 2 AND 8388608),
 snapshot_digest text NOT NULL CHECK (snapshot_digest ~ '^sha256:[0-9a-f]{64}$'),
 purging boolean NOT NULL DEFAULT false,
 state text NOT NULL CHECK (state IN ('active','cancelling','completed','cancelled')),
 total integer NOT NULL CHECK (total BETWEEN 1 AND 1000),
 pending integer NOT NULL CHECK (pending >= 0),
 admitted integer NOT NULL DEFAULT 0 CHECK (admitted >= 0),
 skipped integer NOT NULL CHECK (skipped >= 0),
 failed integer NOT NULL DEFAULT 0 CHECK (failed >= 0),
 cancelled integer NOT NULL DEFAULT 0 CHECK (cancelled >= 0),
 unsettled integer NOT NULL CHECK (unsettled BETWEEN 0 AND total),
 created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 settled_at timestamptz,
 next_tick_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 epoch bigint NOT NULL DEFAULT 0 CHECK (epoch >= 0),
 holder_id text,
 claimed_at timestamptz,
 expires_at timestamptz,
 CHECK (pending + admitted + skipped + failed + cancelled = total),
 CHECK ((state IN ('completed','cancelled')) = (settled_at IS NOT NULL)),
 CHECK ((holder_id IS NULL AND claimed_at IS NULL AND expires_at IS NULL) OR
        (holder_id IS NOT NULL AND claimed_at IS NOT NULL AND expires_at > claimed_at))
);
CREATE INDEX scan_sweeps_owner_page_idx ON scan_sweeps(owner_id,sweep_id);
CREATE INDEX scan_sweeps_ready_idx ON scan_sweeps(next_tick_at,sweep_id) WHERE state IN ('active','cancelling');
CREATE INDEX scan_sweeps_retention_idx ON scan_sweeps(purging DESC,settled_at,sweep_id) WHERE settled_at IS NOT NULL;
CREATE TABLE scan_sweep_members (
 sweep_id text NOT NULL REFERENCES scan_sweeps(sweep_id) ON DELETE CASCADE,
 ordinal integer NOT NULL CHECK (ordinal BETWEEN 0 AND 999),
 project_id text NOT NULL CHECK (project_id <> '' AND octet_length(project_id) <= 256),
 target text NOT NULL CHECK (octet_length(target) <= 2048),
 child bytea NOT NULL CHECK (octet_length(child) BETWEEN 2 AND 16384),
 child_key text NOT NULL CHECK (child_key ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'),
 child_digest text NOT NULL CHECK (child_digest ~ '^sha256:[0-9a-f]{64}$'),
 state text NOT NULL CHECK (state IN ('pending','admitted','skipped','failed','cancelled')),
 reason text NOT NULL,
 run_id text,
 settled boolean NOT NULL,
 attempts integer NOT NULL DEFAULT 0 CHECK (attempts BETWEEN 0 AND 100),
 next_attempt_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 PRIMARY KEY(sweep_id,ordinal), UNIQUE(sweep_id,project_id),
 CHECK ((state = 'admitted') = (run_id IS NOT NULL)),
 CHECK (state NOT IN ('skipped','failed','cancelled') OR settled)
);
CREATE INDEX scan_sweep_members_work_idx ON scan_sweep_members(sweep_id,next_attempt_at,ordinal) WHERE NOT settled;

-- Immutable columns are protected even while claims and counters advance.
CREATE FUNCTION contractor_scan_sweep_immutable() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF TG_TABLE_NAME = 'scan_sweeps' THEN
  IF ROW(NEW.sweep_id,NEW.owner_id,NEW.workflow,NEW.snapshot,NEW.snapshot_digest,NEW.total,NEW.created_at)
    IS DISTINCT FROM ROW(OLD.sweep_id,OLD.owner_id,OLD.workflow,OLD.snapshot,OLD.snapshot_digest,OLD.total,OLD.created_at) THEN
   RAISE EXCEPTION 'scan sweep snapshot is immutable' USING ERRCODE = '23514';
  END IF;
 ELSE
  IF ROW(NEW.sweep_id,NEW.ordinal,NEW.project_id,NEW.target,NEW.child,NEW.child_key,NEW.child_digest)
    IS DISTINCT FROM ROW(OLD.sweep_id,OLD.ordinal,OLD.project_id,OLD.target,OLD.child,OLD.child_key,OLD.child_digest)
    OR (OLD.run_id IS NOT NULL AND NEW.run_id IS DISTINCT FROM OLD.run_id)
    OR (OLD.settled AND NOT NEW.settled) THEN
   RAISE EXCEPTION 'scan sweep member fact is immutable' USING ERRCODE = '23514';
  END IF;
 END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER scan_sweeps_immutable BEFORE UPDATE OF sweep_id,owner_id,workflow,snapshot,snapshot_digest,total,created_at ON scan_sweeps FOR EACH ROW EXECUTE FUNCTION contractor_scan_sweep_immutable();
CREATE TRIGGER scan_sweep_members_immutable BEFORE UPDATE OF sweep_id,ordinal,project_id,target,child,child_key,child_digest,run_id,settled ON scan_sweep_members FOR EACH ROW EXECUTE FUNCTION contractor_scan_sweep_immutable();

-- This is another caller of KeyedMutation, not another receipt mechanism.
ALTER TABLE mutation_receipts DROP CONSTRAINT mutation_receipts_family_check;
ALTER TABLE mutation_receipts ADD CONSTRAINT mutation_receipts_family_check CHECK (
 CASE key_scope[1]
 WHEN 'audit' THEN owner_id IS NOT NULL AND audit_id IS NOT NULL AND project_id IS NULL
  AND key_scope = ARRAY['audit',owner_id,operation] AND idempotency_key ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'
 WHEN 'eval' THEN owner_id IS NOT NULL AND project_id IS NOT NULL AND audit_id IS NULL
  AND key_scope = ARRAY['eval',owner_id,project_id,resource_id,operation]
 WHEN 'runtime-management' THEN owner_id IS NULL AND project_id IS NULL AND audit_id IS NULL
  AND key_scope = ARRAY['runtime-management'] AND idempotency_key ~ '^sha256:[0-9a-f]{64}$'
 WHEN 'scan-sweep' THEN owner_id IS NOT NULL AND project_id IS NULL AND audit_id IS NULL
  AND operation = 'scan-sweep.create' AND key_scope = ARRAY['scan-sweep',owner_id,operation]
  AND idempotency_key ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'
 ELSE false END
);
CREATE OR REPLACE FUNCTION contractor_lifecycle_purge_enabled() RETURNS boolean LANGUAGE sql STABLE AS $$
 SELECT COALESCE(current_setting('contractor.lifecycle_purge',true),'') IN ('run','project','audit','scan-sweep')
$$;

-- The owner library skips retained sweep inputs by index before its page
-- limit; a growing sweep history must not turn each picker page into a scan.
CREATE INDEX artifact_bindings_user_listing_idx ON artifact_bindings(scope_id,namespace,name)
WHERE scope_kind='user' AND namespace NOT IN ('audit-standards','contractor-system');

-- Reject oversized all-Project scopes after 1001 rows, independent of the
-- containing portfolio's size. Explicit selections use the Project PK.
CREATE INDEX projects_sweep_selection_idx ON projects(owner_id,kind,project_id COLLATE "C");
