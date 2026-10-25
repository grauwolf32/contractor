-- A settled Round captures one coherent source generation in bounded pages.
-- Staged pages are not ArtifactRefs and grant no consumer availability. Once
-- sealed, their content is immutable while ordinary later reviews may proceed.
ALTER TABLE audit_rounds
    ADD COLUMN results_generation bigint,
    ADD COLUMN results_sealed boolean NOT NULL DEFAULT false,
    ADD COLUMN results_item_count integer NOT NULL DEFAULT 0 CHECK (results_item_count BETWEEN 0 AND 4096),
    ADD COLUMN results_metadata_bytes bigint NOT NULL DEFAULT 0 CHECK (results_metadata_bytes BETWEEN 0 AND 67108864),
    ADD COLUMN results_page_count integer NOT NULL DEFAULT 0 CHECK (results_page_count BETWEEN 0 AND 256),
    ADD COLUMN results_published_pages jsonb NOT NULL DEFAULT '[]'::jsonb
        CHECK (jsonb_typeof(results_published_pages)='array' AND jsonb_array_length(results_published_pages)<=256
            AND octet_length(results_published_pages::text)<=262144),
    ADD COLUMN results_manifest jsonb,
    ADD CONSTRAINT audit_rounds_results_generation_check CHECK (results_generation IS NULL OR results_generation > 0),
    ADD CONSTRAINT audit_rounds_results_manifest_check CHECK (
        results_manifest IS NULL OR (results_sealed AND octet_length(results_manifest::text) <= 4096));

CREATE TABLE audit_round_result_pages (
    audit_id text NOT NULL REFERENCES audits(audit_id) ON DELETE CASCADE,
    round_id text NOT NULL REFERENCES audit_rounds(round_id) ON DELETE CASCADE,
    source_generation bigint NOT NULL CHECK (source_generation > 0),
    ordinal integer NOT NULL CHECK (ordinal BETWEEN 0 AND 255),
    last_item_ordinal integer NOT NULL CHECK (last_item_ordinal BETWEEN 0 AND 4095),
    item_count integer NOT NULL CHECK (item_count BETWEEN 1 AND 128),
    content_bytes integer NOT NULL CHECK (content_bytes BETWEEN 1 AND 262144),
    content bytea CHECK (content IS NULL OR octet_length(content)=content_bytes),
    artifact jsonb CHECK (artifact IS NULL OR octet_length(artifact::text) <= 4096),
    CHECK (content IS NOT NULL OR artifact IS NOT NULL),
    PRIMARY KEY (round_id, ordinal),
    UNIQUE (round_id, last_item_ordinal)
);
CREATE INDEX audit_round_result_pages_audit_idx ON audit_round_result_pages(audit_id, round_id, ordinal);

-- Attribution lookup selects only descriptors of the selected attempt, never
-- every link of the containing Audit or every result of a shared batch.
CREATE INDEX audit_artifact_links_execution_item_idx
    ON audit_artifact_links(audit_id,
        (COALESCE(source_provenance->>'executionItemId',source_provenance->'source'->>'executionItemId')), logical_key);
CREATE INDEX audit_artifact_links_digest_idx ON audit_artifact_links(audit_id, artifact_digest);
CREATE INDEX audit_finding_assessments_round_results_idx
    ON audit_finding_assessments(audit_id, execution_item_id, assessment_id);

-- Stored conservative JSON sizes let capture choose its row/byte slice before
-- loading coverage or descriptor payloads. Multiplying strings by six covers
-- every JSON escape; pages still enforce their exact encoded byte ceiling.
ALTER TABLE audit_items ADD COLUMN results_projection_bytes bigint GENERATED ALWAYS AS (
    4096 + 6 * (octet_length(item_key) + octet_length(kind) + octet_length(subject_key)
        + octet_length(coverage_rationale) + octet_length(coverage_requested::text)
        + octet_length(coverage_completed::text) + octet_length(coverage_gaps::text))
) STORED;
-- Every argument is JSON, text or integer, so this wrapper is immutable even
-- though polymorphic jsonb_build_object is conservatively STABLE in PostgreSQL.
-- JSONB's spaces only enlarge Go's compact JSON; account for its HTML/U+2028
-- escapes before taking the byte length, without a sixfold whole-object guess.
CREATE FUNCTION contractor_round_results_descriptor_bytes(
    key text, ref jsonb, digest text, media text, size bigint, provenance jsonb, display text
) RETURNS bigint LANGUAGE sql IMMUTABLE AS $$
SELECT octet_length(replace(replace(replace(replace(replace(
    jsonb_strip_nulls(jsonb_build_object(
      'logical_key',key,'classification','accepted-attachment',
      'artifact',jsonb_build_object('ref',ref,'digest',digest,'mediaType',media,'sizeBytes',size),
      'kind',provenance->'evidence'->>'kind','evidence_id',provenance->'evidence'->>'id',
      'content_member_id',provenance->'evidence'->>'content_member_id','display_ref',display,
      'original',CASE WHEN provenance->'evidence'->'artifact' IS NOT NULL
        AND provenance->'evidence'->'artifact'<>'null'::jsonb
        THEN jsonb_build_object('ref',provenance->'evidence'->'artifact',
          'digest',digest,'mediaType',media,'sizeBytes',size) END
    ))::text,'<','\u003c'),'>','\u003e'),'&','\u0026'),U&'\2028','\u2028'),U&'\2029','\u2029'))
$$;
ALTER TABLE audit_artifact_links ADD COLUMN results_descriptor_bytes bigint GENERATED ALWAYS AS (
    contractor_round_results_descriptor_bytes(logical_key,artifact_ref,artifact_digest,
      media_type,size_bytes,source_provenance,display_ref)
) STORED;

CREATE FUNCTION contractor_protect_round_results_page() RETURNS trigger
    LANGUAGE plpgsql AS $$
BEGIN
    IF ROW(NEW.audit_id, NEW.round_id, NEW.source_generation, NEW.ordinal,
           NEW.last_item_ordinal, NEW.item_count, NEW.content_bytes)
       IS DISTINCT FROM ROW(OLD.audit_id, OLD.round_id, OLD.source_generation, OLD.ordinal,
           OLD.last_item_ordinal, OLD.item_count, OLD.content_bytes)
       OR (OLD.artifact IS NOT NULL AND NEW.artifact IS DISTINCT FROM OLD.artifact) THEN
        RAISE EXCEPTION 'Round result page is immutable' USING ERRCODE = '23514',
            CONSTRAINT = 'audit_round_result_pages_immutable';
    END IF;
    IF NEW.content IS DISTINCT FROM OLD.content AND NOT (
        OLD.content IS NOT NULL AND NEW.content IS NULL AND OLD.artifact IS NULL AND NEW.artifact IS NOT NULL
    ) THEN
        RAISE EXCEPTION 'Round result page content is immutable until transferred to its frozen Artifact'
            USING ERRCODE='23514', CONSTRAINT='audit_round_result_pages_content_immutable';
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER audit_round_result_pages_protect_update BEFORE UPDATE ON audit_round_result_pages
    FOR EACH ROW EXECUTE FUNCTION contractor_protect_round_results_page();

CREATE FUNCTION contractor_protect_round_results_root() RETURNS trigger
    LANGUAGE plpgsql AS $$
BEGIN
    IF OLD.results_sealed AND ROW(NEW.results_generation, NEW.results_sealed,
        NEW.results_item_count, NEW.results_metadata_bytes, NEW.results_page_count)
        IS DISTINCT FROM ROW(OLD.results_generation, OLD.results_sealed,
        OLD.results_item_count, OLD.results_metadata_bytes, OLD.results_page_count)
        OR OLD.results_manifest IS NOT NULL AND NEW.results_manifest IS DISTINCT FROM OLD.results_manifest THEN
        RAISE EXCEPTION 'Round results source generation is immutable' USING ERRCODE = '23514',
            CONSTRAINT = 'audit_rounds_results_immutable';
    END IF;
    IF NEW.results_published_pages IS DISTINCT FROM OLD.results_published_pages
       AND (NOT OLD.results_sealed OR OLD.results_manifest IS NOT NULL
         OR jsonb_array_length(NEW.results_published_pages)<>jsonb_array_length(OLD.results_published_pages)+1
         OR NEW.results_published_pages - (jsonb_array_length(NEW.results_published_pages)-1)
            IS DISTINCT FROM OLD.results_published_pages) THEN
        RAISE EXCEPTION 'Round results publication progress is append-only' USING ERRCODE = '23514',
            CONSTRAINT = 'audit_rounds_results_publication_immutable';
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER audit_rounds_protect_results BEFORE UPDATE OF results_generation, results_sealed,
    results_item_count, results_metadata_bytes, results_page_count, results_manifest, results_published_pages ON audit_rounds
    FOR EACH ROW EXECUTE FUNCTION contractor_protect_round_results_root();
-- Durable bounded Audit-owned input initialization; no target allocation can
-- run until its explicit exact local closure is complete.
CREATE FUNCTION contractor_valid_audit_input_materialization(progress json) RETURNS boolean
 LANGUAGE sql IMMUTABLE AS $$
SELECT COALESCE(
 octet_length(progress::text)<=262144 AND jsonb_typeof(value)='object'
 AND value ?& ARRAY['identity','revision','complete','deadlineAt','checkpoint','refs']
 AND value-ARRAY['identity','revision','complete','deadlineAt','checkpoint','refs']='{}'::jsonb
 AND jsonb_typeof(value->'identity')='string'
 AND value->>'identity' ~ '^sha256:[0-9a-f]{64}$'
 AND jsonb_typeof(value->'revision')='number'
 AND (value->>'revision')::bigint BETWEEN 1 AND 4294967295
 AND jsonb_typeof(value->'complete')='boolean'
 AND jsonb_typeof(value->'deadlineAt')='string'
 AND jsonb_typeof(value->'checkpoint')='object'
 AND jsonb_typeof(value->'refs')='array'
 AND jsonb_array_length(value->'refs')<=1024, false)
 FROM (SELECT progress::jsonb AS value) AS parsed
$$;
ALTER TABLE workflow_runs ADD COLUMN audit_input_materialization json;
ALTER TABLE workflow_runs ADD CONSTRAINT workflow_runs_audit_input_materialization_check CHECK (
 audit_input_materialization IS NULL OR (publication_mode='audit-managed'
 AND contractor_valid_audit_input_materialization(audit_input_materialization)));

CREATE FUNCTION contractor_protect_audit_input_materialization() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF NEW.audit_input_materialization::text IS DISTINCT FROM OLD.audit_input_materialization::text THEN
  IF NEW.publication_mode<>'audit-managed' OR NEW.state<>'initializing'
   OR NEW.audit_input_materialization IS NULL THEN
   RAISE EXCEPTION 'Audit input progress is writable only during initialization' USING ERRCODE='23514';
  END IF;
  IF OLD.audit_input_materialization IS NOT NULL AND (
   (OLD.audit_input_materialization->>'complete')::boolean
   OR NEW.audit_input_materialization->>'identity' IS DISTINCT FROM OLD.audit_input_materialization->>'identity'
   OR NEW.audit_input_materialization->>'deadlineAt' IS DISTINCT FROM OLD.audit_input_materialization->>'deadlineAt'
   OR (NEW.audit_input_materialization->>'revision')::bigint<>(OLD.audit_input_materialization->>'revision')::bigint+1
   OR NOT ((NEW.audit_input_materialization::jsonb->'refs') @> (OLD.audit_input_materialization::jsonb->'refs'))) THEN
   RAISE EXCEPTION 'Audit input progress cannot change authority or release its closure' USING ERRCODE='23514';
  END IF;
 END IF;
 RETURN NEW;
END;
$$;
CREATE TRIGGER workflow_runs_protect_audit_input_materialization
 BEFORE UPDATE OF audit_input_materialization ON workflow_runs
 FOR EACH ROW WHEN (OLD.audit_input_materialization::text IS DISTINCT FROM NEW.audit_input_materialization::text)
 EXECUTE FUNCTION contractor_protect_audit_input_materialization();

CREATE FUNCTION contractor_audit_run_input_holds(run_id text, progress jsonb) RETURNS jsonb
 LANGUAGE sql IMMUTABLE AS $$
SELECT contractor_artifact_key_set(ARRAY(
 SELECT contractor_artifact_ref_key('run',run_id,ref)
 FROM contractor_jsonb_elements(progress->'refs') AS ref
))
$$;
CREATE FUNCTION contractor_hold_audit_run_inputs() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE acquired jsonb:='[]'; released jsonb:='[]';
BEGIN
 IF TG_OP<>'DELETE' THEN acquired:=contractor_audit_run_input_holds(NEW.run_id,NEW.audit_input_materialization::jsonb); END IF;
 IF TG_OP<>'INSERT' THEN released:=contractor_audit_run_input_holds(OLD.run_id,OLD.audit_input_materialization::jsonb); END IF;
 PERFORM contractor_artifact_hold_apply(acquired,released,true);
 RETURN NULL;
END;
$$;
CREATE TRIGGER workflow_runs_hold_audit_inputs_on_insert
 AFTER INSERT ON workflow_runs
 FOR EACH ROW WHEN (NEW.audit_input_materialization IS NOT NULL)
 EXECUTE FUNCTION contractor_hold_audit_run_inputs();
CREATE TRIGGER workflow_runs_hold_audit_inputs_on_update
 AFTER UPDATE OF audit_input_materialization ON workflow_runs
 FOR EACH ROW WHEN (OLD.audit_input_materialization::text IS DISTINCT FROM NEW.audit_input_materialization::text)
 EXECUTE FUNCTION contractor_hold_audit_run_inputs();
CREATE TRIGGER workflow_runs_release_audit_inputs
 AFTER DELETE ON workflow_runs
 FOR EACH ROW WHEN (OLD.audit_input_materialization IS NOT NULL)
 EXECUTE FUNCTION contractor_hold_audit_run_inputs();
