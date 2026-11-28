-- One-time, user-authorized in-place format transition. No history revisions or
-- archive copies are published. Ordinary immutable guards remain unchanged.
CREATE TEMP TABLE finding_http_rewrites (
 old_digest bytea PRIMARY KEY, old_size bigint NOT NULL,
 new_digest bytea NOT NULL, new_payload bytea NOT NULL, new_size bigint NOT NULL
) ON COMMIT DROP;

-- The affected owned documents have ASCII member names and safe integer numeric
-- fields. Serialize that closed grammar canonically; never approximate a float
-- or a non-ASCII member ordering. The upgrade test compares this with RFC 8785.
CREATE FUNCTION pg_temp.finding_http_canonical(value jsonb, depth integer DEFAULT 0)
RETURNS text LANGUAGE plpgsql AS $$
DECLARE result text; number numeric;
BEGIN
 IF depth > 64 THEN RAISE EXCEPTION 'Finding format migration JSON depth exceeded' USING ERRCODE='23514'; END IF;
 CASE jsonb_typeof(value)
 WHEN 'object' THEN
  IF EXISTS(SELECT 1 FROM jsonb_object_keys(value) k WHERE k !~ '^[ -~]*$') THEN
   RAISE EXCEPTION 'Unsupported Finding migration JSON member name' USING ERRCODE='23514';
  END IF;
  SELECT '{'||coalesce(string_agg(to_jsonb(key)::text||':'||pg_temp.finding_http_canonical(v,depth+1),',' ORDER BY key COLLATE "C"),'')||'}'
    INTO result FROM jsonb_each(value) e(key,v);
 WHEN 'array' THEN
  SELECT '['||coalesce(string_agg(pg_temp.finding_http_canonical(v,depth+1),',' ORDER BY n),'')||']'
    INTO result FROM jsonb_array_elements(value) WITH ORDINALITY e(v,n);
 WHEN 'number' THEN
  number := (value#>>'{}')::numeric;
  IF number <> trunc(number) OR abs(number)>9007199254740991 THEN
   RAISE EXCEPTION 'Unsupported numeric value in Finding format migration' USING ERRCODE='23514';
  END IF;
  result:=trunc(number)::text;
 ELSE result:=value::text;
 END CASE;
 RETURN result;
END $$;

CREATE FUNCTION pg_temp.finding_http_refs(value jsonb, depth integer DEFAULT 0)
RETURNS jsonb LANGUAGE plpgsql AS $$
DECLARE result jsonb; replacement record; field text; fragment_name text; delta bigint;
BEGIN
 IF depth > 64 THEN RAISE EXCEPTION 'Finding format migration JSON depth exceeded' USING ERRCODE='23514'; END IF;
 IF jsonb_typeof(value)='array' THEN
  SELECT coalesce(jsonb_agg(pg_temp.finding_http_refs(v,depth+1) ORDER BY n),'[]') INTO result
    FROM jsonb_array_elements(value) WITH ORDINALITY e(v,n);
 ELSIF jsonb_typeof(value)='object' THEN
  SELECT coalesce(jsonb_object_agg(k,pg_temp.finding_http_refs(v,depth+1)),'{}') INTO result FROM jsonb_each(value) e(k,v);
  -- Exact Artifact descriptors and the existing compact checks projection.
  IF result ? 'digest' AND result ? 'ref' AND (result ? 'sizeBytes' OR result ? 'size_bytes') THEN
   field:='digest';
  ELSIF result->>'schema'='contractor.audit.finding-checks-projection.v1' THEN
   field:='proposal_digest';
  END IF;
  IF field IS NOT NULL AND result->>field ~ '^sha256:[0-9a-f]{64}$' THEN
   SELECT * INTO replacement FROM pg_temp.finding_http_rewrites WHERE old_digest=decode(substr(result->>field,8),'hex');
   IF FOUND THEN
    IF coalesce(result->>'sizeBytes',result->>'size_bytes','') !~ '^[0-9]{1,16}$' THEN
     RAISE EXCEPTION 'Invalid Finding migration descriptor size' USING ERRCODE='23514';
    END IF;
    IF coalesce(result->>'sizeBytes',result->>'size_bytes')::bigint IS DISTINCT FROM replacement.old_size THEN
     RAISE EXCEPTION 'Finding migration descriptor size disagrees with stored bytes' USING ERRCODE='23514';
    END IF;
    result:=jsonb_set(result,ARRAY[field],to_jsonb('sha256:'||encode(replacement.new_digest,'hex')));
    IF result ? 'sizeBytes' THEN result:=jsonb_set(result,'{sizeBytes}',to_jsonb(replacement.new_size)); END IF;
    IF result ? 'size_bytes' THEN result:=jsonb_set(result,'{size_bytes}',to_jsonb(replacement.new_size)); END IF;
    IF field='proposal_digest' THEN
     -- All indexed subject/limitations/check fragments follow http_exchange
     -- in canonical member order. Renaming the key and adding [] adds 3 bytes.
     delta:=replacement.new_size-replacement.old_size;
     IF delta<>3 THEN RAISE EXCEPTION 'Unexpected indexed Finding format delta' USING ERRCODE='23514'; END IF;
     FOREACH fragment_name IN ARRAY ARRAY['subject','limitations'] LOOP
      IF jsonb_typeof(result->fragment_name)='object' THEN
       result:=jsonb_set(result,ARRAY[fragment_name,'offset'],to_jsonb((result#>>ARRAY[fragment_name,'offset'])::bigint+delta));
      END IF;
     END LOOP;
     SELECT jsonb_set(result,'{checks}',coalesce(jsonb_agg(
      jsonb_set(v,'{fragment,offset}',to_jsonb((v#>>'{fragment,offset}')::bigint+delta)) ORDER BY n),'[]')) INTO result
      FROM jsonb_array_elements(result->'checks') WITH ORDINALITY e(v,n);
    END IF;
   END IF;
  END IF;
 ELSE result:=value;
 END IF;
 RETURN result;
END $$;


-- Mutation receipts may also contain an opaque/empty response, unrelated to
-- Finding JSON. Preserve it byte-for-byte when there is no typed descriptor.
CREATE FUNCTION pg_temp.finding_http_bytes(payload bytea) RETURNS bytea LANGUAGE plpgsql AS $$
DECLARE original jsonb; replacement jsonb;
BEGIN
 BEGIN original:=convert_from(payload,'UTF8')::jsonb;
 EXCEPTION WHEN data_exception THEN RETURN payload;
 END;
 replacement:=pg_temp.finding_http_refs(original);
 IF replacement IS NOT DISTINCT FROM original THEN RETURN payload; END IF;
 RETURN convert_to(pg_temp.finding_http_canonical(replacement),'UTF8');
END $$;

CREATE TEMP TABLE finding_http_inputs (digest bytea PRIMARY KEY, bytes bigint, document jsonb) ON COMMIT DROP;
CREATE TEMP TABLE finding_http_triggers (relation regclass, name text, mode "char") ON COMMIT DROP;
CREATE TEMP TABLE finding_http_versions (version_id text PRIMARY KEY) ON COMMIT DROP;
CREATE TEMP TABLE finding_http_columns (relation regclass, name text, kind text) ON COMMIT DROP;

DO $migration$
DECLARE source record; column_row record; guard record; doc jsonb; payload bytea;
 changed bigint; total_rows bigint; total_bytes bigint; inspected_bytes bigint:=0;
 pass integer; schema_oid oid; inventory_rows integer:=0;
BEGIN
 SELECT relnamespace INTO schema_oid FROM pg_class WHERE oid='finding_proposal_receipts'::regclass;
 -- Bound work before parsing documents. Invalid unrelated JSON remains opaque.
 FOR source IN
  SELECT b.sha256,b.payload,b.size_bytes FROM artifact_blobs b
   WHERE b.backend='postgresql' AND EXISTS(SELECT 1 FROM artifact_versions v
    WHERE v.blob_sha256=b.sha256 AND (v.media_type='application/json' OR v.media_type LIKE 'application/%+json'))
   LIMIT 10001
 LOOP
  inspected_bytes:=inspected_bytes+source.size_bytes; inventory_rows:=inventory_rows+1;
  IF inspected_bytes>67108864 OR inventory_rows>10000 THEN
   RAISE EXCEPTION 'Finding migration JSON inventory bound exceeded' USING ERRCODE='23514';
  END IF;
  BEGIN doc:=convert_from(source.payload,'UTF8')::jsonb;
  EXCEPTION WHEN data_exception THEN doc:=NULL;
  END;
  INSERT INTO pg_temp.finding_http_inputs VALUES(source.sha256,source.size_bytes,doc);
 END LOOP;

 FOR source IN SELECT * FROM pg_temp.finding_http_inputs
  WHERE document->>'schema'='contractor.audit.finding-proposal.v1' AND document ? 'http_exchange'
 LOOP
  IF jsonb_typeof(source.document->'http_exchange') IS DISTINCT FROM 'object' OR source.document ? 'http_exchanges' THEN
   RAISE EXCEPTION 'Ambiguous stored HTTP Finding format' USING ERRCODE='23514';
  END IF;
  IF sha256(convert_to(pg_temp.finding_http_canonical(source.document),'UTF8'))<>source.digest THEN
   RAISE EXCEPTION 'Stored Finding proposal is not canonical' USING ERRCODE='23514';
  END IF;
  doc:=(source.document-'http_exchange')||jsonb_build_object('http_exchanges',jsonb_build_array(source.document->'http_exchange'));
  payload:=convert_to(pg_temp.finding_http_canonical(doc),'UTF8');
  IF octet_length(payload)>8388608 THEN RAISE EXCEPTION 'Finding migration document size exceeded' USING ERRCODE='23514'; END IF;
  INSERT INTO pg_temp.finding_http_rewrites VALUES(source.digest,source.bytes,sha256(payload),payload,octet_length(payload));
  SELECT count(*),sum(new_size) INTO total_rows,total_bytes FROM pg_temp.finding_http_rewrites;
  IF total_rows>512 OR total_bytes>33554432 THEN RAISE EXCEPTION 'Finding migration rewrite bound exceeded' USING ERRCODE='23514'; END IF;
 END LOOP;
 IF EXISTS(SELECT 1 FROM finding_proposal_receipts r
  WHERE r.retention_state<>'discarded' AND r.report_projection::jsonb ? 'http_exchange'
   AND NOT EXISTS(SELECT 1 FROM pg_temp.finding_http_rewrites m WHERE m.old_digest=decode(substr(r.proposal_digest,8),'hex'))
   AND NOT EXISTS(SELECT 1 FROM pg_temp.finding_http_inputs i WHERE i.digest=decode(substr(r.proposal_digest,8),'hex')
     AND i.document->>'schema'='contractor.audit.finding-proposal.v1' AND i.document ? 'http_exchanges' AND NOT i.document ? 'http_exchange')
 ) THEN RAISE EXCEPTION 'Missing or unsupported stored Finding proposal' USING ERRCODE='23514'; END IF;
 IF NOT EXISTS(SELECT 1 FROM pg_temp.finding_http_rewrites) AND NOT EXISTS(
  SELECT 1 FROM finding_proposal_receipts WHERE report_projection::jsonb ? 'http_exchange'
 ) THEN RETURN; END IF;

 -- SQL cannot inspect an external object or rewrite an opaque ZIP snapshot.
 -- Refuse that topology atomically instead of leaving stale exact references.
 IF EXISTS(SELECT 1 FROM artifact_versions v JOIN artifact_blobs b ON b.sha256=v.blob_sha256
  WHERE (b.backend<>'postgresql' AND (v.media_type='application/json' OR v.media_type LIKE 'application/%+json'))
     OR v.media_type='application/vnd.contractor.findings-collection+zip') THEN
  RAISE EXCEPTION 'Finding migration requires inspection of an unsupported stored package' USING ERRCODE='23514';
 END IF;

 -- Immutable source-history documents refer to the old exact proposal. Follow
 -- their bounded dependency closure before publishing any replacement bytes.
 FOR pass IN 1..16 LOOP
  changed:=0;
  FOR source IN SELECT * FROM pg_temp.finding_http_inputs WHERE document->>'schema'='contractor.finding.source-history.v1' LOOP
   doc:=pg_temp.finding_http_refs(source.document);
   IF doc IS DISTINCT FROM source.document THEN
    payload:=convert_to(pg_temp.finding_http_canonical(doc),'UTF8');
    IF octet_length(payload)>8388608 THEN RAISE EXCEPTION 'Finding migration history size exceeded' USING ERRCODE='23514'; END IF;
    INSERT INTO pg_temp.finding_http_rewrites VALUES(source.digest,source.bytes,sha256(payload),payload,octet_length(payload))
    ON CONFLICT(old_digest) DO UPDATE SET new_digest=excluded.new_digest,new_payload=excluded.new_payload,new_size=excluded.new_size
      WHERE finding_http_rewrites.new_digest<>excluded.new_digest;
    GET DIAGNOSTICS total_rows=ROW_COUNT;
    changed:=changed+total_rows;
    SELECT count(*),sum(new_size) INTO total_rows,total_bytes FROM pg_temp.finding_http_rewrites;
    IF total_rows>512 OR total_bytes>33554432 THEN RAISE EXCEPTION 'Finding migration rewrite bound exceeded' USING ERRCODE='23514'; END IF;
   END IF;
  END LOOP;
  SELECT count(*),coalesce(sum(new_size),0) INTO total_rows,total_bytes FROM pg_temp.finding_http_rewrites;
  IF total_rows>512 OR total_bytes>33554432 THEN RAISE EXCEPTION 'Finding migration rewrite bound exceeded' USING ERRCODE='23514'; END IF;
  EXIT WHEN changed=0;
  IF pass=16 THEN RAISE EXCEPTION 'Finding migration reference depth exceeded' USING ERRCODE='23514'; END IF;
 END LOOP;
 FOR source IN SELECT * FROM pg_temp.finding_http_inputs
  WHERE document IS NOT NULL AND coalesce(document->>'schema','') NOT IN ('contractor.audit.finding-proposal.v1','contractor.finding.source-history.v1')
 LOOP
  IF pg_temp.finding_http_refs(source.document) IS DISTINCT FROM source.document THEN
   RAISE EXCEPTION 'Unsupported Artifact refers to a migrated Finding document' USING ERRCODE='23514';
  END IF;
 END LOOP;

 -- Discover typed metadata only, not credentials or arbitrary binary payloads.
 -- Generated columns are recalculated by PostgreSQL, never assigned directly.
 INSERT INTO pg_temp.finding_http_columns
 SELECT c.oid::regclass,a.attname,CASE WHEN t.typname='json' THEN 'json' ELSE 'jsonb' END
 FROM pg_class c JOIN pg_attribute a ON a.attrelid=c.oid JOIN pg_type t ON t.oid=a.atttypid
 WHERE c.relnamespace=schema_oid AND c.relkind='r' AND a.attnum>0 AND NOT a.attisdropped
 AND a.attgenerated='' AND t.typname IN ('json','jsonb');
 INSERT INTO pg_temp.finding_http_columns VALUES('mutation_receipts'::regclass,'response','bytes');
 FOR column_row IN SELECT * FROM pg_temp.finding_http_columns LOOP
  EXECUTE format('SELECT count(*),coalesce(sum(pg_column_size(v)),0) FROM (SELECT %I v FROM %s WHERE %I IS NOT NULL LIMIT 10001) bounded',column_row.name,column_row.relation,column_row.name)
    INTO total_rows,total_bytes;
  inspected_bytes:=inspected_bytes+total_bytes;
  IF total_rows>10000 OR inspected_bytes>67108864 THEN RAISE EXCEPTION 'Finding migration metadata bound exceeded' USING ERRCODE='23514'; END IF;
 END LOOP;
 -- Non-JSON Eval documents have their own canonical identities and pins. Do
 -- not silently rewrite one: fail if a migrated digest is embedded there.
 FOR column_row IN
  SELECT c.oid::regclass relation,a.attname name FROM pg_class c JOIN pg_attribute a ON a.attrelid=c.oid
  WHERE c.relnamespace=schema_oid AND c.relname LIKE 'eval_%' AND a.atttypid='bytea'::regtype AND a.attnum>0 AND NOT a.attisdropped
 LOOP
  EXECUTE format('SELECT count(*),coalesce(sum(pg_column_size(v)),0) FROM (SELECT %I v FROM %s WHERE %I IS NOT NULL LIMIT 10001) bounded',column_row.name,column_row.relation,column_row.name) INTO total_rows,total_bytes;
  inspected_bytes:=inspected_bytes+total_bytes;
  IF total_rows>10000 OR inspected_bytes>67108864 THEN RAISE EXCEPTION 'Finding migration Eval inventory bound exceeded' USING ERRCODE='23514'; END IF;
  EXECUTE format('SELECT EXISTS(SELECT 1 FROM %s s CROSS JOIN pg_temp.finding_http_rewrites m WHERE position(convert_to(encode(m.old_digest,''hex''),''UTF8'') in s.%I)>0)',column_row.relation,column_row.name) INTO source;
  IF source.exists THEN RAISE EXCEPTION 'Pinned Eval document refers to a migrated Finding' USING ERRCODE='23514'; END IF;
 END LOOP;

 -- All changes below, including guard suspension, belong to the migrator's
 -- single transaction. Any exception restores both data and trigger modes.
 INSERT INTO pg_temp.finding_http_triggers
 SELECT t.tgrelid::regclass,t.tgname,t.tgenabled FROM pg_trigger t
 WHERE NOT t.tgisinternal AND t.tgrelid IN (
  SELECT relation::oid FROM pg_temp.finding_http_columns
  UNION SELECT 'artifact_blobs'::regclass::oid
  UNION SELECT 'artifact_versions'::regclass::oid
  UNION SELECT 'artifact_bindings'::regclass::oid
 );
 FOR guard IN SELECT * FROM pg_temp.finding_http_triggers LOOP
  EXECUTE format('ALTER TABLE %s DISABLE TRIGGER %I',guard.relation,guard.name);
 END LOOP;

 INSERT INTO artifact_blobs(sha256,payload,size_bytes,created_at,backend)
 SELECT m.new_digest,m.new_payload,m.new_size,b.created_at,'postgresql'
 FROM pg_temp.finding_http_rewrites m JOIN artifact_blobs b ON b.sha256=m.old_digest
 ON CONFLICT(sha256) DO NOTHING;
 INSERT INTO pg_temp.finding_http_versions SELECT v.version_id FROM artifact_versions v JOIN pg_temp.finding_http_rewrites m ON m.old_digest=v.blob_sha256;
 UPDATE artifact_versions v SET blob_sha256=m.new_digest FROM pg_temp.finding_http_rewrites m WHERE v.blob_sha256=m.old_digest;

 -- The scalar receipt identity and its generated checks size must move together.
 UPDATE finding_proposal_receipts r SET proposal_digest='sha256:'||encode(m.new_digest,'hex'),proposal_size_bytes=m.new_size,
  checks_projection=pg_temp.finding_http_canonical(pg_temp.finding_http_refs(r.checks_projection::jsonb))::json
 FROM pg_temp.finding_http_rewrites m WHERE r.proposal_digest='sha256:'||encode(m.old_digest,'hex');
 UPDATE finding_proposal_receipts SET report_projection=pg_temp.finding_http_canonical(
  (report_projection::jsonb-'http_exchange')||jsonb_build_object('http_exchanges',jsonb_build_array(report_projection::jsonb->'http_exchange')))
 WHERE report_projection::jsonb ? 'http_exchange' AND NOT report_projection::jsonb ? 'http_exchanges'
 AND jsonb_typeof(report_projection::jsonb->'http_exchange')='object';
 IF EXISTS(SELECT 1 FROM finding_proposal_receipts WHERE report_projection::jsonb ? 'http_exchange') THEN
  RAISE EXCEPTION 'Ambiguous stored Finding report format' USING ERRCODE='23514';
 END IF;

 FOR column_row IN SELECT * FROM pg_temp.finding_http_columns LOOP
  IF column_row.kind='bytes' THEN
   EXECUTE format('WITH rewritten AS MATERIALIZED (SELECT ctid rid,%I old_value,pg_temp.finding_http_bytes(%I) new_value FROM %s WHERE %I IS NOT NULL) UPDATE %s t SET %I=r.new_value FROM rewritten r WHERE t.ctid=r.rid AND r.new_value IS DISTINCT FROM r.old_value',column_row.name,column_row.name,column_row.relation,column_row.name,column_row.relation,column_row.name);
  ELSE
   EXECUTE format('WITH parsed AS MATERIALIZED (SELECT ctid rid,%I::jsonb value FROM %s WHERE %I IS NOT NULL), rewritten AS MATERIALIZED (SELECT rid,value,pg_temp.finding_http_refs(value) replacement FROM parsed) UPDATE %s t SET %I=r.replacement::%s FROM rewritten r WHERE t.ctid=r.rid AND r.replacement IS DISTINCT FROM r.value',column_row.name,column_row.relation,column_row.name,column_row.relation,column_row.name,column_row.kind);
  END IF;
 END LOOP;
 UPDATE artifact_bindings b SET retained_bytes=(
  SELECT sum(bl.size_bytes) FROM artifact_binding_revisions r JOIN artifact_versions v USING(version_id)
   JOIN artifact_blobs bl ON bl.sha256=v.blob_sha256
   WHERE (r.scope_kind,r.scope_id,r.namespace,r.name)=(b.scope_kind,b.scope_id,b.namespace,b.name))
 WHERE EXISTS(SELECT 1 FROM artifact_binding_revisions r JOIN pg_temp.finding_http_versions v USING(version_id)
  WHERE (r.scope_kind,r.scope_id,r.namespace,r.name)=(b.scope_kind,b.scope_id,b.namespace,b.name));
 DELETE FROM artifact_blobs b USING pg_temp.finding_http_rewrites m
  WHERE b.sha256=m.old_digest AND NOT EXISTS(SELECT 1 FROM artifact_versions v WHERE v.blob_sha256=b.sha256);

 FOR guard IN SELECT * FROM pg_temp.finding_http_triggers LOOP
  EXECUTE format('ALTER TABLE %s %s TRIGGER %I',guard.relation,
   CASE guard.mode WHEN 'D' THEN 'DISABLE' WHEN 'R' THEN 'ENABLE REPLICA' WHEN 'A' THEN 'ENABLE ALWAYS' ELSE 'ENABLE' END,guard.name);
 END LOOP;
END $migration$;
DROP FUNCTION pg_temp.finding_http_bytes(bytea);
DROP FUNCTION pg_temp.finding_http_refs(jsonb,integer);
DROP FUNCTION pg_temp.finding_http_canonical(jsonb,integer);
