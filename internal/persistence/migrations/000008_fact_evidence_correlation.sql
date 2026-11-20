-- Informational equality extends the existing qualified primary-key path.
-- No receipt, grouping table, comparison scan or key rewrite is introduced.
CREATE OR REPLACE FUNCTION contractor_finding_primary_key(selected_receipt text) RETURNS TABLE(algorithm text, value text, reason text)
    LANGUAGE plpgsql
    AS $$
DECLARE
 fingerprint jsonb;
 anchors jsonb;
 issue_keys text[];
BEGIN
 SELECT fingerprint_projection::jsonb, anchor_projection::jsonb INTO fingerprint, anchors
 FROM finding_proposal_receipts WHERE receipt_id = selected_receipt;
 IF NOT FOUND THEN
  RAISE EXCEPTION 'finding first receipt is missing' USING ERRCODE='23503';
 END IF;
 IF fingerprint->>'availability' = 'available'
  AND fingerprint->>'algorithm' = 'contractor/fact-evidence/v1' THEN
  IF NOT EXISTS (SELECT 1 FROM finding_proposal_receipts
     WHERE receipt_id=selected_receipt AND report_projection::jsonb->>'severity_suggestion'='informational') THEN
   RAISE EXCEPTION 'fact equality requires informational producer severity' USING ERRCODE='23514';
  END IF;
  RETURN QUERY SELECT 'contractor/fact-evidence/v1'::text, fingerprint->>'value', ''::text;
  RETURN;
 END IF;
 IF fingerprint->>'availability' = 'available' THEN
  IF fingerprint->>'algorithm' IS DISTINCT FROM 'contractor/producer-issue/v1'
   OR fingerprint->>'adapter' NOT IN ('pinned-sarif-native/v1','normalized-candidate/v1') THEN
   RAISE EXCEPTION 'unsupported primary producer policy' USING ERRCODE='23514';
  END IF;
  RETURN QUERY SELECT 'contractor/producer-issue/v1'::text, fingerprint->>'value', ''::text;
  RETURN;
 END IF;
 IF anchors->>'availability' = 'available' THEN
  SELECT array_agg(key) INTO issue_keys FROM (
   SELECT DISTINCT item->>'issueKey' AS key FROM jsonb_array_elements(anchors->'keys') AS item
   WHERE item ? 'issueKey' LIMIT 2
  ) AS qualified;
 END IF;
 IF cardinality(issue_keys) = 1 THEN
  RETURN QUERY SELECT 'contractor/snapshot-issue-anchor/v1'::text, issue_keys[1], ''::text;
 ELSIF cardinality(issue_keys) > 1 THEN
  RETURN QUERY SELECT ''::text, ''::text, 'primary_key_ambiguous'::text;
 ELSIF fingerprint->>'reason' = 'not_recorded' AND anchors->>'reason' = 'not_recorded' THEN
  RETURN QUERY SELECT ''::text, ''::text, 'not_recorded'::text;
 ELSE
  RETURN QUERY SELECT ''::text, ''::text, 'no_qualified_key'::text;
 END IF;
END;
$$;

ALTER TABLE project_findings DROP CONSTRAINT project_finding_primary_shape,
 ADD CONSTRAINT project_finding_primary_shape CHECK ((((correlation_algorithm = ''::text) AND (correlation_value = ''::text) AND (correlation_group_key = ''::text) AND (correlation_reason = ANY (ARRAY['not_recorded'::text, 'no_qualified_key'::text, 'primary_key_ambiguous'::text]))) OR ((correlation_algorithm = ANY (ARRAY['contractor/fact-evidence/v1'::text, 'contractor/producer-issue/v1'::text, 'contractor/snapshot-issue-anchor/v1'::text])) AND (correlation_value ~ '^[0-9a-f]{64}$'::text) AND (correlation_reason = ''::text) AND (correlation_group_key = contractor_finding_group_key(project_id, correlation_algorithm, correlation_value)))));

ALTER TABLE finding_proposal_receipts ADD CONSTRAINT finding_fact_fingerprint_shape CHECK (
 (fingerprint_projection::jsonb->>'algorithm' IS DISTINCT FROM 'contractor/fact-evidence/v1') OR
 coalesce((fingerprint_projection::jsonb->>'availability'='available'
  AND fingerprint_projection::jsonb ?& ARRAY['schema','availability','algorithm','value']
  AND (fingerprint_projection::jsonb - ARRAY['schema','availability','algorithm','value'])='{}'::jsonb
  AND jsonb_typeof(fingerprint_projection::jsonb->'value')='string'
  AND fingerprint_projection::jsonb->>'value' ~ '^[0-9a-f]{64}$'
  AND report_projection::jsonb->>'severity_suggestion'='informational'
  AND jsonb_array_length(evidence) BETWEEN 1 AND 4), false)
);
