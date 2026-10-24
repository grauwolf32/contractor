-- Audit snapshots use explicit label maps and derived placement arrays
-- (AUD-R229). Missing metadata means no placement constraint, not JSON null.
-- Existing members, including explicit nulls that strict readers reject, are
-- never replaced. This helper exists only inside the migration transaction.
CREATE FUNCTION contractor_migrate_audit_snapshot_labels(snapshot jsonb)
    RETURNS jsonb LANGUAGE plpgsql STRICT AS $$
DECLARE
    placement jsonb;
BEGIN
    IF NOT snapshot ? 'labels' THEN
        snapshot := snapshot || '{"labels":{}}'::jsonb;
    END IF;
    IF NOT snapshot ? 'requiredPlacementLabels' THEN
        placement := '[]'::jsonb;
        IF jsonb_typeof(snapshot -> 'labels') = 'object' THEN
            SELECT COALESCE(jsonb_agg(key || CASE WHEN value = '' THEN '' ELSE '/' || value END ORDER BY key), '[]'::jsonb)
              INTO placement
              FROM jsonb_each_text(snapshot -> 'labels')
             WHERE key IN ('cloud', 'local');
        END IF;
        snapshot := snapshot || jsonb_build_object('requiredPlacementLabels', placement);
    END IF;
    RETURN snapshot;
END;
$$;

-- Upgrade the immutable JSON representation once, with the same transaction
-- boundary as the migration ledger. Runtime readers keep their strict shape.
ALTER TABLE audits DISABLE TRIGGER audits_protect_preparation_baseline;
UPDATE audits
   SET input_selection = contractor_migrate_audit_snapshot_labels(input_selection),
       baseline_snapshot = contractor_migrate_audit_snapshot_labels(baseline_snapshot)
 WHERE NOT input_selection ?& ARRAY['labels', 'requiredPlacementLabels']
    OR (baseline_snapshot IS NOT NULL AND NOT baseline_snapshot ?& ARRAY['labels', 'requiredPlacementLabels']);
ALTER TABLE audits ENABLE TRIGGER audits_protect_preparation_baseline;

DROP FUNCTION contractor_migrate_audit_snapshot_labels(jsonb);
