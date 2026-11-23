package pgstore

// SQL statements for finding_intake.go.

// linkDirectFindingAssessmentSQL links two Artifacts to Audit $1: the direct
// verification result (logical key $2, ref $3, digest $4, media type $5, size
// $6, provenance $7, display ref $8) and its contract ($9-$15).
// Used by FindingEventsTx.CommitDirectFindingAssessment.
var linkDirectFindingAssessmentSQL = `
INSERT INTO audit_artifact_links (
    audit_id, logical_key, artifact_ref, artifact_digest,
    media_type, size_bytes, source_provenance, display_ref
) VALUES
    ($1, $2, $3::jsonb, $4, $5, $6, $7::jsonb, $8),
    ($1, $9, $10::jsonb, $11, $12, $13, $14::jsonb, $15)`
