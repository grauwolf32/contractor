package evalstore

import (
	"context"
	"encoding/json"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/evaldomain"
)

const runOutputQuery = `
SELECT binding.current_revision, 'sha256:' || encode(version.blob_sha256,
                                                  'hex'), version.media_type, blob.size_bytes
FROM workflow_runs AS run
JOIN artifact_bindings AS binding ON binding.scope_kind = 'run'
AND binding.scope_id = run.run_id
JOIN artifact_binding_revisions AS revision ON revision.scope_kind = binding.scope_kind
AND revision.scope_id = binding.scope_id
AND revision.namespace = binding.namespace
AND revision.name = binding.name
AND revision.revision = binding.current_revision
JOIN artifact_versions AS VERSION USING (version_id)
JOIN artifact_blobs AS blob ON blob.sha256 = version.blob_sha256
WHERE run.owner_id = $1
    AND run.run_id = $2
    AND binding.namespace = 'outputs'
    AND binding.name = $3
`

// RunOutput resolves one frozen output binding after authenticating its Run.
func (s *Store) RunOutput(ctx context.Context, owner, runID, name string) (evaldomain.Artifact, error) {
	ref := evaldomain.Artifact{
		Scope: "run", ScopeID: runID, Namespace: "outputs", Name: name,
	}
	err := s.db.QueryRow(ctx, runOutputQuery, owner, runID, name).Scan(
		&ref.Revision, &ref.SHA256, &ref.MediaType, &ref.SizeBytes,
	)
	return ref, normalize(err)
}

const auditOutputQuery = `
SELECT audit.project_id, link.artifact_ref, link.artifact_digest, link.media_type, link.size_bytes
FROM audits AS AUDIT
JOIN audit_artifact_links AS LINK USING (audit_id)
WHERE audit.owner_id = $1
    AND audit.audit_id = $2
    AND link.logical_key = $3
`

// AuditOutput resolves only the importer's accepted output links. Workspace
// bindings with a similar name cannot substitute for that authority.
func (s *Store) AuditOutput(ctx context.Context, owner, auditID, logicalKey string) (evaldomain.Artifact, error) {
	ref := evaldomain.Artifact{Scope: "project"}
	var encodedRef []byte
	err := s.db.QueryRow(ctx, auditOutputQuery, owner, auditID, logicalKey).Scan(
		&ref.ScopeID, &encodedRef, &ref.SHA256, &ref.MediaType, &ref.SizeBytes,
	)
	if err != nil {
		return ref, normalize(err)
	}

	var artifactRef contracts.ArtifactRef
	if err := json.Unmarshal(encodedRef, &artifactRef); err != nil {
		return ref, err
	}
	if artifactRef.Revision == nil {
		return ref, evaldomain.Failure("eval_evidence_unavailable")
	}
	ref.Namespace, ref.Name, ref.Revision = artifactRef.Namespace, artifactRef.Name, *artifactRef.Revision
	return ref, s.VerifyEvidence(ctx, ref)
}

const evidenceMetadataQuery = `
SELECT 'sha256:' || encode(version.blob_sha256, 'hex'), version.media_type, blob.size_bytes
FROM artifact_binding_revisions AS revision
JOIN artifact_versions AS VERSION USING (version_id)
JOIN artifact_blobs AS blob ON blob.sha256 = version.blob_sha256
WHERE revision.scope_kind = $1
    AND revision.scope_id = $2
    AND revision.namespace = $3
    AND revision.name = $4
    AND revision.revision = $5
`

// VerifyEvidence verifies metadata for an already-authorized exact reference.
// Public ingestion must establish ownership and member association first.
func (s *Store) VerifyEvidence(ctx context.Context, ref evaldomain.Artifact) error {
	var digest, mediaType string
	var sizeBytes int64
	err := s.db.QueryRow(ctx, evidenceMetadataQuery,
		ref.Scope, ref.ScopeID, ref.Namespace, ref.Name, ref.Revision,
	).Scan(&digest, &mediaType, &sizeBytes)
	if err != nil {
		if evaldomain.IsCode(normalize(err), "eval_not_found") {
			return evaldomain.Failure("eval_evidence_unavailable")
		}
		return err
	}
	if digest != ref.SHA256 || mediaType != ref.MediaType || sizeBytes != ref.SizeBytes {
		return evaldomain.Failure("eval_member_conflict")
	}
	return nil
}

// AuthorizeEvidence follows retained member associations; a same-owner Run from
// another case is not evidence for this member. Audit project evidence must be
// one of the importer's exact accepted links.
func (s *Store) AuthorizeEvidence(ctx context.Context, owner, id, member string, ref evaldomain.Artifact) error {
	var allowed bool
	err := s.db.QueryRow(ctx, `
SELECT EXISTS (
    SELECT 1 FROM eval_submissions s
    JOIN eval_experiments e USING (experiment_id)
    WHERE e.owner_id = $1 AND s.experiment_id = $2 AND s.member_id = $3
        AND (
            ($4 = 'run' AND (
                (s.execution_kind = 'run' AND s.execution_id = $5)
                OR EXISTS (SELECT 1 FROM audit_executions x
                    WHERE s.execution_kind = 'audit'
                        AND x.audit_id = s.execution_id AND x.run_id = $5)
            ))
            OR ($4 = 'project' AND EXISTS (
                SELECT 1 FROM audits a
                JOIN audit_artifact_links l USING (audit_id)
                WHERE s.execution_kind = 'audit' AND a.audit_id = s.execution_id
                    AND a.owner_id = e.owner_id AND a.project_id = $5
                    AND l.artifact_ref ->> 'namespace' = $6
                    AND l.artifact_ref ->> 'name' = $7
                    AND l.artifact_ref ->> 'revision' = $8
            ))
        )
)
`, owner, id, member, ref.Scope, ref.ScopeID, ref.Namespace, ref.Name, ref.Revision).Scan(&allowed)
	if err != nil {
		return err
	}
	if !allowed {
		return evaldomain.Failure("eval_not_found")
	}
	return nil
}
