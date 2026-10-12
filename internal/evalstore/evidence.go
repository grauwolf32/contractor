package evalstore

import (
	"context"
	"encoding/json"
	"errors"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/evaldomain"
)

// RunOutput resolves one frozen output binding after authenticating its Run.
func (s *Store) RunOutput(ctx context.Context, owner, runID, name string) (evaldomain.Artifact, error) {
	ref := evaldomain.Artifact{
		Scope: "run", ScopeID: runID, Namespace: "outputs", Name: name,
	}
	var owned bool
	if err := s.db.QueryRow(ctx, `
SELECT EXISTS(SELECT 1 FROM workflow_runs WHERE owner_id = $1 AND run_id = $2)`, owner, runID,
	).Scan(&owned); err != nil {
		return ref, normalize(err)
	}
	if !owned {
		return ref, evaldomain.Failure("eval_not_found")
	}
	metadata, err := s.artifactMetadata(ctx, ref, nil)
	if artifactAbsent(err) {
		return ref, evaldomain.Failure("eval_not_found")
	}
	if err != nil {
		return ref, err
	}
	ref.Revision, ref.SHA256 = *metadata.Ref.Revision, metadata.Digest
	ref.MediaType, ref.SizeBytes = metadata.MediaType, metadata.Size
	return ref, nil
}

// artifactMetadata reads, through the ArtifactStore, the metadata of the
// binding ref names: of its exact revision when revision is set, otherwise of
// its current revision.
func (s *Store) artifactMetadata(
	ctx context.Context, ref evaldomain.Artifact, revision *string,
) (artifacts.Metadata, error) {
	var scope artifacts.Scope
	var err error
	switch ref.Scope {
	case "run":
		scope, err = artifacts.RunScope(ref.ScopeID)
	case "project":
		scope, err = artifacts.ProjectScope(ref.ScopeID)
	case "user":
		scope, err = artifacts.UserScope(ref.ScopeID)
	default:
		err = artifacts.ErrInvalidScope
	}
	if err != nil {
		return artifacts.Metadata{}, err
	}
	return artifacts.NewPostgresRepository(s.db).Metadata(ctx, scope, artifacts.ArtifactRef{
		Namespace: ref.Namespace, Name: ref.Name, Revision: revision,
	})
}

// artifactAbsent reports whether an artifactMetadata error means that no such
// binding revision exists, as for a reference no binding can have.
func artifactAbsent(err error) bool {
	return errors.Is(err, artifacts.ErrArtifactNotFound) || errors.Is(err, artifacts.ErrInvalidScope) ||
		errors.Is(err, artifacts.ErrInvalidName)
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

// VerifyEvidence verifies metadata for an already-authorized exact reference.
// Public ingestion must establish ownership and member association first.
func (s *Store) VerifyEvidence(ctx context.Context, ref evaldomain.Artifact) error {
	revision := ref.Revision
	metadata, err := s.artifactMetadata(ctx, ref, &revision)
	if artifactAbsent(err) {
		return evaldomain.Failure("eval_evidence_unavailable")
	}
	if err != nil {
		return err
	}
	if metadata.Digest != ref.SHA256 || metadata.MediaType != ref.MediaType || metadata.Size != ref.SizeBytes {
		return evaldomain.Failure("eval_member_conflict")
	}
	return nil
}

// AuthorizeEvidence follows retained member associations; a same-owner Run from
// another case is not evidence for this member. Audit project evidence must be
// one of the importer's exact accepted links.
func (s *Store) AuthorizeEvidence(ctx context.Context, owner, id, member string, ref evaldomain.Artifact) error {
	var allowed bool
	err := s.db.QueryRow(ctx, authorizeEvidenceSQL, owner, id, member, ref.Scope, ref.ScopeID, ref.Namespace, ref.Name, ref.Revision).Scan(&allowed)
	if err != nil {
		return err
	}
	if !allowed {
		return evaldomain.Failure("eval_not_found")
	}
	return nil
}
