package pgstore

import (
	"context"
	"errors"

	"github.com/jackc/pgx/v5"

	"github.com/grauwolf32/contractor/internal/auditstore"
)

func (r *admissionReader) GetArtifactLink(ctx context.Context, auditID, key string) (auditstore.ArtifactLink, error) {
	if auditID != r.state.AuditID {
		return auditstore.ArtifactLink{}, auditstore.ErrInvalid
	}
	var size int
	if err := r.db.QueryRow(ctx, `SELECT octet_length(artifact_ref::text)+octet_length(source_provenance::text)+length(logical_key)+length(COALESCE(display_ref,''))+512 FROM audit_artifact_links WHERE audit_id=$1 AND logical_key=$2`, auditID, key).Scan(&size); err != nil {
		return auditstore.ArtifactLink{}, admissionReadError(err)
	}
	if err := r.work.add(0, 1, size, 0); err != nil {
		return auditstore.ArtifactLink{}, admissionReadError(err)
	}
	return r.QueryStore.GetArtifactLink(ctx, auditID, key)
}
func (r *admissionReader) GetPreparationOutput(ctx context.Context, auditID, role, name string) (auditstore.AcceptedPreparationOutput, error) {
	if auditID != r.state.AuditID {
		return auditstore.AcceptedPreparationOutput{}, auditstore.ErrInvalid
	}
	var size, related int
	if err := r.db.QueryRow(ctx, admissionPreparationCostSQL, auditID, role, name).Scan(&size, &related); err != nil {
		return auditstore.AcceptedPreparationOutput{}, admissionReadError(err)
	}
	if err := r.work.add(0, related, size, 0); err != nil {
		return auditstore.AcceptedPreparationOutput{}, admissionReadError(err)
	}
	return r.QueryStore.GetPreparationOutput(ctx, auditID, role, name)
}
func (r *admissionReader) DependencyResultSource(ctx context.Context, auditID, executionID, subject string) (auditstore.DependencyResultSource, error) {
	if auditID != r.state.AuditID {
		return auditstore.DependencyResultSource{}, auditstore.ErrInvalid
	}
	var size, related int
	if err := r.db.QueryRow(ctx, `SELECT octet_length(collection_retained_refs::text)+octet_length(collection_source_output_ref::text)+4096,2+jsonb_array_length(collection_retained_refs) FROM audit_executions WHERE audit_id=$1 AND execution_id=$2`, auditID, executionID).Scan(&size, &related); err != nil {
		return auditstore.DependencyResultSource{}, admissionReadError(err)
	}
	if err := r.work.add(0, related, size, 0); err != nil {
		return auditstore.DependencyResultSource{}, admissionReadError(err)
	}
	return r.QueryStore.DependencyResultSource(ctx, auditID, executionID, subject)
}

func admissionReadError(err error) error {
	if errors.Is(err, pgx.ErrNoRows) {
		return auditstore.ErrNotFound
	}
	return err
}
