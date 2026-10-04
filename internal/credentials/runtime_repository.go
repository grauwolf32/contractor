package credentials

import (
	"context"
	"errors"

	"github.com/grauwolf32/contractor/internal/contentdigest"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

const maximumRuntimeCredentialPageSize = 200

type RuntimeCredentialRepository struct{ db persistencepostgres.DBTX }

func NewRuntimeCredentialRepository(db persistencepostgres.DBTX) *RuntimeCredentialRepository {
	return &RuntimeCredentialRepository{db: db}
}

// Get exposes only active metadata through this repository's connection. It
// also serves transaction-scoped placement without involving the pool service.
func (r *RuntimeCredentialRepository) Get(ctx context.Context, credentialID string) (RuntimeCredentialMetadata, error) {
	record, err := r.GetActiveRecord(ctx, credentialID)
	if err != nil {
		return RuntimeCredentialMetadata{}, err
	}
	return record.Metadata, nil
}

func (r *RuntimeCredentialRepository) CountStored(ctx context.Context) (int64, error) {
	var count int64
	if r == nil || r.db == nil || r.db.QueryRow(ctx, `SELECT count(*) FROM runtime_credentials`).Scan(&count) != nil {
		return 0, errors.New("count stored Runtime credentials")
	}
	return count, nil
}

func (r *RuntimeCredentialRepository) VerifyStoredKey(ctx context.Context, keyID string) error {
	if !contentdigest.Valid(keyID) {
		return ErrKeyUnavailable
	}
	var mismatch bool
	if r == nil || r.db == nil || r.db.QueryRow(ctx, `
SELECT EXISTS (SELECT 1 FROM runtime_credentials WHERE key_id <> $1)`, keyID).Scan(&mismatch) != nil {
		return errors.New("verify stored Runtime credential encryption key")
	}
	if mismatch {
		return ErrKeyUnavailable
	}
	return nil
}

// InsertRecord stores a credential together with its create receipt. It
// reports false, without error, when the credential ID or idempotency key is
// already taken; the caller resolves that through the receipt.
func (r *RuntimeCredentialRepository) InsertRecord(
	ctx context.Context, record RuntimeCredentialRecord, creation RuntimeCredentialCreation,
) (bool, error) {
	if err := validateRuntimeCredentialRecord(record); err != nil {
		return false, err
	}
	if err := validateRuntimeCredentialCreation(creation); err != nil {
		return false, err
	}
	if creation.CredentialID != record.Metadata.CredentialID || creation.Kind != record.Metadata.Kind ||
		creation.ActorID != record.Metadata.CreatedBy || !creation.CreatedAt.Equal(record.Metadata.CreatedAt) {
		return false, runtimeInvalid("Runtime credential receipt does not match its record")
	}
	command, err := r.db.Exec(ctx, `
INSERT INTO runtime_credentials (
    credential_id, credential_kind, encryption_schema_version,
    key_id, nonce, ciphertext, created_by, created_at,
    idempotency_key_digest, request_mac
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
ON CONFLICT DO NOTHING`,
		record.Metadata.CredentialID, string(record.Metadata.Kind), record.Envelope.SchemaVersion,
		record.Envelope.KeyID, record.Envelope.Nonce, record.Envelope.Ciphertext,
		record.Metadata.CreatedBy, persistencepostgres.Timestamp(record.Metadata.CreatedAt),
		creation.IdempotencyKeyDigest, creation.RequestMAC,
	)
	if err != nil {
		return false, classifyRuntimeCredentialWrite(err)
	}
	return command.RowsAffected() == 1, nil
}

func (r *RuntimeCredentialRepository) GetActiveRecord(ctx context.Context, credentialID string) (RuntimeCredentialRecord, error) {
	if err := validateRuntimeCredentialID(credentialID); err != nil {
		return RuntimeCredentialRecord{}, err
	}
	return scanRuntimeCredentialRecord(r.db.QueryRow(ctx, runtimeCredentialRecordSelect+`
WHERE c.credential_id = $1 AND c.deleted_at IS NULL`, credentialID))
}

func (r *RuntimeCredentialRepository) GetAnyRecord(ctx context.Context, credentialID string) (RuntimeCredentialRecord, error) {
	if err := validateRuntimeCredentialID(credentialID); err != nil {
		return RuntimeCredentialRecord{}, err
	}
	return scanRuntimeCredentialRecord(r.db.QueryRow(ctx, runtimeCredentialRecordSelect+`
WHERE c.credential_id = $1`, credentialID))
}

func (r *RuntimeCredentialRepository) ListActiveMetadata(
	ctx context.Context, afterCredentialID string, limit int,
) ([]RuntimeCredentialMetadata, error) {
	if limit < 1 || limit > maximumRuntimeCredentialPageSize ||
		(afterCredentialID != "" && validateRuntimeCredentialID(afterCredentialID) != nil) {
		return nil, runtimeInvalid("Runtime credential page is invalid")
	}
	rows, err := r.db.Query(ctx, `
SELECT c.credential_id, c.credential_kind, c.created_by, c.created_at
FROM runtime_credentials c
WHERE c.credential_id > $1 AND c.deleted_at IS NULL
ORDER BY c.credential_id
LIMIT $2`, afterCredentialID, limit)
	if err != nil {
		return nil, errors.New("list Runtime credential metadata")
	}
	defer rows.Close()
	result := make([]RuntimeCredentialMetadata, 0)
	for rows.Next() {
		var metadata RuntimeCredentialMetadata
		if err := rows.Scan(&metadata.CredentialID, &metadata.Kind, &metadata.CreatedBy, &metadata.CreatedAt); err != nil {
			return nil, errors.New("read Runtime credential metadata")
		}
		result = append(result, metadata)
	}
	if rows.Err() != nil {
		return nil, errors.New("iterate Runtime credential metadata")
	}
	return result, nil
}

func (r *RuntimeCredentialRepository) GetCreation(ctx context.Context, keyDigest string) (RuntimeCredentialCreation, error) {
	if !contentdigest.Valid(keyDigest) {
		return RuntimeCredentialCreation{}, runtimeInvalid("Runtime credential idempotency digest is invalid")
	}
	var result RuntimeCredentialCreation
	err := r.db.QueryRow(ctx, `
SELECT idempotency_key_digest, request_mac, credential_id,
       credential_kind, created_by, created_at
FROM runtime_credentials
WHERE idempotency_key_digest = $1`, keyDigest).Scan(
		&result.IdempotencyKeyDigest, &result.RequestMAC, &result.CredentialID,
		&result.Kind, &result.ActorID, &result.CreatedAt,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return RuntimeCredentialCreation{}, ErrRuntimeCredentialNotFound
	}
	if err != nil {
		return RuntimeCredentialCreation{}, errors.New("get Runtime credential creation replay")
	}
	return result, nil
}

func (r *RuntimeCredentialRepository) InsertTombstone(
	ctx context.Context, tombstone RuntimeCredentialTombstone,
) (bool, error) {
	if validateRuntimeCredentialID(tombstone.CredentialID) != nil || !validActorID(tombstone.ActorID) || tombstone.DeletedAt.IsZero() {
		return false, runtimeInvalid("Runtime credential tombstone is invalid")
	}
	command, err := r.db.Exec(ctx, `
UPDATE runtime_credentials SET deleted_by = $2, deleted_at = $3
WHERE credential_id = $1 AND deleted_at IS NULL`,
		tombstone.CredentialID, tombstone.ActorID, persistencepostgres.Timestamp(tombstone.DeletedAt))
	if err != nil {
		return false, classifyRuntimeCredentialWrite(err)
	}
	return command.RowsAffected() == 1, nil
}

func (r *RuntimeCredentialRepository) GetTombstone(ctx context.Context, credentialID string) (RuntimeCredentialTombstone, error) {
	if err := validateRuntimeCredentialID(credentialID); err != nil {
		return RuntimeCredentialTombstone{}, err
	}
	var result RuntimeCredentialTombstone
	err := r.db.QueryRow(ctx, `
SELECT credential_id, deleted_by, deleted_at
FROM runtime_credentials
WHERE credential_id = $1 AND deleted_at IS NOT NULL`, credentialID).Scan(&result.CredentialID, &result.ActorID, &result.DeletedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return RuntimeCredentialTombstone{}, ErrRuntimeCredentialNotFound
	}
	if err != nil {
		return RuntimeCredentialTombstone{}, errors.New("get Runtime credential tombstone")
	}
	return result, nil
}

func (r *RuntimeCredentialRepository) InspectRuntimeCredentialUsage(
	ctx context.Context, credentialID string, limit int,
) (RuntimeCredentialUsage, error) {
	if err := validateRuntimeCredentialID(credentialID); err != nil || limit < 1 || limit > 128 {
		return RuntimeCredentialUsage{}, runtimeInvalid("Runtime credential usage query is invalid")
	}
	rows, err := r.db.Query(ctx, runtimeCredentialLabelUsageSQL, credentialID, limit)
	if err != nil {
		return RuntimeCredentialUsage{}, errors.New("inspect active Runtime credential bindings")
	}
	defer rows.Close()
	var usage RuntimeCredentialUsage
	for rows.Next() {
		var label string
		if err := rows.Scan(&label); err != nil {
			return RuntimeCredentialUsage{}, errors.New("read active Runtime credential binding")
		}
		usage.BindingLabels = append(usage.BindingLabels, label)
	}
	if rows.Err() != nil {
		return RuntimeCredentialUsage{}, errors.New("iterate active Runtime credential bindings")
	}
	projectRows, err := r.db.Query(ctx, `
SELECT project_id
FROM projects
WHERE http_target_credential_id = $1
ORDER BY project_id
LIMIT $2`, credentialID, limit)
	if err != nil {
		return RuntimeCredentialUsage{}, errors.New("inspect Runtime credential Project targets")
	}
	defer projectRows.Close()
	for projectRows.Next() {
		var projectID string
		if err := projectRows.Scan(&projectID); err != nil {
			return RuntimeCredentialUsage{}, errors.New("read Runtime credential Project target")
		}
		usage.ProjectIDs = append(usage.ProjectIDs, projectID)
	}
	if projectRows.Err() != nil {
		return RuntimeCredentialUsage{}, errors.New("iterate Runtime credential Project targets")
	}
	runRows, err := r.db.Query(ctx, `
SELECT run_id
FROM workflow_runs
WHERE state IN ('initializing', 'pending', 'running', 'waiting', 'cancelling')
  AND (
      (runtime_config_snapshot->'runtimeCredentialIds') ? $1
      OR project_http_target_snapshot#>>'{credential,credentialId}' = $1
  )
ORDER BY created_at, run_id
LIMIT $2`, credentialID, limit)
	if err != nil {
		return RuntimeCredentialUsage{}, errors.New("inspect Runtime credential Run snapshots")
	}
	defer runRows.Close()
	for runRows.Next() {
		var runID string
		if err := runRows.Scan(&runID); err != nil {
			return RuntimeCredentialUsage{}, errors.New("read Runtime credential Run snapshot")
		}
		usage.RunIDs = append(usage.RunIDs, runID)
	}
	if runRows.Err() != nil {
		return RuntimeCredentialUsage{}, errors.New("iterate Runtime credential Run snapshots")
	}
	auditRows, err := r.db.Query(ctx, `
SELECT audit_id
FROM audits
WHERE hold_state = 'held'
  AND (baseline_snapshot->'runtimeCredentialIds') ? $1
ORDER BY created_at, audit_id
LIMIT $2`, credentialID, limit)
	if err != nil {
		return RuntimeCredentialUsage{}, errors.New("inspect Runtime credential Audit holds")
	}
	defer auditRows.Close()
	for auditRows.Next() {
		var auditID string
		if err := auditRows.Scan(&auditID); err != nil {
			return RuntimeCredentialUsage{}, errors.New("read Runtime credential Audit hold")
		}
		usage.AuditIDs = append(usage.AuditIDs, auditID)
	}
	if auditRows.Err() != nil {
		return RuntimeCredentialUsage{}, errors.New("iterate Runtime credential Audit holds")
	}
	allocationRows, err := r.db.Query(ctx, runtimeCredentialAllocationUsageSQL, credentialID, limit)
	if err != nil {
		return RuntimeCredentialUsage{}, errors.New("inspect Runtime credential allocation snapshots")
	}
	defer allocationRows.Close()
	for allocationRows.Next() {
		var allocationID string
		if err := allocationRows.Scan(&allocationID); err != nil {
			return RuntimeCredentialUsage{}, errors.New("read Runtime credential allocation snapshot")
		}
		usage.AllocationIDs = append(usage.AllocationIDs, allocationID)
	}
	if allocationRows.Err() != nil {
		return RuntimeCredentialUsage{}, errors.New("iterate Runtime credential allocation snapshots")
	}
	return usage, nil
}

// ValidateRuntimeCredential lets transaction-bound consumers validate an
// exact active credential without acquiring the process lifecycle barrier a
// second time. The caller must already hold the shared reference side.
func (r *RuntimeCredentialRepository) ValidateRuntimeCredential(
	ctx context.Context, credentialID string, allowedKinds ...string,
) error {
	record, err := r.GetActiveRecord(ctx, credentialID)
	if err != nil {
		return err
	}
	return validateAllowedRuntimeKinds(record.Metadata.Kind, allowedKinds)
}

// ValidateRuntimeCredentialUse additionally requires that user may reference
// the credential from owner-scoped configuration. A credential the user may
// not use is reported exactly like a missing one, so its existence is not
// disclosed. The caller must already hold the shared reference side.
func (r *RuntimeCredentialRepository) ValidateRuntimeCredentialUse(
	ctx context.Context, user RuntimeCredentialUser, credentialID string, allowedKinds ...string,
) error {
	record, err := r.GetActiveRecord(ctx, credentialID)
	if err != nil {
		return err
	}
	if err := validateAllowedRuntimeKinds(record.Metadata.Kind, allowedKinds); err != nil {
		return err
	}
	if !user.MayUse(record.Metadata) {
		return ErrRuntimeCredentialNotFound
	}
	return nil
}

const runtimeCredentialRecordSelect = `
SELECT c.credential_id, c.credential_kind, c.created_by, c.created_at,
       c.encryption_schema_version, c.key_id, c.nonce, c.ciphertext
FROM runtime_credentials c`

func scanRuntimeCredentialRecord(row rowScanner) (RuntimeCredentialRecord, error) {
	var result RuntimeCredentialRecord
	err := row.Scan(
		&result.Metadata.CredentialID, &result.Metadata.Kind,
		&result.Metadata.CreatedBy, &result.Metadata.CreatedAt,
		&result.Envelope.SchemaVersion, &result.Envelope.KeyID,
		&result.Envelope.Nonce, &result.Envelope.Ciphertext,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return RuntimeCredentialRecord{}, ErrRuntimeCredentialNotFound
	}
	if err != nil {
		return RuntimeCredentialRecord{}, persistencepostgres.WrapError("read Runtime credential record", err)
	}
	if err := validateRuntimeCredentialRecord(result); err != nil {
		return RuntimeCredentialRecord{}, errors.New("stored Runtime credential failed integrity validation")
	}
	return result, nil
}

func validateRuntimeCredentialID(value string) error {
	if err := validateCredentialID(value); err != nil {
		return runtimeInvalid("Runtime credential ID is invalid")
	}
	return nil
}

func validateRuntimeCredentialRecord(record RuntimeCredentialRecord) error {
	if validateRuntimeCredentialID(record.Metadata.CredentialID) != nil || !validRuntimeCredentialKind(record.Metadata.Kind) ||
		!validActorID(record.Metadata.CreatedBy) || record.Metadata.CreatedAt.IsZero() ||
		record.Envelope.SchemaVersion != RuntimeCredentialSchemaVersion ||
		!contentdigest.Valid(record.Envelope.KeyID) || len(record.Envelope.Nonce) != 12 ||
		len(record.Envelope.Ciphertext) < 17 || len(record.Envelope.Ciphertext) > MaximumRuntimePlaintextBytes+16 {
		return runtimeInvalid("Runtime credential record is invalid")
	}
	return nil
}

func validateRuntimeCredentialCreation(value RuntimeCredentialCreation) error {
	if !contentdigest.Valid(value.IdempotencyKeyDigest) || len(value.RequestMAC) != 32 ||
		validateRuntimeCredentialID(value.CredentialID) != nil || !validRuntimeCredentialKind(value.Kind) ||
		!validActorID(value.ActorID) || value.CreatedAt.IsZero() {
		return runtimeInvalid("Runtime credential creation replay is invalid")
	}
	return nil
}

func classifyRuntimeCredentialWrite(err error) error {
	if class := persistencepostgres.ConstraintError(err, ErrRuntimeCredentialConflict, ErrRuntimeCredentialInvalid); class != nil {
		return class
	}
	return errors.New("persist Runtime credential state")
}

func runtimeCredentialKeyDigest(value string) (string, error) {
	if !idempotencyKeyPattern.MatchString(value) {
		return "", runtimeInvalid("Runtime credential idempotency key is invalid")
	}
	return operationRequestDigest(value)
}

func validateAllowedRuntimeKinds(actual RuntimeCredentialKind, allowed []string) error {
	if len(allowed) == 0 || len(allowed) > 4 {
		return runtimeInvalid("expected Runtime credential kinds are invalid")
	}
	seen := make(map[RuntimeCredentialKind]struct{}, len(allowed))
	found := false
	for _, candidate := range allowed {
		kind := RuntimeCredentialKind(candidate)
		if !validRuntimeCredentialKind(kind) {
			return runtimeInvalid("expected Runtime credential kinds are invalid")
		}
		if _, exists := seen[kind]; exists {
			return runtimeInvalid("expected Runtime credential kinds contain a duplicate")
		}
		seen[kind] = struct{}{}
		found = found || kind == actual
	}
	if found {
		return nil
	}
	return ErrRuntimeCredentialNotFound
}

func safeRuntimeCredentialStoreError(err error) error {
	if err == nil {
		return nil
	}
	switch {
	case errors.Is(err, ErrRuntimeCredentialInvalid), errors.Is(err, ErrRuntimeCredentialConflict),
		errors.Is(err, ErrRuntimeCredentialNotFound), errors.Is(err, ErrRuntimeCredentialInUse),
		errors.Is(err, ErrKeyUnavailable), errors.Is(err, ErrCrypto),
		errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded):
		return err
	default:
		return errors.New("Runtime credential store failure")
	}
}

var _ RuntimeCredentialUsageChecker = (*RuntimeCredentialRepository)(nil)
