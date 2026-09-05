package credentials

import (
	"context"
	"crypto/hmac"
	"errors"
	"regexp"
	"time"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

const maximumRuntimeCredentialPageSize = 200

var runtimeCredentialDigestPattern = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)

type RuntimeCredentialRepository struct{ db persistencepostgres.DBTX }

func NewRuntimeCredentialRepository(db persistencepostgres.DBTX) *RuntimeCredentialRepository {
	return &RuntimeCredentialRepository{db: db}
}

func (r *RuntimeCredentialRepository) CountActive(ctx context.Context) (int64, error) {
	var count int64
	if r == nil || r.db == nil || r.db.QueryRow(ctx, `
SELECT count(*)
FROM runtime_credentials c
WHERE NOT EXISTS (
    SELECT 1 FROM runtime_credential_tombstones t WHERE t.credential_id = c.credential_id
)`).Scan(&count) != nil {
		return 0, errors.New("count active Runtime credentials")
	}
	return count, nil
}

func (r *RuntimeCredentialRepository) CountStored(ctx context.Context) (int64, error) {
	var count int64
	if r == nil || r.db == nil || r.db.QueryRow(ctx, `SELECT count(*) FROM runtime_credentials`).Scan(&count) != nil {
		return 0, errors.New("count stored Runtime credentials")
	}
	return count, nil
}

func (r *RuntimeCredentialRepository) VerifyActiveKey(ctx context.Context, keyID string) error {
	if !runtimeCredentialDigestPattern.MatchString(keyID) {
		return ErrKeyUnavailable
	}
	var mismatch bool
	if r == nil || r.db == nil || r.db.QueryRow(ctx, `
SELECT EXISTS (
    SELECT 1
    FROM runtime_credentials c
    WHERE c.key_id <> $1
      AND NOT EXISTS (
          SELECT 1 FROM runtime_credential_tombstones t WHERE t.credential_id = c.credential_id
      )
)`, keyID).Scan(&mismatch) != nil {
		return errors.New("verify Runtime credential encryption key")
	}
	if mismatch {
		return ErrKeyUnavailable
	}
	return nil
}

func (r *RuntimeCredentialRepository) VerifyStoredKey(ctx context.Context, keyID string) error {
	if !runtimeCredentialDigestPattern.MatchString(keyID) {
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

func (r *RuntimeCredentialRepository) InsertRecord(ctx context.Context, record RuntimeCredentialRecord) (bool, error) {
	if err := validateRuntimeCredentialRecord(record); err != nil {
		return false, err
	}
	command, err := r.db.Exec(ctx, `
INSERT INTO runtime_credentials (
    credential_id, credential_kind, encryption_schema_version,
    key_id, nonce, ciphertext, created_by, created_at
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
ON CONFLICT DO NOTHING`,
		record.Metadata.CredentialID, string(record.Metadata.Kind), record.Envelope.SchemaVersion,
		record.Envelope.KeyID, record.Envelope.Nonce, record.Envelope.Ciphertext,
		record.Metadata.CreatedBy, runtimeDatabaseTime(record.Metadata.CreatedAt),
	)
	if err != nil {
		return false, classifyRuntimeCredentialWrite(err)
	}
	if command.RowsAffected() == 1 {
		return true, nil
	}
	existing, getErr := r.GetAnyRecord(ctx, record.Metadata.CredentialID)
	if getErr == nil && runtimeCredentialRecordsEqual(existing, record) {
		return false, nil
	}
	return false, ErrRuntimeCredentialConflict
}

func (r *RuntimeCredentialRepository) GetActiveRecord(ctx context.Context, credentialID string) (RuntimeCredentialRecord, error) {
	if err := validateRuntimeCredentialID(credentialID); err != nil {
		return RuntimeCredentialRecord{}, err
	}
	return scanRuntimeCredentialRecord(r.db.QueryRow(ctx, runtimeCredentialRecordSelect+`
WHERE c.credential_id = $1
  AND NOT EXISTS (
      SELECT 1 FROM runtime_credential_tombstones t WHERE t.credential_id = c.credential_id
  )`, credentialID))
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
WHERE c.credential_id > $1
  AND NOT EXISTS (
      SELECT 1 FROM runtime_credential_tombstones t WHERE t.credential_id = c.credential_id
  )
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

func (r *RuntimeCredentialRepository) InsertCreation(ctx context.Context, creation RuntimeCredentialCreation) (bool, error) {
	if err := validateRuntimeCredentialCreation(creation); err != nil {
		return false, err
	}
	command, err := r.db.Exec(ctx, `
INSERT INTO runtime_credential_creations (
    idempotency_key_digest, request_mac, credential_id,
    credential_kind, actor_id, created_at
) VALUES ($1, $2, $3, $4, $5, $6)
ON CONFLICT DO NOTHING`,
		creation.IdempotencyKeyDigest, creation.RequestMAC, creation.CredentialID,
		string(creation.Kind), creation.ActorID, runtimeDatabaseTime(creation.CreatedAt),
	)
	if err != nil {
		return false, classifyRuntimeCredentialWrite(err)
	}
	if command.RowsAffected() == 1 {
		return true, nil
	}
	existing, getErr := r.GetCreation(ctx, creation.IdempotencyKeyDigest)
	if getErr == nil && runtimeCredentialCreationsEqual(existing, creation) {
		return false, nil
	}
	return false, ErrRuntimeCredentialConflict
}

func (r *RuntimeCredentialRepository) GetCreation(ctx context.Context, keyDigest string) (RuntimeCredentialCreation, error) {
	if !runtimeCredentialDigestPattern.MatchString(keyDigest) {
		return RuntimeCredentialCreation{}, runtimeInvalid("Runtime credential idempotency digest is invalid")
	}
	var result RuntimeCredentialCreation
	err := r.db.QueryRow(ctx, `
SELECT idempotency_key_digest, request_mac, credential_id,
       credential_kind, actor_id, created_at
FROM runtime_credential_creations
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
INSERT INTO runtime_credential_tombstones (credential_id, actor_id, deleted_at)
VALUES ($1, $2, $3)
ON CONFLICT DO NOTHING`, tombstone.CredentialID, tombstone.ActorID, runtimeDatabaseTime(tombstone.DeletedAt))
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
SELECT credential_id, actor_id, deleted_at
FROM runtime_credential_tombstones
WHERE credential_id = $1`, credentialID).Scan(&result.CredentialID, &result.ActorID, &result.DeletedAt)
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
	rows, err := r.db.Query(ctx, `
SELECT b.label
FROM runtime_label_bindings b
JOIN runtime_config_versions c
  ON c.name = b.config_name
 AND c.version = b.config_version
 AND c.digest = b.config_digest
WHERE c.canonical_document::jsonb #>> '{spec,worker,telemetry,credential}' = $1
   OR c.canonical_document::jsonb #>> '{spec,worker,httpProxy,credential}' = $1
   OR c.canonical_document::jsonb #>> '{spec,worker,caido,credential}' = $1
   OR c.canonical_document::jsonb #>> '{spec,planner,telemetry,credential}' = $1
ORDER BY b.label
LIMIT $2`, credentialID, limit)
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
WHERE state IN ('initializing', 'running', 'cancelling')
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
	allocationRows, err := r.db.Query(ctx, `
SELECT a.allocation_id
FROM stage_allocations a
JOIN stage_executions e ON e.stage_execution_id = a.stage_execution_id
JOIN workflow_runs r ON r.run_id = e.run_id
WHERE a.release_completed_at IS NULL
  AND (
      a.runtime_configuration->'provenance'->'runtimeCredentialRefs'
          @> jsonb_build_array(jsonb_build_object('credentialId', $1::text))
      OR r.project_http_target_snapshot#>>'{credential,credentialId}' = $1
  )
ORDER BY a.created_at, a.allocation_id
LIMIT $2`, credentialID, limit)
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
		return RuntimeCredentialRecord{}, errors.New("read Runtime credential record")
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
		!runtimeCredentialDigestPattern.MatchString(record.Envelope.KeyID) || len(record.Envelope.Nonce) != 12 ||
		len(record.Envelope.Ciphertext) < 17 || len(record.Envelope.Ciphertext) > MaximumRuntimePlaintextBytes+16 {
		return runtimeInvalid("Runtime credential record is invalid")
	}
	return nil
}

func validateRuntimeCredentialCreation(value RuntimeCredentialCreation) error {
	if !runtimeCredentialDigestPattern.MatchString(value.IdempotencyKeyDigest) || len(value.RequestMAC) != 32 ||
		validateRuntimeCredentialID(value.CredentialID) != nil || !validRuntimeCredentialKind(value.Kind) ||
		!validActorID(value.ActorID) || value.CreatedAt.IsZero() {
		return runtimeInvalid("Runtime credential creation replay is invalid")
	}
	return nil
}

func runtimeCredentialRecordsEqual(left, right RuntimeCredentialRecord) bool {
	return left.Metadata.CredentialID == right.Metadata.CredentialID && left.Metadata.Kind == right.Metadata.Kind &&
		left.Metadata.CreatedBy == right.Metadata.CreatedBy && runtimeDatabaseTime(left.Metadata.CreatedAt).Equal(runtimeDatabaseTime(right.Metadata.CreatedAt)) &&
		left.Envelope.SchemaVersion == right.Envelope.SchemaVersion && left.Envelope.KeyID == right.Envelope.KeyID &&
		string(left.Envelope.Nonce) == string(right.Envelope.Nonce) && string(left.Envelope.Ciphertext) == string(right.Envelope.Ciphertext)
}

func runtimeCredentialCreationsEqual(left, right RuntimeCredentialCreation) bool {
	return left.IdempotencyKeyDigest == right.IdempotencyKeyDigest && hmac.Equal(left.RequestMAC, right.RequestMAC) &&
		left.CredentialID == right.CredentialID && left.Kind == right.Kind && left.ActorID == right.ActorID &&
		runtimeDatabaseTime(left.CreatedAt).Equal(runtimeDatabaseTime(right.CreatedAt))
}

func runtimeDatabaseTime(value time.Time) time.Time { return value.UTC().Truncate(time.Microsecond) }

func classifyRuntimeCredentialWrite(err error) error {
	var postgresError *pgconn.PgError
	if errors.As(err, &postgresError) {
		switch postgresError.Code {
		case "23505":
			return ErrRuntimeCredentialConflict
		case "23503", "23514", "22001", "22P02":
			return ErrRuntimeCredentialInvalid
		}
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
