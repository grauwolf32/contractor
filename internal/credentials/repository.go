package credentials

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"reflect"
	"regexp"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

const (
	maximumOperationRequestBytes = 64 * 1024
	maximumGatewayPolicyBytes    = 64 * 1024
	maximumCredentialPageSize    = 201
)

var (
	credentialDigestPattern = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)
	operationIDPattern      = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$`)
	idempotencyKeyPattern   = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$`)
)

// Repository exposes transaction-neutral primitives. A pgx.Tx may be passed
// to compose reservation, active-row/tombstone transitions and operation phase
// changes atomically in V4-003A.
type Repository struct {
	db persistencepostgres.DBTX
}

func NewRepository(db persistencepostgres.DBTX) *Repository { return &Repository{db: db} }

func (r *Repository) CountCredentials(ctx context.Context) (int64, error) {
	if r == nil || r.db == nil {
		return 0, errors.New("credential repository is not configured")
	}
	var count int64
	if err := r.db.QueryRow(ctx, `SELECT count(*) FROM llm_credentials`).Scan(&count); err != nil {
		return 0, errors.New("count encrypted credentials")
	}
	return count, nil
}

// VerifyActiveKey proves that every active row was sealed by the configured
// key. Comparing non-secret fingerprints at startup catches a wrong key before
// either Planner or Worker execution reaches the decryption boundary.
func (r *Repository) VerifyActiveKey(ctx context.Context, keyID string) error {
	if r == nil || r.db == nil {
		return errors.New("credential repository is not configured")
	}
	if !credentialDigestPattern.MatchString(keyID) {
		return ErrKeyUnavailable
	}
	var mismatch bool
	if err := r.db.QueryRow(ctx, `
SELECT EXISTS (SELECT 1 FROM llm_credentials WHERE key_id <> $1)`, keyID).Scan(&mismatch); err != nil {
		return errors.New("verify encrypted credential key")
	}
	if mismatch {
		return ErrKeyUnavailable
	}
	return nil
}

func (r *Repository) GetCredential(ctx context.Context, credentialID string) (Record, error) {
	if err := validateCredentialID(credentialID); err != nil {
		return Record{}, err
	}
	return scanRecord(r.db.QueryRow(ctx, `
SELECT credential_id, llm_gateway_id, llm_gateway_version, llm_gateway_digest,
       remote_key_id, COALESCE(label, ''), gateway_policy,
       encryption_schema_version, key_id, nonce, ciphertext, created_at
FROM llm_credentials
WHERE credential_id = $1`, credentialID))
}

func (r *Repository) ListCredentials(
	ctx context.Context, afterCredentialID string, limit int,
) ([]Record, error) {
	if afterCredentialID != "" {
		if err := validateCredentialID(afterCredentialID); err != nil {
			return nil, err
		}
	}
	if limit < 1 || limit > maximumCredentialPageSize {
		return nil, fmt.Errorf("%w: credential page limit is invalid", ErrInvalid)
	}
	rows, err := r.db.Query(ctx, `
SELECT credential_id, llm_gateway_id, llm_gateway_version, llm_gateway_digest,
       remote_key_id, COALESCE(label, ''), gateway_policy,
       encryption_schema_version, key_id, nonce, ciphertext, created_at
FROM llm_credentials
WHERE credential_id > $1
ORDER BY credential_id
LIMIT $2`, afterCredentialID, limit)
	if err != nil {
		return nil, errors.New("list encrypted credentials")
	}
	defer rows.Close()
	result := make([]Record, 0)
	for rows.Next() {
		record, scanErr := scanRecord(rows)
		if scanErr != nil {
			return nil, scanErr
		}
		result = append(result, record)
	}
	if err := rows.Err(); err != nil {
		return nil, errors.New("iterate encrypted credentials")
	}
	return result, nil
}

func (r *Repository) ReserveCredentialID(
	ctx context.Context, credentialID string, reservedAt time.Time,
) error {
	if err := validateCredentialID(credentialID); err != nil || reservedAt.IsZero() {
		return fmt.Errorf("%w: credential reservation is invalid", ErrInvalid)
	}
	command, err := r.db.Exec(ctx, `
INSERT INTO llm_credential_identities (credential_id, reserved_at)
VALUES ($1, $2)
ON CONFLICT DO NOTHING`, credentialID, databaseTime(reservedAt))
	if err != nil {
		return classifyRepositoryWrite(err)
	}
	if command.RowsAffected() != 1 {
		return ErrConflict
	}
	return nil
}

func (r *Repository) InsertCredential(ctx context.Context, record Record) error {
	policy, err := encodeRecord(record)
	if err != nil {
		return err
	}
	command, err := r.db.Exec(ctx, `
INSERT INTO llm_credentials (
    credential_id, llm_gateway_id, llm_gateway_version, llm_gateway_digest,
    remote_key_id, label, gateway_policy, encryption_schema_version,
    key_id, nonce, ciphertext, created_at
) VALUES ($1, $2, $3, $4, $5, NULLIF($6, ''), $7::jsonb, $8, $9, $10, $11, $12)
ON CONFLICT DO NOTHING`,
		record.CredentialID, record.LLMGateway.GatewayID, record.LLMGateway.Version,
		record.LLMGateway.Digest, record.RemoteKeyID, record.Label, policy,
		record.Envelope.SchemaVersion, record.Envelope.KeyID,
		record.Envelope.Nonce, record.Envelope.Ciphertext, databaseTime(record.CreatedAt),
	)
	if err != nil {
		return classifyRepositoryWrite(err)
	}
	if command.RowsAffected() == 1 {
		return nil
	}
	existing, getErr := r.GetCredential(ctx, record.CredentialID)
	if getErr == nil && recordsEqual(existing, record) {
		return nil
	}
	return ErrConflict
}

func (r *Repository) DeleteCredential(ctx context.Context, credentialID string) error {
	if err := validateCredentialID(credentialID); err != nil {
		return err
	}
	command, err := r.db.Exec(ctx, `DELETE FROM llm_credentials WHERE credential_id = $1`, credentialID)
	if err != nil {
		return classifyRepositoryWrite(err)
	}
	if command.RowsAffected() != 1 {
		return ErrNotFound
	}
	return nil
}

func (r *Repository) InsertTombstone(ctx context.Context, tombstone Tombstone) error {
	if err := validateCredentialID(tombstone.CredentialID); err != nil ||
		strings.TrimSpace(tombstone.ActorID) == "" || len(tombstone.ActorID) > 256 ||
		tombstone.DeletedAt.IsZero() {
		return fmt.Errorf("%w: credential tombstone is invalid", ErrInvalid)
	}
	command, err := r.db.Exec(ctx, `
INSERT INTO llm_credential_tombstones (credential_id, actor_id, deleted_at)
VALUES ($1, $2, $3)
ON CONFLICT DO NOTHING`, tombstone.CredentialID, tombstone.ActorID, databaseTime(tombstone.DeletedAt))
	if err != nil {
		return classifyRepositoryWrite(err)
	}
	if command.RowsAffected() == 1 {
		return nil
	}
	existing, getErr := r.GetTombstone(ctx, tombstone.CredentialID)
	if getErr == nil && existing.CredentialID == tombstone.CredentialID &&
		existing.ActorID == tombstone.ActorID &&
		databaseTime(existing.DeletedAt).Equal(databaseTime(tombstone.DeletedAt)) {
		return nil
	}
	return ErrConflict
}

func (r *Repository) GetTombstone(ctx context.Context, credentialID string) (Tombstone, error) {
	if err := validateCredentialID(credentialID); err != nil {
		return Tombstone{}, err
	}
	var result Tombstone
	err := r.db.QueryRow(ctx, `
SELECT credential_id, actor_id, deleted_at
FROM llm_credential_tombstones
WHERE credential_id = $1`, credentialID).Scan(
		&result.CredentialID, &result.ActorID, &result.DeletedAt,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return Tombstone{}, ErrNotFound
	}
	if err != nil {
		return Tombstone{}, errors.New("read credential tombstone")
	}
	result.DeletedAt = result.DeletedAt.UTC()
	return result, nil
}

func (r *Repository) InsertOperation(ctx context.Context, operation Operation) error {
	if err := validateOperation(operation); err != nil {
		return err
	}
	command, err := r.db.Exec(ctx, `
INSERT INTO credential_operations (
    operation_id, idempotency_key, request_hash, credential_id,
    operation_kind, phase, request_schema_version, request, created_at, updated_at
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, $10)
ON CONFLICT DO NOTHING`,
		operation.OperationID, operation.IdempotencyKey, operation.RequestHash,
		operation.CredentialID, operation.Kind, operation.Phase,
		CredentialSchemaVersion, operation.Request,
		databaseTime(operation.CreatedAt), databaseTime(operation.UpdatedAt),
	)
	if err != nil {
		return classifyRepositoryWrite(err)
	}
	if command.RowsAffected() == 1 {
		return nil
	}
	existing, getErr := r.GetOperationByIdempotency(ctx, operation.Kind, operation.IdempotencyKey)
	if getErr == nil && operationsEqual(existing, operation) {
		return nil
	}
	return ErrConflict
}

func (r *Repository) GetOperationByIdempotency(
	ctx context.Context, kind OperationKind, idempotencyKey string,
) (Operation, error) {
	if !validOperationKind(kind) || !idempotencyKeyPattern.MatchString(idempotencyKey) {
		return Operation{}, fmt.Errorf("%w: operation lookup is invalid", ErrInvalid)
	}
	return scanOperation(r.db.QueryRow(ctx, `
SELECT operation_id, idempotency_key, request_hash, credential_id,
       operation_kind, phase, request, created_at, updated_at
FROM credential_operations
WHERE operation_kind = $1 AND idempotency_key = $2`, kind, idempotencyKey))
}

func (r *Repository) CompleteOperation(
	ctx context.Context, operationID string, completedAt time.Time,
) error {
	if !operationIDPattern.MatchString(operationID) || completedAt.IsZero() {
		return fmt.Errorf("%w: operation completion is invalid", ErrInvalid)
	}
	command, err := r.db.Exec(ctx, `
UPDATE credential_operations
SET phase = 'completed', updated_at = $2
WHERE operation_id = $1 AND phase = 'prepared' AND updated_at <= $2`,
		operationID, databaseTime(completedAt),
	)
	if err != nil {
		return classifyRepositoryWrite(err)
	}
	if command.RowsAffected() == 1 {
		return nil
	}
	var phase OperationPhase
	if err := r.db.QueryRow(ctx, `SELECT phase FROM credential_operations WHERE operation_id = $1`, operationID).Scan(&phase); errors.Is(err, pgx.ErrNoRows) {
		return ErrNotFound
	} else if err != nil {
		return errors.New("read credential operation phase")
	}
	if phase == OperationCompleted {
		return nil
	}
	return ErrConflict
}

func (r *Repository) ListPreparedOperations(ctx context.Context, limit int) ([]Operation, error) {
	if limit < 1 || limit > maximumCredentialPageSize {
		return nil, fmt.Errorf("%w: operation page limit is invalid", ErrInvalid)
	}
	rows, err := r.db.Query(ctx, `
SELECT operation_id, idempotency_key, request_hash, credential_id,
       operation_kind, phase, request, created_at, updated_at
FROM credential_operations
WHERE phase = 'prepared'
ORDER BY created_at, operation_id
LIMIT $1`, limit)
	if err != nil {
		return nil, errors.New("list prepared credential operations")
	}
	defer rows.Close()
	result := make([]Operation, 0)
	for rows.Next() {
		operation, scanErr := scanOperation(rows)
		if scanErr != nil {
			return nil, scanErr
		}
		result = append(result, operation)
	}
	if err := rows.Err(); err != nil {
		return nil, errors.New("iterate prepared credential operations")
	}
	return result, nil
}

type rowScanner interface{ Scan(...any) error }

func scanRecord(row rowScanner) (Record, error) {
	var result Record
	var policy []byte
	err := row.Scan(
		&result.CredentialID, &result.LLMGateway.GatewayID, &result.LLMGateway.Version,
		&result.LLMGateway.Digest, &result.RemoteKeyID, &result.Label, &policy,
		&result.Envelope.SchemaVersion, &result.Envelope.KeyID,
		&result.Envelope.Nonce, &result.Envelope.Ciphertext, &result.CreatedAt,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return Record{}, ErrNotFound
	}
	if err != nil {
		return Record{}, errors.New("read encrypted credential")
	}
	if err := decodeStrictJSON(policy, &result.EffectivePolicy); err != nil || validateRecord(result) != nil {
		return Record{}, errors.New("stored encrypted credential is invalid")
	}
	result.CreatedAt = result.CreatedAt.UTC()
	result.Envelope.Nonce = append([]byte(nil), result.Envelope.Nonce...)
	result.Envelope.Ciphertext = append([]byte(nil), result.Envelope.Ciphertext...)
	return result, nil
}

func scanOperation(row rowScanner) (Operation, error) {
	var result Operation
	err := row.Scan(
		&result.OperationID, &result.IdempotencyKey, &result.RequestHash,
		&result.CredentialID, &result.Kind, &result.Phase, &result.Request,
		&result.CreatedAt, &result.UpdatedAt,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return Operation{}, ErrNotFound
	}
	if err != nil {
		return Operation{}, errors.New("read credential operation")
	}
	result.CreatedAt = result.CreatedAt.UTC()
	result.UpdatedAt = result.UpdatedAt.UTC()
	result.Request = append(json.RawMessage(nil), result.Request...)
	if err := validateOperation(result); err != nil {
		return Operation{}, errors.New("stored credential operation is invalid")
	}
	return result, nil
}

func encodeRecord(record Record) ([]byte, error) {
	if err := validateRecord(record); err != nil {
		return nil, err
	}
	encoded, err := json.Marshal(record.EffectivePolicy)
	if err != nil {
		return nil, fmt.Errorf("%w: effective policy cannot be encoded", ErrInvalid)
	}
	if len(encoded) > maximumGatewayPolicyBytes {
		return nil, fmt.Errorf("%w: effective policy is too large", ErrInvalid)
	}
	return encoded, nil
}

func validateRecord(record Record) error {
	if err := validateCredentialID(record.CredentialID); err != nil {
		return err
	}
	if err := record.LLMGateway.ValidateRef(); err != nil {
		return fmt.Errorf("%w: credential Gateway ref is invalid", ErrInvalid)
	}
	if strings.TrimSpace(record.RemoteKeyID) == "" || len(record.RemoteKeyID) > MaximumRemoteKeyIDBytes ||
		!utf8.ValidString(record.RemoteKeyID) {
		return fmt.Errorf("%w: remote key ID is invalid", ErrInvalid)
	}
	if record.Label != "" && (strings.TrimSpace(record.Label) == "" ||
		!utf8.ValidString(record.Label) || utf8.RuneCountInString(record.Label) > MaximumCredentialLabel) {
		return fmt.Errorf("%w: credential label is invalid", ErrInvalid)
	}
	if err := validateEffectiveGatewayPolicy(record.EffectivePolicy); err != nil {
		return err
	}
	if record.Envelope.SchemaVersion != CredentialSchemaVersion ||
		!credentialDigestPattern.MatchString(record.Envelope.KeyID) ||
		len(record.Envelope.Nonce) != 12 || len(record.Envelope.Ciphertext) < 17 ||
		len(record.Envelope.Ciphertext) > MaximumTokenBytes+16 || record.CreatedAt.IsZero() {
		return fmt.Errorf("%w: encrypted credential envelope is invalid", ErrInvalid)
	}
	return nil
}

func validateCredentialID(value string) error {
	if len(value) > 128 {
		return fmt.Errorf("%w: credential ID is invalid", ErrInvalid)
	}
	// Reuse the wire contract's exact shared configuration-ID grammar.
	if err := (contracts.LLMCredentialRef{CredentialID: value}).Validate(); err != nil {
		return fmt.Errorf("%w: credential ID is invalid", ErrInvalid)
	}
	return nil
}

func validateOperation(operation Operation) error {
	if !operationIDPattern.MatchString(operation.OperationID) ||
		!idempotencyKeyPattern.MatchString(operation.IdempotencyKey) ||
		!credentialDigestPattern.MatchString(operation.RequestHash) ||
		validateCredentialID(operation.CredentialID) != nil || !validOperationKind(operation.Kind) ||
		(operation.Phase != OperationPrepared && operation.Phase != OperationCompleted) ||
		operation.CreatedAt.IsZero() || operation.UpdatedAt.Before(operation.CreatedAt) ||
		len(operation.Request) == 0 || len(operation.Request) > maximumOperationRequestBytes {
		return fmt.Errorf("%w: credential operation is invalid", ErrInvalid)
	}
	var object map[string]json.RawMessage
	if err := decodeStrictJSON(operation.Request, &object); err != nil || object == nil {
		return fmt.Errorf("%w: credential operation request is invalid", ErrInvalid)
	}
	return nil
}

func validOperationKind(kind OperationKind) bool {
	return kind == OperationCreate || kind == OperationDelete
}

func decodeStrictJSON(data []byte, target any) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return err
	}
	if err := decoder.Decode(&json.RawMessage{}); !errors.Is(err, io.EOF) {
		return errors.New("trailing JSON value")
	}
	return nil
}

func recordsEqual(left, right Record) bool {
	leftPolicy, leftErr := json.Marshal(left.EffectivePolicy)
	rightPolicy, rightErr := json.Marshal(right.EffectivePolicy)
	return leftErr == nil && rightErr == nil && left.CredentialID == right.CredentialID &&
		left.LLMGateway == right.LLMGateway && left.RemoteKeyID == right.RemoteKeyID &&
		left.Label == right.Label && bytes.Equal(leftPolicy, rightPolicy) &&
		left.Envelope.SchemaVersion == right.Envelope.SchemaVersion &&
		left.Envelope.KeyID == right.Envelope.KeyID && bytes.Equal(left.Envelope.Nonce, right.Envelope.Nonce) &&
		bytes.Equal(left.Envelope.Ciphertext, right.Envelope.Ciphertext) &&
		databaseTime(left.CreatedAt).Equal(databaseTime(right.CreatedAt))
}

func operationsEqual(left, right Operation) bool {
	return left.OperationID == right.OperationID && left.IdempotencyKey == right.IdempotencyKey &&
		left.RequestHash == right.RequestHash && left.CredentialID == right.CredentialID &&
		left.Kind == right.Kind && left.Phase == right.Phase &&
		jsonSemanticEqual(left.Request, right.Request) &&
		databaseTime(left.CreatedAt).Equal(databaseTime(right.CreatedAt)) &&
		databaseTime(left.UpdatedAt).Equal(databaseTime(right.UpdatedAt))
}

func jsonSemanticEqual(left, right []byte) bool {
	var leftValue, rightValue any
	return json.Unmarshal(left, &leftValue) == nil && json.Unmarshal(right, &rightValue) == nil &&
		reflect.DeepEqual(leftValue, rightValue)
}

func databaseTime(value time.Time) time.Time { return value.UTC().Truncate(time.Microsecond) }

func classifyRepositoryWrite(err error) error {
	var postgresError *pgconn.PgError
	if errors.As(err, &postgresError) {
		switch postgresError.Code {
		case "23505":
			return ErrConflict
		case "23503", "23514", "22001", "22P02":
			return ErrInvalid
		}
	}
	return errors.New("persist encrypted credential state")
}
