package auditstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

type PostgresStore struct{ db persistencepostgres.DBTX }

var _ OwnerRepository = (*PostgresStore)(nil)
var _ ControllerRepository = (*PostgresStore)(nil)

func NewPostgresStore(db persistencepostgres.DBTX) *PostgresStore {
	return &PostgresStore{db: db}
}

// ListHeldAuditIDsByLLMCredential exposes only the durable dispatch-hold
// projection needed by the managed credential lifecycle. A held Audit keeps
// its exact baseline dependencies available until dispatch is closed and the
// hold is explicitly released by the Audit controller.
func (s *PostgresStore) ListHeldAuditIDsByLLMCredential(
	ctx context.Context, credentialID string, limit int,
) ([]string, error) {
	if err := validateText("credentialID", credentialID, 128, true); err != nil || limit < 1 || limit > 128 {
		return nil, invalidf("Audit credential hold query is invalid")
	}
	rows, err := s.db.Query(ctx, `
SELECT audit_id
  FROM audits
 WHERE hold_state = 'held'
   AND (baseline_snapshot->'llmCredentialIds') ? $1
 ORDER BY created_at, audit_id
 LIMIT $2`, credentialID, limit)
	if err != nil {
		return nil, fmt.Errorf("inspect Audit LLM credential holds: %w", err)
	}
	defer rows.Close()
	result := make([]string, 0, limit)
	for rows.Next() {
		var auditID string
		if err := rows.Scan(&auditID); err != nil {
			return nil, fmt.Errorf("read Audit LLM credential hold: %w", err)
		}
		result = append(result, auditID)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate Audit LLM credential holds: %w", err)
	}
	return result, nil
}

// LookupMutationReplay is the owner-safe pre-resolution boundary used by the
// public API. A matching replay can be returned even when a mutable profile,
// credential, Runtime label binding, or Skill binding has since disappeared.
func (s *PostgresStore) LookupMutationReplay(
	ctx context.Context,
	ownerID string,
	operation MutationOperation,
	key string,
	digest string,
) (Audit, bool, error) {
	if err := validateMutationReplay(ownerID, operation, key, digest); err != nil {
		return Audit{}, false, err
	}
	return s.lookupAuditReplay(ctx, ownerID, string(operation), key, digest)
}

const auditColumns = `
audit_id, owner_id, project_id,
profile_name, profile_version, profile_digest, profile_snapshot, input_selection,
baseline_snapshot, state, revision, current_round_id, dispatch_state, hold_state,
deadline_at, max_rounds, batch_size, max_items_per_round, max_items_total,
max_submitted_runs, max_item_run_attempts, max_evidence_bytes,
reserved_run_count, submitted_run_count, outstanding_run_count, retained_evidence_bytes,
next_event_sequence - 1, stop_reason_code, stop_reason_message,
created_at, updated_at, started_at, finished_at`

func prefixedAuditColumns(prefix string) string {
	return prefix + ".audit_id, " + prefix + ".owner_id, " + prefix + ".project_id, " +
		prefix + ".profile_name, " + prefix + ".profile_version, " + prefix + ".profile_digest, " +
		prefix + ".profile_snapshot, " + prefix + ".input_selection, " + prefix + ".baseline_snapshot, " +
		prefix + ".state, " + prefix + ".revision, " + prefix + ".current_round_id, " +
		prefix + ".dispatch_state, " + prefix + ".hold_state, " + prefix + ".deadline_at, " +
		prefix + ".max_rounds, " + prefix + ".batch_size, " + prefix + ".max_items_per_round, " +
		prefix + ".max_items_total, " + prefix + ".max_submitted_runs, " +
		prefix + ".max_item_run_attempts, " + prefix + ".max_evidence_bytes, " +
		prefix + ".reserved_run_count, " + prefix + ".submitted_run_count, " +
		prefix + ".outstanding_run_count, " + prefix + ".retained_evidence_bytes, " + prefix + ".next_event_sequence - 1, " +
		prefix + ".stop_reason_code, " + prefix + ".stop_reason_message, " +
		prefix + ".created_at, " + prefix + ".updated_at, " + prefix + ".started_at, " + prefix + ".finished_at"
}

func (s *PostgresStore) CreateDraft(ctx context.Context, params CreateDraftParams) (Audit, bool, error) {
	if err := validateCreate(params); err != nil {
		return Audit{}, false, err
	}
	if replay, found, err := s.lookupAuditReplay(
		ctx, params.OwnerID, "audit.create", params.IdempotencyKey, params.RequestDigest,
	); err != nil || found {
		return replay, false, err
	}
	response, _ := json.Marshal(map[string]string{"auditId": params.AuditID})
	audit, err := scanAudit(s.db.QueryRow(ctx, `
WITH project_gate AS MATERIALIZED (
    SELECT contractor_require_active_audit_project($3, $2)
), inserted AS (
    INSERT INTO audits (
        audit_id, owner_id, project_id,
        profile_name, profile_version, profile_digest, profile_snapshot, input_selection,
        max_rounds, batch_size, max_items_per_round, max_items_total,
        max_submitted_runs, max_item_run_attempts, max_evidence_bytes,
        next_event_sequence
    )
    SELECT $1, $2, $3, $4, $5, $6, $7::jsonb, $8::jsonb,
           $9, $10, $11, $12, $13, $14, $15, 2
      FROM project_gate
    ON CONFLICT DO NOTHING
    RETURNING *
), claim_row AS (
    INSERT INTO audit_controller_claims (audit_id)
    SELECT audit_id FROM inserted
), idempotency_row AS (
    INSERT INTO audit_idempotency (
        owner_id, operation, idempotency_key, request_digest,
        audit_id, resource_id, response_snapshot
    )
    SELECT $2, 'audit.create', $16, $17, audit_id, audit_id, $18::jsonb
      FROM inserted
), event_row AS (
    INSERT INTO audit_events (
        audit_id, sequence_number, kind, entity_id, entity_revision, summary
    )
    SELECT audit_id, 1, 'audit.created', audit_id, revision,
           jsonb_build_object('state', state)
      FROM inserted
)
SELECT `+prefixedAuditColumns("inserted")+` FROM inserted`,
		params.AuditID, params.OwnerID, params.ProjectID,
		params.Profile.Name, params.Profile.Version, params.Profile.Digest,
		[]byte(params.ProfileSnapshot), []byte(params.InputSelection),
		params.Limits.MaxRounds, params.Limits.BatchSize,
		params.Limits.MaxItemsPerRound, params.Limits.MaxItemsTotal,
		params.Limits.MaxSubmittedRuns, params.Limits.MaxItemRunAttempts, params.Limits.MaxEvidenceBytes,
		params.IdempotencyKey, params.RequestDigest, response,
	))
	if err == nil {
		return audit, true, nil
	}
	switch persistencepostgres.SQLState(err) {
	case "55000":
		return Audit{}, false, ErrProjectDeleting
	case "23503":
		return Audit{}, false, ErrNotFound
	}
	if !errors.Is(err, pgx.ErrNoRows) && persistencepostgres.SQLState(err) != "23505" {
		return Audit{}, false, fmt.Errorf("create Audit: %w", err)
	}
	return s.replayAudit(ctx, params.OwnerID, "audit.create", params.IdempotencyKey, params.RequestDigest)
}

func (s *PostgresStore) replayAudit(
	ctx context.Context, ownerID, operation, key, digest string,
) (Audit, bool, error) {
	audit, found, err := s.lookupAuditReplay(ctx, ownerID, operation, key, digest)
	if err != nil {
		return Audit{}, false, err
	}
	if !found {
		return Audit{}, false, ErrConflict
	}
	return audit, false, nil
}

func (s *PostgresStore) lookupAuditReplay(
	ctx context.Context, ownerID, operation, key, digest string,
) (Audit, bool, error) {
	var auditID, storedDigest string
	err := s.db.QueryRow(ctx, `
SELECT audit_id, request_digest
  FROM audit_idempotency
 WHERE owner_id = $1 AND operation = $2 AND idempotency_key = $3`,
		ownerID, operation, key,
	).Scan(&auditID, &storedDigest)
	if errors.Is(err, pgx.ErrNoRows) {
		return Audit{}, false, nil
	}
	if err != nil {
		return Audit{}, false, fmt.Errorf("resolve Audit idempotency: %w", err)
	}
	if storedDigest != digest {
		return Audit{}, true, ErrConflict
	}
	audit, err := s.Get(ctx, ownerID, auditID)
	return audit, true, err
}

func (s *PostgresStore) Get(ctx context.Context, ownerID, auditID string) (Audit, error) {
	if err := validateText("ownerID", ownerID, 256, true); err != nil {
		return Audit{}, err
	}
	if err := validateID("auditID", auditID); err != nil {
		return Audit{}, err
	}
	audit, err := scanAudit(s.db.QueryRow(ctx, `
SELECT `+auditColumns+`
  FROM audits
 WHERE owner_id = $1 AND audit_id = $2`, ownerID, auditID))
	if errors.Is(err, pgx.ErrNoRows) {
		return Audit{}, ErrNotFound
	}
	if err != nil {
		return Audit{}, fmt.Errorf("read Audit: %w", err)
	}
	return audit, nil
}

func (s *PostgresStore) List(ctx context.Context, params ListParams) ([]Audit, error) {
	if err := validateList(params); err != nil {
		return nil, err
	}
	var state *string
	if params.State != nil {
		value := string(*params.State)
		state = &value
	}
	rows, err := s.db.Query(ctx, `
SELECT `+auditColumns+`
  FROM audits
 WHERE owner_id = $1
   AND ($2::text IS NULL OR project_id = $2)
   AND ($3::text IS NULL OR state = $3)
   AND ($4::text IS NULL OR (profile_name = $4 AND profile_version = $5))
   AND ($6::timestamptz IS NULL OR (created_at, audit_id) < ($6, $7))
 ORDER BY created_at DESC, audit_id DESC
 LIMIT $8`, params.OwnerID, params.ProjectID, state,
		params.ProfileName, params.ProfileVersion,
		params.BeforeCreatedAt, params.BeforeAuditID, params.Limit)
	if err != nil {
		return nil, fmt.Errorf("list Audits: %w", err)
	}
	defer rows.Close()
	result := make([]Audit, 0, params.Limit)
	for rows.Next() {
		audit, scanErr := scanAudit(rows)
		if scanErr != nil {
			return nil, fmt.Errorf("scan Audit page: %w", scanErr)
		}
		result = append(result, audit)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate Audit page: %w", err)
	}
	return result, nil
}

type scanner interface{ Scan(...any) error }

func scanAudit(row scanner) (Audit, error) {
	var result Audit
	var state string
	var revision, eventSequence int64
	var stopCode, stopMessage *string
	var baseline []byte
	err := row.Scan(
		&result.AuditID, &result.OwnerID, &result.ProjectID,
		&result.Profile.Name, &result.Profile.Version, &result.Profile.Digest,
		&result.ProfileSnapshot, &result.InputSelection, &baseline,
		&state, &revision, &result.CurrentRoundID, &result.Dispatch, &result.Hold,
		&result.DeadlineAt, &result.Limits.MaxRounds, &result.Limits.BatchSize,
		&result.Limits.MaxItemsPerRound, &result.Limits.MaxItemsTotal,
		&result.Limits.MaxSubmittedRuns,
		&result.Limits.MaxItemRunAttempts, &result.Limits.MaxEvidenceBytes,
		&result.ReservedRunCount, &result.SubmittedRunCount, &result.OutstandingRunCount,
		&result.RetainedEvidenceBytes,
		&eventSequence, &stopCode, &stopMessage,
		&result.CreatedAt, &result.UpdatedAt, &result.StartedAt, &result.FinishedAt,
	)
	if err != nil {
		return Audit{}, err
	}
	result.State = AuditState(state)
	if revision <= 0 || eventSequence < 0 || !result.State.Valid() ||
		(result.Dispatch != DispatchOpen && result.Dispatch != DispatchClosed) ||
		(result.Hold != HoldPending && result.Hold != HoldHeld && result.Hold != HoldReleased) {
		return Audit{}, errors.New("stored Audit state is invalid")
	}
	result.Revision = uint64(revision)
	result.EventSequence = uint64(eventSequence)
	if baseline != nil {
		result.BaselineSnapshot = append(json.RawMessage(nil), baseline...)
	}
	if (stopCode == nil) != (stopMessage == nil) {
		return Audit{}, errors.New("stored Audit stop reason is invalid")
	}
	if stopCode != nil {
		result.StopReason = &StopReason{Code: *stopCode, Message: *stopMessage}
	}
	return result, nil
}
