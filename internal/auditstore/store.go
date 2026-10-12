package auditstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
)

type PostgresStore struct{ db persistencepostgres.DBTX }

func NewPostgresStore(db persistencepostgres.DBTX) *PostgresStore {
	return &PostgresStore{db: db}
}

// PostgresTxStore is a PostgresStore bound to the caller's transaction. Its
// own methods take row locks that must last until the caller commits, so they
// exist only on a store that cannot be backed by a pool.
type PostgresTxStore struct {
	*PostgresStore
	tx pgx.Tx
}

// NewPostgresTxStore binds a PostgresStore to the caller's transaction tx.
func NewPostgresTxStore(tx pgx.Tx) *PostgresTxStore {
	return &PostgresTxStore{PostgresStore: NewPostgresStore(tx), tx: tx}
}

// ListHeldAuditIDsByLLMCredential exposes only the durable dispatch-hold
// projection needed by the managed credential lifecycle. A held Audit keeps
// its exact baseline dependencies available until dispatch is closed and the
// hold is explicitly released by the Audit controller.
func (s *PostgresStore) ListHeldAuditIDsByLLMCredential(
	ctx context.Context, credentialID string, limit int,
) ([]string, error) {
	return s.listHeldAuditIDsByCredential(ctx, runtimeconfig.LLMCredentialIDsMember, "LLM", credentialID, limit)
}

// ListHeldAuditIDsByRuntimeCredential is ListHeldAuditIDsByLLMCredential for
// a Runtime credential, which Runtime credential deletion checks (AUD-R16).
func (s *PostgresStore) ListHeldAuditIDsByRuntimeCredential(
	ctx context.Context, credentialID string, limit int,
) ([]string, error) {
	return s.listHeldAuditIDsByCredential(ctx, runtimeconfig.RuntimeCredentialIDsMember, "Runtime", credentialID, limit)
}

// listHeldAuditIDsByCredential lists, oldest first, at most limit held
// Audits whose baseline names credentialID in its credential list member.
func (s *PostgresStore) listHeldAuditIDsByCredential(
	ctx context.Context, member, kind, credentialID string, limit int,
) ([]string, error) {
	if err := validateText("credentialID", credentialID, 128, true); err != nil || limit < 1 || limit > 128 {
		return nil, invalidf("Audit credential hold query is invalid")
	}
	// An Audit baseline repeats the RunSnapshot credential members.
	rows, err := s.db.Query(ctx, `
SELECT audit_id
  FROM audits
 WHERE hold_state = 'held'
   AND (baseline_snapshot->'`+member+`') ? $1
 ORDER BY created_at, audit_id
 LIMIT $2`, credentialID, limit)
	if err != nil {
		return nil, fmt.Errorf("inspect Audit %s credential holds: %w", kind, err)
	}
	defer rows.Close()
	result := make([]string, 0, limit)
	for rows.Next() {
		var auditID string
		if err := rows.Scan(&auditID); err != nil {
			return nil, fmt.Errorf("read Audit %s credential hold: %w", kind, err)
		}
		result = append(result, auditID)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate Audit %s credential holds: %w", kind, err)
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
	return s.lookupAuditReplay(ctx, ownerID, operation, key, digest)
}

func (s *PostgresStore) CreateDraft(ctx context.Context, params CreateDraftParams) (Audit, bool, error) {
	if err := validateCreate(params); err != nil {
		return Audit{}, false, err
	}
	if replay, found, err := s.lookupAuditReplay(
		ctx, params.OwnerID, MutationCreate, params.IdempotencyKey, params.RequestDigest,
	); err != nil || found {
		return replay, false, err
	}
	response, _ := json.Marshal(map[string]string{"auditId": params.AuditID})
	audit, err := scanAudit(persistencepostgres.SavepointQueryRow(ctx, s.db, createDraftSQL+prefixedAuditColumns("inserted")+` FROM inserted`,
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
	switch persistencepostgres.Classify(err) {
	case persistencepostgres.ClassDeleting:
		return Audit{}, false, ErrProjectDeleting
	case persistencepostgres.ClassReferenceViolation:
		return Audit{}, false, ErrNotFound
	}
	if !errors.Is(err, pgx.ErrNoRows) && persistencepostgres.Classify(err) != persistencepostgres.ClassDuplicate {
		return Audit{}, false, fmt.Errorf("create Audit: %w", err)
	}
	return s.replayAudit(ctx, params.OwnerID, MutationCreate, params.IdempotencyKey, params.RequestDigest)
}

func (s *PostgresStore) replayAudit(
	ctx context.Context, ownerID string, operation MutationOperation, key, digest string,
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
	ctx context.Context, ownerID string, operation MutationOperation, key, digest string,
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
		return Audit{}, true, fmt.Errorf("%w: Audit %s key", contracts.ErrIdempotencyConflict, operation)
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
SELECT `+prefixedAuditColumns("audits")+`
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
	rows, err := s.db.Query(ctx, listAuditsSQL, params.OwnerID, params.ProjectID, params.States,
		params.ProfileName, params.ProfileVersion,
		params.BeforeCreatedAt, params.BeforeAuditID, params.Limit, params.VisibleProjects)
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
		&result.DeletionRequestedAt, &result.PausedAt, &result.Phase,
	)
	if err != nil {
		return Audit{}, err
	}
	result.State = AuditState(state)
	if revision <= 0 || eventSequence < 0 || !result.State.Valid() || !result.Phase.Valid() ||
		(result.Dispatch != DispatchOpen && result.Dispatch != DispatchClosed) ||
		(result.Hold != HoldPending && result.Hold != HoldHeld && result.Hold != HoldReleased) ||
		(result.DeletionRequestedAt == nil && result.State == AuditDeleting) ||
		(result.DeletionRequestedAt != nil && result.State != AuditCancelling && result.State != AuditDeleting) {
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
