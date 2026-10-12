package pgstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
)

// ListHeldAuditIDsByLLMCredential exposes only the durable dispatch-hold
// projection needed by the managed credential lifecycle. A held Audit keeps
// its exact baseline dependencies available until dispatch is closed and the
// hold is explicitly released by the Audit controller.
func (s *CredentialHolds) ListHeldAuditIDsByLLMCredential(
	ctx context.Context, credentialID string, limit int,
) ([]string, error) {
	if err := validateText("credentialID", credentialID, 128, true); err != nil || limit < 1 || limit > 128 {
		return nil, invalidf("Audit credential hold query is invalid")
	}
	// An Audit baseline repeats the RunSnapshot credential members.
	rows, err := s.db.Query(ctx, `
SELECT audit_id
  FROM audits
 WHERE hold_state = 'held'
   AND (baseline_snapshot->'`+runtimeconfig.LLMCredentialIDsMember+`') ? $1
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
func (s *OwnerStore) LookupMutationReplay(
	ctx context.Context,
	ownerID string,
	operation auditstore.MutationOperation,
	key string,
	digest string,
) (auditstore.Audit, bool, error) {
	if err := validateMutationReplay(ownerID, operation, key, digest); err != nil {
		return auditstore.Audit{}, false, err
	}
	return s.lookupAuditReplay(ctx, ownerID, operation, key, digest)
}

func (s *OwnerStore) CreateDraft(ctx context.Context, params auditstore.CreateDraftParams) (auditstore.Audit, bool, error) {
	if err := validateCreate(params); err != nil {
		return auditstore.Audit{}, false, err
	}
	if replay, found, err := s.lookupAuditReplay(
		ctx, params.OwnerID, auditstore.MutationCreate, params.IdempotencyKey, params.RequestDigest,
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
		return auditstore.Audit{}, false, auditstore.ErrProjectDeleting
	case persistencepostgres.ClassReferenceViolation:
		return auditstore.Audit{}, false, auditstore.ErrNotFound
	}
	if !errors.Is(err, pgx.ErrNoRows) && persistencepostgres.Classify(err) != persistencepostgres.ClassDuplicate {
		return auditstore.Audit{}, false, fmt.Errorf("create Audit: %w", err)
	}
	return s.replayAudit(ctx, params.OwnerID, auditstore.MutationCreate, params.IdempotencyKey, params.RequestDigest)
}

func (s *OwnerStore) replayAudit(
	ctx context.Context, ownerID string, operation auditstore.MutationOperation, key, digest string,
) (auditstore.Audit, bool, error) {
	audit, found, err := s.lookupAuditReplay(ctx, ownerID, operation, key, digest)
	if err != nil {
		return auditstore.Audit{}, false, err
	}
	if !found {
		return auditstore.Audit{}, false, auditstore.ErrConflict
	}
	return audit, false, nil
}

// operationLiteral quotes an owner mutation for a statement that records it,
// so the stored operation names come from the auditstore constants only.
func operationLiteral(operation auditstore.MutationOperation) string {
	return "'" + string(operation) + "'"
}

func (s *OwnerStore) lookupAuditReplay(
	ctx context.Context, ownerID string, operation auditstore.MutationOperation, key, digest string,
) (auditstore.Audit, bool, error) {
	var auditID, storedDigest string
	err := s.db.QueryRow(ctx, `
SELECT audit_id, request_digest
  FROM audit_idempotency
 WHERE owner_id = $1 AND operation = $2 AND idempotency_key = $3`,
		ownerID, operation, key,
	).Scan(&auditID, &storedDigest)
	if errors.Is(err, pgx.ErrNoRows) {
		return auditstore.Audit{}, false, nil
	}
	if err != nil {
		return auditstore.Audit{}, false, fmt.Errorf("resolve Audit idempotency: %w", err)
	}
	if storedDigest != digest {
		return auditstore.Audit{}, true, fmt.Errorf("%w: Audit %s key", contracts.ErrIdempotencyConflict, operation)
	}
	audit, err := s.get(ctx, ownerID, auditID)
	return audit, true, err
}

func (s *store) get(ctx context.Context, ownerID, auditID string) (auditstore.Audit, error) {
	if err := validateText("ownerID", ownerID, 256, true); err != nil {
		return auditstore.Audit{}, err
	}
	if err := validateID("auditID", auditID); err != nil {
		return auditstore.Audit{}, err
	}
	audit, err := scanAudit(s.db.QueryRow(ctx, `
SELECT `+prefixedAuditColumns("audits")+`
  FROM audits
 WHERE owner_id = $1 AND audit_id = $2`, ownerID, auditID))
	if errors.Is(err, pgx.ErrNoRows) {
		return auditstore.Audit{}, auditstore.ErrNotFound
	}
	if err != nil {
		return auditstore.Audit{}, fmt.Errorf("read Audit: %w", err)
	}
	return audit, nil
}

func (s *QueryStore) List(ctx context.Context, params auditstore.ListParams) ([]auditstore.Audit, error) {
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
	result := make([]auditstore.Audit, 0, params.Limit)
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

func scanAudit(row scanner) (auditstore.Audit, error) {
	var result auditstore.Audit
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
		return auditstore.Audit{}, err
	}
	result.State = auditstore.AuditState(state)
	if revision <= 0 || eventSequence < 0 || !result.State.Valid() || !result.Phase.Valid() ||
		(result.Dispatch != auditstore.DispatchOpen && result.Dispatch != auditstore.DispatchClosed) ||
		(result.Hold != auditstore.HoldPending && result.Hold != auditstore.HoldHeld && result.Hold != auditstore.HoldReleased) ||
		(result.DeletionRequestedAt == nil && result.State == auditstore.AuditDeleting) ||
		(result.DeletionRequestedAt != nil && result.State != auditstore.AuditCancelling && result.State != auditstore.AuditDeleting) {
		return auditstore.Audit{}, errors.New("stored Audit state is invalid")
	}
	result.Revision = uint64(revision)
	result.EventSequence = uint64(eventSequence)
	if baseline != nil {
		result.BaselineSnapshot = append(json.RawMessage(nil), baseline...)
	}
	if (stopCode == nil) != (stopMessage == nil) {
		return auditstore.Audit{}, errors.New("stored Audit stop reason is invalid")
	}
	if stopCode != nil {
		result.StopReason = &auditstore.StopReason{Code: *stopCode, Message: *stopMessage}
	}
	return result, nil
}
