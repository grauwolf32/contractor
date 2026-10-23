package pgstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/auditstore"
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
	return s.listHeldAuditIDsByCredential(ctx, runtimeconfig.LLMCredentialIDsMember, "LLM", credentialID, limit)
}

// ListHeldAuditIDsByRuntimeCredential is ListHeldAuditIDsByLLMCredential for
// a Runtime credential, which Runtime credential deletion checks (AUD-R16).
func (s *CredentialHolds) ListHeldAuditIDsByRuntimeCredential(
	ctx context.Context, credentialID string, limit int,
) ([]string, error) {
	return s.listHeldAuditIDsByCredential(ctx, runtimeconfig.RuntimeCredentialIDsMember, "Runtime", credentialID, limit)
}

// listHeldAuditIDsByCredential lists, oldest first, at most limit held
// Audits whose baseline names credentialID in its credential list member.
func (s *CredentialHolds) listHeldAuditIDsByCredential(
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

func (s *OwnerStore) CreateDraft(ctx context.Context, params auditstore.CreateDraftParams) (auditstore.Audit, bool, error) {
	if err := validateCreate(params); err != nil {
		return auditstore.Audit{}, false, err
	}
	return s.keyedAuditMutation(ctx, params.OwnerID, auditstore.MutationCreate, params.IdempotencyKey, params.RequestDigest,
		func(tx pgx.Tx) (auditMutationResult, error) {
			audit, err := scanAudit(persistencepostgres.SavepointQueryRow(ctx, tx, createDraftSQL+prefixedAuditColumns("inserted")+` FROM inserted`,
				params.AuditID, params.OwnerID, params.ProjectID,
				params.Profile.Name, params.Profile.Version, params.Profile.Digest,
				[]byte(params.ProfileSnapshot), []byte(params.InputSelection),
				params.Limits.MaxRounds, params.Limits.BatchSize,
				params.Limits.MaxItemsPerRound, params.Limits.MaxItemsTotal,
				params.Limits.MaxSubmittedRuns, params.Limits.MaxItemRunAttempts, params.Limits.MaxEvidenceBytes,
			))
			if err == nil {
				return auditMutationResult{
					audit: audit, resourceID: audit.AuditID, response: map[string]string{"auditId": audit.AuditID},
				}, nil
			}
			switch persistencepostgres.Classify(err) {
			case persistencepostgres.ClassDeleting:
				return auditMutationResult{}, auditstore.ErrProjectDeleting
			case persistencepostgres.ClassReferenceViolation:
				return auditMutationResult{}, auditstore.ErrNotFound
			case persistencepostgres.ClassDuplicate:
				return auditMutationResult{}, auditstore.ErrConflict
			}
			if errors.Is(err, pgx.ErrNoRows) {
				// The Audit ID belongs to an Audit of another request.
				return auditMutationResult{}, auditstore.ErrConflict
			}
			return auditMutationResult{}, fmt.Errorf("create Audit: %w", err)
		})
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
		&result.OutstandingToolRunCount, &result.RetainedEvidenceBytes,
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
