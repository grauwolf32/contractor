package auditservice

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/credentials"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
)

func validateDeadlineSeconds(seconds *int) error {
	if seconds != nil && (*seconds < 0 || *seconds > config.MaxAuditDeadlineSeconds) {
		return fmt.Errorf("%w: Audit time limit must be between 0 (unlimited) and %d seconds", ErrInvalid, config.MaxAuditDeadlineSeconds)
	}
	return nil
}

// Resume keeps the pinned baseline and every accepted result. The only terminal
// states it can reopen are legacy deadline closures; other failures stay final.
func (s *Service) Resume(ctx context.Context, params MutationParams) (MutationResult, error) {
	if err := validateMutationParams(params); err != nil {
		return MutationResult{}, err
	}
	if err := validateDeadlineSeconds(params.DeadlineSeconds); err != nil {
		return MutationResult{}, err
	}
	store := auditstore.NewPostgresStore(s.pool)
	if replay, found, err := store.LookupMutationReplay(ctx, params.OwnerID, auditstore.MutationTransition, params.IdempotencyKey, params.RequestDigest); err != nil || found {
		return MutationResult{Audit: replay, Replayed: found}, err
	}
	var result MutationResult
	err := s.credentialGuard.WithRunCreation(ctx, func() error {
		return persistencepostgres.InTxWithRetry(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
			var err error
			result, err = s.resumeInTransaction(ctx, tx, params)
			return err
		})
	})
	return result, err
}

func (s *Service) resumeInTransaction(ctx context.Context, tx pgx.Tx, params MutationParams) (MutationResult, error) {
	store := auditstore.NewPostgresStore(tx)
	// Serialize retries before any mutable dependency checks.
	if _, err := tx.Exec(ctx, `SELECT pg_advisory_xact_lock(hashtextextended($1, 0))`, "audit-resume:"+params.OwnerID+":"+params.IdempotencyKey); err != nil {
		return MutationResult{}, err
	}
	if replay, found, err := store.LookupMutationReplay(ctx, params.OwnerID, auditstore.MutationTransition, params.IdempotencyKey, params.RequestDigest); err != nil || found {
		return MutationResult{Audit: replay, Replayed: found}, err
	}
	audit, err := store.Get(ctx, params.OwnerID, params.AuditID)
	if err != nil {
		return MutationResult{}, err
	}
	if _, err := projectstore.NewPostgresStore(tx).LockActiveAuditProject(ctx, params.OwnerID, audit.ProjectID); err != nil {
		return MutationResult{}, err
	}
	if _, err := tx.Exec(ctx, `SELECT audit_id FROM audits WHERE audit_id=$1 AND owner_id=$2 FOR UPDATE`, params.AuditID, params.OwnerID); err != nil {
		return MutationResult{}, err
	}
	audit, err = store.Get(ctx, params.OwnerID, params.AuditID)
	if err != nil {
		return MutationResult{}, err
	}
	legacy := (audit.State == auditstore.AuditCompleted || audit.State == auditstore.AuditFailed) && audit.StopReason != nil && audit.StopReason.Code == "deadline_exhausted"
	if audit.Revision != params.ExpectedRevision || (audit.State != auditstore.AuditPaused && !legacy) || audit.CurrentRoundID == nil {
		return MutationResult{}, auditstore.ErrPrecondition
	}
	var candidate bool
	if err := tx.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM audit_report_candidates WHERE audit_id=$1)`, audit.AuditID).Scan(&candidate); err != nil {
		return MutationResult{}, err
	}
	if candidate {
		return MutationResult{}, auditstore.ErrPrecondition
	}
	now := s.now().UTC()
	deadline, err := resumeDeadline(audit, params.DeadlineSeconds, now)
	if err != nil {
		return MutationResult{}, err
	}
	if legacy {
		if audit.OutstandingRunCount != 0 {
			return MutationResult{}, auditstore.ErrPrecondition
		}
		var unfinished bool
		if err := tx.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM audit_executions WHERE audit_id=$1 AND state <> 'collected')`, audit.AuditID).Scan(&unfinished); err != nil {
			return MutationResult{}, err
		}
		if unfinished {
			return MutationResult{}, auditstore.ErrPrecondition
		}
		if err := s.validateContinuationDependencies(ctx, tx, audit); err != nil {
			return MutationResult{}, err
		}
		if err := reopenDeadlineItems(ctx, tx, audit); err != nil {
			return MutationResult{}, err
		}
		// Historical report links and exact bytes remain available. New report
		// publication uses a distinct artifact name for this continuation.
		if _, err := tx.Exec(ctx, `UPDATE audit_artifact_links SET logical_key = 'report/history/' || $2::text || '/' || split_part(logical_key, '/', 2) WHERE audit_id=$1 AND logical_key IN ('report/machine','report/summary')`, audit.AuditID, fmt.Sprint(audit.Revision)); err != nil {
			return MutationResult{}, err
		}
		// Re-enter the existing round; accepted discovery results are reused.
		if _, err := tx.Exec(ctx, `UPDATE audit_rounds SET state='accepted', revision=revision+1, updated_at=clock_timestamp() WHERE audit_id=$1 AND round_id=$2`, audit.AuditID, *audit.CurrentRoundID); err != nil {
			return MutationResult{}, err
		}
	}
	if err := renewExpiredItemReviews(ctx, tx, audit, deadline, now); err != nil {
		return MutationResult{}, err
	}
	if _, err := tx.Exec(ctx, `UPDATE audits SET state='active', dispatch_state='open', hold_state='held', deadline_at=$2, paused_at=NULL, finished_at=NULL, stop_reason_code=NULL, stop_reason_message=NULL, continuation_count=continuation_count+$3, revision=revision+1, next_event_sequence=next_event_sequence+1, updated_at=GREATEST(clock_timestamp(),updated_at+interval '1 microsecond') WHERE audit_id=$1`, audit.AuditID, deadline, boolInt(legacy)); err != nil {
		return MutationResult{}, err
	}
	summary, _ := json.Marshal(map[string]any{"from": audit.State, "to": "active", "previousStopReason": audit.StopReason, "deadlineAt": deadline, "continued": legacy})
	if _, err := tx.Exec(ctx, `INSERT INTO audit_events(audit_id,sequence_number,kind,entity_id,entity_revision,summary) SELECT audit_id,next_event_sequence-1,'audit.resumed',audit_id,revision,$2::jsonb FROM audits WHERE audit_id=$1`, audit.AuditID, summary); err != nil {
		return MutationResult{}, err
	}
	response, _ := json.Marshal(map[string]string{"auditId": audit.AuditID})
	if _, err := tx.Exec(ctx, `INSERT INTO audit_idempotency(owner_id,operation,idempotency_key,request_digest,audit_id,resource_id,response_snapshot) VALUES($1,'audit.transition',$2,$3,$4,$4,$5::jsonb)`, params.OwnerID, params.IdempotencyKey, params.RequestDigest, audit.AuditID, response); err != nil {
		return MutationResult{}, err
	}
	updated, err := store.Get(ctx, params.OwnerID, audit.AuditID)
	return MutationResult{Audit: updated}, err
}

func boolInt(value bool) int {
	if value {
		return 1
	}
	return 0
}

func resumeDeadline(audit auditstore.Audit, seconds *int, now time.Time) (*time.Time, error) {
	if seconds != nil {
		if *seconds == 0 {
			return nil, nil
		}
		deadline := now.Add(timeDurationSeconds(*seconds))
		return &deadline, nil
	}
	if audit.State == auditstore.AuditPaused && audit.DeadlineAt == nil {
		return nil, nil
	}
	if audit.State == auditstore.AuditPaused && audit.PausedAt != nil && audit.DeadlineAt != nil && audit.DeadlineAt.After(*audit.PausedAt) {
		deadline := now.Add(audit.DeadlineAt.Sub(*audit.PausedAt))
		return &deadline, nil
	}
	profile, err := config.DecodeResolvedAuditProfileSnapshot(audit.ProfileSnapshot)
	if err != nil {
		return nil, err
	}
	deadline := now.Add(timeDurationSeconds(profile.Execution.DeadlineSeconds))
	return &deadline, nil
}

func (s *Service) validateContinuationDependencies(ctx context.Context, tx pgx.Tx, audit auditstore.Audit) error {
	baseline, err := DecodeBaseline(audit.BaselineSnapshot)
	if err != nil {
		return err
	}
	profile, err := config.DecodeResolvedAuditProfileSnapshot(audit.ProfileSnapshot)
	if err != nil {
		return err
	}
	selection, err := DecodeDraftSelection(audit.InputSelection)
	if err != nil {
		return err
	}
	if _, err := readAndVerifyInputs(ctx, artifacts.NewService(artifacts.NewPostgresRepository(tx)), audit.ProjectID, profile, selection); err != nil {
		return err
	}
	lookup, err := runtimeconfig.BindTransactionLLMCredentialLookup(tx, s.transactionLLMCredentials)
	if err != nil {
		return err
	}
	if _, _, _, err := s.validateProfileDependencies(ctx, profile, lookup); err != nil {
		return err
	}
	for _, id := range baseline.LLMCredentialIDs {
		if _, err := lookup.LookupLLMCredential(ctx, id); err != nil {
			return err
		}
	}
	repository := credentials.NewRuntimeCredentialRepository(tx)
	for _, id := range baseline.RuntimeCredentialIDs {
		if _, err := repository.GetActiveRecord(ctx, id); err != nil {
			return err
		}
	}
	return nil
}

func reopenDeadlineItems(ctx context.Context, tx pgx.Tx, audit auditstore.Audit) error {
	_, err := tx.Exec(ctx, `WITH reopened AS (
        UPDATE audit_items AS item SET state=CASE WHEN approval_kind='none' THEN 'ready' ELSE 'awaiting_review' END, final_disposition=NULL, updated_at=clock_timestamp()
        WHERE item.audit_id=$1 AND item.round_id=$2 AND item.state='settled'
          AND ((item.final_disposition='excluded' AND EXISTS(SELECT 1 FROM audit_coverage_rows c WHERE c.item_id=item.item_id AND c.gaps ? 'audit-closed-before-dispatch'))
            OR (item.final_disposition='execution-cancelled' AND item.updated_at >= $3))
          AND (SELECT COALESCE(max(ei.item_attempt),0) FROM audit_execution_items ei WHERE ei.item_id=item.item_id) < $4
        RETURNING item.item_id
    ) UPDATE audit_coverage_rows AS c SET status='not-tested', gaps=c.gaps-'audit-closed-before-dispatch', rationale='Awaiting continuation after the Audit time limit.', updated_at=clock_timestamp() FROM reopened WHERE c.item_id=reopened.item_id`, audit.AuditID, *audit.CurrentRoundID, audit.DeadlineAt, audit.Limits.MaxItemRunAttempts)
	return err
}

// Expired human decisions are never silently extended by resuming an Audit.
// Require a fresh decision for the same exact task before another submission.
func renewExpiredItemReviews(ctx context.Context, tx pgx.Tx, audit auditstore.Audit, deadline *time.Time, now time.Time) error {
	_, err := tx.Exec(ctx, `UPDATE audit_review_requests SET state='expired',revision=revision+1,updated_at=clock_timestamp() WHERE audit_id=$1 AND subject_kind='audit-item-action' AND state='pending' AND expires_at <= $2`, audit.AuditID, now)
	if err != nil {
		return err
	}
	_, err = tx.Exec(ctx, `WITH needs_review AS (
        UPDATE audit_items AS item SET state='awaiting_review',updated_at=clock_timestamp()
        WHERE item.audit_id=$1 AND item.state IN ('ready','awaiting_review') AND item.approval_kind <> 'none'
          AND NOT EXISTS (SELECT 1 FROM audit_review_requests r WHERE r.audit_id=item.audit_id AND r.subject_id=item.item_id AND r.subject_kind='audit-item-action' AND r.subject_digest=item.approval_subject_digest AND (r.expires_at IS NULL OR r.expires_at>$2) AND (r.state='pending' OR (r.state='decided' AND EXISTS(SELECT 1 FROM audit_review_decisions d WHERE d.request_id=r.request_id AND d.action='approve'))))
        RETURNING item.*
    ) INSERT INTO audit_review_requests(request_id,audit_id,finding_id,subject_kind,subject_id,kind,subject_revision,subject_digest,requested_actions,state,expires_at,idempotency_key,request_digest)
      SELECT 'review-resume-'||$3::text||'-'||item.item_id,item.audit_id,NULL,'audit-item-action',item.item_id,item.approval_kind,1,item.approval_subject_digest,
        CASE WHEN item.approval_kind='requirement-applicability' THEN '["approve","reject","not_applicable"]'::jsonb ELSE '["approve","reject"]'::jsonb END,'pending',$4,'resume:'||$3::text||':'||item.item_id,item.approval_subject_digest FROM needs_review item`, audit.AuditID, now, fmt.Sprint(audit.Revision), deadline)
	return err
}
