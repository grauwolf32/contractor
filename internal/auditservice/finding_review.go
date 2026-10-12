package auditservice

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/reviewstore"
	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/findingintake"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

func (s *Service) CreateFindingReview(
	ctx context.Context, params CreateFindingReviewParams,
) (FindingReviewResult, error) {
	if err := validateCreateFindingReview(params, s.now()); err != nil {
		return FindingReviewResult{}, err
	}
	var requestID string
	var replayed bool
	err := persistencepostgres.InTx(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		reviews := reviewstore.NewTx(tx)
		if err := reviews.LockFindingReviews(ctx, params.AuditID); err != nil {
			return err
		}
		storedID, storedDigest, found, err := reviews.FindReviewRequestReplay(ctx, params.AuditID, params.IdempotencyKey)
		if err != nil {
			return err
		}
		if found {
			requestID = storedID
			if storedDigest != params.RequestDigest {
				return fmt.Errorf("%w: finding review key", contracts.ErrIdempotencyConflict)
			}
			replayed = true
			return nil
		}
		row, state, err := reviews.LockFinding(ctx, params.OwnerID, params.AuditID, params.FindingID)
		if err != nil {
			return err
		}
		// Report finalization hashes Audit revision into immutable report
		// bytes. Freeze review mutations for this short phase so a report
		// artifact cannot be stranded by a later CAS failure. Review remains
		// available again after completion, as required by the public model.
		reportFrozen, err := reviews.PendingReportReview(ctx, params.AuditID)
		if err != nil {
			return err
		}
		if state == auditstore.AuditDeleting || state == auditstore.AuditFinalizing || reportFrozen ||
			row.Revision != params.ExpectedRevision {
			return auditstore.ErrPrecondition
		}
		if err := reviews.ExpireFindingReviews(ctx, params.AuditID, params.FindingID); err != nil {
			return err
		}
		subjectDigest := findingSubjectDigest(row)
		pendingID, found, err := reviews.PendingFindingReview(ctx, params.AuditID, params.FindingID, row.Revision, subjectDigest)
		if err != nil {
			return err
		}
		if found {
			requestID = pendingID
			replayed = true
			return nil
		}
		var expiresAt *time.Time
		if params.ExpiresAt != nil {
			value := params.ExpiresAt.UTC()
			expiresAt = &value
		}
		actions, _ := json.Marshal([]auditstore.AnalystVerdict{
			auditstore.VerdictTruePositive, auditstore.VerdictFalsePositive, auditstore.VerdictDuplicate,
			auditstore.VerdictReopen, auditstore.VerdictNeedsEvidence,
		})
		requestID = params.RequestID
		// The default window starts at the PostgreSQL clock that later decides
		// whether the request has expired.
		if err := reviews.InsertFindingReviewRequest(ctx, auditstore.FindingReviewRequestParams{
			RequestID: requestID, AuditID: params.AuditID, FindingID: params.FindingID, SubjectRevision: row.Revision,
			SubjectDigest: subjectDigest, RequestedActions: actions, ExpiresAt: expiresAt,
			IdempotencyKey: params.IdempotencyKey, RequestDigest: params.RequestDigest,
			DefaultTTLSeconds: int64(defaultReviewTTL / time.Second),
		}); err != nil {
			return err
		}
		return appendAuditReviewEvent(ctx, tx, params.AuditID, "review.requested", requestID,
			map[string]any{
				"subjectKind": auditstore.ReviewSubjectFinding,
				"subjectId":   params.FindingID,
				"findingId":   params.FindingID,
				"kind":        auditstore.FindingReviewKind,
			})
	})
	if err != nil {
		return FindingReviewResult{}, err
	}
	request, err := s.GetReview(ctx, params.OwnerID, params.AuditID, requestID)
	return FindingReviewResult{Request: request, Replayed: replayed}, err
}

func (s *Service) DecideFinding(
	ctx context.Context, params DecideFindingParams,
) (FindingDecisionResult, error) {
	if err := validateFindingDecision(params); err != nil {
		return FindingDecisionResult{}, err
	}
	var decisionID, requestID, findingID string
	var replayed bool
	var expired bool
	err := persistencepostgres.InTx(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		reviews := reviewstore.NewTx(tx)
		if err := reviews.LockFindingReviews(ctx, params.AuditID); err != nil {
			return err
		}
		replay, found, err := reviews.FindReviewDecisionReplay(ctx, params.AuditID, params.IdempotencyKey)
		if err != nil {
			return err
		}
		if found {
			decisionID, requestID, findingID = replay.DecisionID, replay.RequestID, replay.FindingID
			if replay.RequestDigest != params.RequestDigest || replay.RequestID != params.RequestID {
				return fmt.Errorf("%w: finding decision key", contracts.ErrIdempotencyConflict)
			}
			replayed = true
			return nil
		}
		locked, err := reviews.LockFindingReview(ctx, params.OwnerID, params.AuditID, params.RequestID)
		if err != nil {
			return err
		}
		auditState, request, row := locked.AuditState, locked.Request, locked.Finding
		reportFrozen, err := reviews.PendingReportReview(ctx, params.AuditID)
		if err != nil {
			return err
		}
		if auditState == auditstore.AuditDeleting || auditState == auditstore.AuditFinalizing || reportFrozen ||
			request.State != auditstore.ReviewPending ||
			request.Revision != params.ExpectedRequestRevision || request.Kind != auditstore.FindingReviewKind {
			return auditstore.ErrPrecondition
		}
		if request.ExpiresAt != nil {
			if expired, err = reviews.ExpireReviewAtDatabaseTime(ctx, request.RequestID); err != nil {
				return err
			}
		}
		if expired {
			// Commit the expiry transition before reporting the failed decision.
			// Returning ErrPrecondition from inside the transaction would roll the
			// update back and leave the request permanently pending.
			return appendAuditReviewEvent(ctx, tx, params.AuditID, "review.expired",
				request.RequestID, map[string]any{
					"subjectKind": auditstore.ReviewSubjectFinding,
					"findingId":   row.FindingID,
					"kind":        auditstore.FindingReviewKind,
				})
		}
		if request.SubjectRevision != row.Revision || request.SubjectDigest != findingSubjectDigest(row) {
			return auditstore.ErrPrecondition
		}
		if params.Verdict == auditstore.VerdictDuplicate {
			if err := validateDuplicateTarget(ctx, reviews, params.AuditID, row.FindingID, *params.DuplicateTargetID); err != nil {
				return err
			}
		}
		decisionID, requestID, findingID = params.DecisionID, request.RequestID, row.FindingID
		var severity, target *string
		if params.Severity != nil {
			value := string(*params.Severity)
			severity = &value
		}
		if params.DuplicateTargetID != nil {
			target = params.DuplicateTargetID
		}
		verdict := string(params.Verdict)
		if err := reviews.RecordReviewDecision(ctx, auditstore.ReviewDecisionWriteParams{
			DecisionID: decisionID, RequestID: requestID, AuditID: params.AuditID, FindingID: &findingID,
			ActorID: params.OwnerID, Action: verdict, Verdict: &verdict, Severity: severity, Rationale: params.Rationale,
			DuplicateTargetID: target, SubjectRevision: request.SubjectRevision, SubjectDigest: request.SubjectDigest,
			IdempotencyKey: params.IdempotencyKey, RequestDigest: params.RequestDigest,
		}); err != nil {
			return err
		}
		state, rejection, duplicate, effectiveDecision := decisionProjection(params.Verdict, decisionID, target)
		if err := reviews.ProjectFindingDecision(ctx, findingID, state, rejection, duplicate, effectiveDecision); err != nil {
			return err
		}
		return appendAuditReviewEvent(ctx, tx, params.AuditID, "review.decided", decisionID,
			map[string]any{"findingId": findingID, "verdict": params.Verdict})
	})
	if err != nil {
		return FindingDecisionResult{}, err
	}
	if expired {
		return FindingDecisionResult{}, auditstore.ErrPrecondition
	}
	request, err := s.GetReview(ctx, params.OwnerID, params.AuditID, requestID)
	if err != nil {
		return FindingDecisionResult{}, err
	}
	finding, err := s.GetFinding(ctx, params.OwnerID, params.AuditID, findingID)
	if err != nil {
		return FindingDecisionResult{}, err
	}
	if request.Decision == nil || request.Decision.DecisionID != decisionID {
		return FindingDecisionResult{}, errors.New("stored finding review decision is missing")
	}
	return FindingDecisionResult{
		Finding: finding, Request: request, Decision: *request.Decision, Replayed: replayed,
	}, nil
}

func (s *Service) GetReview(
	ctx context.Context, ownerID, auditID, requestID string,
) (auditstore.ReviewRequest, error) {
	return s.reads().GetReview(ctx, ownerID, auditID, requestID)
}

func (s *Service) ListReviews(
	ctx context.Context, params auditstore.ReviewListParams,
) ([]auditstore.ReviewRequest, error) {
	return s.reads().ListReviews(ctx, params)
}

func (s *Service) ListFindingProvenance(
	ctx context.Context, params auditstore.ProvenanceListParams,
) ([]FindingProvenance, error) {
	reads := s.reads()
	beforeAudit, beforeFinding, err := reads.ReadProvenanceRevisions(ctx, params)
	if err != nil {
		return nil, err
	}
	if (params.AuditRevision != nil && *params.AuditRevision != beforeAudit) ||
		(params.FindingRevision != nil && *params.FindingRevision != beforeFinding) {
		return nil, auditstore.ErrConflict
	}
	finding, err := s.GetFinding(ctx, params.OwnerID, params.AuditID, params.FindingID)
	if err != nil {
		return nil, err
	}
	records, err := reads.ListFindingProvenance(ctx, params)
	if err != nil {
		return nil, err
	}
	result := make([]FindingProvenance, len(records))
	for index, record := range records {
		result[index] = FindingProvenance{
			RecordID: record.RecordID, Kind: record.Kind, ReceiptID: record.ReceiptID,
			Relation: record.Relation, Proposal: record.Proposal, Assessment: record.Assessment,
			Attempt: record.Attempt, CreatedAt: record.CreatedAt,
			SupportsCurrent: record.Assessment != nil && finding.CurrentAssessment != nil &&
				finding.CurrentAssessment.AssessmentID == record.Assessment.AssessmentID,
		}
	}
	ids := make([]string, 0, len(result))
	origins := make(map[string]findingintake.Origin, len(result))
	for _, value := range result {
		if _, seen := origins[value.ReceiptID]; !seen {
			ids = append(ids, value.ReceiptID)
			origins[value.ReceiptID] = findingintake.Origin{}
		}
	}
	// A page can include a cursor sentinel; chunk unique receipts by the
	// existing intake bound without dropping or reordering provenance records.
	for start := 0; start < len(ids); start += findingintake.MaxAuditReceiptBatchSize {
		end := min(start+findingintake.MaxAuditReceiptBatchSize, len(ids))
		receipts, err := s.findings.GetAuditReceipts(ctx, params.OwnerID, params.AuditID, ids[start:end])
		if err != nil {
			return nil, err
		}
		for _, receipt := range receipts {
			origins[receipt.ReceiptID] = receipt.Origin
		}
	}
	for index := range result {
		result[index].Origin = origins[result[index].ReceiptID]
	}
	// The final joined snapshot covers rows, exact receipt hydration and the
	// finding assessment used for SupportsCurrent, even without caller pins.
	afterAudit, afterFinding, err := reads.ReadProvenanceRevisions(ctx, params)
	if err != nil {
		return nil, err
	}
	if afterAudit != beforeAudit || afterFinding != beforeFinding {
		return nil, auditstore.ErrConflict
	}
	return result, nil
}

func findingSubjectDigest(row auditstore.FindingRecord) string {
	value := struct {
		Schema              string  `json:"schema"`
		FindingID           string  `json:"findingId"`
		AuditID             string  `json:"auditId"`
		Revision            uint64  `json:"revision"`
		State               string  `json:"state"`
		FirstReceiptID      string  `json:"firstReceiptId"`
		CurrentAssessmentID *string `json:"currentAssessmentId"`
		CurrentDecisionID   *string `json:"currentDecisionId"`
		DuplicateTargetID   *string `json:"duplicateTargetId"`
	}{
		Schema: "contractor.audit.finding-subject.v1", FindingID: row.FindingID,
		AuditID: row.AuditID, Revision: row.Revision, State: string(row.State),
		FirstReceiptID: row.FirstReceiptID, CurrentAssessmentID: row.CurrentAssessmentID,
		CurrentDecisionID: row.CurrentDecisionID, DuplicateTargetID: row.DuplicateTargetID,
	}
	encoded, _ := json.Marshal(value)
	return contentdigest.Bytes(encoded)
}

func decisionProjection(
	verdict auditstore.AnalystVerdict, decisionID string, target *string,
) (string, *string, *string, *string) {
	switch verdict {
	case auditstore.VerdictTruePositive:
		return string(auditstore.FindingConfirmed), nil, nil, &decisionID
	case auditstore.VerdictFalsePositive:
		reason := "false-positive"
		return string(auditstore.FindingRejected), &reason, nil, &decisionID
	case auditstore.VerdictDuplicate:
		return string(auditstore.FindingDuplicate), nil, target, nil
	case auditstore.VerdictNeedsEvidence:
		return string(auditstore.FindingNeedsEvidence), nil, nil, nil
	default:
		return string(auditstore.FindingProposed), nil, nil, nil
	}
}

func validateDuplicateTarget(
	ctx context.Context, reviews *reviewstore.TxStore, auditID, findingID, targetID string,
) error {
	if targetID == findingID || !auditstore.ValidReviewIdentity(targetID, 256) {
		return auditstore.ErrInvalid
	}
	exists, cycle, err := reviews.DuplicateTarget(ctx, auditID, targetID, findingID)
	if err != nil {
		return err
	}
	if !exists {
		return auditstore.ErrNotFound
	}
	if cycle {
		return auditstore.ErrConflict
	}
	return nil
}

func appendAuditReviewEvent(
	ctx context.Context, tx pgx.Tx, auditID, kind, entityID string, summary map[string]any,
) error {
	return reviewstore.New(tx).AppendReviewEvent(ctx, auditstore.ReviewEventParams{
		AuditID: auditID, Kind: kind, EntityID: entityID, Summary: summary,
	})
}

func validateCreateFindingReview(params CreateFindingReviewParams, now time.Time) error {
	if !auditstore.ValidReviewIdentity(params.OwnerID, 256) || !auditstore.ValidReviewIdentity(params.AuditID, 256) ||
		!auditstore.ValidReviewIdentity(params.FindingID, 256) || !auditstore.ValidReviewIdentity(params.RequestID, 256) ||
		params.ExpectedRevision < 1 || !validIdempotencyKey(params.IdempotencyKey) ||
		!contentdigest.Valid(params.RequestDigest) {
		return auditstore.ErrInvalid
	}
	if params.ExpiresAt != nil && (!params.ExpiresAt.After(now) || params.ExpiresAt.After(now.Add(maximumReviewTTL))) {
		return auditstore.ErrInvalid
	}
	return nil
}

func validateFindingDecision(params DecideFindingParams) error {
	if !auditstore.ValidReviewIdentity(params.OwnerID, 256) || !auditstore.ValidReviewIdentity(params.AuditID, 256) ||
		!auditstore.ValidReviewIdentity(params.RequestID, 256) || !auditstore.ValidReviewIdentity(params.DecisionID, 256) ||
		params.ExpectedRequestRevision < 1 || !params.Verdict.Valid() ||
		!validIdempotencyKey(params.IdempotencyKey) || !contentdigest.Valid(params.RequestDigest) ||
		!validRationale(params.Rationale) {
		return auditstore.ErrInvalid
	}
	if params.Verdict == auditstore.VerdictTruePositive {
		if params.Severity == nil || !params.Severity.Valid() {
			return auditstore.ErrInvalid
		}
	} else if params.Severity != nil {
		return auditstore.ErrInvalid
	}
	if (params.Verdict == auditstore.VerdictDuplicate) != (params.DuplicateTargetID != nil) {
		return auditstore.ErrInvalid
	}
	return nil
}

func validRationale(value string) bool {
	return strings.TrimSpace(value) != "" && utf8.ValidString(value) &&
		!strings.ContainsRune(value, 0) && len([]byte(value)) <= MaxReviewRationale
}

func validIdempotencyKey(value string) bool {
	if value == "" || len(value) > 128 {
		return false
	}
	for index, character := range value {
		if (character >= 'A' && character <= 'Z') || (character >= 'a' && character <= 'z') ||
			(character >= '0' && character <= '9') || (index > 0 && strings.ContainsRune("._:-", character)) {
			continue
		}
		return false
	}
	return true
}
