package auditservice

import (
	"context"
	"encoding/json"
	"errors"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/ownerstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

type ProjectFindingEditParams struct {
	auditstore.ProjectFindingEdit
	IdempotencyKey string
	RequestDigest  string
}

type ProjectFindingEditResult struct {
	Finding  auditstore.ProjectFinding `json:"finding"`
	Replayed bool                      `json:"replayed"`
}

func (s *Service) GetProjectFinding(ctx context.Context, ownerID, projectID, findingID string) (auditstore.ProjectFinding, error) {
	finding, err := s.reads().GetProjectFinding(ctx, ownerID, projectID, findingID)
	if err != nil {
		return finding, err
	}
	return s.hydrateProjectFindingClaim(ctx, ownerID, finding)
}

func (s *Service) GetAuditProjectFinding(ctx context.Context, ownerID, auditID, findingID string) (auditstore.ProjectFinding, error) {
	finding, err := s.reads().GetAuditProjectFinding(ctx, ownerID, auditID, findingID)
	if err != nil {
		return finding, err
	}
	return s.hydrateProjectFindingClaim(ctx, ownerID, finding)
}

func (s *Service) hydrateProjectFindingClaim(ctx context.Context, ownerID string, finding auditstore.ProjectFinding) (auditstore.ProjectFinding, error) {
	source, err := s.findings.GetProjectReceipt(ctx, ownerID, finding.ProjectID, finding.FirstReceiptID)
	if err != nil {
		return finding, err
	}
	if source.ProjectProposal == nil || source.ProjectProposal.Verify(finding.FirstProposal) != nil || source.Document.Title != finding.Title {
		return finding, errors.New("Project Finding claim disagrees with its immutable first source")
	}
	claim := auditdomain.ProjectFindingReport(source.Document)
	finding.Claim = &claim
	return finding, nil
}

// EditProjectFinding is the sole analyst mutation, regardless of which Audit
// the user opened. Source collection never invokes it.
func (s *Service) EditProjectFinding(ctx context.Context, params ProjectFindingEditParams) (ProjectFindingEditResult, error) {
	if err := validateProjectFindingEdit(params); err != nil {
		return ProjectFindingEditResult{}, err
	}
	var result ProjectFindingEditResult
	err := persistencepostgres.InTxWithRetry(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		key := persistencepostgres.ProjectReceiptKey(params.OwnerID, params.ProjectID, params.FindingID, "finding.assess", params.IdempotencyKey)
		receipt, replayed, err := persistencepostgres.KeyedMutation(ctx, tx, key, params.RequestDigest, func() (persistencepostgres.Receipt, error) {
			finding, err := ownerstore.NewTx(tx, s.storage).AssessProjectFinding(ctx, params.ProjectFindingEdit)
			if err != nil {
				return persistencepostgres.Receipt{}, err
			}
			encoded, err := json.Marshal(finding)
			return persistencepostgres.Receipt{Operation: "finding.assess", ResourceID: finding.FindingID,
				OwnerID: params.OwnerID, ProjectID: params.ProjectID, Response: encoded}, err
		})
		if errors.Is(err, persistencepostgres.ErrReceiptMismatch) {
			return contracts.ErrIdempotencyConflict
		}
		if err != nil {
			return err
		}
		if receipt.Operation != "finding.assess" || receipt.ResourceID != params.FindingID ||
			receipt.ProjectID != params.ProjectID || receipt.OwnerID != params.OwnerID ||
			strictjson.Decode(receipt.Response, &result.Finding) != nil ||
			result.Finding.ProjectID != params.ProjectID || result.Finding.FindingID != params.FindingID || result.Finding.Validate() != nil {
			return errors.New("stored Project Finding mutation receipt is invalid")
		}
		result.Replayed = replayed
		return nil
	})
	return result, err
}

type ProjectFindingListParams struct {
	auditstore.OwnerFindingListParams
	ProjectRevision *uint64
}
type ProjectFindingPage struct {
	ContinueBefore  *auditstore.ProjectFindingSummary  `json:"-"`
	Items           []auditstore.ProjectFindingSummary `json:"items"`
	ProjectRevision uint64                             `json:"projectRevision"`
}

func (s *Service) ListProjectFindings(ctx context.Context, params ProjectFindingListParams) (ProjectFindingPage, error) {
	var result ProjectFindingPage
	before, err := s.reads().CorrelationFence(ctx, params.OwnerID, params.ProjectID, nil)
	if err != nil {
		return result, err
	}
	if params.ProjectRevision != nil && *params.ProjectRevision != before.ProjectRevision {
		return result, auditstore.ErrConflict
	}
	window, readErr := s.reads().ListOwnerFindings(ctx, params.OwnerFindingListParams)
	err = readErr
	result.Items, result.ContinueBefore = window.Items, window.ContinueBefore
	if err != nil {
		return result, err
	}
	after, err := s.reads().CorrelationFence(ctx, params.OwnerID, params.ProjectID, nil)
	if err != nil {
		return result, err
	}
	if before.ProjectRevision != after.ProjectRevision {
		return result, auditstore.ErrConflict
	}
	result.ProjectRevision = before.ProjectRevision
	return result, nil
}

func validateProjectFindingEdit(params ProjectFindingEditParams) error {
	if !auditstore.ValidFindingIdentity(params.OwnerID, auditstore.MaxFindingActorBytes) || !contracts.ValidResourceID(params.ProjectID) ||
		!contracts.ValidResourceID(params.FindingID) || params.ExpectedRevision < 1 ||
		params.ExpectedRevision > contracts.MaxExactInteger || !params.Verdict.Valid() ||
		!contracts.ValidIdempotencyKey(params.IdempotencyKey) || !contentdigest.Valid(params.RequestDigest) ||
		!validRationale(params.Rationale) ||
		(params.Verdict == auditstore.VerdictTruePositive) != (params.Severity != nil) ||
		(params.Severity != nil && !params.Severity.Valid()) ||
		(params.Verdict == auditstore.VerdictDuplicate) != (params.DuplicateTargetID != nil) ||
		(params.DuplicateTargetID != nil && (!contracts.ValidResourceID(*params.DuplicateTargetID) || *params.DuplicateTargetID == params.FindingID)) {
		return auditstore.ErrInvalid
	}
	return nil
}
