package findingintake

import (
	"context"
	"errors"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	artifactpolicy "github.com/grauwolf32/contractor/internal/artifacts/policy"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"github.com/jackc/pgx/v5"
)

// MaxProjectSourcePage bounds compact source previews. Storage can read one
// cursor sentinel; evidence and result payloads have separate detail reads.
const MaxProjectSourcePage = 64

type ProjectSourcePreview struct {
	ReceiptID            string        `json:"receiptId"`
	ProposalID           string        `json:"proposalId"`
	ProjectID            string        `json:"projectId"`
	FindingID            *string       `json:"findingId,omitempty"`
	Proposal             ExactArtifact `json:"proposal"`
	Origin               Origin        `json:"origin"`
	EvidenceCount        int           `json:"evidenceCount"`
	PublicationRequested bool          `json:"publicationRequested"`
	CreatedAt            time.Time     `json:"createdAt"`
}

type ProjectSourceQuery struct {
	ListQuery
	ProjectRevision *uint64
}

type ProjectSourcePage struct {
	Items           []ProjectSourcePreview `json:"items"`
	ProjectRevision uint64                 `json:"projectRevision"`
}

type ProjectSourceHistoryPage struct {
	Items           []artifacts.Metadata `json:"items"`
	ProjectRevision uint64               `json:"projectRevision"`
}

func (s *Service) GetProjectReceipt(ctx context.Context, ownerID, projectID, receiptID string) (Receipt, error) {
	if !auditstore.ValidFindingIdentity(ownerID, 256) || contracts.ValidateResourceID("project", projectID) != nil ||
		contracts.ValidateResourceID("receipt", receiptID) != nil {
		return Receipt{}, ErrInvalid
	}
	receipt, err := scanReceiptRow(s.pool.QueryRow(ctx, getProjectReceiptSQL, ownerID, projectID, receiptID))
	if errors.Is(err, pgx.ErrNoRows) {
		return Receipt{}, ErrNotFound
	}
	if err != nil {
		return Receipt{}, err
	}
	receipt.AuditHolds = []AuditHold{}
	values, err := s.hydrateAuditReceiptBatch(ctx, []Receipt{receipt})
	if err != nil {
		return Receipt{}, err
	}
	return values[0], nil
}

// ListProjectSources traverses retained receipts even when the Project has
// no Finding cards. Finding selection uses its own indexed source window.
func (s *Service) ListProjectSources(ctx context.Context, ownerID, projectID, findingID string, query ProjectSourceQuery) (ProjectSourcePage, error) {
	if !auditstore.ValidFindingIdentity(ownerID, 256) || contracts.ValidateResourceID("project", projectID) != nil ||
		!validListQuery(query.ListQuery) || query.Limit > MaxProjectSourcePage+1 ||
		(findingID != "" && contracts.ValidateResourceID("finding", findingID) != nil) {
		return ProjectSourcePage{}, ErrInvalid
	}
	revision, err := s.projectSourceRevision(ctx, ownerID, projectID, findingID, query.ProjectRevision)
	if err != nil {
		return ProjectSourcePage{}, err
	}
	statement := listProjectSourcesSQL
	if findingID != "" {
		statement = listCanonicalFindingSourcesSQL
	}
	rows, err := s.pool.Query(ctx, statement, ownerID, projectID, findingID, query.AfterCreatedAt, query.AfterReceiptID, query.Limit)
	if err != nil {
		return ProjectSourcePage{}, err
	}
	defer rows.Close()
	result := make([]ProjectSourcePreview, 0, query.Limit)
	for rows.Next() {
		var value ProjectSourcePreview
		var proposal, configuration []byte
		var auditID, executionID, taskID *string
		var hypothesis, hypothesisSnapshot, scanOrigin []byte
		var deleted *time.Time
		if err := rows.Scan(&value.ReceiptID, &value.ProposalID, &value.ProjectID, &value.FindingID,
			&proposal, &value.EvidenceCount, &value.PublicationRequested, &value.CreatedAt,
			&value.Origin.RunID, &value.Origin.StageExecutionID, &value.Origin.AllocationID,
			&value.Origin.InvocationID, &value.Origin.LogicalAgentName, &value.Origin.Workflow.Name,
			&value.Origin.Workflow.Version, &value.Origin.Workflow.SchemaVersion, &configuration, &value.Origin.Workflow.ClosureDigest,
			&auditID, &executionID, &taskID, &hypothesis, &hypothesisSnapshot, &deleted, &value.Origin.Operation, &scanOrigin); err != nil {
			return ProjectSourcePage{}, err
		}
		if strictjson.Decode(proposal, &value.Proposal) != nil || value.Proposal.Validate() != nil ||
			!value.Proposal.HasMetadata() || strictjson.Decode(configuration, &value.Origin.Workflow.ConfigurationRef) != nil {
			return ProjectSourcePage{}, ErrInvalid
		}
		if scanOrigin != nil {
			var origin artifacts.ExecutionOrigin
			if strictjson.Decode(scanOrigin, &origin) != nil {
				return ProjectSourcePage{}, ErrInvalid
			}
			value.Origin.Scan = &origin
		}
		value.Origin.RunDeleted = deleted != nil
		if auditID != nil {
			if executionID == nil || (taskID == nil && len(hypothesis) != 0) {
				return ProjectSourcePage{}, ErrInvalid
			}
			value.Origin.Audit = &AuditOrigin{AuditID: *auditID, ExecutionID: *executionID}
			if taskID != nil {
				value.Origin.Audit.TaskID = *taskID
			}
			if len(hypothesis) != 0 {
				var ref auditcontract.HypothesisRef
				if strictjson.Decode(hypothesis, &ref) != nil || !contracts.ValidResourceID(ref.ID) || ref.Revision == 0 || ref.Revision > contracts.MaxExactInteger {
					return ProjectSourcePage{}, ErrInvalid
				}
				value.Origin.Audit.Hypothesis = &ref
				if hypothesisSnapshot != nil {
					var exact ExactArtifact
					if strictjson.Decode(hypothesisSnapshot, &exact) != nil || exact.Validate() != nil || !exact.HasMetadata() {
						return ProjectSourcePage{}, ErrInvalid
					}
					value.Origin.Audit.HypothesisSnapshot = &exact
				}
			}

		}
		result = append(result, value)
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return ProjectSourcePage{}, err
	}
	if _, err := s.projectSourceRevision(ctx, ownerID, projectID, findingID, &revision); err != nil {
		return ProjectSourcePage{}, err
	}
	return ProjectSourcePage{Items: result, ProjectRevision: revision}, nil
}

func (s *Service) projectSourceRevision(ctx context.Context, ownerID, projectID, findingID string, expected *uint64) (uint64, error) {
	if !auditstore.ValidFindingIdentity(ownerID, 256) || !contracts.ValidResourceID(projectID) || (expected != nil && (*expected < 1 || *expected > contracts.MaxExactInteger)) {
		return 0, ErrInvalid
	}
	var revision uint64
	err := s.pool.QueryRow(ctx, authorizeProjectSourcesSQL, ownerID, projectID, findingID).Scan(&revision)
	if errors.Is(err, pgx.ErrNoRows) {
		return 0, ErrNotFound
	}
	if err != nil {
		return 0, err
	}
	if revision < 1 || revision > contracts.MaxExactInteger {
		return 0, ErrInvalid
	}
	if expected != nil && *expected != revision {
		return 0, ErrConflict
	}
	return revision, nil
}

// ListProjectSourceHistory returns exact ordinary Artifact metadata. Reading
// history never instantiates or decodes an old Finding model.
func (s *Service) ListProjectSourceHistory(ctx context.Context, ownerID, projectID, receiptID, after string, limit int, expected *uint64) (ProjectSourceHistoryPage, error) {
	if contracts.ValidateResourceID("receipt", receiptID) != nil || limit < 1 || limit > MaxProjectSourcePage+1 {
		return ProjectSourceHistoryPage{}, ErrInvalid
	}
	revision, err := s.projectSourceRevision(ctx, ownerID, projectID, "", expected)
	if err != nil {
		return ProjectSourceHistoryPage{}, err
	}
	var owned bool
	if err := s.pool.QueryRow(ctx, authorizeProjectSourceHistorySQL, ownerID, projectID, receiptID).Scan(&owned); err != nil {
		return ProjectSourceHistoryPage{}, err
	}
	if !owned {
		return ProjectSourceHistoryPage{}, ErrNotFound
	}
	project, err := artifacts.NewService(s.storage.Repository(s.pool)).Project(projectID)
	if err != nil {
		return ProjectSourceHistoryPage{}, err
	}
	namespace := artifactpolicy.FindingHistoryNamespace
	query := artifacts.BindingPageQuery{Namespace: &namespace, NamePrefix: auditdomain.FindingSourceHistoryPrefix(receiptID), Limit: limit}
	if after != "" {
		query.AfterNamespace = namespace
		query.AfterName = after
	}
	items, err := project.ListMetadata(ctx, query)
	if err != nil {
		return ProjectSourceHistoryPage{}, err
	}
	if _, err := s.projectSourceRevision(ctx, ownerID, projectID, "", &revision); err != nil {
		return ProjectSourceHistoryPage{}, err
	}
	return ProjectSourceHistoryPage{Items: items, ProjectRevision: revision}, nil
}
