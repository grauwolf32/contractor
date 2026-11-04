package findingintake

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"slices"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	artifactpolicy "github.com/grauwolf32/contractor/internal/artifacts/policy"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/querystore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// AllocationFindingPage is a bounded live view. Only explicit selection retains bytes.
type AllocationFindingPage struct {
	Items           []auditstore.ProjectFindingSummary `json:"items"`
	ProjectRevision uint64                             `json:"projectRevision"`
	NextCursor      *string                            `json:"nextCursor"`
}

type findingCursor struct {
	ProjectID       string    `json:"projectId"`
	CreatedAt       time.Time `json:"createdAt"`
	FindingID       string    `json:"findingId"`
	ProjectRevision uint64    `json:"projectRevision"`
}

type FindingSelection struct {
	ClientKey  string                       `json:"clientKey" strictjson:"required"`
	Findings   []CollectionFindingSelection `json:"findings" strictjson:"required"`
	ReceiptIDs []string                     `json:"receiptIds" strictjson:"required"`
}

type readerDB interface {
	QueryRow(context.Context, string, ...any) pgx.Row
}

// allocationReaderContext checks both the live grant and its durable allocation,
// and derives ownership and source restrictions without accepting caller IDs.
func (p *CollectionPublisher) allocationReaderContext(ctx context.Context, db readerDB, grant controlplane.AllocationGrant, tool string) (string, string, bool, error) {
	if grant.Lost || grant.WriteFenced || grant.ReadPolicy != controlplane.ReadCurrentRun || !slices.Contains(grant.FindingTools, tool) {
		return "", "", false, ErrAccessDenied
	}
	var owner, project, mode string
	err := db.QueryRow(ctx, allocationFindingContextSQL, grant.AllocationID, grant.StageExecutionID, grant.RunID,
		grant.LogicalAgentName, grant.Namespace, grant.RuntimeAgentID, grant.RuntimeInstanceID).Scan(&owner, &project, &mode)
	if errors.Is(err, pgx.ErrNoRows) {
		return "", "", false, ErrAccessDenied
	}
	if err != nil {
		return "", "", false, err
	}
	if mode != "" && mode != "blackbox" && mode != "source-assisted" {
		return "", "", false, ErrAccessDenied
	}
	return owner, project, mode == "blackbox", nil
}

func (p *CollectionPublisher) ListAllocationFindings(ctx context.Context, grant controlplane.AllocationGrant, limit int, cursor string) (AllocationFindingPage, error) {
	result := AllocationFindingPage{Items: []auditstore.ProjectFindingSummary{}}
	if limit < 1 || limit > 100 || len(cursor) > 1366 {
		return result, ErrInvalid
	}
	err := postgres.InTxWithRetry(ctx, p.pool, pgx.TxOptions{IsoLevel: pgx.RepeatableRead, AccessMode: pgx.ReadOnly}, func(tx pgx.Tx) error {
		result = AllocationFindingPage{Items: []auditstore.ProjectFindingSummary{}}
		owner, project, blackbox, err := p.allocationReaderContext(ctx, tx, grant, "list_project_findings")
		if err != nil {
			return err
		}
		reads := querystore.New(tx, p.storage)
		fence, err := reads.CorrelationFence(ctx, owner, project, nil)
		if err != nil {
			return collectionStoreError(err)
		}
		params := auditstore.OwnerFindingListParams{OwnerID: owner, ProjectID: project, Limit: limit + 1}
		if cursor != "" {
			data, err := base64.RawURLEncoding.Strict().DecodeString(cursor)
			var value findingCursor
			if err != nil || len(data) > 1024 || strictjson.Decode(data, &value) != nil || value.ProjectID != project || !contracts.ValidResourceID(value.FindingID) || value.CreatedAt.IsZero() || value.ProjectRevision < 1 || value.ProjectRevision > contracts.MaxExactInteger {
				return ErrInvalid
			}
			if value.ProjectRevision != fence.ProjectRevision {
				return ErrConflict
			}
			params.BeforeCreatedAt, params.BeforeFindingID = &value.CreatedAt, value.FindingID
		}
		window, err := reads.ListOwnerFindings(ctx, params)
		if err != nil {
			return collectionStoreError(err)
		}
		items := window.Items
		var next *auditstore.ProjectFindingSummary
		if len(items) > limit {
			next = &items[limit-1]
			items = items[:limit]
		} else if window.ContinueBefore != nil {
			next = window.ContinueBefore
		}
		if blackbox {
			ids := make([]string, len(items))
			for i, item := range items {
				ids[i] = item.FindingID
			}
			if err := checkBlackboxFindings(ctx, tx, owner, project, ids); err != nil {
				return err
			}
		}
		if next != nil {
			data, err := json.Marshal(findingCursor{project, next.CreatedAt, next.FindingID, fence.ProjectRevision})
			if err != nil {
				return err
			}
			encoded := base64.RawURLEncoding.EncodeToString(data)
			result.NextCursor = &encoded
		}
		result.Items, result.ProjectRevision = items, fence.ProjectRevision
		return nil
	})
	if err != nil {
		return AllocationFindingPage{}, err
	}
	return result, readerResponseBound(result)
}

// ReadAllocationFinding reads canonical current metadata, with no evidence hold
// or foreign Artifact read capability. Full bytes require explicit selection.
func (p *CollectionPublisher) ReadAllocationFinding(ctx context.Context, grant controlplane.AllocationGrant, findingID string, revision uint64) (auditstore.ProjectFinding, error) {
	var result auditstore.ProjectFinding
	if !contracts.ValidResourceID(findingID) || revision < 1 || revision > contracts.MaxExactInteger {
		return result, ErrInvalid
	}
	err := postgres.InTxWithRetry(ctx, p.pool, pgx.TxOptions{IsoLevel: pgx.RepeatableRead, AccessMode: pgx.ReadOnly}, func(tx pgx.Tx) error {
		owner, project, blackbox, err := p.allocationReaderContext(ctx, tx, grant, "read_project_finding")
		if err != nil {
			return err
		}
		if blackbox {
			if err := checkBlackboxFindings(ctx, tx, owner, project, []string{findingID}); err != nil {
				return err
			}
		}
		result, err = querystore.New(tx, p.storage).GetProjectFinding(ctx, owner, project, findingID)
		if err != nil {
			return collectionStoreError(err)
		}
		if result.Revision != revision {
			return ErrConflict
		}
		return nil
	})
	if err != nil {
		return auditstore.ProjectFinding{}, err
	}
	return result, readerResponseBound(result)
}

// SelectAllocationFindings retains one explicit selection in the ordinary Run
// Artifact plane. Publication replays before consulting deleted source records.
func (p *CollectionPublisher) SelectAllocationFindings(ctx context.Context, grant controlplane.AllocationGrant, input FindingSelection) (PublishedCollection, error) {
	if err := input.validate(); err != nil {
		return PublishedCollection{}, err
	}
	if artifactpolicy.IsModelHiddenBinding(grant.Namespace, input.ClientKey) || artifactpolicy.IsModelHiddenBinding(CollectionNamespace, input.ClientKey) {
		return PublishedCollection{}, ErrAccessDenied
	}
	owner, project, _, err := p.allocationReaderContext(ctx, p.pool, grant, "select_project_findings")
	if err != nil {
		return PublishedCollection{}, err
	}
	return p.publishCollection(ctx, PublishCollectionParams{OwnerID: owner, Request: PublishCollectionRequest{ClientKey: input.ClientKey, Sources: []CollectionSelection{{Kind: "project", ID: project, Findings: input.Findings, ReceiptIDs: input.ReceiptIDs}}}}, &grant)
}

func collectionStoreError(err error) error {
	switch {
	case errors.Is(err, auditstore.ErrInvalid):
		return ErrInvalid
	case errors.Is(err, auditstore.ErrNotFound):
		return ErrNotFound
	case errors.Is(err, auditstore.ErrConflict):
		return ErrConflict
	}
	return err
}

func checkBlackboxFindings(ctx context.Context, tx pgx.Tx, owner, project string, ids []string) error {
	var allowed bool
	if err := tx.QueryRow(ctx, blackboxFindingsSQL, owner, project, ids).Scan(&allowed); err != nil {
		return err
	}
	if !allowed {
		return ErrAccessDenied
	}
	return nil
}
func checkBlackboxReceipts(ctx context.Context, tx pgx.Tx, owner string, ids []string) error {
	var allowed bool
	if err := tx.QueryRow(ctx, blackboxReceiptsSQL, owner, ids).Scan(&allowed); err != nil {
		return err
	}
	if !allowed {
		return ErrAccessDenied
	}
	return nil
}

func readerResponseBound(value any) error {
	encoded, err := json.Marshal(value)
	if err != nil {
		return err
	}
	if len(encoded) > 256<<10 {
		return blob.ErrTooLarge
	}
	return nil
}

func (input FindingSelection) validate() error {
	if input.Findings == nil || input.ReceiptIDs == nil || len(input.Findings)+len(input.ReceiptIDs) > 128 {
		return ErrInvalid
	}
	_, _, err := canonicalCollectionRequest(PublishCollectionRequest{ClientKey: input.ClientKey, Sources: []CollectionSelection{{Kind: "project", ID: "selection", Findings: input.Findings, ReceiptIDs: input.ReceiptIDs}}})
	return err
}
