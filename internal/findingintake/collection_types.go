package findingintake

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/jackc/pgx/v5"
)

const (
	CollectionNamespace         = "finding-collections"
	CollectionReceiptNamespace  = "finding-collection-receipts"
	collectionPublicationSchema = "contractor.findings.publication.v1"
)

// CollectionSelection contains explicit identities, never a live list query.
// Empty arrays select nothing; all pages must be enumerated by the caller.
type CollectionSelection struct {
	Kind       string                       `json:"kind"`
	ID         string                       `json:"id"`
	ReceiptIDs []string                     `json:"receiptIds"`
	Findings   []CollectionFindingSelection `json:"findings"`
}

type CollectionFindingSelection struct {
	FindingID string `json:"findingId"`
	Revision  uint64 `json:"revision"`
}

type PublishCollectionRequest struct {
	ClientKey string                `json:"clientKey"`
	Sources   []CollectionSelection `json:"sources"`
}

type PublishCollectionParams struct {
	OwnerID string
	Request PublishCollectionRequest
}

type PublishedCollection struct {
	Artifact   ExactArtifact `json:"artifact"`
	SnapshotAt string        `json:"snapshotAt"`
	EntryCount int           `json:"entryCount"`
	Replayed   bool          `json:"replayed"`
}

// CollectionReviews keeps Audit selection/review ownership outside intake while
// reading in the publisher's database snapshot. It must never mutate an Audit.
type CollectionReviews interface {
	SelectCollectionFindings(context.Context, pgx.Tx, string, string, []CollectionFindingSelection) ([]string, error)
	ReadCollectionReviews(context.Context, pgx.Tx, string, []string) (map[string][]auditdomain.FindingCollectionReview, error)
}

func canonicalCollectionRequest(input PublishCollectionRequest) (PublishCollectionRequest, string, error) {
	if input.ClientKey == "" || len(input.ClientKey) > 128 ||
		(contracts.ArtifactRef{Namespace: CollectionNamespace, Name: input.ClientKey}).Validate() != nil ||
		len(input.Sources) == 0 || len(input.Sources) > auditdomain.MaximumCollectionSources {
		return PublishCollectionRequest{}, "", ErrInvalid
	}
	result := PublishCollectionRequest{ClientKey: input.ClientKey, Sources: make([]CollectionSelection, len(input.Sources))}
	selections := 0
	for i, source := range input.Sources {
		if !validIdentity(source.ID) || source.Kind != "run" && source.Kind != "audit" ||
			source.ReceiptIDs == nil || source.Findings == nil || source.Kind == "run" && len(source.Findings) != 0 {
			return PublishCollectionRequest{}, "", ErrInvalid
		}
		selections += len(source.ReceiptIDs) + len(source.Findings)
		if selections > auditdomain.MaximumCollectionEntries {
			return PublishCollectionRequest{}, "", fmt.Errorf("%w: collection selection limit", ErrInvalid)
		}
		source.ReceiptIDs = append([]string{}, source.ReceiptIDs...)
		source.Findings = append([]CollectionFindingSelection{}, source.Findings...)
		sort.Strings(source.ReceiptIDs)
		sort.Slice(source.Findings, func(i, j int) bool { return source.Findings[i].FindingID < source.Findings[j].FindingID })
		for j, id := range source.ReceiptIDs {
			if !validIdentity(id) || j > 0 && id == source.ReceiptIDs[j-1] {
				return PublishCollectionRequest{}, "", ErrInvalid
			}
		}
		for j, finding := range source.Findings {
			if !validIdentity(finding.FindingID) || finding.Revision == 0 || finding.Revision > 1<<53-1 ||
				j > 0 && finding.FindingID == source.Findings[j-1].FindingID {
				return PublishCollectionRequest{}, "", ErrInvalid
			}
		}
		result.Sources[i] = source
	}
	sort.Slice(result.Sources, func(i, j int) bool {
		a, b := result.Sources[i], result.Sources[j]
		return a.Kind < b.Kind || a.Kind == b.Kind && a.ID < b.ID
	})
	for i := 1; i < len(result.Sources); i++ {
		if result.Sources[i].Kind == result.Sources[i-1].Kind && result.Sources[i].ID == result.Sources[i-1].ID {
			return PublishCollectionRequest{}, "", ErrInvalid
		}
	}
	encoded, err := json.Marshal(result)
	if err != nil {
		return PublishCollectionRequest{}, "", err
	}
	return result, digestBytes(encoded), nil
}
