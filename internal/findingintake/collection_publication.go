package findingintake

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contracts"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
)

type CollectionPublisher struct {
	pool    *pgxpool.Pool
	reviews CollectionReviews
}

func NewCollectionPublisher(pool *pgxpool.Pool, reviews CollectionReviews) (*CollectionPublisher, error) {
	if pool == nil || reviews == nil {
		return nil, errors.New("finding collection dependencies are incomplete")
	}
	return &CollectionPublisher{pool: pool, reviews: reviews}, nil
}

type collectionPublication struct {
	Schema        string        `json:"schema"`
	RequestDigest string        `json:"request_digest"`
	Artifact      ExactArtifact `json:"artifact"`
}

// PublishCollection captures an explicitly enumerated selection and publishes
// its complete ZIP and replay receipt in one Artifact-plane transaction.
// Published snapshots replay before consulting mutable or deleted sources.
func (p *CollectionPublisher) PublishCollection(ctx context.Context, params PublishCollectionParams) (PublishedCollection, error) {
	request, requestDigest, err := canonicalCollectionRequest(params.Request)
	if err != nil {
		return PublishedCollection{}, err
	}
	if params.OwnerID == "" {
		return PublishedCollection{}, ErrInvalid
	}
	for attempt := 0; attempt < 3; attempt++ {
		var result PublishedCollection
		err = postgres.InTx(ctx, p.pool, pgx.TxOptions{IsoLevel: pgx.RepeatableRead}, func(tx pgx.Tx) error {
			// A competing transaction may have taken its snapshot before this
			// lock was released. PostgreSQL's serialization error is retried below.
			key := deterministicID("collection-lock", params.OwnerID, request.ClientKey)
			if _, err := tx.Exec(ctx, `SELECT pg_advisory_xact_lock(hashtextextended($1, 0))`, key); err != nil {
				return err
			}
			service := artifacts.NewService(artifacts.NewPostgresRepository(tx))
			user, err := service.User(params.OwnerID)
			if err != nil {
				return err
			}
			if replay, found, err := readCollectionReplay(ctx, user, request.ClientKey, requestDigest); err != nil || found {
				result = replay
				return err
			}
			var captured time.Time
			if err := tx.QueryRow(ctx, `SELECT transaction_timestamp()`).Scan(&captured); err != nil {
				return err
			}
			value, contents, err := p.captureCollection(ctx, tx, service, params.OwnerID, request, captured)
			if err != nil {
				return err
			}
			payload, err := auditdomain.BuildFindingCollectionPackage(value, contents)
			if err != nil {
				return err
			}
			written, err := user.Write(ctx, contracts.ArtifactRef{Namespace: CollectionNamespace, Name: request.ClientKey},
				artifacts.Payload{MediaType: auditdomain.FindingCollectionMediaType, Data: payload}, nil)
			if err != nil {
				return err
			}
			exact := ExactArtifact{Ref: written.Ref, Digest: digestBytes(payload), MediaType: written.MediaType, SizeBytes: written.Size}
			receipt, err := json.Marshal(collectionPublication{Schema: collectionPublicationSchema, RequestDigest: requestDigest, Artifact: exact})
			if err != nil {
				return err
			}
			if _, err := user.Write(ctx, contracts.ArtifactRef{Namespace: CollectionReceiptNamespace, Name: request.ClientKey},
				artifacts.Payload{MediaType: "application/json", Data: receipt}, nil); err != nil {
				return err
			}
			result = PublishedCollection{Artifact: exact, SnapshotAt: value.SnapshotAt, EntryCount: len(value.Entries)}
			return nil
		})
		var postgresError *pgconn.PgError
		if !errors.As(err, &postgresError) || postgresError.Code != "40001" {
			return result, err
		}
	}
	return PublishedCollection{}, ErrConflict
}

func readCollectionReplay(ctx context.Context, user artifacts.ScopedStore, key, requestDigest string) (PublishedCollection, bool, error) {
	receiptRef := contracts.ArtifactRef{Namespace: CollectionReceiptNamespace, Name: key}
	receiptMetadata, err := user.Metadata(ctx, receiptRef)
	if errors.Is(err, artifacts.ErrArtifactNotFound) {
		// An orphan/caller-created target must not be mistaken for an absent
		// publication and overwritten. Only the paired receipt proves replay.
		_, targetErr := user.Metadata(ctx, contracts.ArtifactRef{Namespace: CollectionNamespace, Name: key})
		if errors.Is(targetErr, artifacts.ErrArtifactNotFound) {
			return PublishedCollection{}, false, nil
		}
		if targetErr != nil {
			return PublishedCollection{}, false, targetErr
		}
		return PublishedCollection{}, false, ErrConflict
	}
	if err != nil {
		return PublishedCollection{}, false, err
	}
	if receiptMetadata.MediaType != "application/json" || receiptMetadata.Size > 4096 {
		return PublishedCollection{}, false, ErrConflict
	}
	read, err := user.Read(ctx, receiptMetadata.Ref)
	if err != nil {
		return PublishedCollection{}, false, err
	}
	var receipt collectionPublication
	if read.Payload.MediaType != "application/json" || len(read.Payload.Data) > 4096 || json.Unmarshal(read.Payload.Data, &receipt) != nil ||
		receipt.Schema != collectionPublicationSchema || receipt.RequestDigest != requestDigest || receipt.Artifact.Ref.ValidateExact() != nil ||
		receipt.Artifact.Ref.Namespace != CollectionNamespace || receipt.Artifact.Ref.Name != key || receipt.Artifact.MediaType != auditdomain.FindingCollectionMediaType ||
		receipt.Artifact.SizeBytes <= 0 || receipt.Artifact.SizeBytes > auditdomain.MaximumArchiveBytes {
		return PublishedCollection{}, false, ErrConflict
	}
	metadata, err := user.Metadata(ctx, receipt.Artifact.Ref)
	if err != nil {
		return PublishedCollection{}, false, err
	}
	if metadata.Digest != receipt.Artifact.Digest || metadata.MediaType != receipt.Artifact.MediaType || metadata.Size != receipt.Artifact.SizeBytes {
		return PublishedCollection{}, false, artifacts.ErrArtifactIntegrity
	}
	zip, err := user.Read(ctx, receipt.Artifact.Ref)
	if err != nil {
		return PublishedCollection{}, false, err
	}
	value, pkg, err := auditdomain.DecodeFindingCollectionPackage(zip.Payload.Data)
	if err != nil {
		return PublishedCollection{}, false, err
	}
	if pkg.Digest != receipt.Artifact.Digest {
		return PublishedCollection{}, false, artifacts.ErrArtifactIntegrity
	}
	return PublishedCollection{Artifact: receipt.Artifact, SnapshotAt: value.SnapshotAt, EntryCount: len(value.Entries), Replayed: true}, true, nil
}

func (p *CollectionPublisher) captureCollection(ctx context.Context, tx pgx.Tx, service *artifacts.Service, ownerID string, request PublishCollectionRequest, captured time.Time) (auditdomain.FindingCollection, map[string][]byte, error) {
	value := auditdomain.FindingCollection{
		Schema: auditdomain.FindingCollectionSchema, SnapshotAt: captured.UTC().Format(time.RFC3339Nano),
		Sources: []auditdomain.FindingCollectionSource{}, Entries: []auditdomain.FindingCollectionEntry{}, Documents: []auditdomain.FindingCollectionDocument{},
	}
	fail := func(err error) (auditdomain.FindingCollection, map[string][]byte, error) {
		return auditdomain.FindingCollection{}, nil, err
	}
	selected := make(map[string]bool)
	for _, source := range request.Sources {
		value.Sources = append(value.Sources, auditdomain.FindingCollectionSource{Kind: source.Kind, ID: source.ID})
		if err := authorizeCollectionSource(ctx, tx, ownerID, source); err != nil {
			return fail(err)
		}
		ids := append([]string{}, source.ReceiptIDs...)
		if source.Kind == "audit" && len(source.Findings) != 0 {
			contributing, err := p.reviews.SelectCollectionFindings(ctx, tx, ownerID, source.ID, source.Findings)
			if err != nil {
				return fail(err)
			}
			ids = append(ids, contributing...)
		}
		if len(ids) > auditdomain.MaximumCollectionEntries*2 {
			return fail(ErrInvalid)
		}
		for _, id := range ids {
			if err := authorizeCollectionReceipt(ctx, tx, ownerID, source, id); err != nil {
				return fail(err)
			}
			selected[id] = true
			if len(selected) > auditdomain.MaximumCollectionEntries {
				return fail(fmt.Errorf("%w: expanded collection selection limit", ErrInvalid))
			}
		}
	}
	ids := make([]string, 0, len(selected))
	for id := range selected {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	reviews, err := p.reviews.ReadCollectionReviews(ctx, tx, ownerID, ids)
	if err != nil {
		return fail(err)
	}
	builder := collectionDocuments{service: service, documents: map[string]auditdomain.FindingCollectionDocument{}, contents: map[string][]byte{}}
	for _, id := range ids {
		receipt, err := readReceiptByID(ctx, tx, id)
		if err != nil {
			return fail(err)
		}
		entry := auditdomain.FindingCollectionEntry{
			ReceiptID: receipt.ReceiptID, ProposalID: receipt.ProposalID, RunID: receipt.Origin.RunID, InvocationID: receipt.Origin.InvocationID,
			Retention: string(receipt.Retention), Evidence: []auditdomain.FindingCollectionEvidence{}, Reviews: append([]auditdomain.FindingCollectionReview{}, reviews[id]...),
		}
		if receipt.Origin.Audit != nil {
			origin := receipt.Origin.Audit
			entry.AuditOrigin = &auditdomain.FindingCollectionAuditOrigin{AuditID: origin.AuditID, ExecutionID: origin.ExecutionID, Role: origin.Role}
		}
		// Retained copies remain usable after the source Run is deleted. Check
		// ownership of each hold rather than interpreting scoped JSON as a grant.
		holds := []AuditHold{}
		for _, hold := range receipt.AuditHolds {
			var allowed bool
			if err := tx.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM audits AS a JOIN projects AS p ON p.project_id = a.project_id WHERE a.audit_id=$1 AND a.owner_id=$2 AND p.owner_id=$2 AND a.project_id=$3)`, hold.AuditID, ownerID, hold.ProjectID).Scan(&allowed); err != nil {
				return fail(err)
			}
			if allowed {
				holds = append(holds, hold)
				entry.AuditHolds = append(entry.AuditHolds, hold.AuditID)
			}
		}
		sort.Strings(entry.AuditHolds)
		sort.Slice(holds, func(i, j int) bool { return holds[i].AuditID < holds[j].AuditID })
		scope := auditdomain.FindingCollectionSource{Kind: "run", ID: receipt.Origin.RunID}
		proposal, evidence := receipt.Proposal, receipt.Evidence
		if !receipt.Origin.RunDeleted {
			if err := authorizeCollectionSource(ctx, tx, ownerID, CollectionSelection{Kind: "run", ID: receipt.Origin.RunID}); err != nil {
				return fail(err)
			}
		}
		if receipt.Origin.RunDeleted {
			if len(holds) == 0 {
				return fail(ErrNotFound)
			}
			hold := holds[0]
			scope = auditdomain.FindingCollectionSource{Kind: "project", ID: hold.ProjectID}
			proposal, evidence = hold.Proposal, hold.Evidence
		}
		if !sameCollectionContent(proposal, receipt.Proposal) || len(evidence) != len(receipt.Evidence) {
			return fail(artifacts.ErrArtifactIntegrity)
		}
		entry.ProposalDocumentID, err = builder.add(ctx, scope, proposal)
		if err != nil {
			return fail(err)
		}
		decoded, err := auditdomain.DecodeFindingProposal(builder.contents[entry.ProposalDocumentID])
		if err != nil || decoded.ClientKey != receipt.ClientKey || len(decoded.EvidenceIDs) != len(evidence) {
			return fail(artifacts.ErrArtifactIntegrity)
		}
		for i, item := range evidence {
			if !sameCollectionContent(item, receipt.Evidence[i]) || decoded.EvidenceIDs[i] != fmt.Sprintf("evidence-%d", i+1) {
				return fail(artifacts.ErrArtifactIntegrity)
			}
			documentID, err := builder.add(ctx, scope, item)
			if err != nil {
				return fail(err)
			}
			entry.Evidence = append(entry.Evidence, auditdomain.FindingCollectionEvidence{EvidenceID: decoded.EvidenceIDs[i], DocumentID: documentID})
		}
		sort.Slice(entry.Evidence, func(i, j int) bool { return entry.Evidence[i].EvidenceID < entry.Evidence[j].EvidenceID })
		value.Entries = append(value.Entries, entry)
	}
	for _, doc := range builder.documents {
		value.Documents = append(value.Documents, doc)
	}
	sort.Slice(value.Documents, func(i, j int) bool { return value.Documents[i].ID < value.Documents[j].ID })
	return value, builder.contents, nil
}

func authorizeCollectionSource(ctx context.Context, tx pgx.Tx, ownerID string, source CollectionSelection) error {
	var exists bool
	var err error
	if source.Kind == "run" {
		err = tx.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM workflow_runs WHERE run_id=$1 AND owner_id=$2)`, source.ID, ownerID).Scan(&exists)
	} else {
		err = tx.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM audits WHERE audit_id=$1 AND owner_id=$2)`, source.ID, ownerID).Scan(&exists)
	}
	if err != nil {
		return err
	}
	if !exists {
		return ErrNotFound
	}
	return nil
}

func authorizeCollectionReceipt(ctx context.Context, tx pgx.Tx, ownerID string, source CollectionSelection, receiptID string) error {
	var allowed bool
	err := tx.QueryRow(ctx, `SELECT EXISTS(
SELECT 1 FROM finding_proposal_receipts AS r WHERE r.owner_id=$1 AND r.receipt_id=$2 AND (
    ($3='run' AND r.run_id=$4) OR ($3='audit' AND (r.audit_id=$4 OR EXISTS(
        SELECT 1 FROM finding_proposal_audit_holds AS h WHERE h.receipt_id=r.receipt_id AND h.audit_id=$4
    ) OR EXISTS(SELECT 1 FROM audit_finding_contributions AS c WHERE c.receipt_id=r.receipt_id AND c.audit_id=$4)))))`,
		ownerID, receiptID, source.Kind, source.ID).Scan(&allowed)
	if err != nil {
		return err
	}
	if !allowed {
		return ErrNotFound
	}
	return nil
}

type collectionDocuments struct {
	service   *artifacts.Service
	documents map[string]auditdomain.FindingCollectionDocument
	contents  map[string][]byte
	size      int64
}

func (b *collectionDocuments) add(ctx context.Context, scope auditdomain.FindingCollectionSource, exact ExactArtifact) (string, error) {
	doc := auditdomain.FindingCollectionDocument{Scope: scope, Ref: exact.Ref, Digest: exact.Digest, MediaType: exact.MediaType, SizeBytes: exact.SizeBytes}
	id, err := auditdomain.FindingCollectionDocumentID(doc)
	if err != nil {
		return "", err
	}
	if _, ok := b.documents[id]; ok {
		return id, nil
	}
	if len(b.documents) >= auditdomain.MaximumCollectionDocuments || exact.SizeBytes > auditdomain.MaximumArchiveBytes-b.size {
		return "", fmt.Errorf("%w: collection document limit", ErrInvalid)
	}
	var store artifacts.ScopedStore
	if scope.Kind == "run" {
		store, err = b.service.Run(scope.ID)
	} else {
		store, err = b.service.Project(scope.ID)
	}
	if err != nil {
		return "", err
	}
	metadata, err := store.Metadata(ctx, exact.Ref)
	if err != nil {
		return "", err
	}
	if metadata.Digest != exact.Digest || metadata.Size != exact.SizeBytes || metadata.MediaType != exact.MediaType {
		return "", artifacts.ErrArtifactIntegrity
	}
	read, err := store.Read(ctx, exact.Ref)
	if err != nil {
		return "", err
	}
	if int64(len(read.Payload.Data)) != exact.SizeBytes || read.Payload.MediaType != exact.MediaType || digestBytes(read.Payload.Data) != exact.Digest {
		return "", artifacts.ErrArtifactIntegrity
	}
	doc.ID = id
	b.documents[id], b.contents[id] = doc, read.Payload.Data
	b.size += exact.SizeBytes
	return id, nil
}

func sameCollectionContent(a, b ExactArtifact) bool {
	return a.Digest == b.Digest && a.MediaType == b.MediaType && a.SizeBytes == b.SizeBytes
}
