package findingintake

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"sort"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contracts"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
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
	var result PublishedCollection
	err = postgres.InTxWithRetry(ctx, p.pool, pgx.TxOptions{IsoLevel: pgx.RepeatableRead}, func(tx pgx.Tx) error {
		result = PublishedCollection{}
		// A competing transaction may have taken its snapshot before this
		// lock was released; InTxWithRetry replays PostgreSQL's serialization error.
		key := postgres.NewAdvisoryKey("finding-collection", params.OwnerID, request.ClientKey)
		if err := postgres.LockAdvisory(ctx, tx, key); err != nil {
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
		value, contents, err := p.captureCollection(ctx, tx, params.OwnerID, request, captured)
		if err != nil {
			return err
		}
		payload, err := auditdomain.BuildFindingCollectionPackage(value, contents)
		if err != nil {
			return err
		}
		written, err := user.Write(ctx, contracts.ArtifactRef{Namespace: CollectionNamespace, Name: request.ClientKey},
			artifacts.Payload{MediaType: contracts.FindingCollectionMediaType, Data: payload}, nil)
		if err != nil {
			return err
		}
		exact := ExactArtifact{Ref: written.Ref, Digest: auditdomain.DigestBytes(payload), MediaType: written.MediaType, SizeBytes: written.Size}
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
	if err != nil {
		// A transaction conflict that outlasts the retries stays one: the
		// caller reports it as the retryable storage_transaction_conflict
		// (PER-R16).
		return PublishedCollection{}, err
	}
	return result, nil
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
		receipt.Artifact.Ref.Namespace != CollectionNamespace || receipt.Artifact.Ref.Name != key || receipt.Artifact.MediaType != contracts.FindingCollectionMediaType ||
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

// captureCollection reads the selection with a fixed number of statements
// per source and per publication, never per entry: one authorization of
// every source, one receipt authorization per source, then the selected
// receipts, their holds, the Audits of those holds and the source Runs in one
// statement each, and the documents in exact-read batches.
func (p *CollectionPublisher) captureCollection(ctx context.Context, tx pgx.Tx, ownerID string, request PublishCollectionRequest, captured time.Time) (auditdomain.FindingCollection, map[string][]byte, error) {
	value := auditdomain.FindingCollection{
		Schema: auditdomain.FindingCollectionSchema, SnapshotAt: captured.UTC().Format(time.RFC3339Nano),
		Sources: []auditdomain.FindingCollectionSource{}, Entries: []auditdomain.FindingCollectionEntry{}, Documents: []auditdomain.FindingCollectionDocument{},
	}
	fail := func(err error) (auditdomain.FindingCollection, map[string][]byte, error) {
		return auditdomain.FindingCollection{}, nil, err
	}
	sources, err := authorizedCollectionSources(ctx, tx, ownerID, request.Sources)
	if err != nil {
		return fail(err)
	}
	selected := make(map[string]bool)
	// requestingAudits names, per receipt, the Audit sources that selected it.
	requestingAudits := make(map[string][]string)
	for _, source := range request.Sources {
		value.Sources = append(value.Sources, auditdomain.FindingCollectionSource{Kind: source.Kind, ID: source.ID})
		if !sources[collectionSourceKey(source.Kind, source.ID)] {
			return fail(ErrNotFound)
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
		allowed, err := authorizedCollectionReceipts(ctx, tx, ownerID, source, ids)
		if err != nil {
			return fail(err)
		}
		for _, id := range ids {
			if !allowed[id] {
				return fail(ErrNotFound)
			}
			selected[id] = true
			if source.Kind == "audit" {
				requestingAudits[id] = append(requestingAudits[id], source.ID)
			}
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
	receipts, err := readCollectionReceipts(ctx, tx, ownerID, ids)
	if err != nil {
		return fail(err)
	}
	holdAudits, err := collectionHoldAudits(ctx, tx, ownerID, receipts)
	if err != nil {
		return fail(err)
	}
	liveRuns := make([]CollectionSelection, 0)
	seenRuns := make(map[string]bool)
	for _, id := range ids {
		if origin := receipts[id].Origin; !origin.RunDeleted && !seenRuns[origin.RunID] {
			seenRuns[origin.RunID] = true
			liveRuns = append(liveRuns, CollectionSelection{Kind: "run", ID: origin.RunID})
		}
	}
	runs, err := authorizedCollectionSources(ctx, tx, ownerID, liveRuns)
	if err != nil {
		return fail(err)
	}
	builder := collectionDocuments{
		repository: artifacts.NewPostgresRepository(tx),
		documents:  map[string]auditdomain.FindingCollectionDocument{}, contents: map[string][]byte{},
	}
	planned := make([]plannedCollectionEntry, 0, len(ids))
	for _, id := range ids {
		receipt := receipts[id]
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
			if project, owned := holdAudits[hold.AuditID]; owned && project == hold.ProjectID {
				holds = append(holds, hold)
				entry.AuditHolds = append(entry.AuditHolds, hold.AuditID)
			}
		}
		sort.Strings(entry.AuditHolds)
		sort.Slice(holds, func(i, j int) bool { return holds[i].AuditID < holds[j].AuditID })
		scope := auditdomain.FindingCollectionSource{Kind: "run", ID: receipt.Origin.RunID}
		proposal, evidence := receipt.Proposal, receipt.Evidence
		if !receipt.Origin.RunDeleted && !runs[collectionSourceKey("run", receipt.Origin.RunID)] {
			return fail(ErrNotFound)
		}
		if receipt.Origin.RunDeleted {
			// As for a receipt read through an Audit, only a requesting
			// Audit's own retained copy is readable after source deletion.
			hold, found := requestedAuditHold(holds, requestingAudits[id])
			if !found {
				return fail(ErrNotFound)
			}
			scope = auditdomain.FindingCollectionSource{Kind: "project", ID: hold.ProjectID}
			proposal, evidence = hold.Proposal, hold.Evidence
		}
		if !sameCollectionContent(proposal, receipt.Proposal) || len(evidence) != len(receipt.Evidence) {
			return fail(artifacts.ErrArtifactIntegrity)
		}
		entry.ProposalDocumentID, err = builder.add(scope, proposal)
		if err != nil {
			return fail(err)
		}
		documentIDs := make([]string, len(evidence))
		for i, item := range evidence {
			if !sameCollectionContent(item, receipt.Evidence[i]) {
				return fail(artifacts.ErrArtifactIntegrity)
			}
			if documentIDs[i], err = builder.add(scope, item); err != nil {
				return fail(err)
			}
		}
		planned = append(planned, plannedCollectionEntry{entry: entry, clientKey: receipt.ClientKey, evidenceDocumentIDs: documentIDs})
	}
	if err := builder.read(ctx); err != nil {
		return fail(err)
	}
	for _, plan := range planned {
		entry := plan.entry
		decoded, err := auditdomain.DecodeFindingProposal(builder.contents[entry.ProposalDocumentID])
		if err != nil || decoded.ClientKey != plan.clientKey || len(decoded.EvidenceIDs) != len(plan.evidenceDocumentIDs) {
			return fail(artifacts.ErrArtifactIntegrity)
		}
		for i, documentID := range plan.evidenceDocumentIDs {
			if decoded.EvidenceIDs[i] != fmt.Sprintf("evidence-%d", i+1) {
				return fail(artifacts.ErrArtifactIntegrity)
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

// plannedCollectionEntry is an entry whose documents are planned but not yet
// read: the proposal is checked against its receipt once every document of
// the publication has been read.
type plannedCollectionEntry struct {
	entry               auditdomain.FindingCollectionEntry
	clientKey           string
	evidenceDocumentIDs []string
}

// requestedAuditHold returns the hold of the first requesting Audit in the
// AuditID-ordered holds, so the published copy is deterministic.
func requestedAuditHold(holds []AuditHold, auditIDs []string) (AuditHold, bool) {
	for _, hold := range holds {
		if slices.Contains(auditIDs, hold.AuditID) {
			return hold, true
		}
	}
	return AuditHold{}, false
}

func collectionSourceKey(kind, id string) string {
	return kind + "\x00" + id
}

// authorizedCollectionSources returns, keyed by collectionSourceKey, the
// sources among selections that the owner may read: its Runs and Audits.
func authorizedCollectionSources(ctx context.Context, tx pgx.Tx, ownerID string, selections []CollectionSelection) (map[string]bool, error) {
	allowed := make(map[string]bool, len(selections))
	if len(selections) == 0 {
		return allowed, nil
	}
	kinds, ids := make([]string, len(selections)), make([]string, len(selections))
	for index, selection := range selections {
		kinds[index], ids[index] = selection.Kind, selection.ID
	}
	rows, err := tx.Query(ctx, authorizedCollectionSourcesSQL, ownerID, kinds, ids)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	for rows.Next() {
		var kind, id string
		if err := rows.Scan(&kind, &id); err != nil {
			return nil, err
		}
		allowed[collectionSourceKey(kind, id)] = true
	}
	return allowed, rows.Err()
}

// authorizedCollectionReceipts returns the receipts among ids that the owner
// may select through source.
func authorizedCollectionReceipts(ctx context.Context, tx pgx.Tx, ownerID string, source CollectionSelection, ids []string) (map[string]bool, error) {
	allowed := make(map[string]bool, len(ids))
	if len(ids) == 0 {
		return allowed, nil
	}
	rows, err := tx.Query(ctx, authorizedCollectionReceiptsSQL, ownerID, ids, source.Kind, source.ID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		allowed[id] = true
	}
	return allowed, rows.Err()
}

// readCollectionReceipts reads the selected receipts of the owner with every
// Audit hold of each.
func readCollectionReceipts(ctx context.Context, tx pgx.Tx, ownerID string, ids []string) (map[string]Receipt, error) {
	receipts := make(map[string]Receipt, len(ids))
	if len(ids) == 0 {
		return receipts, nil
	}
	rows, err := tx.Query(ctx, `
SELECT `+receiptProjection+`
  FROM finding_proposal_receipts AS receipt
 WHERE receipt.receipt_id = ANY($1::text[]) AND receipt.owner_id = $2`, ids, ownerID)
	if err != nil {
		return nil, fmt.Errorf("read selected finding receipts: %w", err)
	}
	scanned, err := scanReceiptRows(rows)
	if err != nil {
		return nil, err
	}
	holds, err := readRunAuditHoldsBatch(ctx, tx, ids)
	if err != nil {
		return nil, err
	}
	for _, receipt := range scanned {
		receipt.AuditHolds = holds[receipt.ReceiptID]
		if receipt.AuditHolds == nil {
			receipt.AuditHolds = []AuditHold{}
		}
		receipts[receipt.ReceiptID] = receipt
	}
	for _, id := range ids {
		if _, ok := receipts[id]; !ok {
			return nil, ErrNotFound
		}
	}
	return receipts, nil
}

// collectionHoldAudits returns the Project of every Audit holding one of the
// receipts that the owner owns in a Project it also owns.
func collectionHoldAudits(ctx context.Context, tx pgx.Tx, ownerID string, receipts map[string]Receipt) (map[string]string, error) {
	projects := make(map[string]string)
	auditIDs := make([]string, 0)
	seen := make(map[string]bool)
	for _, receipt := range receipts {
		for _, hold := range receipt.AuditHolds {
			if !seen[hold.AuditID] {
				seen[hold.AuditID] = true
				auditIDs = append(auditIDs, hold.AuditID)
			}
		}
	}
	if len(auditIDs) == 0 {
		return projects, nil
	}
	rows, err := tx.Query(ctx, collectionHoldAuditsSQL, auditIDs, ownerID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	for rows.Next() {
		var auditID, projectID string
		if err := rows.Scan(&auditID, &projectID); err != nil {
			return nil, err
		}
		projects[auditID] = projectID
	}
	return projects, rows.Err()
}

// collectionDocuments plans the documents of a publication, deduplicated and
// within the collection limits, and then reads them all in exact-read
// batches.
type collectionDocuments struct {
	repository *artifacts.PostgresRepository
	documents  map[string]auditdomain.FindingCollectionDocument
	contents   map[string][]byte
	planned    []plannedCollectionDocument
	size       int64
}

type plannedCollectionDocument struct {
	id       string
	request  artifacts.ExactReadRequest
	expected ExactArtifact
}

// add plans one document and returns its ID; a document already planned is
// planned once.
func (b *collectionDocuments) add(scope auditdomain.FindingCollectionSource, exact ExactArtifact) (string, error) {
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
	var artifactScope artifacts.Scope
	if scope.Kind == "run" {
		artifactScope, err = artifacts.RunScope(scope.ID)
	} else {
		artifactScope, err = artifacts.ProjectScope(scope.ID)
	}
	if err != nil {
		return "", err
	}
	doc.ID = id
	b.documents[id] = doc
	b.planned = append(b.planned, plannedCollectionDocument{
		id: id, request: artifacts.ExactReadRequest{Scope: artifactScope, Ref: exact.Ref}, expected: exact,
	})
	b.size += exact.SizeBytes
	return id, nil
}

// read reads every planned document, at most MaxExactReadBatchSize per
// statement, and keeps its bytes once they match the planned identity. The
// planned sizes stay within MaximumArchiveBytes, below one batch's payload
// bound.
func (b *collectionDocuments) read(ctx context.Context) error {
	for start := 0; start < len(b.planned); start += artifacts.MaxExactReadBatchSize {
		batch := b.planned[start:min(start+artifacts.MaxExactReadBatchSize, len(b.planned))]
		requests := make([]artifacts.ExactReadRequest, len(batch))
		for index, document := range batch {
			requests[index] = document.request
		}
		reads, err := b.repository.ReadExactBatch(ctx, requests)
		if errors.Is(err, artifacts.ErrPayloadTooLarge) {
			// The planned sizes fit one batch, so stored sizes disagree with
			// the receipts.
			return artifacts.ErrArtifactIntegrity
		}
		if err != nil {
			return err
		}
		for index, read := range reads {
			expected := batch[index].expected
			if int64(len(read.Payload.Data)) != expected.SizeBytes || read.Payload.MediaType != expected.MediaType ||
				auditdomain.DigestBytes(read.Payload.Data) != expected.Digest {
				return artifacts.ErrArtifactIntegrity
			}
			b.contents[batch[index].id] = read.Payload.Data
		}
	}
	return nil
}

func sameCollectionContent(a, b ExactArtifact) bool {
	return a.Digest == b.Digest && a.MediaType == b.MediaType && a.SizeBytes == b.SizeBytes
}
