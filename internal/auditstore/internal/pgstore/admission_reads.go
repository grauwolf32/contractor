package pgstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/artifacts"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"github.com/jackc/pgx/v5"
)

type admissionReader struct {
	*QueryStore
	db             pgx.Tx
	state          auditstore.AdmissionProgress
	owner, project string
	work           *admissionWork
	indexes        map[string]auditdomain.FindingChecksProjection
	sources        map[string]auditstore.AdmissionFindingSource
	payloads       map[string][]byte
	contexts       map[string]auditdomain.FindingCheckContext
	readPage       bool
	pageCache      map[int]admissionCachedPage
	cursors        map[auditstore.AdmissionCursor]bool
}

func newAdmissionReader(tx pgx.Tx, state auditstore.AdmissionProgress, owner, project string, storage *artifactpostgres.Storage) *admissionReader {
	return &admissionReader{QueryStore: NewQueryStore(tx, storage), db: tx, state: state, owner: owner, project: project,
		work: &admissionWork{}, indexes: map[string]auditdomain.FindingChecksProjection{}, sources: map[string]auditstore.AdmissionFindingSource{},
		payloads: map[string][]byte{}, contexts: map[string]auditdomain.FindingCheckContext{}, pageCache: map[int]admissionCachedPage{}, cursors: map[auditstore.AdmissionCursor]bool{state.Cursor: true}}
}

func (r *admissionReader) ReadCandidates(ctx context.Context, cursor auditstore.AdmissionCursor, limit int) (auditstore.AdmissionCandidatePage, error) {
	result := auditstore.AdmissionCandidatePage{Candidates: []auditstore.AdmissionCandidate{}}
	if r.readPage || limit < 1 || limit > auditstore.MaxAdmissionPrimary || cursor != r.state.Cursor {
		return result, auditstore.ErrInvalid
	}
	r.readPage = true
	rows, err := r.db.Query(ctx, admissionCandidateHeadersSQL, r.state.AuditID, r.project, cursor.CreatedAt, cursor.ReceiptID,
		r.state.SourceUpper.CreatedAt, r.state.SourceUpper.ReceiptID, r.state.SourceCutoffAt, limit, cursor.Ordinal)
	if err != nil {
		return result, err
	}
	type header struct {
		id           string
		at           auditstore.AdmissionCursor
		late         bool
		bytes, count *int
		proposal     auditstore.ExactArtifact
	}
	var headers []header
	for rows.Next() {
		var h header
		var proposal []byte
		if err := rows.Scan(&h.id, &h.at.CreatedAt, &h.late, &proposal, &h.bytes, &h.count); err != nil {
			rows.Close()
			return result, err
		}
		if h.bytes == nil || h.count == nil || *h.bytes < 1 || *h.count < 0 || *h.count > 512 || strictjson.Decode(proposal, &h.proposal) != nil || h.proposal.Validate() != nil {
			rows.Close()
			return result, auditstore.ErrInvalid
		}
		if err := r.work.add(0, 1, len(proposal), 0); err != nil {
			rows.Close()
			return result, err
		}
		h.at.ReceiptID = h.id
		headers = append(headers, h)
	}
	err = rows.Err()
	rows.Close()
	if err != nil {
		return result, err
	}
	if err := r.work.add(0, len(headers), 0, 0); err != nil {
		return result, err
	}
	complete := true
	for _, h := range headers {
		if len(result.Candidates) == limit {
			complete = false
			break
		}
		start := 0
		if h.id == cursor.ReceiptID {
			start = cursor.Ordinal + 1
		}
		if *h.count > 0 && start >= *h.count && !h.late {
			continue
		}
		if h.late || *h.count == 0 {
			candidate := auditstore.AdmissionCandidate{ReceiptID: h.id, Proposal: h.proposal, CreatedAt: h.at.CreatedAt, Ordinal: -1,
				LateSource: h.late, NoChecks: *h.count == 0,
				Cursor: auditstore.AdmissionCursor{CreatedAt: h.at.CreatedAt, ReceiptID: h.id, Ordinal: 512}}
			if err := r.work.add(1, 0, 256, 0); err != nil {
				return result, err
			}
			result.Candidates = append(result.Candidates, candidate)
			continue
		}
		if *h.bytes > auditstore.MaxAdmissionMetadataBytes-r.work.metadata {
			complete = false
			break
		}
		projection, err := r.readIndex(ctx, h.id, *h.bytes)
		if err != nil {
			return result, err
		}
		for ordinal := start; ordinal < len(projection.Checks); ordinal++ {
			if len(result.Candidates) == limit {
				complete = false
				break
			}
			entry := projection.Checks[ordinal]
			if err := r.work.add(1, 0, 0, 0); err != nil {
				return result, err
			}
			candidate := auditstore.AdmissionCandidate{ReceiptID: h.id, Proposal: h.proposal, CreatedAt: h.at.CreatedAt, Ordinal: ordinal, Header: entry,
				ProjectionBytes: *h.bytes, Cursor: auditstore.AdmissionCursor{CreatedAt: h.at.CreatedAt, ReceiptID: h.id, Ordinal: ordinal}}

			result.Candidates = append(result.Candidates, candidate)
		}
	}
	if err := r.readCandidateStatus(ctx, result.Candidates); err != nil {
		return result, err
	}
	for _, candidate := range result.Candidates {
		r.cursors[candidate.Cursor] = true
	}
	result.Exhausted = complete && len(headers) < limit
	if err := r.db.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM finding_proposal_audit_holds WHERE audit_id=$1 AND created_at>$2 LIMIT 1)`, r.state.AuditID, r.state.SourceCutoffAt).Scan(&result.LateSources); err != nil {
		return result, err
	}
	return result, nil
}

func (r *admissionReader) readIndex(ctx context.Context, id string, size int) (auditdomain.FindingChecksProjection, error) {
	if value, ok := r.indexes[id]; ok {
		return value, nil
	}
	if size < 1 || size > auditdomain.MaximumFindingChecksProjectionBytes {
		return auditdomain.FindingChecksProjection{}, auditstore.ErrInvalid
	}
	if err := r.work.add(0, 0, size, 0); err != nil {
		return auditdomain.FindingChecksProjection{}, err
	}
	var data []byte
	if err := r.db.QueryRow(ctx, `SELECT checks_projection FROM finding_proposal_receipts WHERE receipt_id=$1`, id).Scan(&data); err != nil {
		return auditdomain.FindingChecksProjection{}, err
	}
	if len(data) != size {
		return auditdomain.FindingChecksProjection{}, auditstore.ErrInvalid
	}
	value, err := auditdomain.DecodeFindingChecksProjection(data)
	if err != nil || value.Discarded {
		return value, auditstore.ErrInvalid
	}
	r.indexes[id] = value
	return value, nil
}

func (r *admissionReader) FindingSource(ctx context.Context, id string) (auditstore.AdmissionFindingSource, error) {
	if value, ok := r.sources[id]; ok {
		return value, nil
	}
	var cost, related int
	var projectionCost *int
	if err := r.db.QueryRow(ctx, admissionFindingSourceCostSQL, r.state.AuditID, r.owner, r.project, id).Scan(&cost, &related, &projectionCost); err != nil {
		return auditstore.AdmissionFindingSource{}, admissionReadError(err)
	}
	if projectionCost == nil {
		return auditstore.AdmissionFindingSource{}, auditstore.ErrInvalid
	}
	if err := r.work.add(0, related, cost, 0); err != nil {
		return auditstore.AdmissionFindingSource{}, err
	}
	projection, err := r.readIndex(ctx, id, *projectionCost)
	if err != nil {
		return auditstore.AdmissionFindingSource{}, err
	}
	var value auditstore.AdmissionFindingSource
	var original, retained, originalEvidence, retainedEvidence []byte
	var projectionBytes *int
	var overflow bool
	err = r.db.QueryRow(ctx, admissionFindingSourceSQL, r.state.AuditID, r.owner, r.project, id).Scan(&value.ReceiptID, &value.ProjectID,
		&original, &value.Original.Digest, &value.Original.MediaType, &value.Original.SizeBytes, &retained, &value.OriginRunID, &value.OriginExecutionID,
		&value.OriginAuditID, &projectionBytes, &originalEvidence, &retainedEvidence, &overflow)
	if errors.Is(err, pgx.ErrNoRows) {
		return value, auditstore.ErrNotFound
	}
	if err != nil {
		return value, err
	}
	if projectionBytes == nil {
		return value, auditstore.ErrInvalid
	}
	if strictjson.Decode(original, &value.Original.Ref) != nil || strictjson.Decode(retained, &value.Retained) != nil || value.Original.Validate() != nil || value.Retained.Validate() != nil ||
		value.Original.Digest != value.Retained.Digest || value.Original.MediaType != value.Retained.MediaType || value.Original.SizeBytes != value.Retained.SizeBytes {
		return value, auditstore.ErrInvalid
	}
	value.EvidenceLimitExceeded = overflow
	value.OriginalEvidence, value.RetainedEvidence = []auditstore.ExactArtifact{}, []auditstore.ExactArtifact{}
	if !overflow && (strictjson.Decode(originalEvidence, &value.OriginalEvidence) != nil || strictjson.Decode(retainedEvidence, &value.RetainedEvidence) != nil || len(value.OriginalEvidence) != len(value.RetainedEvidence)) {
		return value, auditstore.ErrInvalid
	}
	value.Projection = projection
	if value.Projection.ProposalDigest != value.Original.Digest || value.Projection.SizeBytes != value.Original.SizeBytes {
		return value, auditstore.ErrInvalid
	}
	r.sources[id] = value
	return value, nil
}

func (r *admissionReader) ReadFindingCheck(ctx context.Context, id string, ordinal int) (auditstore.AdmissionFindingCheck, error) {
	var result auditstore.AdmissionFindingCheck
	source, err := r.FindingSource(ctx, id)
	if err != nil {
		return result, err
	}
	if ordinal < 0 || ordinal >= len(source.Projection.Checks) {
		return result, auditstore.ErrInvalid
	}
	entry := source.Projection.Checks[ordinal]
	if err := r.work.add(0, 0, int(entry.Fragment.Length), 0); err != nil {
		return result, err
	}
	data, ok := r.payloads[id]
	if !ok {
		if err := r.work.add(0, 0, 0, source.Retained.SizeBytes); err != nil {
			return result, err
		}
		store, _ := artifacts.NewService(r.storage.Repository(r.db)).Project(r.project)
		read, err := store.ReadExact(ctx, source.Retained)
		if err != nil {
			return result, err
		}
		data = read.Payload.Data
		r.payloads[id] = data
	}
	contextValue, ok := r.contexts[id]
	if !ok {
		if source.Projection.Subject == nil || source.Projection.Limitations == nil {
			return result, auditstore.ErrInvalid
		}
		if err := r.work.add(0, 0, int(source.Projection.Subject.Length+source.Projection.Limitations.Length), 0); err != nil {
			return result, err
		}
		contextValue, err = auditdomain.ExtractFindingChecksContext(data, *source.Projection.Subject, *source.Projection.Limitations)
		if err != nil {
			return result, err
		}
		r.contexts[id] = contextValue
	}
	check, plan, err := auditdomain.ExtractFindingCheck(data, entry)
	if err != nil {
		return result, err
	}
	result.Source, result.Check, result.Plan, result.Context = source, check, plan, contextValue
	return result, nil
}

func (r *admissionReader) ReadPage(ctx context.Context, ordinal int) (auditstore.ExactArtifact, []byte, error) {
	if cached, ok := r.pageCache[ordinal]; ok {
		return cached.exact, append([]byte(nil), cached.data...), nil
	}
	var exact auditstore.ExactArtifact
	var data []byte
	if ordinal < 0 || ordinal >= r.state.PublishedPages {
		return exact, nil, auditstore.ErrInvalid
	}
	if err := r.db.QueryRow(ctx, `SELECT artifact_ref,artifact_digest,media_type,size_bytes FROM audit_artifact_links WHERE audit_id=$1 AND logical_key=$2`, r.state.AuditID, admissionPageKey(r.state.Generation, "page", ordinal)).Scan(&data, &exact.Digest, &exact.MediaType, &exact.SizeBytes); err != nil {
		return exact, nil, err
	}
	if strictjson.Decode(data, &exact.Ref) != nil || exact.Validate() != nil || exact.MediaType != auditdomain.PreparedRoundPageMediaType || !exact.HasMetadata() || exact.Ref.Namespace != auditdomain.ArtifactNamespace(r.state.AuditID) || exact.SizeBytes < 1 || exact.SizeBytes > auditdomain.MaximumPreparedRoundPageBytes {
		return exact, nil, auditstore.ErrInvalid
	}
	if err := r.work.add(0, 1, int(exact.SizeBytes), 0); err != nil {
		return exact, nil, err
	}
	store, _ := artifacts.NewService(r.storage.Repository(r.db)).Project(r.project)
	read, err := store.ReadExact(ctx, exact)
	if err != nil {
		return exact, nil, err
	}
	r.pageCache[ordinal] = admissionCachedPage{exact: exact, data: read.Payload.Data}
	return exact, append([]byte(nil), read.Payload.Data...), nil
}

func admissionSourceKey(receipt string, ordinal int, proposal, fragment string) string {
	return auditdomain.DigestBytes([]byte(fmt.Sprintf("%s\x00%d\x00%s\x00%s", receipt, ordinal, proposal, fragment)))
}
func admissionPolicyDigest(state auditstore.AdmissionProgress) string {
	return auditdomain.DigestBytes([]byte(state.ProfileDigest + "\n" + state.BaselineDigest))
}
func admissionPageKey(generation, kind string, ordinal int) string {
	return fmt.Sprintf("admission/%s/%s/%06d", generation, kind, ordinal)
}

var _ auditstore.AdmissionReader = (*admissionReader)(nil)

type admissionCachedPage struct {
	exact auditstore.ExactArtifact
	data  []byte
}

func (r *admissionReader) readCandidateStatus(ctx context.Context, candidates []auditstore.AdmissionCandidate) error {
	type statusInput struct {
		Position  int    `json:"position"`
		ReceiptID string `json:"receipt_id"`
		Ordinal   int    `json:"ordinal"`
		SourceKey string `json:"source_key"`
	}
	input := []statusInput{}
	for position, c := range candidates {
		if c.Ordinal < 0 {
			continue
		}
		projection := r.indexes[c.ReceiptID]
		input = append(input, statusInput{position, c.ReceiptID, c.Ordinal, admissionSourceKey(c.ReceiptID, c.Ordinal, projection.ProposalDigest, c.Header.Fragment.Digest)})
	}
	if len(input) == 0 {
		return nil
	}
	encoded, err := json.Marshal(input)
	if err != nil {
		return err
	}
	if err = r.work.add(0, 0, len(encoded), 0); err != nil {
		return err
	}
	rows, err := r.db.Query(ctx, admissionSourceStatusesSQL, r.state.AuditID, encoded, admissionPolicyDigest(r.state))
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		var position int
		var used, rejected bool
		if err = rows.Scan(&position, &used, &rejected); err != nil {
			return err
		}
		if position < 0 || position >= len(candidates) {
			return auditstore.ErrInvalid
		}
		candidates[position].AlreadyConsumed = used
		candidates[position].AlreadyRejected = rejected
	}
	return rows.Err()
}
