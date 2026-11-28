package auditservice

import (
	"context"
	"encoding/json"
	"errors"
	"strings"

	"github.com/grauwolf32/contractor/internal/artifacts"
	artifactpolicy "github.com/grauwolf32/contractor/internal/artifacts/policy"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/hookstore"
	"github.com/grauwolf32/contractor/internal/auditstore/ownerstore"
	"github.com/grauwolf32/contractor/internal/auditstore/querystore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/helpers/pagelimit"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
)

type resultMember struct {
	id         string
	objective  string
	hypothesis *auditcontract.Hypothesis
}

// resultOwner adapts the two concrete durable owners to the same acceptance
// algorithm. Audit alone owns Tasks, aggregate budgets and its retained links.
type resultOwner struct {
	run          runstore.WorkflowRun
	scope        artifacts.Scope
	namespace    string
	group        string
	audit        *auditstore.Audit
	members      map[string]resultMember
	historyBytes int64
	reads        *querystore.Store
	writer       *ownerstore.TxStore
	events       *hookstore.FindingEventsTx
	artifacts    *artifacts.Service
}

func loadResultOwner(ctx context.Context, db postgres.DBTX, storage *artifactpostgres.Storage, run runstore.WorkflowRun) (*resultOwner, error) {
	o := &resultOwner{run: run, reads: querystore.New(db, storage), artifacts: artifacts.NewService(storage.Repository(db)), members: map[string]resultMember{}}
	if run.AuditCompletion != nil && run.AuditCompletion.Finding != nil {
		if run.PublicationMode != runstore.PublicationOrdinary || run.ProjectID == nil || run.AuditExecutionID != nil {
			return nil, ErrInvalid
		}
		check := run.AuditCompletion.Finding
		if err := findingintake.CheckSource(ctx, db, storage, run.OwnerID, *run.ProjectID, *check); err != nil {
			return nil, err
		}
		o.scope, _ = artifacts.ProjectScope(*run.ProjectID)
		o.namespace = artifactpolicy.FindingMaterialsNamespace
		o.group = run.RunID
		o.members[auditcontract.ExecutionItemKey(0)] = resultMember{id: check.FindingID, objective: check.Assertion}
	} else {
		if run.AuditExecutionID == nil {
			return nil, ErrInvalid
		}
		id, err := o.reads.ExecutionAuditOrigin(ctx, *run.AuditExecutionID, run.RunID)
		if err != nil {
			return nil, err
		}
		audit, err := o.reads.Get(ctx, run.OwnerID, id)
		if err != nil {
			return nil, err
		}
		o.audit = &audit
		o.group = *run.AuditExecutionID
		o.scope, _ = artifacts.AuditScope(id)
		o.namespace = auditdomain.ArtifactNamespace(id)
		tasks, err := o.reads.ListExecutionTasks(ctx, id, o.group)
		if err != nil {
			return nil, err
		}
		for i, t := range tasks {
			o.members[auditcontract.ExecutionItemKey(i)] = resultMember{id: t.Document.TaskID, objective: t.Document.Objective, hypothesis: t.Hypothesis}
		}
	}
	return o, nil
}

// Lock the same owner used by replay. Finding fences precede Run locks to
// share Project deletion's lock order; Audit locking retains its old order.
func (o *resultOwner) lock(ctx context.Context, tx pgx.Tx, storage *artifactpostgres.Storage) error {
	if o.audit == nil {
		o.events = hookstore.NewFindingEventsTx(tx, storage)
		if err := o.events.LockFindingCorrelationFence(ctx, *o.run.ProjectID); err != nil {
			return err
		}
	}
	if _, err := runstore.NewPostgresTxStore(tx, storage).LockRunState(ctx, o.run.RunID, runstore.RunRunning); err != nil {
		return err
	}
	o.writer = ownerstore.NewTx(tx, storage)
	if o.audit != nil {
		audit, err := o.writer.LockTaskInputs(ctx, o.run.OwnerID, o.audit.AuditID, nil)
		if err != nil {
			return err
		}
		o.audit = &audit
	}
	o.artifacts = artifacts.NewService(storage.TxRepository(tx))
	o.reads = querystore.New(tx, storage)
	return nil
}

func (o *resultOwner) material(ref contracts.ArtifactRef, payload artifacts.Payload) auditcontract.Material {
	return auditcontract.Material{ScopeKind: string(o.scope.Kind()), ScopeID: o.scope.ID(), Ref: ref, Digest: auditdomain.DigestBytes(payload.Data), MediaType: payload.MediaType, SizeBytes: int64(len(payload.Data))}
}
func (o *resultOwner) origin(grantID, stage, agent, invocation, submission, key string) auditcontract.EvidenceOrigin {
	member := o.members[key]
	v := auditcontract.EvidenceOrigin{AllocationID: grantID, StageExecutionID: stage, LogicalAgentName: agent, InvocationID: invocation, SubmissionID: submission, RunID: o.run.RunID}
	if o.audit != nil {
		v.AuditID = o.audit.AuditID
		v.ExecutionID = o.group
		v.TaskID = member.id
		if member.hypothesis != nil {
			v.Hypothesis = &auditcontract.HypothesisRef{ID: member.hypothesis.ID, Revision: member.hypothesis.Revision}
		}
	} else {
		v.Finding = o.run.AuditCompletion.Finding
	}
	return v
}
func (o *resultOwner) prefix() string {
	// 132-bit run partition; each decoded receipt additionally checks the full Run.
	return auditdomain.FindingSourceHistoryPrefix(o.run.AuditCompletion.Finding.ReceiptID) + "check-" + strings.TrimPrefix(auditdomain.DigestBytes([]byte(o.run.RunID)), "sha256:")[:33] + "."
}
func (o *resultOwner) historyRef(key string) contracts.ArtifactRef {
	return contracts.ArtifactRef{Namespace: artifactpolicy.FindingHistoryNamespace, Name: o.prefix() + strings.TrimPrefix(auditdomain.DigestBytes([]byte(key)), "sha256:")[:40]}
}

type findingResultLink struct {
	Schema      string                  `json:"schema"`
	ReceiptID   string                  `json:"receiptId"`
	Kind        string                  `json:"kind"`
	RunID       string                  `json:"runId"`
	Link        auditstore.ArtifactLink `json:"link"`
	SourceCount int                     `json:"sourceCount"`
	SourceBytes int64                   `json:"sourceBytes"`
}

func (o *resultOwner) readLink(ctx context.Context, key string) (auditstore.ArtifactLink, bool, error) {
	if o.audit != nil {
		link, err := o.reads.GetArtifactLink(ctx, o.audit.AuditID, key)
		if errors.Is(err, auditstore.ErrNotFound) {
			return link, false, nil
		}
		return link, err == nil, err
	}
	store, _ := o.artifacts.Scoped(o.scope)
	saved, err := store.Read(ctx, o.historyRef(key))
	if errors.Is(err, artifacts.ErrArtifactNotFound) {
		return auditstore.ArtifactLink{}, false, nil
	}
	if err != nil {
		return auditstore.ArtifactLink{}, false, err
	}
	var record findingResultLink
	if strictjson.Decode(saved.Payload.Data, &record) != nil || record.Schema != auditdomain.FindingSourceHistorySchema || record.RunID != o.run.RunID || record.ReceiptID != o.run.AuditCompletion.Finding.ReceiptID || record.Link.LogicalKey != key {
		return auditstore.ArtifactLink{}, false, ErrInvalid
	}
	return record.Link, true, nil
}
func (o *resultOwner) list(ctx context.Context, kind string) ([]auditstore.ArtifactLink, error) {
	if o.audit != nil {
		if kind == "evidence" {
			return o.reads.ListExecutionEvidence(ctx, o.audit.AuditID, o.group)
		}
		return o.reads.ListExecutionJudgments(ctx, o.audit.AuditID, o.group)
	}
	records, err := o.findingRecords(ctx)
	if err != nil {
		return nil, err
	}
	links := []auditstore.ArtifactLink{}
	for _, r := range records {
		if r.Kind == kind {
			links = append(links, r.Link)
		}
	}
	return links, nil
}
func (o *resultOwner) findingRecords(ctx context.Context) ([]findingResultLink, error) {
	store, _ := o.artifacts.Scoped(o.scope)
	ns := artifactpolicy.FindingHistoryNamespace
	query := artifacts.BindingPageQuery{Namespace: &ns, NamePrefix: o.prefix()}
	var metas []artifacts.Metadata
	for len(metas) < 257 {
		query.Limit = min(pagelimit.MaxFetchRows, 257-len(metas))
		page, err := store.ListMetadata(ctx, query)
		if err != nil {
			return nil, err
		}
		metas = append(metas, page...)
		if len(page) < query.Limit {
			break
		}
		query.AfterNamespace = ns
		query.AfterName = page[len(page)-1].Ref.Name
	}
	if len(metas) > 256 {
		return nil, auditstore.ErrEvidenceBudgetExhausted
	}
	var size int64
	for _, meta := range metas {
		size += meta.Size
	}
	if size > auditcontract.MaximumDocumentBytes {
		return nil, auditstore.ErrEvidenceBudgetExhausted
	}
	o.historyBytes = size
	// The existing Artifact batch reader avoids one payload query per receipt.
	rows, err := store.ReadPrefixBatch(ctx, ns, o.prefix(), 256)
	if err != nil {
		return nil, err
	}
	if len(rows) != len(metas) {
		return nil, auditstore.ErrConflict
	}
	result := make([]findingResultLink, 0, len(rows))
	for _, saved := range rows {
		var record findingResultLink
		if strictjson.Decode(saved.Payload.Data, &record) != nil || record.Schema != auditdomain.FindingSourceHistorySchema || record.RunID != o.run.RunID || record.ReceiptID != o.run.AuditCompletion.Finding.ReceiptID || record.SourceCount < 0 || record.SourceCount > 256 || record.SourceBytes < 0 || record.SourceBytes > auditcontract.MaximumDocumentBytes || record.Link.Scope != o.scope.Kind() || (record.Kind != "evidence" && record.Kind != "judgment") || record.Link.Artifact.Validate() != nil {
			return nil, ErrInvalid
		}
		result = append(result, record)
	}
	return result, nil
}
func (o *resultOwner) admit(ctx context.Context, kind string, count int, size int64) error {
	if o.audit != nil {
		if kind == "evidence" {
			return o.writer.AdmitEvidenceCollection(ctx, o.audit.AuditID, o.group, count, size)
		}
		return o.writer.AdmitJudgmentCollection(ctx, o.audit.AuditID, o.group, size)
	}
	records, err := o.findingRecords(ctx)
	if err != nil {
		return err
	}
	rows := count + 1
	bytes := size
	for _, r := range records {
		rows += 1 + r.SourceCount
		bytes += r.Link.Artifact.SizeBytes + r.SourceBytes
	}
	rowMax, byteMax := 256, int64(auditcontract.MaximumDocumentBytes)
	if kind == "evidence" {
		rowMax--
		byteMax -= 1 << 20
	}
	if rows > rowMax || bytes > byteMax {
		return auditstore.ErrEvidenceBudgetExhausted
	}
	return nil
}
func (o *resultOwner) retain(ctx context.Context, kind string, links []auditstore.ArtifactLink, count int, sourceBytes int64) error {
	if o.audit != nil {
		if kind == "evidence" {
			return o.writer.RetainEvidenceLinks(ctx, *o.audit, links)
		}
		return o.writer.RetainJudgmentLinks(ctx, *o.audit, links)
	}
	// Source forks already own their exact bytes in Project. Only the final link
	// is the immutable acknowledgement/history record; it commits with all forks.
	link := links[len(links)-1]
	record := findingResultLink{Schema: auditdomain.FindingSourceHistorySchema, ReceiptID: o.run.AuditCompletion.Finding.ReceiptID, Kind: kind, RunID: o.run.RunID, Link: link, SourceCount: count, SourceBytes: sourceBytes}
	data, err := json.Marshal(record)
	if int64(len(data))+o.historyBytes > auditcontract.MaximumDocumentBytes {
		return auditstore.ErrEvidenceBudgetExhausted
	}
	if err != nil {
		return err
	}
	_, err = o.artifacts.CreateFrozen(ctx, o.scope, o.historyRef(link.LogicalKey), artifacts.Payload{MediaType: "application/json", Data: data})
	if err != nil {
		return err
	}
	return o.events.AdvanceFindingHistory(ctx, *o.run.ProjectID)
}

func (o *resultOwner) ownsEvidence(record auditcontract.EvidenceRecord) bool {
	for _, source := range record.Sources {
		if source.ScopeKind != string(o.scope.Kind()) || source.ScopeID != o.scope.ID() {
			return false
		}
	}
	return true
}
