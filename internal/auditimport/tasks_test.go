package auditimport

import (
	"context"
	"errors"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// Script only the collection seam. The actual collection mutation, Artifact
// fork, budget and receipt validation run on PG in runservice's Task tests.
type collectionScript struct {
	Store
	Reader
	RunReader
	ArtifactAccess
	collected *auditstore.CollectParams
	receipt   *auditstore.CollectionReceipt
	run       runstore.WorkflowRun
	readErr   error
	runReads  int
}

func (s *collectionScript) WithClaim(auditstore.ControllerClaim) ArtifactAccess { return s }
func (s *collectionScript) CollectionReceipt(context.Context, string, string) (auditstore.CollectionReceipt, error) {
	if s.receipt != nil {
		return *s.receipt, nil
	}
	return auditstore.CollectionReceipt{}, auditstore.ErrNotFound
}
func (s *collectionScript) ListExecutionTasks(context.Context, string, string) ([]auditstore.Task, error) {
	return []auditstore.Task{{Document: auditcontract.Task{TaskID: "task-1"}}}, nil
}
func (s *collectionScript) Collect(_ context.Context, p auditstore.CollectParams) (auditstore.CollectionReceipt, bool, error) {
	s.collected = &p
	s.receipt = &auditstore.CollectionReceipt{RequestDigest: p.RequestDigest}
	return *s.receipt, true, nil
}
func (s *collectionScript) GetRun(context.Context, string) (runstore.WorkflowRun, error) {
	s.runReads++
	return s.run, s.readErr
}
func commonFailedCollection() (*Importer, *collectionScript, auditstore.ReconcileSnapshot, auditstore.Execution) {
	id, project, execution, key := "run-1", "project-1", "execution-1", "submission-1"
	outcome := auditstore.TerminalFailed
	s := &collectionScript{run: runstore.WorkflowRun{RunID: id, OwnerID: "owner", ProjectID: &project, PublicationMode: runstore.PublicationAuditManaged, AuditExecutionID: &execution, AuditSubmissionKey: &key, State: runstore.RunFailed, StateReason: runstore.Reason{Code: runstore.AuditInputInitializationFailedReason}}}
	i := &Importer{store: s, reads: s, runs: s, artifacts: s}
	return i, s, auditstore.ReconcileSnapshot{Audit: auditstore.Audit{AuditID: "audit-1", OwnerID: "owner", ProjectID: project}}, auditstore.Execution{AuditID: "audit-1", ExecutionID: execution, RunID: &id, SubmissionKey: key, Role: auditstore.ExecutionTask, State: auditstore.ExecutionCollecting, TerminalOutcome: &outcome}
}
func TestCommonImporterPermanentInputFailureAndReplayBeforeSourceReads(t *testing.T) {
	for _, state := range []auditstore.AuditState{auditstore.AuditActive, auditstore.AuditPaused, auditstore.AuditCancelling} {
		t.Run(string(state), func(t *testing.T) {
			i, s, snapshot, execution := commonFailedCollection()
			snapshot.Audit.State = state
			claim := auditstore.ControllerClaim{AuditID: execution.AuditID}
			changed, err := i.Collect(t.Context(), claim, snapshot, execution)
			if err != nil || !changed || s.collected == nil || s.collected.ErrorCode == nil || *s.collected.ErrorCode != runstore.AuditInputInitializationFailedReason || len(s.collected.Tasks) != 1 || s.collected.Tasks[0].Retryable || s.runReads != 1 {
				t.Fatalf("permanent input: changed=%t error=%v reads=%d receipt=%+v", changed, err, s.runReads, s.collected)
			}
			s.readErr = errors.New("Run was purged")
			changed, err = i.Collect(t.Context(), claim, snapshot, execution)
			if err != nil || changed || s.runReads != 1 {
				t.Fatalf("receipt replay read source: changed=%t error=%v reads=%d", changed, err, s.runReads)
			}
		})
	}
}
func TestCommonImporterDefiniteSubmissionRejectionNeverRetries(t *testing.T) {
	i, s, snapshot, execution := commonFailedCollection()
	outcome := auditstore.TerminalSubmissionFailed
	execution.TerminalOutcome = &outcome
	execution.RunID = nil
	changed, err := i.Collect(t.Context(), auditstore.ControllerClaim{AuditID: execution.AuditID}, snapshot, execution)
	if err != nil || !changed || s.collected == nil || *s.collected.ErrorCode != "submission-failed" || s.collected.Tasks[0].Retryable || s.runReads != 0 {
		t.Fatalf("rejected submission: changed=%t error=%v receipt=%+v", changed, err, s.collected)
	}
}
func TestCommonImporterReadOutageCreatesNoTechnicalReceipt(t *testing.T) {
	i, s, snapshot, execution := commonFailedCollection()
	s.readErr = errors.New("Run store unavailable")
	changed, err := i.Collect(t.Context(), auditstore.ControllerClaim{AuditID: execution.AuditID}, snapshot, execution)
	if changed || !errors.Is(err, s.readErr) || s.collected != nil {
		t.Fatalf("read outage: changed=%t error=%v receipt=%+v", changed, err, s.collected)
	}
}
func TestCommonImporterInputFailureRequiresExactOwnedChild(t *testing.T) {
	for name, change := range map[string]func(*runstore.WorkflowRun){
		"Run": func(r *runstore.WorkflowRun) { r.RunID = "other" }, "owner": func(r *runstore.WorkflowRun) { r.OwnerID = "other" }, "Project": func(r *runstore.WorkflowRun) { r.ProjectID = nil }, "publication": func(r *runstore.WorkflowRun) { r.PublicationMode = runstore.PublicationOrdinary }, "execution": func(r *runstore.WorkflowRun) { r.AuditExecutionID = nil }, "submission": func(r *runstore.WorkflowRun) { r.AuditSubmissionKey = nil }, "state": func(r *runstore.WorkflowRun) { r.State = runstore.RunPending },
	} {
		t.Run(name, func(t *testing.T) {
			i, s, snapshot, execution := commonFailedCollection()
			change(&s.run)
			changed, err := i.Collect(t.Context(), auditstore.ControllerClaim{AuditID: execution.AuditID}, snapshot, execution)
			if err != nil || !changed || *s.collected.ErrorCode != "collection-contract-invalid" || s.collected.Tasks[0].Retryable {
				t.Fatalf("foreign child: %t,%v,%+v", changed, err, s.collected)
			}
		})
	}
}

type findingScript struct {
	FindingRetention
	candidates []findingintake.CollectionReceipt
	retained   []findingintake.ImportRequest
	rejected   []string
	retainErr  error
	references map[string]findingintake.ReceiptReferences
}

func (s *findingScript) ListAuditCollection(_ context.Context, _, _, _ string, q findingintake.ListQuery) ([]findingintake.CollectionReceipt, error) {
	if q.Limit != findingintake.MaxAuditReceiptBatchSize {
		panic("unbounded finding page")
	}
	return s.candidates, nil
}
func (s *findingScript) RetainAuditCollectionBatch(_ context.Context, p []findingintake.ImportRequest) error {
	s.retained = p
	return s.retainErr
}
func (s *findingScript) RejectAuditCollection(_ context.Context, _ findingintake.ImportRequest, reason string) error {
	s.rejected = append(s.rejected, reason)
	return nil
}
func (s *findingScript) AuditExecutionReceiptReferences(context.Context, string, string, string, string) (map[string]findingintake.ReceiptReferences, error) {
	return s.references, nil
}
func TestCommonImporterFindingRetentionRejectsOnlyMalformedOwnedCandidate(t *testing.T) {
	i, _, snapshot, execution := commonFailedCollection()
	s := &findingScript{}
	for index := range 3 {
		revision := "revision-1"
		r := findingintake.Receipt{Origin: findingintake.Origin{RunID: *execution.RunID, Audit: &findingintake.AuditOrigin{AuditID: execution.AuditID, ExecutionID: execution.ExecutionID, TaskID: "task-1"}}}
		r.Proposal.Ref = contracts.ArtifactRef{Namespace: "worker", Name: []string{"first", "invalid", "last"}[index], Revision: &revision}
		candidate := findingintake.CollectionReceipt{Receipt: r}
		if index == 1 {
			candidate.ReadFailure = "invalid-proposal"
		}
		s.candidates = append(s.candidates, candidate)
	}
	i.findings = s
	more, err := i.retainTaskFindings(t.Context(), snapshot, execution)
	if err != nil || !more || len(s.retained) != 2 || len(s.rejected) != 1 {
		t.Fatalf("finding page: more=%t error=%v retained=%v rejected=%v", more, err, s.retained, s.rejected)
	}
	s.candidates = s.candidates[:1]
	s.candidates[0].Receipt.Origin.Audit.TaskID = ""
	if _, err := i.retainTaskFindings(t.Context(), snapshot, execution); !errors.Is(err, ErrPermanent) {
		t.Fatal("foreign finding accepted", err)
	}
	s.candidates[0].Receipt.Origin.Audit.TaskID = "task-1"
	s.retainErr = findingintake.ErrAuditClosed
	if more, err := i.retainTaskFindings(t.Context(), snapshot, execution); more || err != nil {
		t.Fatalf("closed owner loops: %t,%v", more, err)
	}
}
