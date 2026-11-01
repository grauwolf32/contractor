//go:build integration

package auditcontroller

import (
	"context"
	"encoding/json"
	"errors"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/artifactstest"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditimport"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/hookstore"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/randomid"
	"github.com/grauwolf32/contractor/internal/planner"
	plannersession "github.com/grauwolf32/contractor/internal/planner/session"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/scanplan"
)

// A failed scan Run is collected from its durable scan journal. Reading the
// retained plan and report keeps its failure class through the production
// controller: a storage failure leaves the execution collecting without a
// receipt, a missing artifact is a coverage gap, and exact retained content
// that can never build a result settles once as collection-contract-invalid.
func TestPostgresScanRecoveryCollectionKeepsFailureClasses(t *testing.T) {
	validReport := &artifacts.Payload{MediaType: auditdomain.JSONMediaType, Data: []byte(`{"tool":"scan_sqlmap","status":"completed"}`)}
	t.Run("transient report read then missing plan gap", func(t *testing.T) {
		scan := newScanCollection(t, nil, validReport)
		transient := errors.New("artifact storage unavailable")
		scan.faults.set(scan.report.Name, transient)
		err := scan.runUntilError(t)
		if !errors.Is(err, transient) || errors.Is(err, auditimport.ErrPermanent) {
			t.Fatalf("collection with lost report read = %v", err)
		}
		scan.requireCollecting(t)
		scan.faults.set(scan.report.Name, nil)
		receipt := scan.runUntilCollected(t)
		if receipt.Disposition != auditstore.CollectionExecutionFailed {
			t.Fatalf("recovered scan receipt = %+v", receipt)
		}
		coverage := scan.coverage(t)
		if !slices.Contains(coverage.Gaps, "scan_plan_unavailable") || len(coverage.Completed) != 1 {
			t.Fatalf("recovered scan coverage = %+v", coverage)
		}
	})
	// Scan history is the accepted exception (AUX-R96): the Planner session
	// validates its journal on every write and read, so a stored plan
	// reference without a revision fails to decode as history before
	// collection reads it. The error stays retryable; the execution stays
	// collecting and every reconcile reports it.
	t.Run("scan history that does not decode stays collecting", func(t *testing.T) {
		scan := newScanCollection(t, nil, validReport)
		tag, err := scan.h.pool.Exec(scan.ctx, `
UPDATE planner_sessions SET state = state #- '{scan,state,plan,revision}'
 WHERE stage_execution_id IN (SELECT stage_execution_id FROM stage_executions WHERE run_id = $1)`,
			*scan.execution.RunID)
		if err != nil || tag.RowsAffected() != 1 {
			t.Fatalf("remove the journal plan revision = (%d, %v)", tag.RowsAffected(), err)
		}
		for range 3 {
			_, err := scan.controller.RunOnce(scan.ctx)
			if err == nil || errors.Is(err, auditimport.ErrPermanent) {
				t.Fatalf("collection with undecodable scan history = %v", err)
			}
			scan.requireCollecting(t)
		}
		if reads := scan.faults.count(scan.plan.Name) + scan.faults.count(scan.report.Name); reads != 0 {
			t.Fatalf("collection read %d retained scan artifacts without a valid history", reads)
		}
	})
	for name, test := range map[string]struct {
		plan, report *artifacts.Payload
		invalid      func(*scanCollection) string
	}{
		"plan that does not decode": {
			plan:    &artifacts.Payload{MediaType: scanplan.PlanMediaType, Data: []byte(`{"schema":"not a scan plan"}`)},
			report:  validReport,
			invalid: func(s *scanCollection) string { return s.plan.Name },
		},
		"report of another scanner": {
			report:  &artifacts.Payload{MediaType: auditdomain.JSONMediaType, Data: []byte(`{"tool":"scan_nuclei"}`)},
			invalid: func(s *scanCollection) string { return s.report.Name },
		},
	} {
		t.Run(name, func(t *testing.T) {
			scan := newScanCollection(t, test.plan, test.report)
			receipt := scan.runUntilCollected(t)
			if receipt.Disposition != auditstore.CollectionContractInvalid {
				t.Fatalf("invalid retained scan receipt = %+v", receipt)
			}
			if scan.coverage(t).Status != auditstore.CoverageBlocked {
				t.Fatalf("contract-invalid coverage = %+v", scan.coverage(t))
			}
			invalid := test.invalid(scan)
			reads := scan.faults.count(invalid)
			// Reconciliation never decodes the settled content again.
			for range 3 {
				if _, err := scan.controller.RunOnce(scan.ctx); err != nil {
					t.Fatal(err)
				}
			}
			if scan.faults.count(invalid) != reads {
				t.Fatalf("settled scan content was decoded again: reads %d -> %d", reads, scan.faults.count(invalid))
			}
			// A retry after a lost acknowledgement sends the same request,
			// and the store replays the one receipt.
			claim := scan.claim(t)
			_, err := scan.collector.Collect(scan.ctx, claim, scan.observed, scan.execution)
			scan.release(t, claim)
			if err != nil {
				t.Fatalf("replayed contract-invalid collection: %v", err)
			}
			if replayed := scan.receipt(t); replayed.ReceiptID != receipt.ReceiptID || replayed.Disposition != receipt.Disposition {
				t.Fatalf("replay changed the receipt: %+v -> %+v", receipt, replayed)
			}
		})
	}
}

type scanCollection struct {
	ctx        context.Context
	h          *postgresControllerHarness
	controller *Controller
	collector  *auditimport.Importer
	faults     *scanReadFaults
	plan       contracts.ArtifactRef
	report     contracts.ArtifactRef
	execution  auditstore.Execution
	observed   auditstore.ReconcileSnapshot
}

// newScanCollection dispatches one prepared OpenAPI scan, journals one
// terminal attempt whose plan and report hold the given payloads (nil is a
// missing artifact) and fails the Run. The execution is observed, collecting.
func newScanCollection(t *testing.T, plan, report *artifacts.Payload) *scanCollection {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), 90*time.Second)
	t.Cleanup(cancel)
	settings := artifacts.Payload{MediaType: "application/json", Data: []byte(`{"schema":"contractor.audit.openapi-scan-settings.v1","scanner":"sqlmap","server":"http://127.0.0.1:8080","testParameters":["q"],"operations":{"#/paths/~1a/get":{"parameters":{"query:q":"one"}}}}`)}
	h := newPostgresControllerHarnessWithPolicy(t, ctx, 0, 1, preparedOpenAPIScanCatalog(t), contracts.ActiveChecksAutomatic, map[string]artifacts.Payload{"settings": settings})
	access, err := auditimport.NewArtifactAccess(h.artifacts, hookstore.NewClaimedArtifacts(h.pool))
	if err != nil {
		t.Fatal(err)
	}
	faults := &scanReadFaults{fail: map[string]error{}, reads: map[string]int{}}
	faulty := scanFaultAccess{ArtifactAccess: access, faults: faults}
	collector, err := auditimport.New(h.audits, h.audits, h.runs, faulty)
	if err != nil {
		t.Fatal(err)
	}
	c := h.controllerWithImportAccess(t, faulty)
	c.options.NewID = randomid.New
	generated := artifacts.Payload{MediaType: "application/json", Data: []byte(`{"openapi":"3.1.0","paths":{"/a":{"get":{"parameters":[{"in":"query","name":"q","schema":{"type":"string"}}],"responses":{"200":{"description":"ok"}}}}}}`)}
	readyGeneratedInventory(t, ctx, h, c, map[string]artifacts.Payload{"checklist": generated})
	c.initialRoundBuilder = h.service
	scan := &scanCollection{ctx: ctx, h: h, controller: c, collector: collector, faults: faults}
	scan.execution = scan.dispatch(t)
	scan.failRun(t, plan, report)
	return scan
}

func (s *scanCollection) dispatch(t *testing.T) auditstore.Execution {
	t.Helper()
	for range 12 {
		executions, err := s.h.audits.ListExecutions(s.ctx, s.h.started.Audit.AuditID)
		if err != nil {
			t.Fatal(err)
		}
		for _, execution := range executions {
			if execution.Role == auditstore.ExecutionCheck && execution.RunID != nil {
				return execution
			}
		}
		if worked, err := s.controller.RunOnce(s.ctx); err != nil || !worked {
			t.Fatalf("scan dispatch = (%t, %v)", worked, err)
		}
	}
	t.Fatal("scan item was not dispatched")
	return auditstore.Execution{}
}

// failRun records the scan journal the scan Planner keeps for one terminal
// attempt, ends the Stage and fails the Run.
func (s *scanCollection) failRun(t *testing.T, plan, report *artifacts.Payload) {
	t.Helper()
	ctx, runs, runID := s.ctx, s.h.runs, *s.execution.RunID
	run, err := runs.GetRun(ctx, runID)
	if err != nil {
		t.Fatal(err)
	}
	if run.State == runstore.RunPending {
		if _, err := runs.TransitionRun(ctx, runID, runstore.RunPending, runstore.RunRunning, runstore.Reason{Code: "test_started"}); err != nil {
			t.Fatalf("start scan Run: %v", err)
		}
	}
	// The scan Planner journals only under the Scheduler's live Run claim.
	claimed, err := runs.ClaimRunnableRun(ctx, "scan-claim", time.Minute, nil, run.SchedulerLaneClass)
	if err != nil || claimed.RunID != runID {
		t.Fatalf("claim scan Run = (%s, %v)", claimed.RunID, err)
	}
	workflow, err := resolved.DecodeResolvedWorkflowSnapshot(run.WorkflowSnapshot)
	if err != nil || workflow.AuditTask == nil {
		t.Fatalf("scan Workflow snapshot: %v", err)
	}
	stage := workflow.Stages[workflow.AuditTask.Stage]
	encoded, err := json.Marshal(stage)
	if err != nil {
		t.Fatal(err)
	}
	stageID := runID + "-scan-1"
	if _, err := runs.CreateStageExecution(ctx, runstore.CreateStageExecutionParams{
		LaneClass: resolved.StageLaneClass(stage), StageExecutionID: stageID, RunID: runID,
		StageName: workflow.AuditTask.Stage, Attempt: 1,
		StageSpecSchemaVersion: contracts.APIVersion, StageSpecSnapshot: encoded,
		StageContextSchemaVersion: contracts.APIVersion,
		StageContext: runstore.StageContextSnapshot{
			Parameters: map[string]string{}, Artifacts: map[string]runstore.PinnedContextArtifact{},
		},
	}); err != nil {
		t.Fatalf("create scan Stage: %v", err)
	}
	runArtifacts, err := s.h.artifacts.Run(runID)
	if err != nil {
		t.Fatal(err)
	}
	retain := func(name string, payload *artifacts.Payload) contracts.ArtifactRef {
		if payload == nil {
			revision := "missing-revision"
			return contracts.ArtifactRef{Namespace: "scan-results", Name: name, Revision: &revision}
		}
		written, err := runArtifacts.Write(ctx, contracts.ArtifactRef{Namespace: "scan-results", Name: name}, *payload, nil)
		if err != nil {
			t.Fatal(err)
		}
		return written.Ref
	}
	s.plan, s.report = retain("plan.fixture", plan), retain("scan.fixture", report)
	planDigest := contentdigest.Bytes([]byte("missing plan"))
	if plan != nil {
		planDigest = contentdigest.Bytes(plan.Data)
	}
	sessions, err := plannersession.New(runs, plannersession.Options{})
	if err != nil {
		t.Fatal(err)
	}
	start, err := sessions.BeginScan(ctx, stageID, "scan-claim")
	if err != nil {
		t.Fatalf("begin scan session: %v", err)
	}
	job := planner.ScanJobRecord{ID: "job-1", Worker: "scanner", Status: planner.ScanJobPending, InputArtifacts: map[string]contracts.ArtifactRef{}}
	if err := sessions.InitializeScan(ctx, start.Identity, planner.ScanState{Plan: &s.plan, PlanDigest: planDigest, Jobs: []planner.ScanJobRecord{job}}); err != nil {
		t.Fatalf("initialize scan journal: %v", err)
	}
	if claimed, err := sessions.ClaimScanJob(ctx, start.Identity, job.ID); err != nil || !claimed {
		t.Fatalf("claim scan job = (%t, %v)", claimed, err)
	}
	job.Status, job.Report = planner.ScanJobCompleted, &s.report
	if err := sessions.FinishScanJob(ctx, start.Identity, job); err != nil {
		t.Fatalf("finish scan job: %v", err)
	}
	if err := runs.EnterAborting(ctx, runstore.EnterAbortingParams{
		StageExecutionID: stageID, ExpectedState: runstore.StageRunning,
		TerminationSchemaVersion: contracts.APIVersion,
		Termination: runstore.StageTermination{
			Outcome: runstore.TerminationInterrupted, Code: "result_publication_failed",
			Message: "The scan result could not be published", Phase: runstore.TerminationRunning,
			OccurredAt: time.Now().UTC(),
		},
		AbortID: stageID + "-abort", Deadline: time.Now().Add(time.Minute),
		Reason: runstore.Reason{Code: "stage_interrupted"},
	}); err != nil {
		t.Fatalf("interrupt scan Stage: %v", err)
	}
	if err := runs.CompleteStageTermination(ctx, stageID); err != nil {
		t.Fatalf("end scan Stage: %v", err)
	}
	if err := runs.ReleaseRunClaim(ctx, runID, "scan-claim"); err != nil {
		t.Fatalf("release scan Run: %v", err)
	}
	if err := artifactstest.FreezeOutputs(ctx, s.h.artifacts, runID); err != nil {
		t.Fatal(err)
	}
	if _, err := runs.TransitionRun(ctx, runID, runstore.RunRunning, runstore.RunFailed, runstore.Reason{Code: "test_failed"}); err != nil {
		t.Fatalf("fail scan Run: %v", err)
	}
	cursor, err := runs.GetRunEventCursor(ctx, runID)
	if err != nil {
		t.Fatal(err)
	}
	claim := s.claim(t)
	if s.execution, err = s.h.audits.ObserveTerminal(ctx, auditstore.ObserveTerminalParams{
		Claim: claim, ExecutionID: s.execution.ExecutionID, RunID: runID,
		Generation: cursor.Generation, Sequence: uint64(cursor.Sequence),
	}); err != nil {
		t.Fatal(err)
	}
	if s.observed, err = s.h.audits.GetReconcileSnapshot(ctx, claim); err != nil {
		t.Fatal(err)
	}
	s.release(t, claim)
	if s.execution.State != auditstore.ExecutionCollecting {
		t.Fatalf("observed scan execution = %+v", s.execution)
	}
}

func (s *scanCollection) claim(t *testing.T) auditstore.ControllerClaim {
	t.Helper()
	claims, err := s.h.audits.Claim(s.ctx, auditstore.ClaimParams{HolderID: "scan-collection-test", Lease: time.Minute, Limit: 1})
	if err != nil || len(claims) != 1 {
		t.Fatalf("claim scan Audit = (%+v, %v)", claims, err)
	}
	return claims[0]
}

func (s *scanCollection) release(t *testing.T, claim auditstore.ControllerClaim) {
	t.Helper()
	if err := s.h.audits.ReleaseClaim(s.ctx, claim); err != nil {
		t.Fatal(err)
	}
}

func (s *scanCollection) runUntilError(t *testing.T) error {
	t.Helper()
	for range 4 {
		if _, err := s.controller.RunOnce(s.ctx); err != nil {
			return err
		}
	}
	t.Fatal("collection did not fail")
	return nil
}

func (s *scanCollection) runUntilCollected(t *testing.T) auditstore.CollectionReceiptSummary {
	t.Helper()
	for range 4 {
		if _, err := s.controller.RunOnce(s.ctx); err != nil {
			t.Fatal(err)
		}
		if receipt, found := s.findReceipt(t); found {
			return receipt
		}
	}
	t.Fatal("scan execution was not collected")
	return auditstore.CollectionReceiptSummary{}
}

func (s *scanCollection) findReceipt(t *testing.T) (auditstore.CollectionReceiptSummary, bool) {
	t.Helper()
	claim := s.claim(t)
	defer s.release(t, claim)
	snapshot, err := s.h.audits.GetReconcileSnapshot(s.ctx, claim)
	if err != nil {
		t.Fatal(err)
	}
	found := []auditstore.CollectionReceiptSummary{}
	for _, receipt := range snapshot.Receipts {
		if receipt.ExecutionID == s.execution.ExecutionID {
			found = append(found, receipt)
		}
	}
	if len(found) > 1 {
		t.Fatalf("execution has %d receipts", len(found))
	}
	if len(found) == 0 {
		return auditstore.CollectionReceiptSummary{}, false
	}
	return found[0], true
}

func (s *scanCollection) receipt(t *testing.T) auditstore.CollectionReceiptSummary {
	t.Helper()
	receipt, found := s.findReceipt(t)
	if !found {
		t.Fatal("scan execution has no receipt")
	}
	return receipt
}

// requireCollecting fails unless the execution still has no receipt and none
// of its members settled, so no item attempt was consumed.
func (s *scanCollection) requireCollecting(t *testing.T) {
	t.Helper()
	if receipt, found := s.findReceipt(t); found {
		t.Fatalf("storage failure produced a receipt: %+v", receipt)
	}
	members, err := s.h.audits.ListExecutionItems(s.ctx, s.execution.ExecutionID)
	if err != nil {
		t.Fatal(err)
	}
	for _, member := range members {
		if member.CollectionDisposition != nil || member.CollectedAt != nil {
			t.Fatalf("storage failure settled member %+v", member)
		}
	}
}

func (s *scanCollection) coverage(t *testing.T) auditstore.Coverage {
	t.Helper()
	if s.execution.RoundID == nil {
		t.Fatal("scan execution has no Round")
	}
	rows, err := s.h.audits.ListCoverage(s.ctx, s.h.started.Audit.AuditID, *s.execution.RoundID, -1, 10)
	if err != nil || len(rows) != 1 {
		t.Fatalf("scan coverage = (%+v, %v)", rows, err)
	}
	return rows[0].Coverage
}

// scanReadFaults injects failures into reads of Run artifacts by name and
// counts every read. The controller reconciles on its own goroutine.
type scanReadFaults struct {
	mu    sync.Mutex
	fail  map[string]error
	reads map[string]int
}

func (f *scanReadFaults) set(name string, err error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.fail[name] = err
}

func (f *scanReadFaults) count(name string) int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.reads[name]
}

type scanFaultAccess struct {
	auditimport.ArtifactAccess
	faults *scanReadFaults
}

func (a scanFaultAccess) WithClaim(claim auditstore.ControllerClaim) auditimport.ArtifactAccess {
	a.ArtifactAccess = a.ArtifactAccess.WithClaim(claim)
	return a
}

func (a scanFaultAccess) ReadRunExact(
	ctx context.Context, runID string, ref contracts.ArtifactRef,
) (auditstore.ExactArtifact, []byte, error) {
	a.faults.mu.Lock()
	a.faults.reads[ref.Name]++
	err := a.faults.fail[ref.Name]
	a.faults.mu.Unlock()
	if err != nil {
		return auditstore.ExactArtifact{}, nil, err
	}
	return a.ArtifactAccess.ReadRunExact(ctx, runID, ref)
}
