//go:build integration

package auditcontroller

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"fmt"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/artifactstest"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func generationBinding() auditcontract.ProgramWorkflow {
	inputs := map[string]auditcontract.InputMapping{"context": {Source: "task-context"}, "generation_context": {Source: "prior-results"}, "material": {Source: "audit-input", Name: "source"}}
	for i := 1; i <= 16; i++ {
		inputs[fmt.Sprintf("source_%02d", i)] = auditcontract.InputMapping{Source: "task-input"}
	}
	return auditcontract.ProgramWorkflow{Ref: "audit-hypothesis-generation@1", Inputs: inputs, Parameters: map[string]string{}, Outputs: map[string]string{"proposals": "proposals"}}
}

func TestPostgresContinuationKeepsExactSourcesPauseResumeAndEmptySettlement(t *testing.T) {
	h := newCommonControllerHarness(t, func(p *auditcontract.Program) {
		p.Workflows["generate"] = generationBinding()
		p.Workflows["check"] = auditcontract.ProgramWorkflow{Ref: "audit-contract-check@1", Inputs: map[string]auditcontract.InputMapping{"source": {Source: "audit-input", Name: "source"}, "context": {Source: "task-context"}}, Parameters: map[string]string{}, Outputs: map[string]string{"result": "result"}}
		p.Hypotheses = map[string]auditcontract.HypothesisSeed{"initial": {Statement: "A selected actor can access another record", Scope: map[string]string{"actor": "selected reader"}, Grounds: "Explicit initial suspicion"}}
		p.InitialWork = []auditcontract.InitialWork{{Key: "first", Workflow: "check", Objective: "Check the initial scope", Hypothesis: "initial"}}
		p.Continuation = &auditcontract.Continuation{Workflow: "generate"}
		p.Limits.MaxTasks = 12
		p.Limits.MaxTasksPerIteration = 12
		p.Limits.MaxSubmittedRuns = 12
		p.Limits.MaxHypotheses = 8
		p.Limits.MaxHypothesisRevisions = 8
		p.Limits.MaxHypothesisBytes = 1 << 20
	})
	ctx := t.Context()
	c := h.commonController(t)
	step := func() {
		t.Helper()
		changed, err := c.RunOnce(ctx)
		if err != nil || !changed {
			t.Fatalf("step=%v error=%v", changed, err)
		}
	}
	executions := func() []auditstore.Execution {
		t.Helper()
		rows, err := h.audits.ListExecutions(ctx, h.started.Audit.AuditID)
		if err != nil {
			t.Fatal(err)
		}
		return rows
	}
	step()
	step()
	step() // seed, initial Task, ordinary checking Run
	first := executions()[0]
	finishGenerationTestCheck(t, h, first, "refuted")
	step()
	step()
	step() // observe, retain result, accept continuation instead of completing
	audit, err := h.service.Get(ctx, "owner", h.started.Audit.AuditID)
	if err != nil || audit.State != auditstore.AuditActive {
		t.Fatalf("premature completion: %+v %v", audit, err)
	}
	page, err := h.service.Tasks(ctx, auditstore.ListTasksParams{OwnerID: "owner", AuditID: audit.AuditID, Limit: 20})
	if err != nil || len(page.Tasks) != 2 {
		t.Fatal(page, err)
	}
	decision := page.Tasks[1]
	if decision.State != auditstore.TaskReady || decision.Source.Kind != "continuation" || decision.Document.Inputs["source_01"].Digest != page.Tasks[0].Result.Digest {
		t.Fatal("continuation lost accepted exact context", decision)
	}
	paused, err := h.service.Pause(ctx, auditservice.MutationParams{OwnerID: "owner", AuditID: audit.AuditID, ExpectedRevision: audit.Revision, IdempotencyKey: "pause-before-generation", RequestDigest: contentdigest.Bytes([]byte("pause-before-generation"))})
	if err != nil {
		t.Fatal(err)
	}
	c = h.commonController(t) // restart the coordinator with no process-local decision memory
	if changed, err := c.RunOnce(ctx); err != nil || changed {
		t.Fatalf("paused dispatch=%v %v", changed, err)
	}
	if len(executions()) != 1 {
		t.Fatal("Pause submitted generator")
	}
	_, _, err = h.audits.Resume(ctx, auditstore.ResumeParams{OwnerID: "owner", AuditID: audit.AuditID, ExpectedRevision: paused.Audit.Revision, IdempotencyKey: "resume-generator", RequestDigest: contentdigest.Bytes([]byte("resume-generator"))})
	if err != nil {
		t.Fatal(err)
	}
	step()
	generator := executions()[1]
	run, _ := h.artifacts.Run(*generator.RunID)
	assignment, err := run.Read(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "context"})
	if err != nil {
		t.Fatal(err)
	}
	var taskContext struct {
		Items []struct {
			TaskID    string `json:"task_id"`
			Objective string `json:"objective"`
		}
	}
	if json.Unmarshal(assignment.Payload.Data, &taskContext) != nil || len(taskContext.Items) != 1 || taskContext.Items[0].TaskID != decision.Document.TaskID || taskContext.Items[0].Objective != decision.Document.Objective {
		t.Fatal("generator lost its trusted ordinary task assignment")
	}
	contextRead, err := run.Read(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "generation_context"})
	if err != nil {
		t.Fatal(err)
	}
	var selected auditcontract.ProposalContext
	if json.Unmarshal(contextRead.Payload.Data, &selected) != nil || len(selected.Observations) != 1 || selected.Materials["source_01"].Digest != page.Tasks[0].Result.Digest {
		t.Fatal("generator context changed")
	}
	prior, err := run.Read(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "source_01"})
	if err != nil {
		t.Fatal(err)
	}
	var negative map[string]any
	if json.Unmarshal(prior.Payload.Data, &negative) != nil || negative["items"].([]any)[0].(map[string]any)["verdict"] != "refuted" {
		t.Fatal("refutation was dropped from generation input")
	}
	output := []byte(`{"schema":"contractor.audit.proposals.v1","hypotheses":{"next":{"statement":"The selected alternate actor can access the record","scope":{"actor":"alternate reader"},"grounds":"The first tested actor was refuted; this tests a distinct condition"}},"work":[{"key":"next-check","workflow":"check","objective":"Check the alternate actor because its condition remains untested","hypothesis":"next"}]}`)
	finishGenerationTestOutput(t, h, *generator.RunID, "proposals", output)
	step()
	step() // observe and atomically accept seed/task plus producer receipt
	page, err = h.service.Tasks(ctx, auditstore.ListTasksParams{OwnerID: "owner", AuditID: audit.AuditID, Limit: 20})
	if err != nil || len(page.Tasks) != 3 {
		t.Fatal(page, err)
	}
	generated := page.Tasks[2]
	if generated.Hypothesis == nil || generated.Hypothesis.Origin.RunID != *generator.RunID || generated.Hypothesis.Origin.TaskID != decision.Document.TaskID {
		t.Fatal("untrusted generated origin", generated)
	}
	c = h.commonController(t)
	step() // restarts never re-admit the proposal batch
	if len(executions()) != 3 {
		t.Fatal("producer replay duplicated work")
	}
	finishGenerationTestCheck(t, h, executions()[2], "inconclusive")
	step()
	step()
	step()
	step() // observe, collect, accept next decision, dispatch it
	empty := executions()[3]
	finishGenerationTestOutput(t, h, *empty.RunID, "proposals", []byte(`{"schema":"contractor.audit.proposals.v1","hypotheses":{},"work":[]}`))
	step()
	step()
	step()
	step()
	step() // collect empty decision, release dispatch hold, finalization, completion
	final, err := h.service.Get(ctx, "owner", audit.AuditID)
	if err != nil || final.State != auditstore.AuditCompleted {
		t.Fatalf("empty decision did not settle: state=%s error=%v", final.State, err)
	}
	if len(executions()) != 4 {
		t.Fatal("empty result triggered another generator")
	}
}

func finishGenerationTestCheck(t *testing.T, h *postgresControllerHarness, execution auditstore.Execution, verdict string) {
	t.Helper()
	run, _ := h.artifacts.Run(*execution.RunID)
	value, err := run.Read(t.Context(), contracts.ArtifactRef{Namespace: "inputs", Name: "context"})
	if err != nil {
		t.Fatal(err)
	}
	var trusted struct {
		Items []struct {
			ItemKey  string                   `json:"item_key"`
			Evidence []auditcontract.Material `json:"evidence"`
		}
	}
	if json.Unmarshal(value.Payload.Data, &trusted) != nil || len(trusted.Items) != 1 {
		t.Fatal("missing check context")
	}
	data, _ := json.Marshal(map[string]any{"schema": "contractor.audit.conclusions.v1", "items": []any{map[string]any{"item_key": trusted.Items[0].ItemKey, "verdict": verdict, "rationale": "Only the selected actor was tested", "tested_conditions": []string{"selected actor and exact fixture"}, "limitations": []string{"Alternate actor remains untested"}, "evidence": trusted.Items[0].Evidence}}})
	finishGenerationTestOutput(t, h, *execution.RunID, "result", data)
}

func finishGenerationTestOutput(t *testing.T, h *postgresControllerHarness, runID, slot string, data []byte) {
	t.Helper()
	ctx := context.Background()
	run, _ := h.artifacts.Run(runID)
	written, err := run.Write(ctx, contracts.ArtifactRef{Namespace: "builder", Name: slot}, artifacts.Payload{MediaType: "application/json", Data: data}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = artifactstest.BindOutput(ctx, h.artifacts, runID, slot, written.Ref); err != nil {
		t.Fatal(err)
	}
	if err = artifactstest.FreezeOutputs(ctx, h.artifacts, runID); err != nil {
		t.Fatal(err)
	}
	for _, pair := range [][2]runstore.WorkflowRunState{{runstore.RunPending, runstore.RunRunning}, {runstore.RunRunning, runstore.RunSucceeded}} {
		if _, err = h.runs.TransitionRun(ctx, runID, pair[0], pair[1], runstore.Reason{Code: "fixture"}); err != nil {
			t.Fatal(err)
		}
	}
}

func TestPostgresProducerRejectsWholeInvalidBatchAndCollectsDuringCancellation(t *testing.T) {
	for _, scenario := range []string{"hypothesis-only", "foreign-material", "unsupported-workflow", "oversized-seed", "cancelled", "producer-self-loop", "task-budget"} {
		t.Run(scenario, func(t *testing.T) {
			h := newCommonControllerHarness(t, func(p *auditcontract.Program) {
				p.Workflows["generate"] = generationBinding()
				p.Workflows["check"] = auditcontract.ProgramWorkflow{Ref: "audit-contract-check@1", Inputs: map[string]auditcontract.InputMapping{"source": {Source: "audit-input", Name: "source"}, "context": {Source: "task-context"}}, Parameters: map[string]string{}, Outputs: map[string]string{"result": "result"}}
				p.InitialWork = []auditcontract.InitialWork{{Key: "initial-generator", Workflow: "generate", Objective: "Propose from exact initial materials"}}
				p.Continuation = &auditcontract.Continuation{Workflow: "generate"}
				p.Limits.MaxTasks = 8
				p.Limits.MaxTasksPerIteration = 8
				p.Limits.MaxSubmittedRuns = 8
				p.Limits.MaxHypotheses = 8
				p.Limits.MaxHypothesisRevisions = 8
				p.Limits.MaxHypothesisBytes = 1 << 20
				if scenario == "task-budget" {
					p.Limits.MaxTasks = 1
					p.Limits.MaxTasksPerIteration = 1
					p.Limits.BatchSize = 1
				}
			})
			ctx := t.Context()
			c := h.commonController(t)
			for range 2 {
				if changed, err := c.RunOnce(ctx); err != nil || !changed {
					t.Fatal(changed, err)
				}
			}
			rows, err := h.audits.ListExecutions(ctx, h.started.Audit.AuditID)
			if err != nil || len(rows) != 1 {
				t.Fatal(rows, err)
			}
			producer := rows[0]
			seed := auditcontract.HypothesisSeed{Statement: "A reader can access the selected record", Scope: map[string]string{"actor": "selected reader"}, Grounds: "Selected exact initial material"}
			work := []auditcontract.InitialWork{}
			if scenario != "hypothesis-only" {
				work = append(work, auditcontract.InitialWork{Key: "valid", Workflow: "check", Objective: "Check the selected actor", Hypothesis: "candidate"})
			}
			switch scenario {
			case "foreign-material":
				work = append(work, auditcontract.InitialWork{Key: "bad", Workflow: "check", Objective: "An invalid sibling must reject the batch", Inputs: map[string]string{"source": "foreign-handle"}})
			case "unsupported-workflow":
				work = append(work, auditcontract.InitialWork{Key: "bad", Workflow: "unknown", Objective: "Reject an unknown binding"})
			case "oversized-seed":
				seed.Statement = strings.Repeat("x", 32768)
			case "producer-self-loop":
				work = append(work, auditcontract.InitialWork{Key: "loop", Workflow: "generate", Objective: "Try to invoke the producer from its own response"})
			case "cancelled":
				audit, err := h.service.Get(ctx, "owner", h.started.Audit.AuditID)
				if err != nil {
					t.Fatal(err)
				}
				if _, err = h.service.Cancel(ctx, auditservice.MutationParams{OwnerID: "owner", AuditID: audit.AuditID, ExpectedRevision: audit.Revision, IdempotencyKey: "cancel-producer", RequestDigest: contentdigest.Bytes([]byte("cancel-producer"))}); err != nil {
					t.Fatal(err)
				}
			}
			data, _ := json.Marshal(auditcontract.Proposals{Schema: auditcontract.ProposalsSchema, Hypotheses: map[string]auditcontract.HypothesisSeed{"candidate": seed}, Work: work})
			finishGenerationTestOutput(t, h, *producer.RunID, "proposals", data)
			for range 2 {
				if changed, err := c.RunOnce(ctx); err != nil || !changed {
					t.Fatal(changed, err)
				}
			}
			page, err := h.service.Tasks(ctx, auditstore.ListTasksParams{OwnerID: "owner", AuditID: h.started.Audit.AuditID, Limit: 10})
			if err != nil || len(page.Tasks) != 1 {
				t.Fatal("partial Task admission", page, err)
			}
			hypotheses, err := h.service.Hypotheses(ctx, auditstore.ListHypothesesParams{OwnerID: "owner", AuditID: h.started.Audit.AuditID, Limit: 10})
			if err != nil {
				t.Fatal(err)
			}
			expected := 0
			if scenario == "hypothesis-only" {
				expected = 1
			}
			if len(hypotheses.Items) != expected {
				t.Fatalf("partial assertion admission: count=%d expected=%d", len(hypotheses.Items), expected)
			}
			if scenario != "hypothesis-only" && scenario != "cancelled" && page.Tasks[0].GapCode == nil {
				t.Fatal("invalid batch lacked a truthful collection gap")
			}
			if scenario == "hypothesis-only" {
				// Recover the accepted batch before looking up source bytes again.
				claims, err := h.audits.Claim(ctx, auditstore.ClaimParams{HolderID: "proposal-replay", Lease: time.Minute, Limit: 1})
				if err != nil || len(claims) != 1 {
					t.Fatal(claims, err)
				}
				audit, err := h.service.Get(ctx, "owner", h.started.Audit.AuditID)
				if err != nil {
					t.Fatal(err)
				}
				run, _ := h.artifacts.Run(*producer.RunID)
				metadata, err := run.Metadata(ctx, contracts.ArtifactRef{Namespace: "outputs", Name: "proposals"})
				if err != nil {
					t.Fatal(err)
				}
				source := auditcontract.Material{ScopeKind: "run", ScopeID: *producer.RunID, Ref: metadata.Ref, Digest: metadata.Digest, MediaType: metadata.MediaType, SizeBytes: metadata.Size}
				result, err := h.service.AcceptProposals(ctx, claims[0], auditstore.ReconcileSnapshot{Audit: audit}, producer, source, data)
				if err != nil || !result.Complete || result.Changed {
					t.Fatal("lost acknowledgement replay", result, err)
				}
				if err := h.audits.ReleaseClaim(ctx, claims[0]); err != nil {
					t.Fatal(err)
				}
			}
		})
	}
}

func TestPostgresContinuationConsumesOnlyItsBoundedSelectedSources(t *testing.T) {
	h := newCommonControllerHarness(t, func(p *auditcontract.Program) {
		p.Workflows["generate"] = generationBinding()
		p.Continuation = &auditcontract.Continuation{Workflow: "generate"}
		p.InitialWork = nil
		for i := 1; i <= 20; i++ {
			p.InitialWork = append(p.InitialWork, auditcontract.InitialWork{Key: fmt.Sprintf("source-%02d", i), Workflow: "inspect", Objective: "Retain one selected ordinary observation"})
		}
		p.Limits.MaxTasks = 32
		p.Limits.MaxTasksPerIteration = 32
		p.Limits.MaxSubmittedRuns = 32
		p.Limits.BatchSize = 1
	})
	ctx := t.Context()
	c := h.commonController(t)
	step := func() {
		t.Helper()
		changed, err := c.RunOnce(ctx)
		if err != nil || !changed {
			t.Fatalf("step=%v error=%v", changed, err)
		}
	}
	step() // initial intake may need another byte-bounded slice
	for i := 1; i <= 20; i++ {
		var rows []auditstore.Execution
		for budget := 0; budget < 8; budget++ {
			var err error
			rows, err = h.audits.ListExecutions(ctx, h.started.Audit.AuditID)
			if err != nil {
				t.Fatal(err)
			}
			if len(rows) == i {
				break
			}
			step()
		}
		if len(rows) != i {
			t.Fatal("ordinary source was not dispatched")
		}
		finishCommonChild(t, h, *rows[i-1].RunID, false)
		step()
		step()
	}
	for decision, count := range []int{16, 4} {
		step() // accept a distinct source slice
		page, err := h.service.Tasks(ctx, auditstore.ListTasksParams{OwnerID: "owner", AuditID: h.started.Audit.AuditID, Limit: 100})
		if err != nil {
			t.Fatal(err)
		}
		task := page.Tasks[len(page.Tasks)-1]
		expected := 16
		if decision == 1 {
			expected = 20
		}
		if task.Source.ThroughOrdinal != expected {
			t.Fatalf("decision consumed unselected history: through=%d", task.Source.ThroughOrdinal)
		}
		step()
		rows, err := h.audits.ListExecutions(ctx, h.started.Audit.AuditID)
		if err != nil {
			t.Fatal(err)
		}
		generator := rows[len(rows)-1]
		run, _ := h.artifacts.Run(*generator.RunID)
		value, err := run.Read(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "generation_context"})
		if err != nil {
			t.Fatal(err)
		}
		var selected auditcontract.ProposalContext
		if json.Unmarshal(value.Payload.Data, &selected) != nil || len(selected.Observations) != count {
			t.Fatalf("selected source count=%d expected=%d", len(selected.Observations), count)
		}
		finishGenerationTestOutput(t, h, *generator.RunID, "proposals", []byte(`{"schema":"contractor.audit.proposals.v1","hypotheses":{},"work":[]}`))
		step()
		step()
	}
	for range 3 {
		step()
	}
	final, err := h.service.Get(ctx, "owner", h.started.Audit.AuditID)
	if err != nil || final.State != auditstore.AuditCompleted {
		t.Fatalf("bounded source exhaustion: state=%s error=%v", final.State, err)
	}
}

func TestPostgresContinuationProjectsLargeAcceptedTaskMetadata(t *testing.T) {
	h := newCommonControllerHarness(t, func(p *auditcontract.Program) {
		inspect := generationBinding()
		inspect.Outputs = map[string]string{"material": "proposals"}
		for i := 0; i < 128; i++ {
			inspect.Parameters[fmt.Sprintf("note_%03d", i)] = strings.Repeat("x", 4096)
		}
		p.Workflows["inspect"] = inspect
		p.Workflows["generate"] = generationBinding()
		p.InitialWork = []auditcontract.InitialWork{{Key: "large-source", Workflow: "inspect", Objective: "Retain an ordinary observation with bounded large accepted parameters"}}
		p.Continuation = &auditcontract.Continuation{Workflow: "generate"}
		p.Limits.MaxTasks = 4
		p.Limits.MaxTasksPerIteration = 4
		p.Limits.MaxSubmittedRuns = 4
		p.Limits.BatchSize = 1
	})
	ctx := t.Context()
	c := h.commonController(t)
	step := func() {
		t.Helper()
		changed, err := c.RunOnce(ctx)
		if err != nil || !changed {
			t.Fatalf("step=%v error=%v", changed, err)
		}
	}
	step()
	step()
	rows, err := h.audits.ListExecutions(ctx, h.started.Audit.AuditID)
	if err != nil || len(rows) != 1 {
		t.Fatal(rows, err)
	}
	finishGenerationTestOutput(t, h, *rows[0].RunID, "proposals", []byte(`{"observation":"Selected ordinary material remains useful"}`))
	step()
	step()
	step()
	page, err := h.service.Tasks(ctx, auditstore.ListTasksParams{OwnerID: "owner", AuditID: h.started.Audit.AuditID, Limit: 10})
	if err != nil || len(page.Tasks) != 2 {
		t.Fatalf("large source was dropped: count=%d error=%v", len(page.Tasks), err)
	}
	if page.Tasks[1].Source.ThroughOrdinal != 1 || page.Tasks[1].Document.Inputs["source_01"].Digest != page.Tasks[0].Result.Digest {
		t.Fatal("source projection lost the exact large Task result")
	}
}
