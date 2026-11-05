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
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func generationBinding() auditcontract.ProgramWorkflow {
	inputs := map[string]auditcontract.InputMapping{"context": {Source: "task-context"}, "analysis": {Source: "prior-results"}, "results": {Source: "task-results"}, "material": {Source: "audit-input", Name: "source"}}
	return auditcontract.ProgramWorkflow{Ref: "audit-hypothesis-generation@1", Inputs: inputs, Parameters: map[string]string{}, Outputs: map[string]string{"proposals": "proposals"}}
}

func TestPostgresContinuationKeepsExactSourcesPauseResumeAndEmptySettlement(t *testing.T) {
	var nativeMembers []contracts.MaterialDescriptor
	var unheld contracts.ExactArtifact
	h := newCommonControllerHarnessWithInputs(t, func(project artifacts.ScopedStore) map[string]auditstore.ExactArtifact {
		for i := range 17 {
			raw := []byte(fmt.Sprint("exact architecture ", i))
			written, err := project.Write(t.Context(), contracts.ArtifactRef{Namespace: "architecture", Name: fmt.Sprint("native-", i)}, artifacts.Payload{MediaType: "text/plain", Data: raw}, nil)
			if err != nil {
				t.Fatal(err)
			}
			nativeMembers = append(nativeMembers, contracts.MaterialDescriptor{ScopeKind: "project", ScopeID: "common-project", Ref: written.Ref, Digest: contentdigest.Bytes(raw), MediaType: "text/plain", SizeBytes: int64(len(raw))})
		}
		written, err := project.Write(t.Context(), contracts.ArtifactRef{Namespace: "architecture", Name: "unselected"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("unheld")}, nil)
		if err != nil {
			t.Fatal(err)
		}
		unheld = contracts.ExactContent(written.Ref, "text/plain", []byte("unheld"))
		raw, _ := json.Marshal(contracts.MaterialSet{Schema: contracts.MaterialSetSchema, Materials: nativeMembers})
		set, err := project.Write(t.Context(), contracts.ArtifactRef{Namespace: "selections", Name: "native"}, artifacts.Payload{MediaType: "application/json", Data: raw}, nil)
		if err != nil {
			t.Fatal(err)
		}
		return map[string]auditstore.ExactArtifact{"materials": contracts.ExactContent(set.Ref, "application/json", raw)}
	}, func(p *auditcontract.Program) {
		p.Inputs["materials"] = auditcontract.ProgramInput{Required: true, MediaTypes: []string{"application/json"}}
		binding := generationBinding()
		binding.Inputs["materials"] = auditcontract.InputMapping{Source: "audit-input", Name: "materials"}
		p.Workflows["generate"] = binding
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
	if decision.State != auditstore.TaskReady || decision.Source.Kind != "continuation" || len(decision.Source.ObservationTaskIDs) != 1 || decision.Source.ObservationTaskIDs[0] != page.Tasks[0].Document.TaskID {
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
	// Force real history maintenance while the accepted continuation has no Run.
	history, err := artifactpostgres.NewHistoryPolicy(3, 1<<30)
	if err != nil {
		t.Fatal(err)
	}
	gc := artifacts.NewService(artifactpgtest.NewConfigured(artifactpgtest.NewRuntime(), history).Repository(h.pool))
	project, _ := gc.Project(audit.ProjectID)
	for _, source := range append([]contracts.MaterialDescriptor{nativeMembers[0]}, contracts.MaterialDescriptor{ScopeKind: "project", ScopeID: audit.ProjectID, Ref: unheld.Ref, Digest: unheld.Digest, MediaType: unheld.MediaType, SizeBytes: unheld.SizeBytes}) {
		current := source.Ref
		for i := range 8 {
			next, err := project.Write(ctx, contracts.ArtifactRef{Namespace: current.Namespace, Name: current.Name}, artifacts.Payload{MediaType: "text/plain", Data: []byte(fmt.Sprint("later ", i))}, current.Revision)
			if err != nil {
				t.Fatal(err)
			}
			current = next.Ref
		}
	}
	if _, err := project.Read(ctx, unheld.Ref); !artifacts.IsExactReferenceRejection(err) {
		t.Fatalf("history did not evict the unheld control: %v", err)
	}
	if _, err := project.Read(ctx, nativeMembers[0].Ref); err != nil {
		t.Fatalf("accepted set member was lost while paused: %v", err)
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
			TaskID    string                   `json:"task_id"`
			Objective string                   `json:"objective"`
			Evidence  []auditcontract.Material `json:"evidence"`
		}
	}
	if json.Unmarshal(assignment.Payload.Data, &taskContext) != nil || len(taskContext.Items) != 1 || taskContext.Items[0].TaskID != decision.Document.TaskID || taskContext.Items[0].Objective != decision.Document.Objective {
		t.Fatal("generator lost its trusted ordinary task assignment")
	}
	for _, material := range taskContext.Items[0].Evidence {
		actual, err := run.ReadExact(ctx, material.Exact())
		if err != nil || material.Exact().Verify(actual.Exact()) != nil {
			t.Fatalf("context is not exact: %+v actual=%+v err=%v", material, actual, err)
		}
	}
	localSet, err := run.Read(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "materials"})
	if err != nil {
		t.Fatal(err)
	}
	preparedSet, err := contracts.DecodeMaterialSet(localSet.Payload.Data)
	if err != nil || len(preparedSet.Materials) != 17 {
		t.Fatal(preparedSet, err)
	}
	for _, m := range preparedSet.Materials {
		if _, err := run.ReadExact(ctx, m.Exact()); err != nil {
			t.Fatalf("Resume lost a selected material: %v", err)
		}
	}
	contextRead, err := run.Read(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "analysis"})
	if err != nil {
		t.Fatal(err)
	}
	var selected auditcontract.ProposalContext
	if json.Unmarshal(contextRead.Payload.Data, &selected) != nil || len(selected.Observations) != 1 || selected.Materials[contracts.MaterialHandle(*page.Tasks[0].Result)].Digest != page.Tasks[0].Result.Digest {
		t.Fatal("generator context changed")
	}
	prior, err := run.Read(ctx, selected.Materials[contracts.MaterialHandle(*page.Tasks[0].Result)].Ref)
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

func finishGenerationTestOutput(t *testing.T, h *postgresControllerHarness, runID, slot string, data []byte, outcome ...runstore.WorkflowRunState) {
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
	terminal := runstore.RunSucceeded
	if len(outcome) > 0 {
		terminal = outcome[0]
	}
	for _, pair := range [][2]runstore.WorkflowRunState{{runstore.RunPending, runstore.RunRunning}, {runstore.RunRunning, terminal}} {
		if _, err = h.runs.TransitionRun(ctx, runID, pair[0], pair[1], runstore.Reason{Code: "fixture"}); err != nil {
			t.Fatal(err)
		}
	}
}

func TestPostgresProducerRejectsWholeInvalidBatchAndCollectsDuringCancellation(t *testing.T) {
	for _, scenario := range []string{"hypothesis-only", "foreign-material", "unsupported-workflow", "oversized-seed", "cancelled", "producer-self-loop", "task-budget", "selected-hidden", "selected-system", "selected-foreign", "selected-over-bound", "failed-run"} {
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
			selected := map[string]contracts.ArtifactRef{}
			run, _ := h.artifacts.Run(*producer.RunID)
			switch scenario {
			case "selected-hidden":
				selected["forbidden"] = contracts.ArtifactRef{Namespace: "skills", Name: "private", Revision: func() *string { s := "r1"; return &s }()}
			case "selected-system":
				m, err := run.Metadata(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "analysis"})
				if err != nil {
					t.Fatal(err)
				}
				selected["forbidden"] = m.Ref
			case "selected-foreign":
				project, _ := h.artifacts.Project(h.started.Audit.ProjectID)
				m, err := project.Write(ctx, contracts.ArtifactRef{Namespace: "foreign", Name: "selected"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("Project content is not an own-Run selection")}, nil)
				if err != nil {
					t.Fatal(err)
				}
				selected["forbidden"] = m.Ref
			case "selected-over-bound":
				for i := range 32 {
					m, err := run.Write(ctx, contracts.ArtifactRef{Namespace: "generator", Name: fmt.Sprint("selection-", i)}, artifacts.Payload{MediaType: "text/plain", Data: []byte("selected content")}, nil)
					if err != nil {
						t.Fatal(err)
					}
					selected[fmt.Sprint("choice-", i)] = m.Ref
				}
			}
			data, _ := json.Marshal(auditcontract.Proposals{Schema: auditcontract.ProposalsSchema, SelectedMaterials: selected, Hypotheses: map[string]auditcontract.ProposedHypothesis{"candidate": {Statement: seed.Statement, Scope: seed.Scope, Grounds: seed.Grounds}}, Work: work})
			if scenario == "failed-run" {
				finishGenerationTestOutput(t, h, *producer.RunID, "proposals", data, runstore.RunFailed)
			} else {
				finishGenerationTestOutput(t, h, *producer.RunID, "proposals", data)
			}
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
		for i := 1; i <= 33; i++ {
			p.InitialWork = append(p.InitialWork, auditcontract.InitialWork{Key: fmt.Sprintf("source-%02d", i), Workflow: "inspect", Objective: "Retain one selected ordinary observation"})
		}
		p.Limits.MaxTasks = 64
		p.Limits.MaxTasksPerIteration = 64
		p.Limits.MaxSubmittedRuns = 64
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
	for i := 1; i <= 33; i++ {
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
	for decision, count := range []int{31, 2} {
		step() // accept a distinct source slice
		page, err := h.service.Tasks(ctx, auditstore.ListTasksParams{OwnerID: "owner", AuditID: h.started.Audit.AuditID, Limit: 100})
		if err != nil {
			t.Fatal(err)
		}
		task := page.Tasks[len(page.Tasks)-1]
		expected := 31
		if decision == 1 {
			expected = 33
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
		if generator.RunID == nil {
			t.Fatalf("generator rejected: %+v", generator)
		}
		run, _ := h.artifacts.Run(*generator.RunID)
		value, err := run.Read(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "analysis"})
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
		inspect.Ref = "large-observation@1"
		delete(inspect.Inputs, "analysis")
		delete(inspect.Inputs, "results")
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
	if page.Tasks[1].Source.ThroughOrdinal != 1 || len(page.Tasks[1].Source.ObservationTaskIDs) != 1 || page.Tasks[1].Source.ObservationTaskIDs[0] != page.Tasks[0].Document.TaskID {
		t.Fatal("source projection lost the exact large Task result")
	}
}
