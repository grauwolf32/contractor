//go:build integration

package auditcontroller

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/rundeletion"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func TestPostgresHypothesisGroundsUseOrdinaryRetentionPublicationAndReplay(t *testing.T) {
	h := newCommonControllerHarness(t, func(p *auditcontract.Program) {
		p.Workflows["explore"] = generationBinding()
		p.InitialWork = []auditcontract.InitialWork{{Key: "explore", Workflow: "explore", Objective: "Select exact grounds during ordinary exploratory work"}}
		p.Limits.MaxHypotheses, p.Limits.MaxHypothesisRevisions = 4, 4
		p.Limits.MaxHypothesisBytes = 1 << 20
	})
	ctx := t.Context()
	c := h.commonController(t)
	step := func() {
		t.Helper()
		if changed, err := c.RunOnce(ctx); err != nil || !changed {
			t.Fatal(changed, err)
		}
	}
	step()
	step()
	executions, err := h.audits.ListExecutions(ctx, h.started.Audit.AuditID)
	if err != nil || len(executions) != 1 || executions[0].RunID == nil {
		t.Fatal(executions, err)
	}
	producer := executions[0]
	run, _ := h.artifacts.Run(*producer.RunID)
	selected, err := run.Write(ctx, contracts.ArtifactRef{Namespace: "generator", Name: "selected-findings"}, artifacts.Payload{MediaType: "application/zip", Data: []byte("exact selected snapshot")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := run.Read(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "analysis"})
	if err != nil {
		t.Fatal(err)
	}
	var contextValue auditcontract.ProposalContext
	if err = json.Unmarshal(raw.Payload.Data, &contextValue); err != nil {
		t.Fatal(err)
	}
	selectedMaterials := map[string]contracts.ArtifactRef{"selection": selected.Ref, "same-selection": selected.Ref}
	// One existing input plus 31 exact own-Run choices reaches the shared
	// ceiling. Intake resolves the finite selection with one metadata read.
	for i := range 30 {
		written, err := run.Write(ctx, contracts.ArtifactRef{Namespace: "generator", Name: fmt.Sprint("selection-", i)}, artifacts.Payload{MediaType: "text/plain", Data: []byte("selected supporting material")}, nil)
		if err != nil {
			t.Fatal(err)
		}
		selectedMaterials[fmt.Sprint("choice-", i)] = written.Ref
	}
	handles := []string{"selection", "same-selection"}
	for key := range contextValue.Materials {
		handles = append(handles, key)
	}
	data, _ := json.Marshal(auditcontract.Proposals{Schema: auditcontract.ProposalsSchema, SelectedMaterials: selectedMaterials, Hypotheses: map[string]auditcontract.ProposedHypothesis{
		"candidate": {Statement: "The selected actor can cross the chosen boundary", Scope: map[string]string{"actor": "reader"}, Grounds: "Exact architecture and selected problems motivate a scoped check", Sources: handles},
		"later":     {Statement: "A second actor can cross the same boundary", Scope: map[string]string{"actor": "alternate"}, Grounds: "This condition remains untested", Sources: []string{"selection"}},
	}, Work: []auditcontract.InitialWork{}})
	finishGenerationTestOutput(t, h, *producer.RunID, "proposals", data)
	step()
	step()
	metadata, err := run.Metadata(ctx, contracts.ArtifactRef{Namespace: "outputs", Name: "proposals"})
	if err != nil {
		t.Fatal(err)
	}
	output := auditcontract.Material{ScopeKind: "run", ScopeID: *producer.RunID, Ref: metadata.Ref, Digest: metadata.Digest, MediaType: metadata.MediaType, SizeBytes: metadata.Size}
	hypotheses, err := h.service.Hypotheses(ctx, auditstore.ListHypothesesParams{OwnerID: "owner", AuditID: h.started.Audit.AuditID, Limit: 10})
	if err != nil || len(hypotheses.Items) != 2 {
		t.Fatal(hypotheses, err)
	}
	var retained auditcontract.Material
	for _, hypothesis := range hypotheses.Items {
		if hypothesis.Document.Origin.RunID != *producer.RunID || len(hypothesis.Document.Sources) == 0 {
			t.Fatal("missing origin or grounds", hypothesis)
		}
		for _, source := range hypothesis.Document.Sources {
			if source.MediaType == "application/zip" {
				retained = source
			}
		}
	}
	audit, err := h.service.Get(ctx, "owner", h.started.Audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	prior := hypotheses.Items[0]
	if _, _, err = h.service.WriteHypothesis(ctx, auditservice.HypothesisMutationParams{MutationParams: auditservice.MutationParams{OwnerID: "owner", AuditID: audit.AuditID, ExpectedRevision: audit.Revision, IdempotencyKey: "refine-grounded", RequestDigest: contentdigest.Bytes([]byte("refine-grounded"))}, HypothesisID: prior.Document.ID, Content: auditservice.HypothesisContent{Key: prior.SourceKey, Statement: prior.Document.Statement + " under explicit conditions", Scope: prior.Document.Scope, Grounds: prior.Document.Grounds}}); err != nil {
		t.Fatal(err)
	}
	edited, err := h.service.Hypothesis(ctx, "owner", audit.AuditID, prior.Document.ID, 2)
	if err != nil || len(edited.Document.Sources) != len(prior.Document.Sources) {
		t.Fatal("content edit lost exact grounds", edited, err)
	}
	for i, m := range edited.Document.Sources {
		if contracts.MaterialKey(m) != contracts.MaterialKey(prior.Document.Sources[i]) {
			t.Fatal("content edit changed exact ground")
		}
	}
	audit, err = h.service.Get(ctx, "owner", audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	if err = rundeletion.New(h.pool, artifactpgtest.NewStorage()).DeleteReleasedTerminalRun(ctx, "owner", *producer.RunID); err != nil {
		t.Fatal(err)
	}
	ownedScope, _ := artifacts.AuditScope(audit.AuditID)
	owned, _ := h.artifacts.Scoped(ownedScope)
	material, err := owned.Read(ctx, retained.Ref)
	if err != nil || string(material.Payload.Data) != "exact selected snapshot" {
		t.Fatal("grounds lost after producer deletion", err)
	}
	publication, err := h.service.PublishMaterial(ctx, "owner", audit.ProjectID, "publish-ground", auditservice.MaterialPublicationRequest{AuditID: audit.AuditID, Source: retained.Ref, Target: contracts.ArtifactRef{Namespace: "analysis", Name: "selected-ground"}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	// Receipt lookup precedes all deleted Run/context reads.
	claims, err := h.audits.Claim(ctx, auditstore.ClaimParams{HolderID: "replay", Lease: time.Minute, Limit: 1})
	if err != nil || len(claims) != 1 {
		t.Fatal(claims, err)
	}
	freshExecutions, err := h.audits.ListExecutions(ctx, audit.AuditID)
	if err != nil || len(freshExecutions) != 1 || freshExecutions[0].RunDeletedAt == nil {
		t.Fatal("producer deletion was not durable", err)
	}
	// Public read projections retain technical attempts and exact selected
	// grounds after the producer's ordinary Run has been deleted.
	taskPage, err := h.service.Tasks(ctx, auditstore.ListTasksParams{OwnerID: "owner", AuditID: audit.AuditID, Limit: 200})
	if err != nil {
		t.Fatal(err)
	}
	matched := false
	for _, task := range taskPage.Tasks {
		if task.LastExecution == nil || *task.LastExecution != producer.ExecutionID {
			continue
		}
		attempts, err := h.service.TaskAttempts(ctx, auditstore.TaskAttemptListParams{OwnerID: "owner", AuditID: audit.AuditID, TaskID: task.Document.TaskID, Limit: 10})
		if err != nil || len(attempts.Items) != 1 || attempts.Items[0].RunID == nil || *attempts.Items[0].RunID != *producer.RunID || !attempts.Items[0].RunDeleted {
			t.Fatalf("retained attempts: %+v %v", attempts, err)
		}
		matched = true
	}
	if !matched {
		t.Fatal("producer Task missing from retained history")
	}
	materialPage, err := h.service.Materials(ctx, auditstore.MaterialListParams{OwnerID: "owner", AuditID: audit.AuditID, Limit: 200})
	if err != nil {
		t.Fatal(err)
	}
	foundGround := false
	for _, entry := range materialPage.Items {
		if !entry.Material.Ref.SameExact(retained.Ref) {
			continue
		}
		read, err := h.service.ReadArtifact(ctx, "owner", audit.AuditID, entry.Key, *entry.Material.Ref.Revision)
		if err != nil || string(read.Payload.Data) != "exact selected snapshot" {
			t.Fatalf("retained public grounds: %v", err)
		}
		foundGround = true
	}
	if !foundGround {
		t.Fatal("selected grounds absent from accepted material page")
	}
	if _, err = h.runs.GetRun(ctx, *producer.RunID); !errors.Is(err, runstore.ErrNotFound) {
		t.Fatalf("producer Run still exists: %v", err)
	}
	replay, err := h.service.AcceptProposals(ctx, claims[0], auditstore.ReconcileSnapshot{Audit: audit}, freshExecutions[0], output, data)
	if err != nil || !replay.Complete || replay.Changed {
		t.Fatal("receipt replay read deleted producer", replay, err)
	}
	if err = h.audits.ReleaseClaim(ctx, claims[0]); err != nil {
		t.Fatal(err)
	}
	step() // close the drained Audit
	current, err := h.service.Get(ctx, "owner", audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = h.service.Delete(ctx, auditservice.MutationParams{OwnerID: "owner", AuditID: audit.AuditID, ExpectedRevision: current.Revision, IdempotencyKey: "delete", RequestDigest: contentdigest.Bytes([]byte("delete"))}); err != nil {
		t.Fatal(err)
	}
	for budget := 0; budget < 30; budget++ {
		changed, err := c.RunOnce(ctx)
		if err != nil {
			t.Fatal(err)
		}
		if !changed {
			break
		}
	}
	if _, err = h.service.Get(ctx, "owner", audit.AuditID); !errors.Is(err, auditstore.ErrNotFound) {
		t.Fatalf("Audit did not finish purge: %v", err)
	}
	project, _ := h.artifacts.Project(audit.ProjectID)
	published, err := project.Read(ctx, publication.Artifact)
	if err != nil || string(published.Payload.Data) != "exact selected snapshot" {
		t.Fatal("explicit publication lost after Audit deletion", err)
	}
}

type invalidContinuationIntake struct {
	TaskIntake
	calls int
}

func (i *invalidContinuationIntake) AcceptContinuation(context.Context, auditstore.ControllerClaim, auditstore.ReconcileSnapshot) (auditstore.TaskIntakeResult, error) {
	i.calls++
	return auditstore.TaskIntakeResult{}, auditservice.ErrInvalid
}

func TestPostgresPermanentContinuationServiceRejectionSettlesOnce(t *testing.T) {
	h := newCommonControllerHarness(t, func(p *auditcontract.Program) {
		p.Continuation = &auditcontract.Continuation{Workflow: "inspect"}
	})
	c := h.commonController(t)
	intake := &invalidContinuationIntake{TaskIntake: h.service}
	c.options.TaskIntake = intake
	for range 2 {
		if _, err := c.RunOnce(t.Context()); err != nil {
			t.Fatal(err)
		}
	}
	executions, err := h.audits.ListExecutions(t.Context(), h.started.Audit.AuditID)
	if err != nil || len(executions) != 1 || executions[0].RunID == nil {
		t.Fatal(executions, err)
	}
	finishCommonChild(t, h, *executions[0].RunID, false)
	for range 5 {
		if _, err := c.RunOnce(t.Context()); err != nil {
			t.Fatal(err)
		}
	}
	current, err := h.service.Get(t.Context(), "owner", h.started.Audit.AuditID)
	if err != nil || current.State != auditstore.AuditCompleted || current.StopReason == nil || current.StopReason.Code != "continuation-invalid" || intake.calls != 1 {
		t.Fatal(current, intake.calls, err)
	}
}

func TestPostgresContinuationOriginBlocksSelfTriggerWithMaterialOutputAlias(t *testing.T) {
	h := newCommonControllerHarness(t, func(p *auditcontract.Program) { p.Continuation = &auditcontract.Continuation{Workflow: "inspect"} })
	c := h.commonController(t)
	ctx := t.Context()
	for n := 0; n < 2; n++ {
		for budget := 0; budget < 6; budget++ {
			if _, err := c.RunOnce(ctx); err != nil {
				t.Fatal(err)
			}
			executions, err := h.audits.ListExecutions(ctx, h.started.Audit.AuditID)
			if err != nil {
				t.Fatal(err)
			}
			if len(executions) == n+1 && executions[n].RunID != nil {
				finishCommonChild(t, h, *executions[n].RunID, false)
				break
			}
		}
	}
	for range 6 {
		if _, err := c.RunOnce(ctx); err != nil {
			t.Fatal(err)
		}
	}
	executions, err := h.audits.ListExecutions(ctx, h.started.Audit.AuditID)
	current, getErr := h.service.Get(ctx, "owner", h.started.Audit.AuditID)
	if err != nil || getErr != nil || len(executions) != 2 || current.State != auditstore.AuditCompleted {
		t.Fatal("continuation consumed its own material output", current, len(executions), err, getErr)
	}
}

func TestPostgresContinuationExposesExhaustedNativeMaterialBudget(t *testing.T) {
	for _, kind := range []string{"count", "document-bytes"} {
		t.Run(kind, func(t *testing.T) {
			ctx := t.Context()
			h := newCommonControllerHarnessWithInputs(t, func(project artifacts.ScopedStore) map[string]auditstore.ExactArtifact {
				members := []contracts.MaterialDescriptor{}
				if kind == "count" {
					for i := range 31 {
						payload := []byte(fmt.Sprint("architecture ", i))
						written, err := project.Write(ctx, contracts.ArtifactRef{Namespace: "architecture", Name: fmt.Sprint("report-", i)}, artifacts.Payload{MediaType: "text/plain", Data: payload}, nil)
						if err != nil {
							t.Fatal(err)
						}
						members = append(members, contracts.MaterialDescriptor{ScopeKind: "project", ScopeID: "common-project", Ref: written.Ref, Digest: contentdigest.Bytes(payload), MediaType: "text/plain", SizeBytes: int64(len(payload))})
					}
				}
				data, _ := json.Marshal(contracts.MaterialSet{Schema: contracts.MaterialSetSchema, Materials: members})
				if kind == "document-bytes" {
					data = append(data, bytes.Repeat([]byte(" "), contracts.MaxMaterialDescriptorBytes-len(data))...)
				}
				written, err := project.Write(ctx, contracts.ArtifactRef{Namespace: "selections", Name: "materials"}, artifacts.Payload{MediaType: "application/json", Data: data}, nil)
				if err != nil {
					t.Fatal(err)
				}
				return map[string]auditstore.ExactArtifact{"materials": contracts.ExactContent(written.Ref, "application/json", data)}
			}, func(p *auditcontract.Program) {
				p.Inputs["materials"] = auditcontract.ProgramInput{Required: false, MediaTypes: []string{"application/json"}}
				binding := generationBinding()
				binding.Inputs["materials"] = auditcontract.InputMapping{Source: "audit-input", Name: "materials"}
				p.Workflows["generate"] = binding
				p.Continuation = &auditcontract.Continuation{Workflow: "generate"}
			})
			c := h.commonController(t)
			for range 2 {
				if _, err := c.RunOnce(ctx); err != nil {
					t.Fatal(err)
				}
			}
			executions, err := h.audits.ListExecutions(ctx, h.started.Audit.AuditID)
			if err != nil || len(executions) != 1 || executions[0].RunID == nil {
				t.Fatal(executions, err)
			}
			finishCommonChild(t, h, *executions[0].RunID, false)
			for range 5 {
				if _, err := c.RunOnce(ctx); err != nil {
					t.Fatal(err)
				}
			}
			current, err := h.service.Get(ctx, "owner", h.started.Audit.AuditID)
			if err != nil || current.State != auditstore.AuditCompleted || current.StopReason == nil || current.StopReason.Code != "material-budget-exhausted" || current.TaskCount != 1 {
				t.Fatal("budget exhaustion was silent or retried", current, err)
			}
		})
	}
}

func TestPostgresContinuationKeepsDistinctHypothesesForSharedBatchResult(t *testing.T) {
	h := newCommonControllerHarness(t, func(p *auditcontract.Program) {
		p.Workflows["generate"] = generationBinding()
		p.Workflows["check"] = auditcontract.ProgramWorkflow{Ref: "audit-contract-check@1", Inputs: map[string]auditcontract.InputMapping{"source": {Source: "audit-input", Name: "source"}, "context": {Source: "task-context"}}, Parameters: map[string]string{}, Outputs: map[string]string{"result": "result"}}
		p.Hypotheses = map[string]auditcontract.HypothesisSeed{"first": {Statement: "Actor A can cross the boundary", Scope: map[string]string{"actor": "A"}, Grounds: "Selected baseline"}, "second": {Statement: "Actor B can cross the boundary", Scope: map[string]string{"actor": "B"}, Grounds: "Selected baseline"}}
		p.InitialWork = []auditcontract.InitialWork{{Key: "first", Workflow: "check", Hypothesis: "first", Objective: "Check actor A"}, {Key: "second", Workflow: "check", Hypothesis: "second", Objective: "Check actor B"}}
		p.Continuation = &auditcontract.Continuation{Workflow: "generate"}
		p.Limits.BatchSize = 2
		p.Limits.MaxHypotheses = 4
		p.Limits.MaxHypothesisRevisions = 4
		p.Limits.MaxHypothesisBytes = 1 << 20
	})
	ctx := t.Context()
	c := h.commonController(t)
	step := func() {
		t.Helper()
		if changed, err := c.RunOnce(ctx); err != nil || !changed {
			t.Fatal(changed, err)
		}
	}
	step()
	step()
	step()
	executions, err := h.audits.ListExecutions(ctx, h.started.Audit.AuditID)
	if err != nil || len(executions) != 1 || executions[0].RunID == nil {
		t.Fatal(executions, err)
	}
	run, _ := h.artifacts.Run(*executions[0].RunID)
	raw, err := run.Read(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "context"})
	if err != nil {
		t.Fatal(err)
	}
	var check struct {
		Items []struct {
			ItemKey    string                   `json:"item_key"`
			TaskID     string                   `json:"task_id"`
			Hypothesis auditcontract.Hypothesis `json:"hypothesis"`
		}
	}
	if json.Unmarshal(raw.Payload.Data, &check) != nil || len(check.Items) != 2 {
		t.Fatal("missing exact batch context")
	}
	items := []any{}
	for _, member := range check.Items {
		items = append(items, map[string]any{"item_key": member.ItemKey, "verdict": "inconclusive", "rationale": "This actor remains unverified", "tested_conditions": []string{}, "limitations": []string{"Explicit gap in selected conditions"}, "evidence": []any{}})
	}
	data, _ := json.Marshal(map[string]any{"schema": "contractor.audit.conclusions.v1", "items": items})
	finishGenerationTestOutput(t, h, *executions[0].RunID, "result", data)
	step()
	step()
	step()
	step()
	executions, err = h.audits.ListExecutions(ctx, h.started.Audit.AuditID)
	if err != nil || len(executions) != 2 || executions[1].RunID == nil {
		t.Fatal("shared result prevented continuation", executions, err)
	}
	generator, _ := h.artifacts.Run(*executions[1].RunID)
	raw, err = generator.Read(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "analysis"})
	if err != nil {
		t.Fatal(err)
	}
	var selected auditcontract.ProposalContext
	if json.Unmarshal(raw.Payload.Data, &selected) != nil || len(selected.Observations) != 2 || len(selected.Materials) != 2 || selected.Observations[0].Key != selected.Observations[1].Key {
		t.Fatal("shared material was duplicated or a member was lost", selected)
	}
	for i, observation := range selected.Observations {
		if observation.Hypothesis == nil || observation.TaskID != check.Items[i].TaskID || observation.Hypothesis.ID != check.Items[i].Hypothesis.ID {
			t.Fatal("exact checked hypothesis rebound", observation)
		}
	}
	finishGenerationTestOutput(t, h, *executions[1].RunID, "proposals", []byte(`{"schema":"contractor.audit.proposals.v1","hypotheses":{},"work":[]}`))
	for range 6 {
		if _, err := c.RunOnce(ctx); err != nil {
			t.Fatal(err)
		}
	}
	current, err := h.service.Get(ctx, "owner", h.started.Audit.AuditID)
	if err != nil || current.State != auditstore.AuditCompleted || current.ReservedRunCount != 2 {
		t.Fatalf("shared result settlement: state=%s Runs=%d error=%v", current.State, current.ReservedRunCount, err)
	}
}

func TestPostgresNativeSetAndSingletonKeepOriginalSourceBudget(t *testing.T) {
	ctx := t.Context()
	var sources []contracts.MaterialDescriptor
	h := newCommonControllerHarnessWithInputs(t, func(project artifacts.ScopedStore) map[string]auditstore.ExactArtifact {
		// Equal bytes are intentionally independent sources. Only the explicit
		// singleton alias of source zero may be deduplicated.
		for i := range 32 {
			raw := []byte("equal independent bytes")
			written, err := project.Write(ctx, contracts.ArtifactRef{Namespace: "architecture", Name: fmt.Sprint("alias-", i)}, artifacts.Payload{MediaType: "text/plain", Data: raw}, nil)
			if err != nil {
				t.Fatal(err)
			}
			sources = append(sources, contracts.MaterialDescriptor{ScopeKind: "project", ScopeID: "common-project", Ref: written.Ref, Digest: contentdigest.Bytes(raw), MediaType: "text/plain", SizeBytes: int64(len(raw))})
		}
		raw, _ := json.Marshal(contracts.MaterialSet{Schema: contracts.MaterialSetSchema, Materials: sources})
		set, err := project.Write(ctx, contracts.ArtifactRef{Namespace: "selections", Name: "ceiling"}, artifacts.Payload{MediaType: "application/json", Data: raw}, nil)
		if err != nil {
			t.Fatal(err)
		}
		return map[string]auditstore.ExactArtifact{"source": sources[0].Exact(), "materials": contracts.ExactContent(set.Ref, "application/json", raw)}
	}, func(p *auditcontract.Program) {
		p.Inputs["materials"] = auditcontract.ProgramInput{Required: true, MediaTypes: []string{"application/json"}}
		binding := generationBinding()
		binding.Inputs["materials"] = auditcontract.InputMapping{Source: "audit-input", Name: "materials"}
		p.Workflows["generate"] = binding
		p.InitialWork = []auditcontract.InitialWork{{Key: "generate-first", Workflow: "generate", Objective: "Research 32 exact independent sources"}}
	})
	c := h.commonController(t)
	for range 3 {
		if _, err := c.RunOnce(ctx); err != nil {
			t.Fatal(err)
		}
	}
	executions, err := h.audits.ListExecutions(ctx, h.started.Audit.AuditID)
	if err != nil || len(executions) != 1 || executions[0].RunID == nil {
		t.Fatalf("32 sources plus their alias did not dispatch: %+v %v", executions, err)
	}
	run, _ := h.artifacts.Run(*executions[0].RunID)
	raw, err := run.Read(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "analysis"})
	if err != nil {
		t.Fatal(err)
	}
	var value auditcontract.ProposalContext
	if err := json.Unmarshal(raw.Payload.Data, &value); err != nil || len(value.Materials) != 32 || len(value.InputSources) != 32 {
		t.Fatalf("source budget changed across Task retention: %+v %v", value, err)
	}
	for _, m := range sources {
		key := contracts.MaterialHandle(m)
		if contracts.MaterialKey(value.InputSources[key]) != contracts.MaterialKey(m) {
			t.Fatalf("original source changed: %+v", value.InputSources[key])
		}
		if _, err := run.ReadExact(ctx, value.Materials[key].Exact()); err != nil {
			t.Fatal(err)
		}
	}
	output, _ := json.Marshal(auditcontract.Proposals{Schema: auditcontract.ProposalsSchema, Hypotheses: map[string]auditcontract.ProposedHypothesis{"grounded": {Statement: "A selected source suggests an access boundary", Scope: map[string]string{}, Grounds: "The exact selected architecture was inspected", Sources: []string{contracts.MaterialHandle(sources[0])}}}, Work: []auditcontract.InitialWork{{Key: "check-origin", Workflow: "check", Objective: "Check the selected original source", Hypothesis: "grounded", Inputs: map[string]string{"source": contracts.MaterialHandle(sources[0])}}}})
	finishGenerationTestOutput(t, h, *executions[0].RunID, "proposals", output)
	for range 2 {
		if _, err := c.RunOnce(ctx); err != nil {
			t.Fatal(err)
		}
	}
	page, err := h.service.Hypotheses(ctx, auditstore.ListHypothesesParams{OwnerID: "owner", AuditID: h.started.Audit.AuditID, Limit: 10})
	if err != nil || len(page.Items) != 1 || len(page.Items[0].Document.Sources) != 1 {
		t.Fatalf("ceiling inputs did not reach proposal admission: %+v %v", page, err)
	}
	tasks, err := h.service.Tasks(ctx, auditstore.ListTasksParams{OwnerID: "owner", AuditID: h.started.Audit.AuditID, Limit: 10})
	if err != nil || len(tasks.Tasks) != 2 {
		t.Fatal(tasks, err)
	}
	expected, err := auditdomain.MaterialArtifactName(sources[0])
	if err != nil {
		t.Fatal(err)
	}
	input := tasks.Tasks[1].Document.Inputs["source"]
	if input.Ref.Name != expected || !input.Ref.SameExact(page.Items[0].Document.Sources[0].Ref) {
		t.Fatalf("proposal acceptance lost the original input identity: %+v", input)
	}

}
