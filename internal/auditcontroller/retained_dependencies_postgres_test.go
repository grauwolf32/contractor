//go:build integration

package auditcontroller

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/rundeletion"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func TestPostgresRetainedDependenciesUseAcceptedPrepareAndItemAfterProducerDeletion(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 120*time.Second)
	defer cancel()
	catalog := preparationControllerConfig(t, false, func(files map[string]string) {
		profile := files["audit-profiles/checklist.yaml"]
		profile = strings.Replace(profile, "maxRounds: 1", "maxRounds: 3", 1)
		profile = strings.Replace(profile, "findingConfirmation: disabled", "findingConfirmation: human-required", 1)
		profile = strings.Replace(profile, "        task: {source: item-package}", "        task: {source: item-package}\n        prepared: {source: prepare-output, role: z-seed, name: list}\n        previous_result: {source: retained-dependency, name: source-result}\n        previous_prepared: {source: retained-dependency, name: prepare.z-seed.list}", 1)

		profile = strings.Replace(profile, "  inventory:\n", "    spare: {required: false, mediaTypes: [application/json]}\n  inventory:\n", 1)
		profile = strings.Replace(profile, "        source: {source: audit-input, name: checklist}", "        source: {source: audit-input, name: checklist}\n        spare: {source: audit-input, name: spare}", 1)
		files["workflows/prepare.yaml"] = strings.Replace(files["workflows/prepare.yaml"], "    previous: {required: false, mediaTypes: [application/json]}", "    previous: {required: false, mediaTypes: [application/json]}\n    spare: {required: false, mediaTypes: [application/json]}", 1)
		files["audit-profiles/checklist.yaml"] = profile
		files["workflows/check.yaml"] = strings.Replace(files["workflows/check.yaml"], "    task: {required: true, mediaTypes: [application/zip]}", "    task: {required: true, mediaTypes: [application/zip]}\n    prepared: {required: true, mediaTypes: [application/json]}\n    previous_result: {required: false, mediaTypes: [application/zip]}\n    previous_prepared: {required: false, mediaTypes: [application/json]}", 1)
		files["agent-templates/worker.yaml"] = strings.Replace(files["agent-templates/worker.yaml"], "  sandboxProfile:", "    - ref: security-findings@1\n      tools: [finding]\n  sandboxProfile:", 1)
	})
	h := newPostgresControllerHarnessWithConfig(t, ctx, 0, 3, catalog)
	intake, err := findingintake.New(h.pool)
	if err != nil {
		t.Fatal(err)
	}
	c := h.controllerWithCollector(t, intake)
	c.initialRoundBuilder = h.service
	readyGeneratedInventory(t, ctx, h, c)
	preparationStep(t, ctx, c)
	preparationStep(t, ctx, c)
	var producer auditstore.Execution
	for i := 0; i < 3; i++ {
		preparationStep(t, ctx, c)
		executions, err := h.audits.ListExecutions(ctx, h.started.Audit.AuditID)
		if err != nil {
			t.Fatal(err)
		}
		var check auditstore.Execution
		for _, execution := range executions {
			if execution.Role == auditstore.ExecutionCheck && execution.State == auditstore.ExecutionSubmitted {
				check = execution
			}
		}
		if check.RunID == nil {
			t.Fatalf("check%d not dispatched: %+v", i, executions)
		}
		if i == 0 {
			producer = check
			submitChildFindingProposal(t, ctx, intake, childFindingGrant(t, ctx, h, *check.RunID), "dependent-plan", nil, auditdomain.ProposedCheck{Objective: "Verify from accepted exact context.", Method: "static-trace"})
		}
		finishGeneratedChecklistCheck(t, ctx, h, check)
		preparationStep(t, ctx, c)
		preparationStep(t, ctx, c)
	}
	claims, err := h.audits.Claim(ctx, auditstore.ClaimParams{HolderID: "retained-dependency-test", Lease: 2 * time.Minute, Limit: 1})
	if err != nil || len(claims) != 1 {
		t.Fatalf("claim: %+v %v", claims, err)
	}
	claim := claims[0]
	snapshot, err := h.audits.GetReconcileSnapshot(ctx, claim)
	if err != nil || snapshot.Round == nil {
		t.Fatal(err)
	}
	round := *snapshot.Round
	for _, state := range []auditstore.RoundState{auditstore.RoundAssessing, auditstore.RoundClosed} {
		round, err = h.audits.TransitionRound(ctx, auditstore.RoundTransitionParams{Claim: claim, RoundID: round.RoundID, ExpectedRevision: round.Revision, ExpectedState: round.State, TargetState: state})
		if err != nil {
			t.Fatal(err)
		}
	}
	if err := h.runs.MarkStageAllocationReleased(ctx, *producer.RunID+"-allocation"); err != nil {
		t.Fatal(err)
	}
	prepared, err := h.audits.GetPreparationOutput(ctx, snapshot.Audit.AuditID, "z-seed", "list")
	if err != nil {
		t.Fatal(err)
	}
	// Delete the actual source Runs before the accepted receipt adapter builds
	// later tasks. Neither exact retained source depends on a live producer.
	for _, runID := range []string{prepared.RunID, *producer.RunID} {
		if err := rundeletion.New(h.pool).DeleteReleasedTerminalRun(ctx, snapshot.Audit.OwnerID, runID); err != nil {
			t.Fatalf("delete producer %s: %v", runID, err)
		}
	}
	snapshot, err = h.audits.GetReconcileSnapshot(ctx, claim)
	if err != nil {
		t.Fatal(err)
	}
	params, reason, err := h.service.PrepareNextRound(ctx, claim, snapshot)
	if err != nil || reason != nil || len(params.Items) != 1 {
		t.Fatalf("production receipt adapter: items=%d reason=%+v error=%v", len(params.Items), reason, err)
	}
	deps := params.Items[0].Dependencies
	if deps == nil || len(deps.Dependencies) != 4 {
		t.Fatalf("accepted closure: %+v", deps)
	}

	// A forged accepted-source claim or omitted profile-declared closure must
	// fail on the real transaction even when its standalone schema is valid.
	for _, mutation := range []string{"foreign-owner", "omitted-prepare-closure"} {
		bad := params
		bad.Items = append([]auditstore.MaterializedItem{}, params.Items...)
		encoded, _ := auditdomain.EncodeRetainedDependencies(*params.Items[0].Dependencies)
		value, _ := auditdomain.DecodeRetainedDependencies(encoded)
		if mutation == "foreign-owner" {
			value.OwnerID = "other-owner"
		} else {
			for index := range value.Dependencies {
				if value.Dependencies[index].Kind == "prepare-output" {
					value.Dependencies[index].Requires = []string{}
				}
			}
		}
		if err := auditdomain.ValidateRetainedDependencies(value); err != nil {
			t.Fatalf("mutation must reach real store origin/closure fence: %v", err)
		}
		bad.Items[0].Dependencies = &value
		if _, _, err := h.audits.AcceptNextRound(ctx, bad); err == nil {
			t.Fatalf("real admission accepted %s", mutation)
		}
	}
	next, _, err := h.audits.AcceptNextRound(ctx, params)
	if err != nil {
		t.Fatal(err)
	}
	next, err = h.audits.TransitionRound(ctx, auditstore.RoundTransitionParams{Claim: claim, RoundID: next.RoundID, ExpectedRevision: next.Revision, ExpectedState: next.State, TargetState: auditstore.RoundExecuting})
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err = h.audits.GetReconcileSnapshot(ctx, claim)
	if err != nil {
		t.Fatal(err)
	}
	items, err := h.audits.ListRoundItems(ctx, snapshot.Audit.AuditID, next.RoundID)
	if err != nil || len(items) != 1 {
		t.Fatalf("later item: %+v %v", items, err)
	}
	built, err := h.builder(t).WithClaim(claim).PrepareBatch(ctx, snapshot, []CheckExecutionMember{{Item: items[0], Attempt: 1}})
	if err != nil {
		t.Fatal(err)
	}
	built.Intent.Claim = claim
	execution, _, err := h.audits.CreateExecutionIntent(ctx, built.Intent)
	if err != nil {
		t.Fatal(err)
	}
	built.Run.Claim = claim
	built.Run.NewRunID = func() (string, error) { return "dependent-consumer", nil }
	child, err := h.runService.CreateAudit(ctx, built.Run)
	if err != nil {
		t.Fatal(err)
	}
	if child.Run.State != runstore.RunPending {
		t.Fatalf("consumer initialization did not complete: %s", child.Run.State)
	}
	local, _ := h.artifacts.Run(child.Run.RunID)
	contextRead, err := local.Read(ctx, contracts.ArtifactRef{Namespace: "audit-context", Name: "retained-dependencies"})
	if err != nil {
		t.Fatalf("consumer mapping is absent: %v", err)
	}
	var mappings struct {
		Schema   string `json:"schema"`
		Mappings []struct {
			ItemID                     string `json:"itemId"`
			Name                       string `json:"name"`
			Source, Retained, Consumer auditstore.ExactArtifact
		} `json:"mappings"`
	}
	if json.Unmarshal(contextRead.Payload.Data, &mappings) != nil || len(mappings.Mappings) != 4 {
		t.Fatalf("consumer mapping: %s", contextRead.Payload.Data)
	}
	project, _ := h.artifacts.Project(snapshot.Audit.ProjectID)
	for _, mapping := range mappings.Mappings {
		original, err := project.ReadExact(ctx, mapping.Retained)
		if err != nil {
			t.Fatal(err)
		}
		consumed, err := local.ReadExact(ctx, mapping.Consumer)
		if err != nil || string(original.Payload.Data) != string(consumed.Payload.Data) || mapping.Retained.Digest != mapping.Consumer.Digest || mapping.Source.Ref.SameExact(mapping.Consumer.Ref) {
			t.Fatalf("original/retained/consumer exact mapping %s: %+v %v", mapping.Name, mapping, err)
		}
	}
	replay, err := h.runService.CreateAudit(ctx, built.Run)
	if err != nil || !replay.Replayed || replay.Run.RunID != child.Run.RunID {
		t.Fatalf("lost fork/reply replay: %+v %v", replay, err)
	}
	members, err := h.audits.ListExecutionItems(ctx, execution.ExecutionID)
	if err != nil || len(members) != 1 || len(members[0].Inputs) < 3 {
		t.Fatalf("complete closure was not durable intent authority: %+v %v", members, err)
	}
}
