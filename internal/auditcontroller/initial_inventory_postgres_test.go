//go:build integration

package auditcontroller

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditbaseline"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func generatedInventoryHarness(t *testing.T) (context.Context, *postgresControllerHarness, *Controller) {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), 60*time.Second)
	t.Cleanup(cancel)
	catalog := preparationControllerConfig(t, false, func(files map[string]string) {
		profile := files["audit-profiles/checklist.yaml"]
		profile = strings.Replace(profile, "    source: {source: audit-input, name: checklist}", "    source: {source: prepare-output, role: z-seed, name: list}", 1)
		profile = strings.Replace(profile, "        task: {source: item-package}", "        task: {source: item-package}\n        prepared: {source: prepare-output, role: z-seed, name: list}", 1)
		files["audit-profiles/checklist.yaml"] = profile
		files["workflows/check.yaml"] = strings.Replace(files["workflows/check.yaml"], "    task: {required: true, mediaTypes: [application/zip]}", "    task: {required: true, mediaTypes: [application/zip]}\n    prepared: {required: true, mediaTypes: [application/json]}", 1)
	})
	h := newPostgresControllerHarnessWithConfig(t, ctx, 0, 3, catalog)
	return ctx, h, preparationController(t, h)
}

func TestPostgresGeneratedOpenAPIOrderingAndExactSource(t *testing.T) {
	documents := []struct{ media, data string }{
		{"application/json", `{"openapi":"3.1.0","paths":{"/z":{"post":{"responses":{"200":{"description":"ok"}}},"get":{"responses":{"200":{"description":"ok"}}}},"/a":{"get":{"responses":{"200":{"description":"ok"}}}}},"webhooks":{"event":{}}}`},
		{"application/yaml", "webhooks: {event: {}}\npaths:\n  /a:\n    get: {responses: {'200': {description: ok}}}\n  /z:\n    get: {responses: {'200': {description: ok}}}\n    post: {responses: {'200': {description: ok}}}\nopenapi: 3.1.0\n"},
	}
	var canonicalDigest string
	for _, document := range documents {
		t.Run(document.media, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 60*time.Second)
			defer cancel()
			catalog := preparationControllerConfig(t, false, func(files map[string]string) {
				profile := files["audit-profiles/checklist.yaml"]
				profile = strings.Replace(profile, "mode: custom-checklist", "mode: operation-tracing", 1)
				profile = strings.Replace(profile, "implementation: checklist@1", "implementation: openapi-operations@1", 1)
				profile = strings.Replace(profile, "    source: {source: audit-input, name: checklist}", "    source: {source: prepare-output, role: z-seed, name: list}", 1)
				files["audit-profiles/checklist.yaml"] = profile
				files["workflows/prepare.yaml"] = strings.ReplaceAll(files["workflows/prepare.yaml"], "mediaTypes: [application/json]", "mediaTypes: [application/json, application/yaml]")
			})
			h := newPostgresControllerHarnessWithConfig(t, ctx, 0, 1, catalog)
			c := preparationController(t, h)
			readyGeneratedInventory(t, ctx, h, c, map[string]artifacts.Payload{"checklist": {MediaType: document.media, Data: []byte(document.data)}})
			claim, snapshot := initialInventoryClaim(t, ctx, h)
			params, reason, err := h.service.PrepareInitialRound(ctx, claim, snapshot)
			if err != nil || reason != nil || len(params.Items) != 3 {
				t.Fatalf("generated OpenAPI: %+v %+v %v", params, reason, err)
			}
			project, _ := h.artifacts.Project(snapshot.Audit.ProjectID)
			read, err := project.Read(ctx, params.Inventory.Artifact.Ref)
			var derived auditbaseline.DerivedInventory
			if err != nil || json.Unmarshal(read.Payload.Data, &derived) != nil || len(derived.Inventory.Gaps) != 1 || !strings.Contains(derived.Inventory.Gaps[0], "unsupported-webhook") {
				t.Fatalf("unsupported surface lost its gap: %+v %v", derived, err)
			}
			if canonicalDigest == "" {
				canonicalDigest = derived.Inventory.CanonicalInventoryDigest
			} else if derived.Inventory.CanonicalInventoryDigest != canonicalDigest {
				t.Fatalf("YAML/JSON ordering changed inventory identity: %s != %s", derived.Inventory.CanonicalInventoryDigest, canonicalDigest)
			}
			accepted, _ := h.audits.GetPreparationOutput(ctx, snapshot.Audit.AuditID, "z-seed", "list")
			for i, item := range params.Items {
				if item.Ordinal != i || item.Origin.SourceRef == nil || !item.Origin.SourceRef.SameExact(accepted.Output.Retained.Ref) || item.Origin.SourceContentDigest != auditdomain.DigestBytes([]byte(document.data)) || item.Origin.SourceMediaType != document.media {
					t.Fatalf("operation source identity: %+v", item)
				}
			}
			if _, inserted, err := h.audits.AcceptInitialRound(ctx, params); err != nil || !inserted {
				t.Fatalf("accept generated OpenAPI: %t %v", inserted, err)
			}
		})
	}
}

func readyGeneratedInventory(t *testing.T, ctx context.Context, h *postgresControllerHarness, c *Controller, outputs ...map[string]artifacts.Payload) {
	t.Helper()
	preparationStep(t, ctx, c)
	finishPreparationRun(t, ctx, h, preparationExecutions(t, ctx, h, 1)[0], runstore.RunSucceeded, false, outputs...)
	preparationStep(t, ctx, c)
	preparationStep(t, ctx, c)
	preparationStep(t, ctx, c)
}

func initialInventoryClaim(t *testing.T, ctx context.Context, h *postgresControllerHarness) (auditstore.ControllerClaim, auditstore.ReconcileSnapshot) {
	t.Helper()
	claims, err := h.audits.Claim(ctx, auditstore.ClaimParams{HolderID: "initial-inventory-test", Lease: time.Minute, Limit: 1})
	if err != nil || len(claims) != 1 {
		t.Fatalf("claim inventory: %+v %v", claims, err)
	}
	snapshot, err := h.audits.GetReconcileSnapshot(ctx, claims[0])
	if err != nil || snapshot.Audit.Phase != auditdomain.AuditPhaseInventory || snapshot.Round != nil {
		t.Fatalf("inventory boundary: %+v %v", snapshot, err)
	}
	return claims[0], snapshot
}

func TestPostgresInitialInventoryAtomicAcceptanceAndPackageReplay(t *testing.T) {
	ctx, h, c := generatedInventoryHarness(t)
	readyGeneratedInventory(t, ctx, h, c)
	prepared := preparationExecutions(t, ctx, h, 1)[0]
	if err := h.runs.DeleteReleasedTerminalRun(ctx, h.started.Audit.OwnerID, *prepared.RunID); err != nil {
		t.Fatalf("release preparation source before inventory: %v", err)
	}
	claim, snapshot := initialInventoryClaim(t, ctx, h)
	first, reason, err := h.service.PrepareInitialRound(ctx, claim, snapshot)
	if err != nil || reason != nil || len(first.Items) != 3 {
		t.Fatalf("build generated checklist: %+v %+v %v", first, reason, err)
	}
	// Simulate a crash after publication but before acceptance.
	replayed, reason, err := h.service.PrepareInitialRound(ctx, claim, snapshot)
	if err != nil || reason != nil || !replayed.Manifest.Ref.SameExact(first.Manifest.Ref) || !replayed.Inventory.Artifact.Ref.SameExact(first.Inventory.Artifact.Ref) || replayed.Manifest.Digest != first.Manifest.Digest {
		t.Fatalf("package publication replay: %+v %+v %v", replayed, reason, err)
	}
	output, _ := h.audits.GetPreparationOutput(ctx, snapshot.Audit.AuditID, "z-seed", "list")
	for _, item := range first.Items {
		if item.Origin.SourceRef == nil || !item.Origin.SourceRef.SameExact(output.Output.Retained.Ref) || item.Origin.SourceContentDigest != output.Output.Retained.Digest || item.Origin.CanonicalInventoryDigest == "" {
			t.Fatalf("generated item lost exact source: %+v", item)
		}
	}
	var wait sync.WaitGroup
	var inserted [2]bool
	var acceptErrors [2]error
	for i := range inserted {
		wait.Add(1)
		go func(i int) {
			defer wait.Done()
			_, inserted[i], acceptErrors[i] = h.audits.AcceptInitialRound(ctx, first)
		}(i)
	}
	wait.Wait()
	if acceptErrors[0] != nil || acceptErrors[1] != nil || inserted[0] == inserted[1] {
		t.Fatalf("concurrent acceptance: inserted=%v errors=%v", inserted, acceptErrors)
	}
	audit, _ := h.audits.Get(ctx, snapshot.Audit.OwnerID, snapshot.Audit.AuditID)
	if audit.Phase != auditdomain.AuditPhaseRounds || audit.CurrentRoundID == nil || *audit.CurrentRoundID != first.RoundID || string(audit.BaselineSnapshot) != string(snapshot.Audit.BaselineSnapshot) || audit.RetainedEvidenceBytes != snapshot.Audit.RetainedEvidenceBytes+first.Inventory.Artifact.SizeBytes {
		t.Fatalf("acceptance changed original baseline or double charged: %+v", audit)
	}
	var rounds, items, links int
	if err := h.pool.QueryRow(ctx, `SELECT (SELECT count(*) FROM audit_rounds), (SELECT count(*) FROM audit_items), (SELECT count(*) FROM audit_artifact_links WHERE logical_key=$1)`, auditstore.InitialInventoryLogicalKey).Scan(&rounds, &items, &links); err != nil || rounds != 1 || items != 3 || links != 1 {
		t.Fatalf("atomic publication: rounds=%d items=%d links=%d %v", rounds, items, links, err)
	}
	first.Items = first.Items[:2]
	if _, _, err := h.audits.AcceptInitialRound(ctx, first); !errors.Is(err, auditstore.ErrConflict) {
		t.Fatalf("replay accepted changed worklist: %v", err)
	}
}

func TestPostgresInitialInventoryFencesControlsAndStaleClaimsAfterPublication(t *testing.T) {
	for _, action := range []string{"pause", "cancel", "delete", "stale-claim", "deadline"} {
		t.Run(action, func(t *testing.T) {
			ctx, h, c := generatedInventoryHarness(t)
			readyGeneratedInventory(t, ctx, h, c)
			claim, snapshot := initialInventoryClaim(t, ctx, h)
			params, reason, err := h.service.PrepareInitialRound(ctx, claim, snapshot)
			if err != nil || reason != nil {
				t.Fatalf("publication: %+v %v", reason, err)
			}
			want := auditstore.ErrPrecondition
			switch action {
			case "pause":
				_, err = h.service.Pause(ctx, preparationMutation(snapshot.Audit, action))
			case "cancel":
				_, err = h.service.Cancel(ctx, preparationMutation(snapshot.Audit, action))
			case "delete":
				_, err = h.service.Delete(ctx, preparationMutation(snapshot.Audit, action))
			case "stale-claim":
				_, err = h.pool.Exec(ctx, `UPDATE audit_controller_claims SET claimed_at=clock_timestamp()-interval '2 seconds',expires_at=clock_timestamp()-interval '1 second' WHERE audit_id=$1`, claim.AuditID)
				want = auditstore.ErrClaimLost
			case "deadline":
				_, err = h.pool.Exec(ctx, `UPDATE audits SET deadline_at=clock_timestamp()-interval '1 second' WHERE audit_id=$1`, claim.AuditID)
			}
			if err != nil {
				t.Fatal(err)
			}
			if _, _, err := h.audits.AcceptInitialRound(ctx, params); !errors.Is(err, want) {
				t.Fatalf("%s did not fence acceptance: %v", action, err)
			}
			var rounds, items, links int
			if err := h.pool.QueryRow(ctx, `SELECT (SELECT count(*) FROM audit_rounds), (SELECT count(*) FROM audit_items), (SELECT count(*) FROM audit_artifact_links WHERE logical_key=$1)`, auditstore.InitialInventoryLogicalKey).Scan(&rounds, &items, &links); err != nil || rounds != 0 || items != 0 || links != 0 {
				t.Fatalf("fenced publication leaked authority: %d %d %d %v", rounds, items, links, err)
			}
		})
	}
}

func TestPostgresInitialInventoryInvalidAndOverLimitFailBeforeRound(t *testing.T) {
	for _, document := range []string{"malformed", "empty", "over-limit"} {
		t.Run(document, func(t *testing.T) {
			ctx, h, c := generatedInventoryHarness(t)
			var overrides map[string]artifacts.Payload
			if document != "over-limit" {
				data := []byte("{invalid")
				if document == "empty" {
					data = []byte(`{"schema":"contractor.audit.checklist.v1","items":[]}`)
				}
				overrides = map[string]artifacts.Payload{"checklist": {MediaType: "application/json", Data: data}}
			}
			readyGeneratedInventory(t, ctx, h, c, overrides)
			if document == "over-limit" {
				if _, err := h.pool.Exec(ctx, `UPDATE audits SET max_items_per_round=1,max_items_total=1 WHERE audit_id=$1`, h.started.Audit.AuditID); err != nil {
					t.Fatal(err)
				}
			}
			c.initialRoundBuilder = h.service
			for step := 0; step < 5; step++ {
				audit, _ := h.audits.Get(ctx, h.started.Audit.OwnerID, h.started.Audit.AuditID)
				if audit.State == auditstore.AuditFailed {
					wantCode := "invalid_inventory"
					if document == "empty" {
						wantCode = "empty_inventory"
					}
					if audit.StopReason == nil || audit.StopReason.Code != wantCode || audit.CurrentRoundID != nil || audit.ReservedRunCount != 1 || audit.OutstandingRunCount != 0 || audit.Hold != auditstore.HoldReleased {
						t.Fatalf("invalid inventory outcome: %+v", audit)
					}
					if _, err := h.audits.GetPreparationOutput(ctx, audit.AuditID, "z-seed", "diagnostics"); err != nil {
						t.Fatalf("failure lost preparation diagnostics: %v", err)
					}
					return
				}
				preparationStep(t, ctx, c)
			}
			t.Fatal("invalid generated inventory did not fail")
		})
	}
}

func TestPostgresGeneratedChecklistRunsChecksAndPublishesReport(t *testing.T) {
	ctx, h, c := generatedInventoryHarness(t)
	readyGeneratedInventory(t, ctx, h, c)
	c.initialRoundBuilder = h.service
	preparationStep(t, ctx, c)
	preparationStep(t, ctx, c) // initial Round executing
	for index := 0; index < 3; index++ {
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
			t.Fatalf("check %d was not submitted: %+v", index, executions)
		}
		finishGeneratedChecklistCheck(t, ctx, h, check)
		preparationStep(t, ctx, c)
		preparationStep(t, ctx, c)
	}
	for step := 0; step < 10; step++ {
		audit, _ := h.audits.Get(ctx, h.started.Audit.OwnerID, h.started.Audit.AuditID)
		if audit.State == auditstore.AuditCompleted {
			report, err := h.service.GetReport(ctx, audit.OwnerID, audit.AuditID)
			if err != nil || report.Status != auditservice.ReportReady || audit.ReservedRunCount != 4 || audit.OutstandingRunCount != 0 || audit.Hold != auditstore.HoldReleased {
				t.Fatalf("prepared Audit report: %+v %+v %v", audit, report, err)
			}
			baseline, _ := auditservice.DecodeBaseline(audit.BaselineSnapshot)
			if baseline.Inventory != nil {
				t.Fatal("report finalization mutated original baseline")
			}
			link, err := h.audits.GetArtifactLink(ctx, audit.AuditID, auditstore.InitialInventoryLogicalKey)
			project, _ := h.artifacts.Project(audit.ProjectID)
			read, readErr := project.Read(ctx, link.Artifact.Ref)
			var derived auditbaseline.DerivedInventory
			if err != nil || readErr != nil || json.Unmarshal(read.Payload.Data, &derived) != nil || len(derived.Inventory.ExecutionManifest.Items) != 3 {
				t.Fatalf("derived immutable inventory: %+v %v %v", derived, err, readErr)
			}
			return
		}
		preparationStep(t, ctx, c)
	}
	t.Fatal("generated checklist did not complete")
}

func finishGeneratedChecklistCheck(t *testing.T, ctx context.Context, h *postgresControllerHarness, execution auditstore.Execution) {
	t.Helper()
	members, err := h.audits.ListExecutionItems(ctx, execution.ExecutionID)
	if err != nil || len(members) != 1 || len(members[0].Inputs) != 1 {
		t.Fatalf("prepared check membership: %+v %v", members, err)
	}
	accepted, _ := h.audits.GetPreparationOutput(ctx, execution.AuditID, "z-seed", "list")
	if !members[0].Inputs[0].Ref.SameExact(accepted.Output.Retained.Ref) {
		t.Fatal("check consumed another preparation revision")
	}
	items, _ := h.audits.ListItemsByIDs(ctx, execution.AuditID, []string{members[0].ItemID})
	encoded, err := auditdomain.EncodeCheckResultSet(auditdomain.CheckResultSet{Schema: auditdomain.CheckResultsSchema, ExecutionManifestDigest: execution.Manifest.Digest,
		Results: []auditdomain.CheckResult{{ItemKey: items[0].ItemKey, SubjectKey: items[0].SubjectKey, Assessment: "satisfied", Summary: "Prepared checklist item passed.", EvidenceIDs: []string{},
			Coverage: auditdomain.ResultCoverage{Requested: []string{}, Completed: []string{}, Gaps: []string{}}, Proposals: []auditdomain.ProposalSelection{}}}})
	if err != nil {
		t.Fatal(err)
	}
	payload, _, err := auditdomain.BuildPackage("prepared-check-result", auditdomain.PackageKindCheckResults, "", []auditdomain.PackageInput{{ID: auditdomain.CheckResultsMemberID, Path: "check-results.json", MediaType: "application/json", Data: encoded}})
	if err != nil {
		t.Fatal(err)
	}
	run, _ := h.artifacts.Run(*execution.RunID)
	written, err := run.Write(ctx, contracts.ArtifactRef{Namespace: "worker", Name: "result"}, artifacts.Payload{MediaType: auditdomain.PackageMediaType, Data: payload}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := h.artifacts.BindOutputExact(ctx, *execution.RunID, "result", written.Ref, nil); err != nil {
		t.Fatal(err)
	}
	if err := h.artifacts.FreezeRunOutputs(ctx, *execution.RunID); err != nil {
		t.Fatal(err)
	}
	if _, err := h.runs.TransitionRun(ctx, *execution.RunID, runstore.RunPending, runstore.RunRunning, runstore.Reason{Code: "test-admitted"}); err != nil {
		t.Fatal(err)
	}
	if _, err := h.runs.TransitionRun(ctx, *execution.RunID, runstore.RunRunning, runstore.RunSucceeded, runstore.Reason{Code: "test-terminal"}); err != nil {
		t.Fatal(err)
	}
}
