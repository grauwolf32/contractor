//go:build integration

package auditcontroller

import (
	"bytes"
	"context"
	"errors"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/hookstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/scan"
)

func TestPostgresPreparedOpenAPIScansKeepAssignedDenominatorAndApproval(t *testing.T) {
	for _, policy := range []contracts.ActiveChecksPolicy{contracts.ActiveChecksAutomatic, contracts.ActiveChecksApprovalRequired} {
		t.Run(string(policy), func(t *testing.T) { testPreparedOpenAPIScanPolicy(t, policy) })
	}
}
func testPreparedOpenAPIScanPolicy(t *testing.T, policy contracts.ActiveChecksPolicy) {
	t.Helper()

	ctx, cancel := context.WithTimeout(t.Context(), 60*time.Second)
	defer cancel()
	catalog := preparedOpenAPIScanCatalog(t)
	settings := artifacts.Payload{MediaType: "application/json", Data: []byte(`{"schema":"contractor.audit.openapi-scan-settings.v1","scanner":"sqlmap","server":"http://127.0.0.1:8080","testParameters":["q"],"operations":{"#/paths/~1a/get":{"parameters":{"query:q":"one"}},"#/paths/~1b/get":{"parameters":{"query:q":"two"}},"#/paths/~1c/get":{"parameters":{"query:q":"three"}}}}`)}
	h := newPostgresControllerHarnessWithPolicy(t, ctx, 0, 1, catalog, policy, map[string]artifacts.Payload{"settings": settings})
	c := preparationController(t, h)
	generated := artifacts.Payload{MediaType: "application/json", Data: []byte(`{"openapi":"3.1.0","paths":{"/a":{"get":{"parameters":[{"in":"query","name":"q","schema":{"type":"string"}}],"responses":{"200":{"description":"ok"}}}},"/b":{"get":{"parameters":[{"in":"query","name":"q","schema":{"type":"string"}}],"responses":{"200":{"description":"ok"}}}},"/c":{"get":{"parameters":[{"in":"query","name":"q","schema":{"type":"string"}}],"responses":{"200":{"description":"ok"}}}},"/unselected":{"get":{"responses":{"200":{"description":"ok"}}}}}}`)}
	readyGeneratedInventory(t, ctx, h, c, map[string]artifacts.Payload{"checklist": generated})
	c.initialRoundBuilder = h.service
	preparationStep(t, ctx, c)
	items, err := h.audits.Items(ctx, h.started.Audit.OwnerID, h.started.Audit.AuditID)
	if err != nil || len(items) != 3 {
		t.Fatalf("scan denominator expanded beyond explicit settings: %+v %v", items, err)
	}
	project, _ := h.artifacts.Project(h.started.Audit.ProjectID)
	wantState, wantKind, wantReviews := auditstore.ItemReady, auditstore.ItemApprovalNone, 0
	if policy == contracts.ActiveChecksApprovalRequired {
		wantState, wantKind, wantReviews = auditstore.ItemAwaitingReview, auditstore.ItemApprovalActiveCheck, 3
	}
	for _, item := range items {
		if item.Kind != "openapi-scan" || item.State != wantState || item.ApprovalKind != wantKind {
			t.Fatalf("generated scan bypassed exact approval: %+v", item)
		}
		read, err := project.Read(ctx, item.Task.Ref)
		if err != nil {
			t.Fatal(err)
		}
		pkg, err := auditdomain.ValidatePackage(read.Payload.Data)
		if err != nil {
			t.Fatal(err)
		}
		member, _ := pkg.MemberByID("task-document")
		task, err := auditdomain.DecodeItemTask(member.Data())
		if err != nil || task.Scan == nil || !task.Scan.Runnable || task.Scan.CoverageRequirement() != "sqlmap-request-scan" || strings.Contains(task.Scan.Operation, "unselected") {
			t.Fatalf("generated scan lost assigned task semantics: %+v %v", task, err)
		}
	}
	audit, err := h.audits.Get(ctx, h.started.Audit.OwnerID, h.started.Audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	page, err := h.service.ListCoverage(ctx, audit.OwnerID, audit.AuditID, *audit.CurrentRoundID, -1, 200, &audit.Revision)
	if err != nil || len(page.Rows) != 3 {
		t.Fatalf("scan coverage projection: %+v %v", page, err)
	}
	for _, row := range page.Rows {
		if row.Details == nil || row.Details.Scan == nil || row.Details.Scan.Scanner != "sqlmap" || row.Details.Scan.RequestDigest == "" || row.Details.Scan.TargetURL != "" || row.Details.Scan.Source.Digest != auditdomain.DigestBytes(generated.Data) {
			t.Fatalf("scan lost exact prepared provenance: %+v", row.Details)
		}
	}
	reviews, err := h.service.ListReviews(ctx, auditstore.ReviewListParams{OwnerID: h.started.Audit.OwnerID, AuditID: h.started.Audit.AuditID, Limit: 10})
	if err != nil || len(reviews) != wantReviews {
		t.Fatalf("generated scan review ledger: %+v %v", reviews, err)
	}
	preparationExecutions(t, ctx, h, 1)
}

// preparedOpenAPIScanCatalog is the preparation catalog whose check role runs
// the SQLMap OpenAPI scan Workflow on the prepared document and the Audit's
// scan settings input.
func preparedOpenAPIScanCatalog(t *testing.T) *config.Snapshot {
	t.Helper()
	return preparationControllerConfig(t, false, func(files map[string]string) {
		for _, path := range []string{"workflows/audit_openapi_sqlmap_scan.yaml", "agent-templates/audit_sqlmap_scan.yaml", "instructions/audit-openapi-scan.md"} {
			data, err := os.ReadFile("../config/testdata/audit-scan-catalog/" + path)
			if err != nil {
				t.Fatal(err)
			}
			files[path] = string(data)
		}
		profile := files["audit-profiles/checklist.yaml"]
		profile = strings.Replace(profile, "mode: custom-checklist", "mode: risk-assessment", 1)
		profile = strings.Replace(profile, "    checklist: {required: true, mediaTypes: [application/json]}", "    checklist: {required: true, mediaTypes: [application/json]}\n    settings: {required: true, mediaTypes: [application/json]}", 1)
		profile = strings.Replace(profile, "implementation: checklist@1", "implementation: openapi-scans@1", 1)
		profile = strings.Replace(profile, "    source: {source: audit-input, name: checklist}", "    source: {source: prepare-output, role: z-seed, name: list}\n    settings: {source: audit-input, name: settings}", 1)
		profile = strings.Replace(profile, "      ref: audit-check@1", "      ref: audit-openapi-sqlmap-scan@1", 1)
		profile = strings.Replace(profile, "        task: {source: item-package}", `        task: {source: item-package}
        execution_manifest: {source: execution-manifest}
        openapi: {source: prepare-output, role: z-seed, name: list}
        settings: {source: audit-input, name: settings}`, 1)
		files["audit-profiles/checklist.yaml"] = strings.Replace(profile, "activeChecks: prohibited", "activeChecks: approval-required", 1)
	})
}

func TestPostgresPreparedRulepackRefusedBeforeScanDispatch(t *testing.T) {
	for _, test := range []struct {
		name    string
		size    int
		refused bool
	}{
		{"boundary", scan.MaxNucleiRulepackFileBytes, false},
		{"oversized", scan.MaxNucleiRulepackFileBytes + 1, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 60*time.Second)
			defer cancel()
			catalog := preparationControllerConfig(t, false, func(files map[string]string) {
				for _, path := range []string{"workflows/audit_web_nuclei_scan.yaml", "agent-templates/audit_web_nuclei_scan.yaml", "instructions/audit-openapi-scan.md"} {
					data, err := os.ReadFile("../auditservice/testdata/catalog/" + path)
					if err != nil {
						t.Fatal(err)
					}
					files[path] = string(data)
				}
				files["workflows/prepare.yaml"] = strings.ReplaceAll(files["workflows/prepare.yaml"], "checklist: {required: true, mediaTypes: [application/json]", "checklist: {required: true, mediaTypes: [application/yaml]")
				profile := files["audit-profiles/checklist.yaml"]
				profile = strings.Replace(profile, "mode: custom-checklist", "mode: risk-assessment", 1)
				profile = strings.Replace(profile, "maxEvidenceBytes: 1048576", "maxEvidenceBytes: 67108864", 1)
				profile = strings.Replace(profile, "    checklist: {required: true, mediaTypes: [application/json]}", "    checklist: {required: true, mediaTypes: [application/json]}\n    target: {required: true, mediaTypes: [application/json]}\n    settings: {required: true, mediaTypes: [application/json]}", 1)
				profile = strings.Replace(profile, "implementation: checklist@1", "implementation: web-scans@1", 1)
				profile = strings.Replace(profile, "    source: {source: audit-input, name: checklist}", "    source: {source: audit-input, name: target}\n    settings: {source: audit-input, name: settings}", 1)
				profile = strings.Replace(profile, "      ref: audit-check@1", "      ref: audit-web-nuclei-scan@1", 1)
				profile = strings.Replace(profile, "        task: {source: item-package}", `        task: {source: item-package}
        execution_manifest: {source: execution-manifest}
        target: {source: audit-input, name: target}
        settings: {source: audit-input, name: settings}
        rulepack: {source: prepare-output, role: z-seed, name: list}`, 1)
				files["audit-profiles/checklist.yaml"] = strings.Replace(profile, "activeChecks: prohibited", "activeChecks: automatic", 1)
			})
			h := newPostgresControllerHarnessWithPolicy(t, ctx, 0, 1, catalog, contracts.ActiveChecksAutomatic, map[string]artifacts.Payload{
				"target":   {MediaType: "application/json", Data: []byte(`{"schema":"contractor.audit.web-target.v1","url":"https://target.test/"}`)},
				"settings": {MediaType: "application/json", Data: []byte(`{"schema":"contractor.audit.web-scan-settings.v1","templateIds":"","tags":"","severity":"","rateLimit":10}`)},
			})
			c := preparationController(t, h)
			readyGeneratedInventory(t, ctx, h, c, map[string]artifacts.Payload{"checklist": {MediaType: "application/yaml", Data: bytes.Repeat([]byte("x"), test.size)}})
			output, err := h.audits.GetPreparationOutput(ctx, h.started.Audit.AuditID, "z-seed", "list")
			if err != nil || output.Output.Retained.SizeBytes != int64(test.size) {
				a, _ := h.audits.Get(ctx, h.started.Audit.OwnerID, h.started.Audit.AuditID)
				e, _ := h.audits.ListPreparationExecutions(ctx, h.started.Audit.AuditID)
				t.Fatalf("accepted prepared output: %+v %v audit=%+v executions=%+v", output, err, a.StopReason, e)
			}
			// The same exact material can be a retained dependency of an
			// ordinary role. Source lookup changes, metadata authority does not.
			access, err := NewProjectArtifactAccess(h.artifacts, hookstore.NewClaimedArtifacts(h.pool))
			if err != nil {
				t.Fatal(err)
			}
			builder, err := NewPinnedSubmissionBuilder(access, &fakeRoleOutputLookup{link: auditstore.ArtifactLink{Scope: "project", Artifact: output.Output.Retained}})
			if err != nil {
				t.Fatal(err)
			}
			profile, err := catalog.AuditProfile("test-checklist@1")
			if err != nil {
				t.Fatal(err)
			}
			binding := profile.Workflows["check"]
			binding.Inputs = map[string]auditprofile.AuditWorkflowInputMapping{"rulepack": {Source: auditprofile.AuditInputFromRetainedOutput, Role: "discover", Name: "rules"}}
			inputs, err := builder.resolveRoleInputs(ctx, auditstore.ReconcileSnapshot{Audit: h.started.Audit, Round: &auditstore.Round{Ordinal: 1}}, "assess", binding, auditservice.BaselineSnapshot{}, auditstore.ExactArtifact{})
			if err != nil {
				t.Fatal(err)
			}
			err = builder.validateRulepackInputs(ctx, h.started.Audit, binding, inputs)
			if test.refused && !errors.Is(err, ErrInvalidSubmission) || !test.refused && err != nil {
				t.Fatalf("retained role admission: %v", err)
			}
			c.initialRoundBuilder = h.service
			preparationStep(t, ctx, c) // accept initial inventory
			audit := h.started.Audit
			count := 1
			// The controller advances one durable Round transition per tick.
			// Bound the attempts by state transitions, not elapsed time.
			for step := 0; step < 8 && count == 1 && audit.StopReason == nil; step++ {
				preparationStep(t, ctx, c)
				audit, err = h.audits.Get(ctx, h.started.Audit.OwnerID, h.started.Audit.AuditID)
				if err != nil {
					t.Fatal(err)
				}
				if err := h.pool.QueryRow(ctx, `SELECT count(*) FROM workflow_runs WHERE project_id=$1`, audit.ProjectID).Scan(&count); err != nil {
					t.Fatal(err)
				}
			}
			if test.refused {
				if count != 1 || audit.StopReason == nil || audit.StopReason.Code != "dispatch_contract_invalid" {
					t.Fatalf("oversized future output dispatched: runs=%d state=%s phase=%s stop=%+v", count, audit.State, audit.Phase, audit.StopReason)
				}
			} else if count != 2 {
				t.Fatalf("valid prepared output not dispatched: runs=%d state=%s phase=%s stop=%+v", count, audit.State, audit.Phase, audit.StopReason)
			}
			scope, _ := artifacts.AuditScope(audit.AuditID)
			project, _ := h.artifacts.Scoped(scope)
			meta, err := project.Metadata(ctx, output.Output.Retained.Ref)
			if err != nil || output.Output.Retained.Verify(meta.Exact()) != nil {
				t.Fatalf("retained output changed: %v", err)
			}
		})
	}
}
