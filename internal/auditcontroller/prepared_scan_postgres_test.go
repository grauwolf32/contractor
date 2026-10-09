//go:build integration

package auditcontroller

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
)

func TestPostgresPreparedOpenAPIScansKeepAssignedDenominatorAndApproval(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 60*time.Second)
	defer cancel()
	catalog := preparationControllerConfig(t, false, func(files map[string]string) {
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
	settings := artifacts.Payload{MediaType: "application/json", Data: []byte(`{"schema":"contractor.audit.openapi-scan-settings.v1","scanner":"sqlmap","server":"http://127.0.0.1:8080","testParameters":["q"],"operations":{"#/paths/~1a/get":{"parameters":{"query:q":"one"}},"#/paths/~1b/get":{"parameters":{"query:q":"two"}},"#/paths/~1c/get":{"parameters":{"query:q":"three"}}}}`)}
	h := newPostgresControllerHarnessWithConfig(t, ctx, 0, 1, catalog, map[string]artifacts.Payload{"settings": settings})
	c := preparationController(t, h)
	generated := artifacts.Payload{MediaType: "application/json", Data: []byte(`{"openapi":"3.1.0","paths":{"/a":{"get":{"parameters":[{"in":"query","name":"q","schema":{"type":"string"}}],"responses":{"200":{"description":"ok"}}}},"/b":{"get":{"parameters":[{"in":"query","name":"q","schema":{"type":"string"}}],"responses":{"200":{"description":"ok"}}}},"/c":{"get":{"parameters":[{"in":"query","name":"q","schema":{"type":"string"}}],"responses":{"200":{"description":"ok"}}}},"/unselected":{"get":{"responses":{"200":{"description":"ok"}}}}}}`)}
	readyGeneratedInventory(t, ctx, h, c, map[string]artifacts.Payload{"checklist": generated})
	c.initialRoundBuilder = h.service
	preparationStep(t, ctx, c)
	items, err := h.audits.ListItems(ctx, h.started.Audit.AuditID)
	if err != nil || len(items) != 3 {
		t.Fatalf("scan denominator expanded beyond explicit settings: %+v %v", items, err)
	}
	project, _ := h.artifacts.Project(h.started.Audit.ProjectID)
	for _, item := range items {
		if item.Kind != "openapi-scan" || item.State != auditstore.ItemAwaitingReview || item.ApprovalKind != auditstore.ItemApprovalActiveCheck {
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
	reviews, err := h.service.ListReviews(ctx, auditservice.ReviewListParams{OwnerID: h.started.Audit.OwnerID, AuditID: h.started.Audit.AuditID, Limit: 10})
	if err != nil || len(reviews) != 3 {
		t.Fatalf("generated scan review ledger: %+v %v", reviews, err)
	}
	preparationExecutions(t, ctx, h, 1)
}
