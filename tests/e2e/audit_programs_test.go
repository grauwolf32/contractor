package e2e

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
)

const auditFixtureRevision = "audit-fixture-r1"

func TestAuditProgramsResolveAsRunnableMVPProfiles(t *testing.T) {
	t.Parallel()

	snapshot, err := config.Load("../../configs", config.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	for _, selector := range []string{"source-checklist@1", "openapi-operation-trace@1"} {
		profile, err := snapshot.AuditProfile(selector)
		if err != nil {
			t.Fatal(err)
		}
		compatibility := auditservice.ProfileCompatibility(profile)
		if !compatibility.ServerCompatible || len(compatibility.Reasons) != 0 {
			t.Fatalf("%s compatibility = %+v", selector, compatibility)
		}
		role := profile.Inventory.ItemWorkflowRole
		binding := profile.Workflows[role]
		if binding.Inputs["execution_manifest"].Source != config.AuditInputFromExecutionManifest ||
			binding.Outputs["result"] != "result" {
			t.Fatalf("%s has no exact execution/result boundary: %+v", selector, binding)
		}
	}
}

func TestAuditProgramsFixturesBuildStableChecklistAndOpenAPIInventories(t *testing.T) {
	t.Parallel()

	checklistBytes := readAuditFixture(t, "checklist.yaml")
	checklistRef := exactFixtureRef("checklists", "source-checklist")
	checklist, err := auditdomain.BuildChecklistInventory(
		checklistBytes, "application/yaml", auditdomain.InventoryOptions{
			Round: 1, WorkflowRole: "check", SourceInputName: "checklist", SourceRef: checklistRef,
			ApprovalRequirement: auditdomain.ApprovalNone,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := auditdomain.ValidateInventory(checklist); err != nil || len(checklist.Tasks) != 2 {
		t.Fatalf("checklist inventory = tasks:%d err:%v", len(checklist.Tasks), err)
	}
	for _, item := range checklist.Worklist.Items {
		if item.ApprovalRequirement != auditdomain.ApprovalNone {
			t.Fatalf("checklist item unexpectedly needs review: %+v", item)
		}
	}

	openAPIBytes := readAuditFixture(t, "openapi.yaml")
	openAPIRef := exactFixtureRef("openapi", "fixture")
	first, err := auditdomain.BuildOpenAPIInventory(
		openAPIBytes, "application/yaml", auditdomain.InventoryOptions{
			Round: 1, WorkflowRole: "trace", SourceInputName: "openapi", SourceRef: openAPIRef,
			ApprovalRequirement: auditdomain.ApprovalNone,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	second, err := auditdomain.BuildOpenAPIInventory(
		openAPIBytes, "application/yaml", auditdomain.InventoryOptions{
			Round: 1, WorkflowRole: "trace", SourceInputName: "openapi", SourceRef: openAPIRef,
			ApprovalRequirement: auditdomain.ApprovalNone,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(first.Tasks) != 2 || first.CanonicalInventoryDigest != second.CanonicalInventoryDigest ||
		!slices.Equal(first.CanonicalInventory, second.CanonicalInventory) {
		t.Fatalf("OpenAPI inventory is not stable: first=%+v second=%+v", first, second)
	}
	if !containsAuditGap(first.Gaps, "unsupported-callback") ||
		!containsAuditGap(first.Gaps, "unsupported-webhook") {
		t.Fatalf("OpenAPI unsupported surfaces were not retained as gaps: %v", first.Gaps)
	}
}

func TestAuditProgramsResultFixturesRemainDistinct(t *testing.T) {
	t.Parallel()

	manifest, err := auditdomain.DecodeExecutionManifest(readAuditFixture(t, "execution-manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	valid := readAuditFixture(t, "result-valid.json")
	evidence := readAuditFixture(t, "evidence-valid.json")
	validPackage, _, err := auditdomain.BuildPackage(
		"fixture-valid", auditdomain.PackageKindCheckResults, "", []auditdomain.PackageInput{
			{ID: auditdomain.CheckResultsMemberID, Path: "check-results.json", MediaType: auditdomain.JSONMediaType, Data: valid},
			{ID: auditdomain.EvidenceMemberID, Path: "evidence.json", MediaType: auditdomain.JSONMediaType, Data: evidence},
			{ID: "ev-content-1", Path: "evidence/ev-1.txt", MediaType: "text/plain", Data: []byte("Authorization call at source/app.py:13.")},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := auditdomain.DecodeCheckResultPackage(validPackage)
	if err != nil {
		t.Fatal(err)
	}
	if err := auditdomain.ValidateResultSet(decoded.Results, manifest); err != nil {
		t.Fatalf("valid result fixture: %v", err)
	}

	inconclusive := readAuditFixture(t, "result-inconclusive.json")
	inconclusivePackage, _, err := auditdomain.BuildPackage(
		"fixture-inconclusive", auditdomain.PackageKindCheckResults, "", []auditdomain.PackageInput{{
			ID: auditdomain.CheckResultsMemberID, Path: "check-results.json", MediaType: auditdomain.JSONMediaType, Data: inconclusive,
		}},
	)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err = auditdomain.DecodeCheckResultPackage(inconclusivePackage)
	if err != nil || decoded.Results.Results[0].Assessment != "inconclusive" {
		t.Fatalf("inconclusive result fixture = (%+v, %v)", decoded.Results, err)
	}
	if err := auditdomain.ValidateResultSet(decoded.Results, manifest); err != nil {
		t.Fatalf("inconclusive fixture lost valid technical membership: %v", err)
	}

	invalid := readAuditFixture(t, "result-invalid.json")
	invalidPackage, _, err := auditdomain.BuildPackage(
		"fixture-invalid", auditdomain.PackageKindCheckResults, "", []auditdomain.PackageInput{{
			ID: auditdomain.CheckResultsMemberID, Path: "check-results.json", MediaType: auditdomain.JSONMediaType, Data: invalid,
		}},
	)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err = auditdomain.DecodeCheckResultPackage(invalidPackage)
	if err != nil {
		t.Fatal(err)
	}
	if err := auditdomain.ValidateResultSet(decoded.Results, manifest); err == nil {
		t.Fatal("invalid result fixture was accepted as a clean result")
	}
}

func exactFixtureRef(namespace, name string) contracts.ArtifactRef {
	revision := auditFixtureRevision
	return contracts.ArtifactRef{Namespace: namespace, Name: name, Revision: &revision}
}

func readAuditFixture(t *testing.T, name string) []byte {
	t.Helper()
	value, err := os.ReadFile(filepath.Join("..", "fixtures", "audits", name))
	if err != nil {
		t.Fatal(err)
	}
	return value
}

func containsAuditGap(gaps []string, prefix string) bool {
	for _, gap := range gaps {
		if len(gap) >= len(prefix) && gap[:len(prefix)] == prefix {
			return true
		}
	}
	return false
}
