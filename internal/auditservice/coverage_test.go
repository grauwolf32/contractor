package auditservice

import (
	"encoding/json"
	"errors"
	"os"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestCoverageDescribesArbitraryRetainedTask(t *testing.T) {
	revision := "source-v1"
	inventory, err := auditdomain.BuildChecklistInventory([]byte(`{
  "schema":"contractor.audit.checklist.v1",
  "items":[{"key":"custom-check","version":"1","statement":"Check that expired invitations cannot be reused.\n\nInclude the edge cases.","applicability":"always","allowed_methods":["custom-method"],"required_evidence":[],"review_policy":"automatic"}]
}`), "application/json", auditdomain.InventoryOptions{
		Round: 1, WorkflowRole: "check", SourceInputName: "checklist",
		ApprovalRequirement: auditdomain.ApprovalNone,
		SourceRef:           contracts.ArtifactRef{Namespace: "inputs", Name: "custom", Revision: &revision},
		Scope:               map[string]string{"objective": "An arbitrary audit task"},
	})
	if err != nil {
		t.Fatal(err)
	}
	task := inventory.Tasks[0]
	row := auditstore.CoverageRow{
		ItemKey: task.Item.ItemKey, SubjectKey: task.Item.SubjectKey,
		Task:    auditstore.ExactArtifact{Digest: task.PackageDigest},
		Details: &auditstore.CoverageDetails{},
	}
	if err := describeCoverageTask(task.Package, &row); err != nil {
		t.Fatal(err)
	}
	if row.Details.Objective != task.Document.Checklist.Statement || len(row.Details.Methods) != 1 || row.Details.Methods[0] != "custom-method" {
		t.Fatalf("lost custom task text: %+v", row.Details)
	}
	var document auditdomain.ItemTask
	if err := json.Unmarshal(row.Details.TaskDocument, &document); err != nil || document.Scope["objective"] != "An arbitrary audit task" || document.Checklist.Statement != task.Document.Checklist.Statement {
		t.Fatalf("exact task projection differs: %+v, %v", document, err)
	}
	row.SubjectKey = "different-check"
	if err := describeCoverageTask(task.Package, &row); !errors.Is(err, artifacts.ErrArtifactIntegrity) {
		t.Fatalf("accepted unrelated task: %v", err)
	}
}

func TestCoverageResultKeepsBatchMembersAndEvidenceSeparate(t *testing.T) {
	pkg := auditdomain.CheckResultPackage{
		Results: auditdomain.CheckResultSet{Results: []auditdomain.CheckResult{
			{ItemKey: "first", SubjectKey: "subject", Summary: "First result", EvidenceIDs: []string{"e1"}},
			{ItemKey: "second", SubjectKey: "subject", Summary: "Second result", EvidenceIDs: []string{"e2"}},
		}},
		Evidence: auditdomain.EvidenceEnvelope{Evidence: []auditdomain.Evidence{
			{ID: "e1", Kind: "observation", Summary: "First evidence"},
			{ID: "e2", Kind: "tool-result", Summary: "Second evidence"},
		}},
	}
	row := auditstore.CoverageRow{ItemKey: "second", SubjectKey: "subject", Coverage: auditstore.Coverage{Status: auditstore.CoverageInconclusive}, Details: &auditstore.CoverageDetails{}}
	if err := describeCoverageResult(pkg, &row); err != nil {
		t.Fatal(err)
	}
	if row.Details.ResultSummary != "Second result" || len(row.Details.Evidence) != 1 || row.Details.Evidence[0].ID != "e2" || row.Coverage.Status != auditstore.CoverageInconclusive {
		t.Fatalf("result or evidence crossed check boundaries: %+v", row)
	}
	row.ItemKey = "missing"
	if err := describeCoverageResult(pkg, &row); !errors.Is(err, artifacts.ErrArtifactIntegrity) {
		t.Fatalf("accepted unrelated result: %v", err)
	}
}

func TestScanCoverageProjectsExactInputsWithoutRequestCredentials(t *testing.T) {
	for _, scanner := range []string{"sqlmap", "nuclei"} {
		t.Run(scanner, func(t *testing.T) {
			source, err := os.ReadFile(openAPIScanInputFixture + "/openapi.json")
			if err != nil {
				t.Fatal(err)
			}
			settings, err := os.ReadFile(openAPIScanInputFixture + "/" + scanner + "-settings.json")
			if err != nil {
				t.Fatal(err)
			}
			sourceRevision, settingsRevision := "source-r1", "settings-r1"
			sourceRef := contracts.ArtifactRef{Namespace: "inputs", Name: "openapi", Revision: &sourceRevision}
			settingsRef := contracts.ArtifactRef{Namespace: "inputs", Name: "settings", Revision: &settingsRevision}
			inventory, err := auditdomain.BuildOpenAPIScanInventory(source, "application/json", settings, auditdomain.ExactInput{Name: "settings", Ref: settingsRef, Digest: auditdomain.DigestBytes(settings)}, auditdomain.InventoryOptions{Round: 1, WorkflowRole: "scan", SourceInputName: "openapi", SourceRef: sourceRef, ApprovalRequirement: auditdomain.ApprovalNone})
			if err != nil {
				t.Fatal(err)
			}
			for _, task := range inventory.Tasks {
				row := auditstore.CoverageRow{ItemKey: task.Item.ItemKey, SubjectKey: task.Item.SubjectKey, Details: &auditstore.CoverageDetails{}}
				if err := describeCoverageTask(task.Package, &row); err != nil {
					t.Fatal(err)
				}
				scan := row.Details.Scan
				if scan == nil || scan.Scanner != scanner || !scan.Source.Ref.SameExact(sourceRef) || scan.Source.Digest != auditdomain.DigestBytes(source) || !scan.Settings.Ref.SameExact(settingsRef) || scan.PreparationDigest != task.Document.Scan.PreparationDigest || len(row.Details.Methods) != 1 || row.Details.Methods[0] != task.Document.Scan.CoverageRequirement() {
					t.Fatalf("lost exact %s provenance: %+v", scanner, row.Details)
				}
			}
		})
	}
}
