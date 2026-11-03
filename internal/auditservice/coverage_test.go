package auditservice

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestCoverageDescribesArbitraryRetainedTask(t *testing.T) {
	task := retainedCoverageTask(t, "Check that expired invitations cannot be reused.\n\nInclude the edge cases.")
	row := auditstore.CoverageRow{
		ItemKey: task.Document.ItemKey, SubjectKey: task.Document.SubjectKey,
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
	if err := describeCoverageTask(task.Package, &row); !errors.Is(err, blob.ErrIntegrity) {
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
	row := auditstore.CoverageRow{ItemKey: "second", SubjectKey: "subject", Coverage: auditstore.Coverage{Status: auditstore.CoverageStatus("inconclusive")}, Details: &auditstore.CoverageDetails{}}
	if err := describeCoverageResult(pkg, &row); err != nil {
		t.Fatal(err)
	}
	if row.Details.ResultSummary != "Second result" || len(row.Details.Evidence) != 1 || row.Details.Evidence[0].ID != "e2" || row.Coverage.Status != auditstore.CoverageStatus("inconclusive") {
		t.Fatalf("result or evidence crossed check boundaries: %+v", row)
	}
	row.ItemKey = "missing"
	if err := describeCoverageResult(pkg, &row); !errors.Is(err, blob.ErrIntegrity) {
		t.Fatalf("accepted unrelated result: %v", err)
	}
}

// The remaining coverage reader projects fixed retained documents. Scanner
// preparation itself is tested through the common operation in auditdomain.
func TestScanCoverageProjectsExactInputsWithoutRequestCredentials(t *testing.T) {
	for _, scanner := range []string{"sqlmap", "nuclei"} {
		t.Run(scanner, func(t *testing.T) {
			task := retainedCoverageTask(t, "Verify custom.")
			task.Document.Kind = "openapi-scan"
			task.Document.Checklist = nil
			settingsRevision := "settings-r1"
			scan := &auditdomain.ScanTask{Scanner: scanner, Operation: "#/paths/~1pets/get", Settings: auditdomain.ExactInput{Name: "settings", Ref: contracts.ArtifactRef{Namespace: "inputs", Name: "settings", Revision: &settingsRevision}, Digest: serviceTestDigest("settings")}, PreparationDigest: serviceTestDigest("preparation"), Runnable: true, TestParameters: []string{}, Gaps: []string{}}
			if scanner == "sqlmap" {
				scan.RequestDigest = serviceTestDigest("request")
				scan.TestParameters = []string{"id"}
			} else {
				scan.TargetURL = "https://example.test/pets"
				scan.Gaps = []string{"url_template_scan_only"}
			}
			task.Document.Scan = scan
			encoded, err := json.Marshal(task.Document)
			if err != nil {
				t.Fatal(err)
			}
			payload, _, err := auditdomain.BuildPackage("retained-scan", auditdomain.PackageKindTask, "", []auditdomain.PackageInput{{ID: "task-document", Path: "task.json", MediaType: auditdomain.JSONMediaType, Data: encoded}})
			if err != nil {
				t.Fatal(err)
			}
			row := auditstore.CoverageRow{ItemKey: task.Document.ItemKey, SubjectKey: task.Document.SubjectKey, Details: &auditstore.CoverageDetails{}}
			if err := describeCoverageTask(payload, &row); err != nil {
				t.Fatal(err)
			}
			projected := row.Details.Scan
			if projected == nil || projected.Scanner != scanner || !projected.Source.Ref.SameExact(task.Document.SourceRef) || projected.Source.Digest != task.Document.SourceContentDigest || !projected.Settings.Ref.SameExact(scan.Settings.Ref) || projected.PreparationDigest != scan.PreparationDigest || len(row.Details.Methods) != 1 || row.Details.Methods[0] != scan.CoverageRequirement() {
				t.Fatalf("lost exact provenance: %+v", row.Details)
			}
		})
	}
}

// retainedCoverageTask builds one fixed document for the surviving reader's
// integrity tests; it performs no checklist interpretation or task admission.
func retainedCoverageTask(t *testing.T, statement string) struct {
	Document      auditdomain.ItemTask
	Package       []byte
	PackageDigest string
} {
	t.Helper()
	revision := "source-v1"
	document := auditdomain.ItemTask{Schema: auditdomain.TaskSchema, ItemKey: "custom-check", SubjectKey: "custom-check", Kind: "checklist", WorkflowRole: "check", SourceContentDigest: serviceTestDigest("source"), SourceMediaType: "application/json", SourceRef: contracts.ArtifactRef{Namespace: "inputs", Name: "custom", Revision: &revision}, CanonicalInventoryDigest: serviceTestDigest("inventory"), Scope: map[string]string{"objective": "An arbitrary audit task"}, Checklist: &auditdomain.ChecklistTask{Version: "1", Statement: statement, Applicability: "always", AllowedMethods: []string{"custom-method"}, RequiredEvidence: []string{}, ReviewPolicy: "automatic"}}
	encoded, err := json.Marshal(document)
	if err != nil {
		t.Fatal(err)
	}
	payload, _, err := auditdomain.BuildPackage("retained-task", auditdomain.PackageKindTask, "", []auditdomain.PackageInput{{ID: "task-document", Path: "task.json", MediaType: auditdomain.JSONMediaType, Data: encoded}})
	if err != nil {
		t.Fatal(err)
	}
	return struct {
		Document      auditdomain.ItemTask
		Package       []byte
		PackageDigest string
	}{document, payload, auditdomain.DigestBytes(payload)}
}
