package auditservice

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/projectstore"
)

func TestCoverageReadsRetainedTasksAndSharedResult(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	pool := isolatedAuditServicePool(t, ctx, databaseURL)
	project, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: "project-coverage", OwnerID: "coverage-owner", Kind: projectstore.KindProject,
		Name: "Coverage", IdempotencyKey: "create-coverage-project", RequestDigest: serviceTestDigest("coverage-project"),
	})
	if err != nil {
		t.Fatal(err)
	}
	store, err := artifacts.NewService(artifactpgtest.NewStorage().Repository(pool)).Project(project.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	task := retainedCoverageTask(t, "Verify custom.")
	write := func(name string, data []byte) auditstore.ExactArtifact {
		t.Helper()
		result, err := store.Write(ctx, contracts.ArtifactRef{Namespace: "retained", Name: name}, artifacts.Payload{MediaType: auditdomain.PackageMediaType, Data: data}, nil)
		if err != nil {
			t.Fatal(err)
		}
		return auditstore.ExactArtifact{Ref: result.Ref, Digest: auditdomain.DigestBytes(data)}
	}
	taskRef := write("task", task.Package)
	results, err := json.Marshal(auditdomain.CheckResultSet{
		Schema: auditdomain.CheckResultsSchema, ExecutionManifestDigest: serviceTestDigest("manifest"),
		Results: []auditdomain.CheckResult{{
			ItemKey: task.Document.ItemKey, SubjectKey: task.Document.SubjectKey, Assessment: "inconclusive",
			Summary: "An arbitrary model conclusion from an existing retained package.", EvidenceIDs: []string{},
			Coverage: auditdomain.ResultCoverage{Requested: []string{}, Completed: []string{}, Gaps: []string{}}, Proposals: []auditdomain.ProposalSelection{},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	payload, _, err := auditdomain.BuildPackage("coverage-result", auditdomain.PackageKindCheckResults, "", []auditdomain.PackageInput{{ID: auditdomain.CheckResultsMemberID, Path: "check-results.json", MediaType: auditdomain.JSONMediaType, Data: results}})
	if err != nil {
		t.Fatal(err)
	}
	resultRef := write("result", payload)
	rows := []auditstore.CoverageRow{
		{ItemKey: task.Document.ItemKey, SubjectKey: task.Document.SubjectKey, Task: taskRef, Result: &resultRef},
		{ItemKey: task.Document.ItemKey, SubjectKey: task.Document.SubjectKey, Task: taskRef, Result: &resultRef},
	}
	service := &Service{storage: artifactpgtest.NewStorage(), pool: pool}
	described, err := service.describeCoverage(ctx, project.ProjectID, rows)
	if err != nil {
		t.Fatal(err)
	}
	for _, row := range described {
		if row.Details.Objective != "Verify custom." || row.Details.ResultSummary != "An arbitrary model conclusion from an existing retained package." {
			t.Fatalf("missing retained text: %+v", row.Details)
		}
	}
	rows[0].Task.Digest = serviceTestDigest("wrong-task")
	if _, err := service.describeCoverage(ctx, project.ProjectID, rows); !errors.Is(err, blob.ErrIntegrity) {
		t.Fatalf("accepted a mismatched task digest: %v", err)
	}
}
