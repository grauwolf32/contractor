package public

import (
	"context"
	"encoding/json"
	"io/fs"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
)

func TestAuditPostgresPublicCreateStartAndQuery(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	pool := isolatedPublicPool(t, ctx)
	configuration := loadPublicAuditConfiguration(t)
	gateway, err := configuration.LLMGateway("local-litellm@1")
	if err != nil {
		t.Fatal(err)
	}
	managedCredentials := newFakeManagedCredentials()
	managedCredentials.lookups["development-worker"] = config.CredentialMetadata{
		Ref:        contracts.LLMCredentialRef{CredentialID: "development-worker"},
		LLMGateway: gateway.Ref, Unrestricted: true,
	}
	audits, err := auditservice.New(auditservice.Options{
		Pool: pool, Profiles: configuration,
		TransactionLLMCredentials: runtimeconfig.TransactionLLMCredentialLookupFactoryFunc(
			func(pgx.Tx) (config.CredentialLookup, error) { return managedCredentials, nil },
		),
		CredentialGuard: managedCredentials,
		Now:             func() time.Time { return time.Date(2026, 9, 5, 20, 0, 0, 0, time.UTC) },
	})
	if err != nil {
		t.Fatal(err)
	}

	projects := projectstore.NewPostgresStore(pool)
	project, _, err := projects.Create(ctx, projectstore.CreateParams{
		ProjectID: "project-public-audit", OwnerID: "user-1", Kind: projectstore.KindProject,
		Name: "Public Audit", IdempotencyKey: "create-public-audit-project",
		RequestDigest: auditHandlerDigest("public-audit-project"),
	})
	if err != nil {
		t.Fatal(err)
	}
	projectArtifacts, err := artifacts.NewService(artifacts.NewPostgresRepository(pool)).Project(project.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	source, err := projectArtifacts.Write(ctx, contracts.ArtifactRef{
		Namespace: "inputs", Name: "source",
	}, artifacts.Payload{MediaType: "application/zip", Data: []byte("fixture source archive")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	checklist, err := projectArtifacts.Write(ctx, contracts.ArtifactRef{
		Namespace: "inputs", Name: "checklist",
	}, artifacts.Payload{MediaType: "application/json", Data: []byte(
		`{"schema":"contractor.audit.checklist.v1","items":[{"key":"check-auth","version":"1","statement":"Verify authentication boundaries.","applicability":"always","allowed_methods":["static"],"required_evidence":[],"review_policy":"automatic"}]}`,
	)}, nil)
	if err != nil {
		t.Fatal(err)
	}

	h := auditTestHandler(audits)
	h.dependencies.Projects = projects
	requestBody, err := json.Marshal(createAuditRequest{
		Profile: auditProfileSelectorRequest{Name: "public-checklist", Version: "1"},
		Inputs: map[string]contracts.ArtifactRef{
			"source": source.Ref, "checklist": checklist.Ref,
		},
		RuntimeLabels: []string{}, Scope: auditservice.Scope{Objective: "Review authentication"},
	})
	if err != nil {
		t.Fatal(err)
	}
	create := auditAuthenticatedRequest(http.MethodPost, "/v1/projects/"+project.ProjectID+"/audits", requestBody)
	create.SetPathValue("projectId", project.ProjectID)
	create.Header.Set("Content-Type", "application/json")
	create.Header.Set("Idempotency-Key", "create-public-audit")
	created := httptest.NewRecorder()
	h.createAudit(created, create)
	if created.Code != http.StatusCreated || created.Header().Get("ETag") != `"1"` {
		t.Fatalf("create Audit = %d headers=%v body=%s", created.Code, created.Header(), created.Body.String())
	}
	var draft auditResponse
	if err := json.Unmarshal(created.Body.Bytes(), &draft); err != nil || draft.State != auditstore.AuditDraft ||
		draft.Inputs["source"].Ref.Revision == nil || draft.Baseline != nil {
		t.Fatalf("draft response = (%+v, %v)", draft, err)
	}

	start := auditAuthenticatedRequest(http.MethodPost, "/v1/audits/"+draft.AuditID+"/start", nil)
	start.SetPathValue("auditId", draft.AuditID)
	start.Header.Set("Idempotency-Key", "start-public-audit")
	start.Header.Set("If-Match", `"1"`)
	startedResponse := httptest.NewRecorder()
	h.startAudit(startedResponse, start)
	if startedResponse.Code != http.StatusOK || startedResponse.Header().Get("ETag") != `"2"` {
		t.Fatalf("start Audit = %d headers=%v body=%s", startedResponse.Code, startedResponse.Header(), startedResponse.Body.String())
	}
	var started auditStartResponse
	if err := json.Unmarshal(startedResponse.Body.Bytes(), &started); err != nil ||
		started.Audit.State != auditstore.AuditActive || started.Audit.HoldState != auditstore.HoldHeld ||
		started.Audit.Baseline == nil || len(started.Items) != 1 || started.Round.ExpectedItemCount != 1 {
		t.Fatalf("started response = (%+v, %v)", started, err)
	}

	replay := auditAuthenticatedRequest(http.MethodPost, "/v1/audits/"+draft.AuditID+"/start", nil)
	replay.SetPathValue("auditId", draft.AuditID)
	replay.Header.Set("Idempotency-Key", "start-public-audit")
	replay.Header.Set("If-Match", `"1"`)
	replayed := httptest.NewRecorder()
	h.startAudit(replayed, replay)
	if replayed.Code != http.StatusOK || replayed.Header().Get("Idempotency-Replayed") != "true" ||
		replayed.Body.String() != startedResponse.Body.String() {
		t.Fatalf("start replay = %d headers=%v body=%s", replayed.Code, replayed.Header(), replayed.Body.String())
	}

	coverage := auditAuthenticatedRequest(http.MethodGet, "/v1/audits/"+draft.AuditID+"/coverage", nil)
	coverage.SetPathValue("auditId", draft.AuditID)
	coverageResponse := httptest.NewRecorder()
	h.listAuditCoverage(coverageResponse, coverage)
	var page auditCoveragePageResponse
	if coverageResponse.Code != http.StatusOK || json.Unmarshal(coverageResponse.Body.Bytes(), &page) != nil ||
		len(page.Items) != 1 || page.Items[0].Coverage.Status != auditstore.CoverageNotTested {
		t.Fatalf("coverage = %d body=%s", coverageResponse.Code, coverageResponse.Body.String())
	}
}

func loadPublicAuditConfiguration(t *testing.T) *config.Snapshot {
	t.Helper()
	sourceRoot := filepath.Clean("../../../configs")
	targetRoot := filepath.Join(t.TempDir(), "configs")
	err := filepath.WalkDir(sourceRoot, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		relative, err := filepath.Rel(sourceRoot, path)
		if err != nil {
			return err
		}
		target := filepath.Join(targetRoot, relative)
		if entry.IsDir() {
			return os.MkdirAll(target, 0o755)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		return os.WriteFile(target, data, 0o600)
	})
	if err != nil {
		t.Fatalf("copy configuration: %v", err)
	}
	profile := []byte(`apiVersion: contractor/v1alpha1
kind: AuditProfile
metadata: {name: public-checklist, version: "1"}
spec:
  mode: custom-checklist
  standards: []
  inputs:
    source: {required: true, mediaTypes: [application/zip]}
    checklist: {required: true, mediaTypes: [application/json]}
  inventory:
    implementation: checklist@1
    sourceInput: checklist
    itemWorkflowRole: check
  workflows:
    check:
      ref: openapi-from-workspace@5
      inputs:
        source: {source: audit-input, name: source}
      parameters:
        objective: {source: item-field, name: subjectKey}
      outputs: {result: openapi}
  execution:
    roundMode: fixed-barrier
    maxRounds: 1
    batchSize: 1
    maxItemsPerRound: 10
    maxItemsTotal: 10
    maxSubmittedRuns: 20
    maxItemRunAttempts: 2
    deadlineSeconds: 3600
    maxEvidenceBytes: 1048576
    incompleteRound: assess-with-gaps
  interaction:
    activeChecks: prohibited
    findingConfirmation: disabled
    notApplicable: profile-rule
    reportAcceptance: automatic
`)
	if err := os.WriteFile(filepath.Join(targetRoot, "audit-profiles", "public_checklist.yaml"), profile, 0o600); err != nil {
		t.Fatal(err)
	}
	snapshot, err := config.Load(targetRoot, config.MVPDescriptors())
	if err != nil {
		t.Fatalf("load public Audit configuration: %v", err)
	}
	return snapshot
}
