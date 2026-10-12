package public

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
)

// Audit start revalidates every pinned Workflow credential inside its start
// transaction. A lookup outage must answer the documented retryable error and
// commit nothing; a confirmed missing credential is a permanent 400.
func TestAuditPostgresStartKeepsWorkflowCredentialLookupClass(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	pool := isolatedPublicPool(t, ctx)
	configuration := loadPublicAuditConfiguration(t)
	gateway, err := configuration.LLMGateway("local-litellm@1")
	if err != nil {
		t.Fatal(err)
	}
	lookup := &scriptedLLMCredentialLookup{metadata: map[string]resolved.CredentialMetadata{
		"development-worker": {
			Ref:        llmgateway.LLMCredentialRef{CredentialID: "development-worker"},
			LLMGateway: gateway.Ref, Unrestricted: true,
		},
	}}
	audits, err := auditservice.New(auditservice.Options{
		Pool: pool, Profiles: configuration,
		TransactionLLMCredentials: runtimeconfig.TransactionLLMCredentialLookupFactoryFunc(
			func(pgx.Tx) (resolved.CredentialLookup, error) { return lookup, nil },
		),
		CredentialGuard: newFakeManagedCredentials(),
		Now:             func() time.Time { return time.Date(2026, 9, 5, 20, 0, 0, 0, time.UTC) },
	})
	if err != nil {
		t.Fatal(err)
	}
	projects := projectstore.NewPostgresStore(pool)
	project, _, err := projects.Create(ctx, projectstore.CreateParams{
		ProjectID: "project-audit-credential-lookup", OwnerID: "user-1", Kind: projectstore.KindProject,
		Name: "Audit credential lookup", IdempotencyKey: "create-audit-credential-lookup-project",
		RequestDigest: auditHandlerDigest("audit-credential-lookup-project"),
	})
	if err != nil {
		t.Fatal(err)
	}
	projectArtifacts, err := artifacts.NewService(artifacts.NewPostgresRepository(pool)).Project(project.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	source, err := projectArtifacts.Write(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "source"},
		artifacts.Payload{MediaType: "application/zip", Data: []byte("fixture source archive")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	checklist, err := projectArtifacts.Write(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "checklist"},
		artifacts.Payload{MediaType: "application/json", Data: []byte(
			`{"schema":"contractor.audit.checklist.v1","items":[{"key":"check-auth","version":"1","statement":"Verify authentication boundaries.","applicability":"always","allowed_methods":["static"],"required_evidence":[],"review_policy":"automatic"}]}`,
		)}, nil)
	if err != nil {
		t.Fatal(err)
	}
	draft, _, err := audits.CreateDraft(ctx, auditservice.CreateDraftParams{
		AuditID: "audit-credential-lookup", OwnerID: project.OwnerID, ProjectID: project.ProjectID,
		Profile:        auditservice.ProfileSelector{Name: "public-checklist", Version: "1"},
		Inputs:         map[string]contracts.ArtifactRef{"source": source.Ref, "checklist": checklist.Ref},
		RuntimeLabels:  []string{},
		Scope:          auditservice.Scope{Objective: "Review authentication"},
		IdempotencyKey: "create-audit-credential-lookup", RequestDigest: auditHandlerDigest("create-audit-credential-lookup"),
	})
	if err != nil {
		t.Fatal(err)
	}

	h := auditTestHandler(audits)
	h.Projects = projects
	start := func() *httptest.ResponseRecorder {
		request := auditAuthenticatedRequest(http.MethodPost, "/v1/audits/"+draft.AuditID+"/start", nil)
		request.SetPathValue("auditId", draft.AuditID)
		request.Header.Set("Idempotency-Key", "start-audit-credential-lookup")
		request.Header.Set("If-Match", `"1"`)
		response := httptest.NewRecorder()
		h.startAudit(response, request)
		return response
	}
	for _, outcome := range workflowCredentialLookupOutcomes {
		t.Run(outcome.name, func(t *testing.T) {
			lookup.fail(outcome.err)
			assertCredentialLookupResponse(t, start(), outcome.status, outcome.code, outcome.retryable)
			current, err := audits.Get(ctx, project.OwnerID, draft.AuditID)
			if err != nil || current.State != auditstore.AuditDraft || current.Revision != draft.Revision {
				t.Fatalf("failed start changed the draft = (%+v, %v)", current, err)
			}
		})
	}

	// The same start request succeeds once the credential store answers.
	lookup.fail(nil)
	response := start()
	var started auditStartResponse
	if response.Code != http.StatusOK || json.Unmarshal(response.Body.Bytes(), &started) != nil ||
		started.Audit.State != auditstore.AuditActive {
		t.Fatalf("start after lookup recovery = %d %s", response.Code, response.Body.String())
	}
}
