package public

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runservice"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

// Every operation that binds an Idempotency-Key reports the key reused with a
// different request as the permanent 409 idempotency_conflict, retryable
// false (FAIL-RUN-1). Each dependency below answers with the sentinel the
// real stores return for that outcome; that the stores return it, and that
// the same request still replays, is tested against PostgreSQL in their
// packages and in the Run and Audit handler tests. An operation that takes
// the IdempotencyKey parameter but has no request here fails the test.
func TestEveryKeyedOperationReportsKeyReuseAsPermanent(t *testing.T) {
	reused := fmt.Errorf("%w: test key", contracts.ErrIdempotencyConflict)
	audits := &reusedKeyAudits{fakeAuditManagement: &fakeAuditManagement{
		audit: auditstore.Audit{AuditID: "audit-one", OwnerID: "user-1", ProjectID: "project-one", Revision: 1},
	}, err: reused}
	var fixture handlerFixture
	fixture = newHandlerFixtureWithAuth(t, "../../config/testdata/valid",
		newTestAuthentication(t), mustTestOrigins(t), false, nil,
		func(dependencies *testServices) {
			dependencies.Audits = audits
			dependencies.RunCreator = reusedKeyRuns{err: reused}
			dependencies.Projects = reusedKeyProjects{ProjectManagement: dependencies.Projects, err: reused}
			dependencies.ConfigurationPublisher = reusedKeyPublisher{err: reused}
			dependencies.RuntimeConfigs = reusedKeyRuntimeConfigs{
				fakeRuntimeConfigManagement: dependencies.RuntimeConfigs.(*fakeRuntimeConfigManagement), err: reused,
			}
			dependencies.RuntimeCredentials = reusedKeyRuntimeCredentials{
				fakeRuntimeCredentialManagement: dependencies.RuntimeCredentials.(*fakeRuntimeCredentialManagement), err: reused,
			}
			dependencies.RuntimeAgentPrincipals = reusedKeyPrincipals{
				fakeRuntimeAgentPrincipalManagement: dependencies.RuntimeAgentPrincipals.(*fakeRuntimeAgentPrincipalManagement), err: reused,
			}
			dependencies.ManagedCredentials = reusedKeyManagedCredentials{
				fakeManagedCredentials: dependencies.ManagedCredentials.(*fakeManagedCredentials), err: reused,
			}
		},
	)
	// createProjectRun reads its Project before it creates the Run.
	if _, _, err := fixture.projects.Create(t.Context(), projectstore.CreateParams{
		ProjectID: "project-one", OwnerID: "user-1", Kind: projectstore.KindProject, Name: "Keyed",
		IdempotencyKey: "seed-project", RequestDigest: auditHandlerDigest("seed-project"),
	}); err != nil {
		t.Fatal(err)
	}
	gateway, err := fixture.configs.Snapshot().LLMGateway("local-litellm@1")
	if err != nil {
		t.Fatal(err)
	}
	policy, err := fixture.configs.Snapshot().ModelPolicy("worker@1")
	if err != nil {
		t.Fatal(err)
	}
	credentialBody, err := json.Marshal(createCredentialRequest{
		CredentialID: "managed-keyed", LLMGateway: gateway.Ref,
		GatewayPolicy: credentials.GatewayPolicy{ModelPolicies: []contracts.ModelPolicyRef{policy.Ref}},
	})
	if err != nil {
		t.Fatal(err)
	}
	labelBody, err := json.Marshal(runtimeLabelMutationRequest{Config: runtimeconfig.Ref{
		Name: "debug", Version: "1", Digest: "sha256:" + strings.Repeat("a", 64),
	}})
	if err != nil {
		t.Fatal(err)
	}
	agent := strings.Repeat("b", 64)
	type keyedRequest struct {
		method, path, body string
		headers            map[string]string
	}
	runBody := `{"workflow":"artifact-copy@1","parameters":{},"artifacts":{"source":{"namespace":"projects","name":"source"}}}`
	requests := map[string]keyedRequest{
		"createProject":    {http.MethodPost, "/v1/projects", `{"kind":"project","name":"Keyed"}`, nil},
		"createRun":        {http.MethodPost, "/v1/runs", runBody, nil},
		"createProjectRun": {http.MethodPost, "/v1/projects/project-one/runs", runBody, nil},
		"createAudit": {http.MethodPost, "/v1/projects/project-one/audits",
			`{"profile":{"name":"checklist","version":"1"},"inputs":{},"runtimeLabels":[],"scope":{"objective":"Keyed"}}`, nil},
		"startAudit":  {http.MethodPost, "/v1/audits/audit-one/start", "", map[string]string{"If-Match": `"1"`}},
		"pauseAudit":  {http.MethodPost, "/v1/audits/audit-one/pause", "", map[string]string{"If-Match": `"2"`}},
		"resumeAudit": {http.MethodPost, "/v1/audits/audit-one/resume", `{"deadlineSeconds":0}`, map[string]string{"If-Match": `"3"`}},
		"cancelAudit": {http.MethodPost, "/v1/audits/audit-one/cancel", "", map[string]string{"If-Match": `"3"`}},
		"deleteAudit": {http.MethodDelete, "/v1/audits/audit-one", "", map[string]string{"If-Match": `"4"`}},
		"createAuditFindingReview": {http.MethodPost, "/v1/audits/audit-one/findings/finding-one/reviews", `{}`,
			map[string]string{"If-Match": `"1"`}},
		"decideAuditReview": {http.MethodPost, "/v1/audits/audit-one/reviews/review-one/decisions",
			`{"verdict":"true-positive","rationale":"Confirmed."}`, map[string]string{"If-Match": `"1"`}},
		"publishConfiguration": {http.MethodPost, "/v1/configurations/model-policies",
			`{"name":"keyed","version":"1","modelPolicy":{"model":"m"}}`, nil},
		"publishRuntimeConfig": {http.MethodPost, "/v1/operations/runtime-configs",
			`{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig","metadata":{"name":"debug","version":"1"},"spec":{"worker":{"telemetry":{"adapter":"otlp-http@1","endpoint":"https://otel.example/v1/traces"}}}}`, nil},
		"putRuntimeLabel":    {http.MethodPut, "/v1/operations/runtime-labels/debug", string(labelBody), map[string]string{"If-None-Match": "*"}},
		"deleteRuntimeLabel": {http.MethodDelete, "/v1/operations/runtime-labels/debug", "", map[string]string{"If-Match": `"1"`}},
		"createRuntimeCredential": {http.MethodPost, "/v1/operations/runtime-credentials",
			`{"credentialId":"otel-keyed","kind":"otlp-headers@1","material":{"headers":{"authorization":"keyed"}}}`, nil},
		"deleteRuntimeAgentPrincipal": {http.MethodDelete, "/v1/operations/runtime-agent-principals/" + agent, "",
			map[string]string{"If-Match": `"1"`}},
		"replaceRuntimeAgentPrincipalLabels": {http.MethodPut, "/v1/operations/runtime-agent-principals/" + agent + "/labels",
			`{"labels":[]}`, map[string]string{"If-Match": `"1"`}},
		"createCredential": {http.MethodPost, "/v1/operations/credentials", string(credentialBody), nil},
		"deleteCredential": {http.MethodDelete, "/v1/operations/credentials/managed-keyed", "", nil},
	}
	// Eval mutations validate frozen documents before the service, and report
	// a reused key as eval_idempotency_conflict through handleError (checked
	// below); TestSharedEvalPostgresAuthorizationCSRFStrictBodiesAndCAS reuses a
	// command key against PostgreSQL.
	evalKeyed := map[string]bool{
		"createEvalExperiment": true, "updateEvalDraft": true, "commandEvalExperiment": true,
		"deleteEvalExperiment": true, "importEvalDataset": true, "submitEvalMember": true,
		"ingestEvalResult": true, "assessEvalMember": true, "selectEvalRecords": true,
	}
	keyed := keyedOperations(t)
	for operation := range keyed {
		if _, ok := requests[operation]; !ok && !evalKeyed[operation] {
			t.Errorf("keyed operation %s has no request in this test", operation)
		}
	}
	for operation, keyed := range requests {
		t.Run(operation, func(t *testing.T) {
			request := authenticatedRequest(keyed.method, keyed.path, bytes.NewReader([]byte(keyed.body)))
			if keyed.body != "" {
				request.Header.Set("Content-Type", "application/json")
			}
			request.Header.Set(idempotencyKeyHeader, "reused-key")
			for name, value := range keyed.headers {
				request.Header.Set(name, value)
			}
			response := httptest.NewRecorder()
			fixture.handler.ServeHTTP(response, request)
			assertIdempotencyConflict(t, response)
		})
	}
	evalResponse := httptest.NewRecorder()
	handleError(evalResponse, evaldomain.Failure("eval_idempotency_conflict"))
	var evalBody errorResponse
	if err := json.Unmarshal(evalResponse.Body.Bytes(), &evalBody); err != nil || evalResponse.Code != http.StatusConflict ||
		evalBody.Code != "eval_idempotency_conflict" || evalBody.Retryable {
		t.Fatalf("Eval idempotency conflict = %d %s", evalResponse.Code, evalResponse.Body.String())
	}
	for operation := range evalKeyed {
		requests[operation] = keyedRequest{}
	}
	for operation := range requests {
		if !keyed[operation] {
			t.Errorf("%s does not take the IdempotencyKey parameter", operation)
		}
	}
	if !audits.reached["DecideFinding"] || !audits.reached["CreateDraft"] {
		t.Fatalf("Audit mutations were not reached: %v", audits.reached)
	}
}

// keyedOperations returns the operationIds that bind a required
// Idempotency-Key (the optional key of a resource-idempotent delete is not
// bound to a request).
func keyedOperations(t *testing.T) map[string]bool {
	t.Helper()
	result := make(map[string]bool)
	for _, item := range loadPublicOpenAPI(t).Paths.Map() {
		for _, operation := range item.Operations() {
			for _, parameter := range operation.Parameters {
				if parameter.Ref == "#/components/parameters/IdempotencyKey" {
					result[operation.OperationID] = true
				}
			}
		}
	}
	if len(result) == 0 {
		t.Fatal("OpenAPI has no operation with the IdempotencyKey parameter")
	}
	return result
}

type reusedKeyProjects struct {
	ProjectManagement
	err error
}

func (p reusedKeyProjects) Create(context.Context, projectstore.CreateParams) (projectstore.Project, bool, error) {
	return projectstore.Project{}, false, p.err
}

type reusedKeyRuns struct{ err error }

func (r reusedKeyRuns) CreatePublic(context.Context, runservice.PublicCreateParams) (runservice.CreateResult, error) {
	return runservice.CreateResult{}, r.err
}

type reusedKeyPublisher struct{ err error }

func (p reusedKeyPublisher) Publish(context.Context, config.PublicationRequest) (config.PublicationResult, error) {
	return config.PublicationResult{}, p.err
}

type reusedKeyRuntimeConfigs struct {
	*fakeRuntimeConfigManagement
	err error
}

func (r reusedKeyRuntimeConfigs) Publish(context.Context, []byte, string, string) (runtimeconfig.PublishResult, error) {
	return runtimeconfig.PublishResult{}, r.err
}

func (r reusedKeyRuntimeConfigs) CreateBinding(
	context.Context, string, runtimeconfig.Ref, string, string, time.Time,
) (runtimeconfig.BindingMutationResult, error) {
	return runtimeconfig.BindingMutationResult{}, r.err
}

func (r reusedKeyRuntimeConfigs) Rebind(
	context.Context, string, uint64, runtimeconfig.Ref, string, string, time.Time,
) (runtimeconfig.BindingMutationResult, error) {
	return runtimeconfig.BindingMutationResult{}, r.err
}

func (r reusedKeyRuntimeConfigs) DeleteBinding(
	context.Context, string, uint64, string, string, time.Time,
) (runtimeconfig.BindingMutationResult, error) {
	return runtimeconfig.BindingMutationResult{}, r.err
}

type reusedKeyRuntimeCredentials struct {
	*fakeRuntimeCredentialManagement
	err error
}

func (r reusedKeyRuntimeCredentials) Create(
	context.Context, credentials.RuntimeCredentialCreateRequest,
) (credentials.RuntimeCredentialCreateResult, error) {
	return credentials.RuntimeCredentialCreateResult{}, r.err
}

type reusedKeyPrincipals struct {
	*fakeRuntimeAgentPrincipalManagement
	err error
}

func (p reusedKeyPrincipals) ReplaceLabels(
	context.Context, string, uint64, []string, string, string, time.Time,
) (controlplane.RuntimeAgentPrincipalProjection, bool, error) {
	return controlplane.RuntimeAgentPrincipalProjection{}, false, p.err
}

func (p reusedKeyPrincipals) Delete(context.Context, string, uint64, string, string, time.Time) (bool, error) {
	return false, p.err
}

type reusedKeyManagedCredentials struct {
	*fakeManagedCredentials
	err error
}

func (c reusedKeyManagedCredentials) Create(context.Context, credentials.CreateRequest) (credentials.CreateResult, error) {
	return credentials.CreateResult{}, c.err
}

func (c reusedKeyManagedCredentials) Delete(context.Context, credentials.DeleteRequest) (credentials.DeleteResult, error) {
	return credentials.DeleteResult{}, c.err
}

// reusedKeyAudits answers every keyed Audit mutation with the reused-key
// error and records which ones the handlers reached.
type reusedKeyAudits struct {
	*fakeAuditManagement
	err     error
	reached map[string]bool
}

func (a *reusedKeyAudits) reach(method string) error {
	if a.reached == nil {
		a.reached = make(map[string]bool)
	}
	a.reached[method] = true
	return a.err
}

func (a *reusedKeyAudits) CreateDraft(context.Context, auditservice.CreateDraftParams) (auditstore.Audit, bool, error) {
	return auditstore.Audit{}, false, a.reach("CreateDraft")
}

func (a *reusedKeyAudits) Start(context.Context, auditservice.StartParams) (auditservice.StartedAudit, error) {
	return auditservice.StartedAudit{}, a.reach("Start")
}

func (a *reusedKeyAudits) Pause(context.Context, auditservice.MutationParams) (auditservice.MutationResult, error) {
	return auditservice.MutationResult{}, a.reach("Pause")
}

func (a *reusedKeyAudits) Resume(context.Context, auditservice.MutationParams) (auditservice.MutationResult, error) {
	return auditservice.MutationResult{}, a.reach("Resume")
}

func (a *reusedKeyAudits) Cancel(context.Context, auditservice.MutationParams) (auditservice.MutationResult, error) {
	return auditservice.MutationResult{}, a.reach("Cancel")
}

func (a *reusedKeyAudits) Delete(context.Context, auditservice.MutationParams) (auditservice.MutationResult, error) {
	return auditservice.MutationResult{}, a.reach("Delete")
}

func (a *reusedKeyAudits) CreateFindingReview(
	context.Context, auditservice.CreateFindingReviewParams,
) (auditservice.FindingReviewResult, error) {
	return auditservice.FindingReviewResult{}, a.reach("CreateFindingReview")
}

func (a *reusedKeyAudits) DecideFinding(
	context.Context, auditservice.DecideFindingParams,
) (auditservice.FindingDecisionResult, error) {
	return auditservice.FindingDecisionResult{}, a.reach("DecideFinding")
}

func (a *reusedKeyAudits) DecideActionReview(
	context.Context, auditservice.DecideActionReviewParams,
) (auditservice.ActionReviewDecisionResult, error) {
	return auditservice.ActionReviewDecisionResult{}, a.reach("DecideActionReview")
}
