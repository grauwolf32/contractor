package public

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/getkin/kin-openapi/openapi3"
	"github.com/getkin/kin-openapi/routers/gorillamux"
	"github.com/grauwolf32/contractor/internal/auth"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/gitimport"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

func TestPublicRequestAndHistoryContracts(t *testing.T) {
	document := loadPublicOpenAPI(t)
	router, err := gorillamux.NewRouter(document)
	if err != nil {
		t.Fatal(err)
	}

	t.Run("finding-decision-conditions", func(t *testing.T) {
		data, err := os.ReadFile("../../../api/testdata/public/finding-decision-cases.json")
		if err != nil {
			t.Fatal(err)
		}
		var cases []struct {
			Name    string
			Valid   bool
			Request any
		}
		if err := json.Unmarshal(data, &cases); err != nil {
			t.Fatal(err)
		}
		for _, tc := range cases {
			t.Run(tc.Name, func(t *testing.T) {
				for _, name := range []string{"DecideAuditFindingRequest", "DecideAuditReviewRequest"} {
					err := document.Components.Schemas[name].Value.VisitJSON(tc.Request, openapi3.EnableJSONSchema2020())
					if (err == nil) != tc.Valid {
						t.Fatalf("%s validity = %v, want %v: %v", name, err == nil, tc.Valid, err)
					}
				}
			})
		}
	})

	t.Run("empty-label-filter-both-run-lists", func(t *testing.T) {
		fixture := newHandlerFixture(t)
		projectID := "project-labels"
		fixture.projects.projects[projectID] = projectstore.Project{ProjectID: projectID, OwnerID: "user-1", Kind: projectstore.KindProject, Lifecycle: projectstore.LifecycleActive}
		now := time.Now().UTC()
		for id, labels := range map[string]runstore.RunMetadataLabels{
			"run-empty": {"a": "", "triaged": ""}, "run-nonempty": {"a": "yes", "triaged": "yes"}, "run-missing": {},
			"run-equals": {"a": "b=c"}, "run-newline": {"a": "line\nbreak"},
		} {
			fixture.runs.runs[id] = runstore.WorkflowRun{RunID: id, OwnerID: "user-1", ProjectID: &projectID, WorkflowName: "artifact-copy", WorkflowVersion: "1", State: runstore.RunSucceeded, MetadataLabels: labels, CreatedAt: now, UpdatedAt: now}
		}
		for _, path := range []string{"/v1/runs", "/v1/projects/" + projectID + "/runs"} {
			for _, selection := range []struct{ selector, runID string }{
				{"a%3D", "run-empty"}, {"triaged%3D", "run-empty"}, {"a%3Db%3Dc", "run-equals"}, {"a%3Dline%0Abreak", "run-newline"},
			} {
				t.Run(path+"/"+selection.selector, func(t *testing.T) {
					response := serveAndValidatePublicContract(t, router, fixture.handler, newPublicContractRequest(http.MethodGet, path+"?label="+selection.selector, nil), true)
					var page runPageResponse
					if response.Code != http.StatusOK || json.Unmarshal(response.Body.Bytes(), &page) != nil || len(page.Items) != 1 || page.Items[0].RunID != selection.runID {
						t.Fatalf("empty label selection = %d %s", response.Code, response.Body.String())
					}
				})
			}
		}
	})

	t.Run("password-UTF8-bytes", func(t *testing.T) {
		passwordSchema := document.Components.Schemas["LoginRequest"].Value.Properties["password"].Value
		if passwordSchema.MinLength != 0 || passwordSchema.MaxLength != nil ||
			fmt.Sprint(passwordSchema.Extensions["x-min-utf8-bytes"]) != fmt.Sprint(auth.MinimumPasswordBytes) ||
			fmt.Sprint(passwordSchema.Extensions["x-max-utf8-bytes"]) != fmt.Sprint(auth.MaximumPasswordBytes) {
			t.Fatal("password schema must describe the server's byte policy, not character bounds")
		}
		for _, tc := range []struct{ name, password string }{{"minimum", "пароль"}, {"maximum", strings.Repeat("🔐", 256)}} {
			t.Run(tc.name, func(t *testing.T) {
				hash, err := auth.HashPassword([]byte(tc.password))
				if err != nil {
					t.Fatal(err)
				}
				bootstrap, err := auth.NewBootstrap("user-1", "admin", hash)
				if err != nil {
					t.Fatal(err)
				}
				authentication, err := auth.NewService(bootstrap, auth.Options{})
				if err != nil {
					t.Fatal(err)
				}
				fixture := newHandlerFixtureWithAuth(t, "../../config/testdata/valid", authentication, mustTestOrigins(t), false, nil)
				for _, attempt := range []struct {
					password string
					status   int
				}{
					{tc.password, http.StatusOK}, {"парол", http.StatusUnauthorized}, {strings.Repeat("🔐", 257), http.StatusUnauthorized},
				} {
					body, _ := json.Marshal(map[string]string{"username": "admin", "password": attempt.password})
					request := newPublicContractRequest(http.MethodPost, "/v1/auth/login", body)
					request.Header.Del("Authorization")
					request.Header.Set("Origin", testBrowserOrigin)
					request.Header.Set("Content-Type", "application/json")
					response := serveAndValidatePublicContract(t, router, fixture.handler, request, true)
					if response.Code != attempt.status {
						t.Fatalf("%d-byte password HTTP status = %d, want %d", len(attempt.password), response.Code, attempt.status)
					}
					if attempt.status == http.StatusOK && len(response.Result().Cookies()) != 1 {
						t.Fatal("valid password did not establish a session")
					}
				}
			})
		}
	})

	t.Run("git-cookie-and-bearer-mutations", func(t *testing.T) {
		git := &requestContractGit{}
		fixture := newHandlerFixtureWithAuth(t, "../../config/testdata/valid", newTestAuthentication(t), mustTestOrigins(t), false, nil, func(d *Dependencies) { d.GitKeys = git; d.GitImports = git })
		login := loginBrowser(t, fixture.handler, "admin", testAuthPassword, testBrowserOrigin)
		if login.response.Code != http.StatusOK {
			t.Fatal("fixture login failed")
		}
		for _, operation := range []struct {
			method, path, template, body string
			status                       int
		}{
			{http.MethodPut, "/v1/settings/git-key", "/v1/settings/git-key", `{"privateKey":"fixture"}`, http.StatusOK},
			{http.MethodDelete, "/v1/settings/git-key", "/v1/settings/git-key", "", http.StatusNoContent},
			{http.MethodPost, "/v1/artifacts/source/repository/git-import", "/v1/artifacts/{namespace}/{name}/git-import", `{"repositoryUrl":"https://example.test/repository.git"}`, http.StatusCreated},
			{http.MethodPost, "/v1/projects/project-1/artifacts/source/repository/git-import", "/v1/projects/{projectId}/artifacts/{namespace}/{name}/git-import", `{"repositoryUrl":"https://example.test/repository.git"}`, http.StatusCreated},
		} {
			t.Run(operation.method+" "+operation.path, func(t *testing.T) {
				parameters := document.Paths.Value(operation.template).GetOperation(operation.method).Parameters
				for _, expected := range []string{"OptionalOrigin", "OptionalCSRFToken"} {
					found := false
					for _, parameter := range parameters {
						if parameter.Ref == "#/components/parameters/"+expected {
							found = true
						}
					}
					if !found {
						t.Fatalf("missing browser parameter %s", expected)
					}
				}
				for _, mode := range []string{"bearer", "cookie", "missing-origin", "missing-csrf", "wrong-csrf"} {
					t.Run(mode, func(t *testing.T) {
						request := newPublicContractRequest(operation.method, operation.path, []byte(operation.body))
						if operation.body != "" {
							request.Header.Set("Content-Type", "application/json")
						}
						if mode != "bearer" {
							request.Header.Del("Authorization")
							request.AddCookie(login.cookie)
							if mode != "missing-origin" {
								request.Header.Set("Origin", testBrowserOrigin)
							}
							if mode != "missing-csrf" {
								request.Header.Set("X-CSRF-Token", login.session.CSRFToken)
							}
							if mode == "wrong-csrf" {
								request.Header.Set("X-CSRF-Token", strings.Repeat("x", 43))
							}
						}
						before := git.calls
						response := serveAndValidatePublicContract(t, router, fixture.handler, request, true)
						allowed := mode == "bearer" || mode == "cookie"
						wantStatus, wantCalls := http.StatusForbidden, before
						if allowed {
							wantStatus, wantCalls = operation.status, before+1
						}
						if response.Code != wantStatus || git.calls != wantCalls {
							t.Fatalf("mutation status=%d calls=%d; want %d/%d", response.Code, git.calls, wantStatus, wantCalls)
						}
					})
				}
			})
		}
	})

	t.Run("complete-automatic-retry-history", func(t *testing.T) {
		const count = 1025
		fixture := newHandlerFixture(t)
		workflow, err := fixture.configs.Snapshot().Workflow("artifact-copy@1")
		if err != nil {
			t.Fatal(err)
		}
		selectedStage := workflow.Stages["copy"]
		selectedStage.On.Failed = config.TransitionAction{Kind: config.TransitionRetry, Retry: &config.RetryTransition{MaxAttempts: count, Then: config.TransitionAction{Kind: config.TransitionFail}}}
		stage, err := json.Marshal(selectedStage)
		if err != nil {
			t.Fatal(err)
		}
		now := time.Now().UTC()
		fixture.runs.runs["run-history"] = runstore.WorkflowRun{RunID: "run-history", OwnerID: "user-1", WorkflowName: "artifact-copy", WorkflowVersion: "1", State: runstore.RunFailed, RuntimeConfig: runtimeconfig.BuiltInRunSnapshot()}
		for i := 1; i <= count; i++ {
			id := fmt.Sprintf("attempt-%d", i)
			fixture.runs.executions["run-history"] = append(fixture.runs.executions["run-history"], runstore.StageExecution{StageExecutionID: id, RunID: "run-history", StageName: "copy", Attempt: i, StageSpecSnapshot: stage, State: runstore.StageFailed, CreatedAt: now, UpdatedAt: now})
			decision := runstore.StageTransitionDecision{SourceExecutionID: id, RunID: "run-history", Action: runstore.StageTransitionRetry, DecidedAt: now}
			if i == count {
				decision.Action = runstore.StageTransitionFail
			} else {
				next := fmt.Sprintf("attempt-%d", i+1)
				stageName := "copy"
				decision.TargetExecutionID, decision.TargetStageName = &next, &stageName
			}
			fixture.runs.decisions["run-history"] = append(fixture.runs.decisions["run-history"], decision)
		}
		response := serveAndValidatePublicContract(t, router, fixture.handler, newPublicContractRequest(http.MethodGet, "/v1/runs/run-history", nil), true)
		var status runStatusResponse
		if response.Code != http.StatusOK || json.Unmarshal(response.Body.Bytes(), &status) != nil || len(status.Attempts) != count || len(status.Transitions) != count {
			t.Fatalf("history was rejected or truncated: HTTP %d", response.Code)
		}
		if status.Attempts[count-1].Attempt != count || status.Transitions[count-1].Action != runstore.StageTransitionFail {
			t.Fatal("last retained records are missing")
		}
	})
}

// Domain storage and remote Git are faked; public routing and session/CSRF checks
// are real. Import precondition store behavior has a separate gitimport test.
type requestContractGit struct{ calls int }

func (g *requestContractGit) Metadata(context.Context, string) (credentials.GitKeyMetadata, error) {
	return credentials.GitKeyMetadata{}, nil
}
func (g *requestContractGit) Replace(_ context.Context, owner string, _ []byte) (credentials.GitKeyMetadata, error) {
	g.calls++
	return credentials.GitKeyMetadata{Configured: true}, nil
}
func (g *requestContractGit) Delete(context.Context, string) error { g.calls++; return nil }
func (g *requestContractGit) DoImport(_ context.Context, request gitimport.ImportRequest, respond func(gitimport.ImportResult)) error {
	g.calls++
	if request.ExpectedRevision != nil {
		return fmt.Errorf("implicit create unexpectedly acquired an update precondition")
	}
	revision := "revision-1"
	ref := request.Target
	ref.Revision = &revision
	result := gitimport.ImportResult{Artifact: ref, MediaType: "application/zip", Size: 1}
	result.GitSource.RepositoryURL = request.RepositoryURL
	result.GitSource.ResolvedCommit = strings.Repeat("a", 40)
	result.GitSource.ImportedAt = time.Now().UTC()
	respond(result)
	return nil
}
