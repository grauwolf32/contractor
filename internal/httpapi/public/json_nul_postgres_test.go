//go:build integration

package public

import (
	"bytes"
	"encoding/json"
	"maps"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/findingintake"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// A genuine data exception from PostgreSQL remains permanent even if a future
// request path misses the input check. The response never includes diagnostics.
func TestPostgresJSONNULFailureIsPermanent(t *testing.T) {
	pool := isolatedPublicPool(t, t.Context())
	_, err := pool.Exec(t.Context(), `SELECT $1::jsonb`, `{"note":"private\u0000text"}`)
	if err == nil || persistencepostgres.Classify(err) != persistencepostgres.ClassInvalid {
		t.Fatalf("expected PostgreSQL data exception, got %v", err)
	}
	response := httptest.NewRecorder()
	handleError(response, persistencepostgres.WrapError("write document", err))
	problem := apiDecode[errorResponse](t, response)
	if response.Code != http.StatusBadRequest || problem.Code != "invalid_request" || problem.Retryable {
		t.Fatalf("data exception = %d %+v", response.Code, problem)
	}
}

func nulWriteCounts(t *testing.T, h *evalAPIHarness) map[string]int {
	t.Helper()
	counts := map[string]int{}
	for _, table := range []string{"projects", "workflow_runs", "audits", "mutation_receipts", "finding_proposal_receipts", "audit_review_decisions"} {
		var count int
		if err := h.pool.QueryRow(t.Context(), "SELECT count(*) FROM "+table).Scan(&count); err != nil {
			t.Fatal(err)
		}
		counts[table] = count
	}
	return counts
}

func TestPostgresPublicJSONNULEscapesBeforeSideEffects(t *testing.T) {
	for _, kind := range []string{"project", "run", "project run", "audit"} {
		t.Run(kind, func(t *testing.T) {
			h := newEvalAPIHarness(t)
			service := artifacts.NewService(artifactpgtest.NewStorage().Repository(h.pool))
			user, err := service.User("user-1")
			if err != nil {
				t.Fatal(err)
			}
			project, err := service.Project("ordinary")
			if err != nil {
				t.Fatal(err)
			}
			write := func(store artifacts.ScopedStore, name, media, text string) contracts.ArtifactRef {
				t.Helper()
				value, err := store.Write(t.Context(), contracts.ArtifactRef{Namespace: "inputs", Name: name}, artifacts.Payload{MediaType: media, Data: []byte(text)}, nil)
				if err != nil {
					t.Fatal(err)
				}
				return value.Ref
			}
			var body any
			path, status := "/v1/projects", http.StatusCreated
			switch kind {
			case "project":
				body = map[string]any{"kind": "project", "name": "NUL project", "description": "note\x00text"}
			case "run", "project run":
				path, status = "/v1/runs", http.StatusAccepted
				store := user
				if kind == "project run" {
					path, store = "/v1/projects/ordinary/runs", project
				}
				source := write(store, "source", "text/plain", "source")
				body = map[string]any{"workflow": "artifact-copy@2", "parameters": map[string]string{"objective": "note\x00text"}, "artifacts": map[string]contracts.ArtifactRef{"source": source}}
			case "audit":
				path = "/v1/projects/ordinary/audits"
				source := write(project, "source", "application/zip", "source")
				checklist := write(project, "checklist", "application/json", `{"schema":"contractor.audit.checklist.v1","items":[{"key":"check","version":"1","statement":"Check the source.","applicability":"always","allowed_methods":["static"],"required_evidence":[],"review_policy":"automatic"}]}`)
				body = createAuditRequest{Labels: map[string]string{}, Profile: auditProfileSelectorRequest{Name: "public-checklist", Version: "1"}, Inputs: map[string]contracts.ArtifactRef{"source": source, "checklist": checklist}, RuntimeLabels: []string{}, Scope: auditservice.Scope{Objective: "note\x00text"}}
			}
			before := nulWriteCounts(t, h)
			response := h.request(t, http.MethodPost, path, body, "nul-write", "", http.StatusBadRequest)
			problem := apiDecode[errorResponse](t, response)
			if problem.Code != "invalid_request" || problem.Retryable {
				t.Fatalf("NUL rejection = %+v", problem)
			}
			if after := nulWriteCounts(t, h); !maps.Equal(before, after) {
				t.Fatalf("rejected request wrote rows: before %v after %v", before, after)
			}
			// Reusing the same key proves no receipt was claimed. Escaped
			// backslash text is still valid and reaches the real store.
			raw, err := json.Marshal(body)
			if err != nil {
				t.Fatal(err)
			}
			var corrected any
			if err := json.Unmarshal(bytes.ReplaceAll(raw, []byte(`\u0000`), []byte(`\\u0000`)), &corrected); err != nil {
				t.Fatal(err)
			}
			created := h.request(t, http.MethodPost, path, corrected, "nul-write", "", status)
			if kind == "run" {
				run := apiDecode[createRunResponse](t, created)
				runs := runstore.NewPostgresStore(h.pool, artifactpgtest.NewStorage())
				cancelHandler := &runHandlers{RunDependencies: RunDependencies{Runs: runs, RunLifecycle: runs, RunNotifier: &recordingRunNotifier{}, NewID: func(string) (string, error) { return "resume-nul", nil }, Now: func() time.Time { return time.Date(2026, 10, 25, 12, 0, 0, 0, time.UTC) }}}
				resume := auditAuthenticatedRequest(http.MethodPost, "/v1/runs/"+run.RunID+"/resume", []byte(`{"stageExecutionId":"stage\u0000source"}`))
				resume.SetPathValue("runId", run.RunID)
				resume.Header.Set("Content-Type", "application/json")
				resumed := httptest.NewRecorder()
				serve(cancelHandler.resumeRun)(resumed, resume)
				if problem := apiDecode[errorResponse](t, resumed); resumed.Code != http.StatusBadRequest || problem.Code != "invalid_request" || problem.Retryable {
					t.Fatalf("NUL resumption = %d %+v", resumed.Code, problem)
				}
				cancelRequest := func(reason string) *httptest.ResponseRecorder {
					t.Helper()
					body, err := json.Marshal(map[string]string{"reason": reason})
					if err != nil {
						t.Fatal(err)
					}
					request := auditAuthenticatedRequest(http.MethodPost, "/v1/runs/"+run.RunID+"/cancel", body)
					request.SetPathValue("runId", run.RunID)
					request.Header.Set("Content-Type", "application/json")
					response := httptest.NewRecorder()
					serve(cancelHandler.cancelRun)(response, request)
					return response
				}
				bad := cancelRequest("reason\x00text")
				if problem := apiDecode[errorResponse](t, bad); bad.Code != http.StatusBadRequest || problem.Code != "invalid_request" || problem.Retryable {
					t.Fatalf("NUL cancellation = %d %+v", bad.Code, problem)
				}
				if stored, err := runs.GetRun(t.Context(), run.RunID); err != nil || stored.Cancellation != nil {
					t.Fatalf("NUL cancellation changed Run: %+v %v", stored, err)
				}
				good := cancelRequest(`reason\u0000text`)
				if good.Code != http.StatusAccepted {
					t.Fatalf("literal cancellation = %d %s", good.Code, good.Body.String())
				}
			}
		})
	}
}

func TestPostgresFindingImportRejectsNULEscapeBeforeSideEffects(t *testing.T) {
	h := newEvalAPIHarness(t)
	intake, err := findingintake.New(h.pool, artifactpgtest.NewStorage())
	if err != nil {
		t.Fatal(err)
	}
	handler := findingHandlers{FindingDependencies: FindingDependencies{Transfers: artifactpgtest.NewStorage(), FindingProposals: intake}}
	before := nulWriteCounts(t, h)
	revision := "1"
	body, err := json.Marshal(importFindingProposalRequest{RunID: "source\x00content", Proposal: contracts.ArtifactRef{Namespace: "finding-proposals", Name: "proposal", Revision: &revision}})
	if err != nil {
		t.Fatal(err)
	}
	request := auditAuthenticatedRequest(http.MethodPost, "/v1/audits/audit/finding-proposal-imports", body)
	request.SetPathValue("auditId", "audit")
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	serve(handler.importAuditFindingProposal)(response, request)
	problem := apiDecode[errorResponse](t, response)
	if response.Code != http.StatusBadRequest || problem.Code != "invalid_request" || problem.Retryable || !maps.Equal(before, nulWriteCounts(t, h)) {
		t.Fatalf("NUL finding import = %d %+v", response.Code, problem)
	}
}
