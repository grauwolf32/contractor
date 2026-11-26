package privateartifacts

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/contracts"
)

type plannerOperationsFunc func(context.Context, string, string, contracts.PlannerOperation) (contracts.PlannerOperationResult, error)

func (f plannerOperationsFunc) ExecuteOperation(ctx context.Context, stage, allocation string, request contracts.PlannerOperation) (contracts.PlannerOperationResult, error) {
	return f(ctx, stage, allocation, request)
}

func TestPrivatePlannerOperationsUseAllocationAuthorityAndRejectMalformedRequests(t *testing.T) {
	const body = `{"apiVersion":"contractor/v1alpha1","sequence":1,"name":"before_model","arguments":{}}`
	for _, test := range []struct {
		name, instance, query, body string
		lost, fenced, trusted       bool
		want                        int
	}{
		{name: "admitted", trusted: true, want: 200},
		{name: "missing-mtls", want: 401},
		{name: "old-instance", instance: "runtime-old", trusted: true, want: 409},
		{name: "lost", lost: true, trusted: true, want: 409},
		{name: "ending", fenced: true, trusted: true, want: 409},
		{name: "scope-query", query: "?stageExecutionId=other-stage", trusted: true, want: 400},
		{name: "duplicate", body: strings.Replace(body, `"sequence":1`, `"sequence":1,"sequence":1`, 1), trusted: true, want: 400},
		{name: "unknown", body: strings.Replace(body, `"sequence":1`, `"Sequence":1`, 1), trusted: true, want: 400},
		{name: "null", body: strings.Replace(body, `"arguments":{}`, `"arguments":null`, 1), trusted: true, want: 400},
	} {
		t.Run(test.name, func(t *testing.T) {
			grant := testGrant("run-a")
			grant.Lost, grant.WriteFenced = test.lost, test.fenced
			registry := &fakeRegistry{grant: grant}
			calls := 0
			operations := plannerOperationsFunc(func(ctx context.Context, stage, allocation string, request contracts.PlannerOperation) (contracts.PlannerOperationResult, error) {
				calls++
				if stage != grant.StageExecutionID || allocation != grant.AllocationID || request.Sequence != 1 {
					t.Fatal("Planner operation trusted request-selected authority")
				}
				// A Registry read here also proves that the callback executes
				// outside WithWriteGrant's mutex, as long dispatches must.
				if _, err := registry.GetGrant(allocation); err != nil {
					t.Fatal(err)
				}
				return contracts.PlannerOperationResult{APIVersion: contracts.APIVersion, Result: json.RawMessage(`{}`)}, nil
			})
			handler, err := NewHandler(Dependencies{Transfers: artifactpgtest.NewRuntime(), Registry: registry, Artifacts: artifacts.NewService(newMemoryRepository()), PlannerOperations: operations})
			if err != nil {
				t.Fatal(err)
			}
			data := body
			if test.body != "" {
				data = test.body
			}
			request := trustedRequest(http.MethodPost, "/private/v1/allocations/allocation-1/planner-operations"+test.query, strings.NewReader(data))
			request.Header.Set("Content-Type", "application/json")
			if !test.trusted {
				request.TLS = nil
			}
			if test.instance != "" {
				request.Header.Set(RuntimeInstanceHeader, test.instance)
			}
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, request)
			wantCalls := 0
			if test.want == 200 {
				wantCalls = 1
			}
			if response.Code != test.want || calls != wantCalls {
				t.Fatalf("status=%d calls=%d body=%s, want %d/%d", response.Code, calls, response.Body.String(), test.want, wantCalls)
			}
		})
	}
}
