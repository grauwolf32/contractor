package public

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func TestPrecomputedAnalysisRunInputsAreExplicitAndStrict(t *testing.T) {
	t.Parallel()

	snapshot, err := config.Load("../../../configs", config.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	for _, workflowRef := range []string{"openapi-from-analysis@1", "likec4-from-analysis@1"} {
		t.Run(workflowRef, func(t *testing.T) {
			workflow, workflowErr := snapshot.Workflow(workflowRef)
			if workflowErr != nil {
				t.Fatal(workflowErr)
			}
			request := createRunRequest{
				Workflow:   workflowRef,
				Parameters: map[string]string{"objective": "Reuse reviewed analysis"},
				Artifacts: map[string]contracts.ArtifactRef{
					"source":            {Namespace: "projects", Name: "source", Revision: analysisRevision("source-r1")},
					"dependency_report": {Namespace: "projects", Name: "dependencies", Revision: analysisRevision("deps-r1")},
					"project_report":    {Namespace: "projects", Name: "project", Revision: analysisRevision("project-r1")},
				},
			}
			if err := validateRunInputs(workflow, request); err != nil {
				t.Fatalf("complete input contract rejected: %v", err)
			}
			delete(request.Artifacts, "dependency_report")
			if err := validateRunInputs(workflow, request); err == nil {
				t.Fatal("missing dependency report was accepted")
			}
			if acceptsMediaType(workflow.Inputs["dependency_report"].MediaTypes, "text/plain") ||
				!acceptsMediaType(workflow.Inputs["dependency_report"].MediaTypes, "text/markdown") {
				t.Fatal("dependency report media-type contract is not strict")
			}
		})
	}
}

func analysisRevision(value string) *string { return &value }

func TestPrecomputedAnalysisRunForksReportsAndRejectsWrongMediaType(t *testing.T) {
	t.Parallel()

	for _, test := range []struct {
		name             string
		dependencyType   string
		wantStatus       int
		wantNotification int
	}{
		{
			name: "complete exact inputs", dependencyType: "text/markdown",
			wantStatus: http.StatusAccepted, wantNotification: 1,
		},
		{
			name: "dependency report has wrong media type", dependencyType: "text/plain",
			wantStatus: http.StatusBadRequest,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			fixture := newHandlerFixtureWithConfig(t, "../../../configs")
			user, err := fixture.artifacts.User("user-1")
			if err != nil {
				t.Fatal(err)
			}
			inputs := map[string]contracts.ArtifactRef{}
			for _, input := range []struct {
				slot, name, mediaType, data string
			}{
				{slot: "source", name: "source", mediaType: "application/zip", data: "zip"},
				{slot: "dependency_report", name: "dependencies", mediaType: test.dependencyType, data: "dependencies"},
				{slot: "project_report", name: "project", mediaType: "text/markdown", data: "project"},
			} {
				written, writeErr := user.Write(
					t.Context(), contracts.ArtifactRef{Namespace: "projects", Name: input.name},
					artifacts.Payload{MediaType: input.mediaType, Data: []byte(input.data)}, nil,
				)
				if writeErr != nil {
					t.Fatal(writeErr)
				}
				inputs[input.slot] = written.Ref
			}
			body, err := json.Marshal(createRunRequest{
				Workflow: "openapi-from-analysis@1", Artifacts: inputs,
			})
			if err != nil {
				t.Fatal(err)
			}
			request := authenticatedRequest(http.MethodPost, "/v1/runs", bytes.NewReader(body))
			response := httptest.NewRecorder()
			fixture.handler.ServeHTTP(response, request)

			if response.Code != test.wantStatus || fixture.notifier.calls != test.wantNotification {
				t.Fatalf(
					"create Run = status %d, notifications %d, body %s",
					response.Code, fixture.notifier.calls, response.Body.String(),
				)
			}
			if response.Code != http.StatusAccepted {
				assertErrorCode(t, response, "invalid_request")
				return
			}
			if fixture.runs.runs["run_fixed"].State != runstore.RunRunning {
				t.Fatalf("Run state = %q", fixture.runs.runs["run_fixed"].State)
			}
			runArtifacts, _ := fixture.artifacts.Run("run_fixed")
			for slot, input := range inputs {
				forked, readErr := runArtifacts.Read(
					t.Context(), contracts.ArtifactRef{Namespace: "inputs", Name: slot},
				)
				if readErr != nil || forked.Ref.Revision == nil || *forked.Ref.Revision == *input.Revision {
					t.Fatalf("forked input %q = (%+v, %v)", slot, forked, readErr)
				}
			}
		})
	}
}
