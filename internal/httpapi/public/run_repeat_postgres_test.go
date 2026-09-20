package public

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifactpolicy"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auth"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runrepeat"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
)

func TestPostgresRunRepeatRequiresRetainedAuthorityWithoutMutation(t *testing.T) {
	ctx := t.Context()
	pool := isolatedPublicPool(t, ctx)
	fixture := newHandlerFixture(t)
	repository := artifacts.NewPostgresRepository(pool)
	service := artifacts.NewService(repository)
	runs := runstore.NewPostgresStore(pool)
	projects := projectstore.NewPostgresStore(pool)
	h := &handler{dependencies: Dependencies{
		Runs: runs, Artifacts: service, Projects: projects,
		Config: fixture.configs, Credentials: fixture.credentials, RuntimeConfigs: fixture.runtimeConfigs,
	}}
	// Include payload bytes, revisions, clocks, pins, lineage and Run events so
	// repeated projections cannot quietly repair history or create another Run.
	storedState := func() map[string]string {
		t.Helper()
		result := make(map[string]string)
		for _, table := range []string{
			"workflow_runs", "workflow_run_events", "workflow_run_metadata_labels", "projects",
			"artifact_scopes", "artifact_blobs", "artifact_versions", "artifact_binding_revisions",
			"artifact_bindings", "artifact_lineage", "artifact_pins",
		} {
			var state string
			if err := pool.QueryRow(ctx, `SELECT COALESCE(jsonb_agg(to_jsonb(row) ORDER BY to_jsonb(row)::text), '[]'::jsonb)::text FROM `+pgx.Identifier{table}.Sanitize()+` AS row`).Scan(&state); err != nil {
				t.Fatal(err)
			}
			result[table] = state
		}
		return result
	}
	for _, scope := range []string{"user", "project"} {
		t.Run(scope, func(t *testing.T) {
			var projectID *string
			sources, err := service.User("user-1")
			if scope == "project" {
				project, _, createErr := projects.Create(ctx, projectstore.CreateParams{
					ProjectID: "repeat-project", OwnerID: "user-1", Kind: projectstore.KindProject,
					Name: "Repeat Project", IdempotencyKey: "repeat-project", RequestDigest: "sha256:" + strings.Repeat("a", 64),
				})
				if createErr != nil {
					t.Fatal(createErr)
				}
				projectID = &project.ProjectID
				sources, err = service.Project(*projectID)
			}
			if err != nil {
				t.Fatal(err)
			}
			source, err := sources.Write(ctx, contracts.ArtifactRef{Namespace: "sources", Name: "original"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("original")}, nil)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := sources.Write(ctx, contracts.ArtifactRef{Namespace: "sources", Name: "original"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("new current source")}, source.Ref.Revision); err != nil {
				t.Fatal(err)
			}
			for _, variant := range []string{"current", "missing", "malformed", "missing-patch", "null-patch", "wrong-workflow", "wrong-project", "wrong-media-type"} {
				t.Run(variant, func(t *testing.T) {
					runID := "repeat-" + scope + "-" + variant
					if _, err := runs.CreateRun(ctx, runstore.CreateRunParams{
						RunID: runID, OwnerID: "user-1", ProjectID: projectID, WorkflowName: "artifact-copy", WorkflowVersion: "1",
						WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: json.RawMessage(`{}`),
						Parameters: map[string]string{"objective": "original objective"}, RuntimeConfig: runtimeconfig.BuiltInRunSnapshot(),
					}); err != nil {
						t.Fatal(err)
					}
					if projectID == nil {
						_, err = service.ForkInput(ctx, "user-1", source.Ref, runID, "source")
					} else {
						_, err = service.ForkProjectInput(ctx, *projectID, source.Ref, runID, "source")
					}
					if err != nil {
						t.Fatal(err)
					}
					retained, err := runrepeat.Encode(runrepeat.Snapshot{
						Workflow: config.WorkflowRef{Name: "artifact-copy", Version: "1"}, ProjectID: projectID,
						Inputs: map[string]contracts.ArtifactRef{"source": source.Ref},
					})
					if err != nil {
						t.Fatal(err)
					}
					var stored map[string]json.RawMessage
					if err := json.Unmarshal(retained, &stored); err != nil {
						t.Fatal(err)
					}
					switch variant {
					case "missing-patch":
						delete(stored, "executionConfig")
					case "null-patch":
						stored["executionConfig"] = json.RawMessage(`null`)
					case "wrong-workflow":
						stored["workflow"] = json.RawMessage(`{"name":"other","version":"1"}`)
					case "wrong-project":
						stored["projectId"] = json.RawMessage(`"other-project"`)
					}
					retained, err = json.Marshal(stored)
					if err != nil {
						t.Fatal(err)
					}
					if variant == "malformed" {
						retained = []byte("{")
					}
					if variant == "wrong-media-type" {
						runScope, _ := artifacts.RunScope(runID)
						_, err = repository.Write(ctx, runScope, contracts.ArtifactRef{Namespace: artifactpolicy.RunSystemNamespace, Name: artifactpolicy.RunRepeatRequestName}, artifacts.Payload{MediaType: "text/plain", Data: retained}, nil)
					} else if variant != "missing" {
						_, err = service.WriteRunRepeatRequest(ctx, runID, retained)
					}
					if err != nil {
						t.Fatal(err)
					}
					if _, err := runs.TransitionRun(ctx, runID, runstore.RunInitializing, runstore.RunFailed, runstore.Reason{Code: "fixture-complete"}); err != nil {
						t.Fatal(err)
					}
					before := storedState()
					var first []byte
					for range 2 {
						request := httptest.NewRequest(http.MethodGet, "/v1/runs/"+runID+"/repeat-draft", nil).WithContext(auth.WithPrincipal(ctx, auth.Principal{UserID: "user-1"}))
						request.SetPathValue("runID", runID)
						response := httptest.NewRecorder()
						h.getRunRepeatDraft(response, request)
						var repeat runRepeatDraftResponse
						decodeQueryResponse(t, response, &repeat)
						if response.Code != http.StatusOK || repeat.Authority != repeatAuthorityOrdinary {
							t.Fatalf("Repeat = %d %s", response.Code, response.Body.String())
						}
						if variant == "current" {
							if repeat.Draft == nil || repeat.Draft.Parameters["objective"] != "original objective" || !sameOptionalString(repeat.ProjectID, projectID) {
								t.Fatalf("current request lost its draft: %+v", repeat)
							}
							input := repeat.Draft.Inputs["source"]
							patch, err := json.Marshal(repeat.Draft.ExecutionConfig.Value)
							if input.Artifact == nil || !sameArtifactRef(*input.Artifact, source.Ref) || input.Metadata == nil || input.Metadata.Current || input.SourceScope != artifacts.ScopeKind(scope) || err != nil || string(patch) != "{}" {
								t.Fatalf("current request lost exact source/empty patch: %+v, %s (%v)", input, patch, err)
							}
						} else {
							code := "repeat_request_invalid"
							if variant == "missing" {
								code = "repeat_request_unavailable"
							}
							if repeat.Draft != nil || len(repeat.Notices) != 1 || repeat.Notices[0].Code != code || repeat.Notices[0].Severity != "blocking" {
								t.Fatalf("untrusted request produced a draft: %+v", repeat)
							}
						}
						if first != nil && !bytes.Equal(first, response.Body.Bytes()) {
							t.Fatal("repeated projection changed its response")
						}
						first = append([]byte(nil), response.Body.Bytes()...)
						if !reflect.DeepEqual(before, storedState()) {
							t.Fatal("Repeat modified retained state")
						}
					}
				})
			}
		})
	}
}
