package public

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifactpolicy"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/configtest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

func TestRunRepeatDraftRetainsExactRequestAndHidesSystemArtifact(t *testing.T) {
	fixture := newHandlerFixtureWithConfig(t, configtest.CopyWithPolicies(t, "../../../testdata/configs"))
	user, _ := fixture.artifacts.User("user-1")
	source, err := user.Write(
		t.Context(), contracts.ArtifactRef{Namespace: "sources", Name: "service"},
		artifacts.Payload{MediaType: "text/plain", Data: []byte("source")}, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	body := []byte(`{"workflow":"artifact-copy@1","labels":{"purpose":"eval","eval.id":"sample-1"},"parameters":{"objective":"copy"},"artifacts":{"source":{"namespace":"sources","name":"service"}},"executionConfig":{"workers":{"modelPolicy":"test-worker@1","credential":null}}}`)
	request := authenticatedRequest(http.MethodPost, "/v1/runs", bytes.NewReader(body))
	created := httptest.NewRecorder()
	fixture.handler.ServeHTTP(created, request)
	if created.Code != http.StatusAccepted {
		t.Fatalf("create Run = %d %s", created.Code, created.Body.String())
	}
	run := fixture.runs.runs["run_fixed"]
	run.State = runstore.RunSucceeded
	run.FinishedAt = timePointer(run.CreatedAt.Add(time.Minute))
	fixture.runs.runs[run.RunID] = run

	response := serveQuery(t, fixture.handler, "/v1/runs/run_fixed/repeat-draft")
	var repeat runRepeatDraftResponse
	decodeQueryResponse(t, response, &repeat)
	if response.Code != http.StatusOK || repeat.Authority != repeatAuthorityOrdinary || repeat.Draft == nil ||
		repeat.Draft.ExecutionConfig.Status != repeatStatusAvailable {
		t.Fatalf("repeat draft = status %d, %+v", response.Code, repeat)
	}
	input := repeat.Draft.Inputs["source"]
	if input.Status != repeatStatusAvailable || input.Artifact == nil || input.Metadata == nil ||
		!sameArtifactRef(*input.Artifact, source.Ref) || !sameArtifactRef(input.Metadata.Ref, source.Ref) {
		t.Fatalf("repeat input = %+v, source = %+v", input, source.Ref)
	}
	encodedPatch, err := json.Marshal(repeat.Draft.ExecutionConfig.Value)
	if err != nil || string(encodedPatch) != `{"workers":{"credential":null,"modelPolicy":"test-worker@1"}}` {
		t.Fatalf("retained executionConfig = %s (%v)", encodedPatch, err)
	}
	if !hasRepeatNotice(repeat.Notices, "evaluation_labels_require_review") {
		t.Fatalf("repeat notices = %+v", repeat.Notices)
	}

	listed := serveQuery(t, fixture.handler, "/v1/runs/run_fixed/artifacts")
	if listed.Code != http.StatusOK || bytes.Contains(listed.Body.Bytes(), []byte(artifactpolicy.RunSystemNamespace)) {
		t.Fatalf("owner Artifact list exposed repeat request = %d %s", listed.Code, listed.Body.String())
	}
	raw := serveQuery(t, fixture.handler,
		"/v1/runs/run_fixed/artifacts/"+artifactpolicy.RunSystemNamespace+"/"+artifactpolicy.RunRepeatRequestName)
	if raw.Code != http.StatusNotFound {
		t.Fatalf("raw repeat request = %d %s", raw.Code, raw.Body.String())
	}
	for _, suffix := range []string{"/metadata", "/versions", "/lineage"} {
		raw = serveQuery(t, fixture.handler,
			"/v1/runs/run_fixed/artifacts/"+artifactpolicy.RunSystemNamespace+"/"+
				artifactpolicy.RunRepeatRequestName+suffix)
		if raw.Code != http.StatusNotFound {
			t.Fatalf("raw repeat request%s = %d %s", suffix, raw.Code, raw.Body.String())
		}
	}
}

func TestRunRepeatDraftRecoversLegacyLineageWithoutGuessingOverrides(t *testing.T) {
	fixture := newHandlerFixture(t)
	user, _ := fixture.artifacts.User("user-1")
	source, err := user.Write(
		t.Context(), contracts.ArtifactRef{Namespace: "sources", Name: "legacy"},
		artifacts.Payload{MediaType: "text/plain", Data: []byte("legacy")}, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	run := queryRun("run-legacy", "user-1", runstore.RunFailed, time.Now().UTC())
	run.PublicationMode = runstore.PublicationOrdinary
	run.RuntimeConfig = runtimeconfig.BuiltInRunSnapshot()
	fixture.runs.runs[run.RunID] = run
	if _, err := fixture.artifacts.ForkInput(t.Context(), "user-1", source.Ref, run.RunID, "source"); err != nil {
		t.Fatal(err)
	}

	response := serveQuery(t, fixture.handler, "/v1/runs/run-legacy/repeat-draft")
	var repeat runRepeatDraftResponse
	decodeQueryResponse(t, response, &repeat)
	if response.Code != http.StatusOK || repeat.Draft == nil ||
		repeat.Draft.ExecutionConfig.Status != repeatStatusUnavailable ||
		repeat.Draft.ExecutionConfig.Value != nil ||
		repeat.Draft.Inputs["source"].Status != repeatStatusAvailable ||
		!hasRepeatNotice(repeat.Notices, "execution_configuration_not_retained") {
		t.Fatalf("legacy repeat draft = status %d, %+v", response.Code, repeat)
	}
}

func TestRunRepeatDraftPreservesAuditAuthorityAndTerminalBoundary(t *testing.T) {
	fixture := newHandlerFixture(t)
	audit := queryRun("run-audit", "user-1", runstore.RunFailed, time.Now().UTC())
	audit.PublicationMode = runstore.PublicationAuditManaged
	audit.MetadataLabels = runstore.RunMetadataLabels{"audit.id": "audit-1", "audit.role": "check"}
	projectID := "project-1"
	audit.ProjectID = &projectID
	fixture.runs.runs[audit.RunID] = audit

	response := serveQuery(t, fixture.handler, "/v1/runs/run-audit/repeat-draft")
	var repeat runRepeatDraftResponse
	decodeQueryResponse(t, response, &repeat)
	if response.Code != http.StatusOK || repeat.Authority != repeatAuthorityAudit || repeat.Draft != nil ||
		repeat.AuditID == nil || *repeat.AuditID != "audit-1" || !hasRepeatNotice(repeat.Notices, "audit_managed_run") {
		t.Fatalf("Audit repeat context = status %d, %+v", response.Code, repeat)
	}

	active := queryRun("run-active", "user-1", runstore.RunRunning, time.Now().UTC())
	fixture.runs.runs[active.RunID] = active
	activeResponse := serveQuery(t, fixture.handler, "/v1/runs/run-active/repeat-draft")
	if activeResponse.Code != http.StatusConflict {
		t.Fatalf("active repeat draft = %d %s", activeResponse.Code, activeResponse.Body.String())
	}
	foreign := queryRun("run-foreign-repeat", "user-2", runstore.RunFailed, time.Now().UTC())
	fixture.runs.runs[foreign.RunID] = foreign
	foreignResponse := serveQuery(t, fixture.handler, "/v1/runs/run-foreign-repeat/repeat-draft")
	if foreignResponse.Code != http.StatusNotFound {
		t.Fatalf("foreign repeat draft = %d %s", foreignResponse.Code, foreignResponse.Body.String())
	}
}

func TestRunRepeatDraftReportsChangedRuntimeBinding(t *testing.T) {
	fixture := newHandlerFixture(t)
	run := queryRun("run-changed-binding", "user-1", runstore.RunCancelled, time.Now().UTC())
	run.PublicationMode = runstore.PublicationOrdinary
	run.RuntimeConfig = runtimeconfig.BuiltInRunSnapshot()
	fixture.runs.runs[run.RunID] = run
	fixture.runtimeConfigs.mu.Lock()
	binding := fixture.runtimeConfigs.bindings[runtimeconfig.DefaultLabel]
	binding.Revision++
	fixture.runtimeConfigs.bindings[runtimeconfig.DefaultLabel] = binding
	fixture.runtimeConfigs.mu.Unlock()

	response := serveQuery(t, fixture.handler, "/v1/runs/run-changed-binding/repeat-draft")
	var repeat runRepeatDraftResponse
	decodeQueryResponse(t, response, &repeat)
	if response.Code != http.StatusOK || !hasRepeatNotice(repeat.Notices, "runtime_binding_changed") {
		t.Fatalf("changed binding repeat draft = status %d, %+v", response.Code, repeat)
	}
}

func TestRunRepeatDraftReportsUnavailableProjectContext(t *testing.T) {
	fixture := newHandlerFixture(t)
	project, _, err := fixture.projects.Create(t.Context(), projectstore.CreateParams{
		ProjectID: "project-repeat", OwnerID: "user-1", Kind: projectstore.KindProject,
		Name: "Repeat", IdempotencyKey: "project-repeat", RequestDigest: "digest-repeat",
	})
	if err != nil {
		t.Fatal(err)
	}
	run := queryRun("run-project-repeat", "user-1", runstore.RunFailed, time.Now().UTC())
	run.PublicationMode = runstore.PublicationOrdinary
	run.RuntimeConfig = runtimeconfig.BuiltInRunSnapshot()
	run.ProjectID = &project.ProjectID
	fixture.runs.runs[run.RunID] = run
	if _, _, err := fixture.projects.BeginDeletion(t.Context(), projectstore.BeginDeletionParams{
		ProjectID: project.ProjectID, OwnerID: run.OwnerID, ExpectedRevision: project.Revision,
	}); err != nil {
		t.Fatal(err)
	}

	response := serveQuery(t, fixture.handler, "/v1/runs/"+run.RunID+"/repeat-draft")
	var repeat runRepeatDraftResponse
	decodeQueryResponse(t, response, &repeat)
	if response.Code != http.StatusOK || !hasRepeatNotice(repeat.Notices, "project_deleting") {
		t.Fatalf("deleting Project repeat draft = status %d, %+v", response.Code, repeat)
	}

	fixture.projects.mu.Lock()
	delete(fixture.projects.projects, project.ProjectID)
	fixture.projects.mu.Unlock()
	response = serveQuery(t, fixture.handler, "/v1/runs/"+run.RunID+"/repeat-draft")
	decodeQueryResponse(t, response, &repeat)
	if response.Code != http.StatusOK || !hasRepeatNotice(repeat.Notices, "project_unavailable") {
		t.Fatalf("missing Project repeat draft = status %d, %+v", response.Code, repeat)
	}
}

func hasRepeatNotice(notices []runRepeatDraftNotice, code string) bool {
	for _, notice := range notices {
		if notice.Code == code {
			return true
		}
	}
	return false
}

func timePointer(value time.Time) *time.Time { return &value }
