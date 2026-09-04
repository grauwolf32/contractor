package public

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func TestWorkflowQueriesArePaginatedAndSafe(t *testing.T) {
	fixture := newHandlerFixtureWithConfig(t, "../../../configs")

	first := serveQuery(t, fixture.handler, "/v1/workflows?limit=1")
	if first.Code != http.StatusOK {
		t.Fatalf("first Workflow page = %d: %s", first.Code, first.Body.String())
	}
	var page workflowPageResponse
	decodeQueryResponse(t, first, &page)
	if len(page.Items) != 1 || !page.Page.HasMore || page.Page.NextCursor == nil {
		t.Fatalf("first Workflow page = %+v", page)
	}
	second := serveQuery(
		t, fixture.handler, "/v1/workflows?limit=1&cursor="+url.QueryEscape(*page.Page.NextCursor),
	)
	if second.Code != http.StatusOK {
		t.Fatalf("second Workflow page = %d: %s", second.Code, second.Body.String())
	}
	var next workflowPageResponse
	decodeQueryResponse(t, second, &next)
	if len(next.Items) != 1 || next.Items[0].Ref == page.Items[0].Ref {
		t.Fatalf("second Workflow page = %+v", next)
	}

	for _, target := range []string{
		"/v1/workflows?limit=201",
		"/v1/workflows?unknown=true",
		"/v1/workflows?cursor=" + url.QueryEscape(tamperCursor(*page.Page.NextCursor)),
	} {
		response := serveQuery(t, fixture.handler, target)
		if response.Code != http.StatusBadRequest {
			t.Errorf("invalid Workflow query %q = %d: %s", target, response.Code, response.Body.String())
		}
	}

	detail := serveQuery(t, fixture.handler, "/v1/workflows/openapi-from-workspace/versions/4")
	if detail.Code != http.StatusOK {
		t.Fatalf("Workflow detail = %d: %s", detail.Code, detail.Body.String())
	}
	var workflow workflowResourceResponse
	decodeQueryResponse(t, detail, &workflow)
	stage := workflow.Stages["openapi_build"]
	if stage.Objective == "" || stage.Instructions.Ref != "instructions/openapi-build-planner.md" ||
		stage.Instructions.Digest == "" || stage.Agents["builder"].Template.TemplateID != "workspace_openapi_builder" ||
		stage.ExecutionConfig.Agents["builder"].Origins.LLMGateway == "" {
		t.Fatalf("safe Workflow projection = %+v", stage)
	}
	if !workflow.Outputs["openapi"].Primary || workflow.Outputs["openapi_validation_report"].Primary ||
		workflow.Inputs["source"].Primary {
		t.Fatalf("Workflow primary output projection = %+v", workflow.Outputs)
	}
	snapshot, err := config.Load("../../../configs", config.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	resolved, err := snapshot.Workflow("openapi-from-workspace@4")
	if err != nil {
		t.Fatal(err)
	}
	unsafeText := resolved.Stages["openapi_build"].Instructions.Text
	if strings.Contains(detail.Body.String(), unsafeText) || strings.Contains(detail.Body.String(), "llmGatewayUrl") ||
		strings.Contains(detail.Body.String(), "http://") {
		t.Fatalf("Workflow detail leaked instruction text or endpoint: %s", detail.Body.String())
	}
}

func TestRunListCursorFilterAndOwnership(t *testing.T) {
	fixture := newHandlerFixture(t)
	base := time.Date(2026, 8, 30, 12, 0, 0, 0, time.UTC)
	fixture.runs.runs["run-old"] = queryRun("run-old", "user-1", runstore.RunSucceeded, base)
	fixture.runs.runs["run-new"] = queryRun("run-new", "user-1", runstore.RunRunning, base.Add(time.Minute))
	fixture.runs.runs["run-foreign"] = queryRun("run-foreign", "user-2", runstore.RunFailed, base.Add(2*time.Minute))
	old := fixture.runs.runs["run-old"]
	old.MetadataLabels = runstore.RunMetadataLabels{
		"purpose": "eval", "eval.id": "eval_01", "eval.leg": "a", "eval.note": "left=right",
	}
	fixture.runs.runs["run-old"] = old
	newest := fixture.runs.runs["run-new"]
	newest.MetadataLabels = runstore.RunMetadataLabels{"purpose": "eval", "eval.id": "eval_01", "eval.leg": "b"}
	fixture.runs.runs["run-new"] = newest
	foreign := fixture.runs.runs["run-foreign"]
	foreign.MetadataLabels = runstore.RunMetadataLabels{"purpose": "eval", "eval.id": "eval_01", "eval.leg": "a"}
	fixture.runs.runs["run-foreign"] = foreign

	first := serveQuery(t, fixture.handler, "/v1/runs?limit=1")
	var page runPageResponse
	decodeQueryResponse(t, first, &page)
	if first.Code != http.StatusOK || len(page.Items) != 1 || page.Items[0].RunID != "run-new" ||
		!page.Page.HasMore || page.Page.NextCursor == nil {
		t.Fatalf("first Run page = status %d, %+v", first.Code, page)
	}
	second := serveQuery(t, fixture.handler, "/v1/runs?limit=1&cursor="+url.QueryEscape(*page.Page.NextCursor))
	var next runPageResponse
	decodeQueryResponse(t, second, &next)
	if second.Code != http.StatusOK || len(next.Items) != 1 || next.Items[0].RunID != "run-old" || next.Page.HasMore {
		t.Fatalf("second Run page = status %d, %+v", second.Code, next)
	}

	filtered := serveQuery(t, fixture.handler, "/v1/runs?state=succeeded")
	var filteredPage runPageResponse
	decodeQueryResponse(t, filtered, &filteredPage)
	if filtered.Code != http.StatusOK || len(filteredPage.Items) != 1 || filteredPage.Items[0].RunID != "run-old" {
		t.Fatalf("filtered Run page = status %d, %+v", filtered.Code, filteredPage)
	}
	evalQuery := url.Values{"label": {"purpose=eval", "eval.id=eval_01"}, "limit": {"1"}}
	evalFirst := serveQuery(t, fixture.handler, "/v1/runs?"+evalQuery.Encode())
	var evalPage runPageResponse
	decodeQueryResponse(t, evalFirst, &evalPage)
	if evalFirst.Code != http.StatusOK || len(evalPage.Items) != 1 ||
		evalPage.Items[0].RunID != "run-new" || evalPage.Items[0].Labels["eval.leg"] != "b" ||
		!evalPage.Page.HasMore || evalPage.Page.NextCursor == nil {
		t.Fatalf("first exact-label Run page = status %d, %+v", evalFirst.Code, evalPage)
	}
	evalNextQuery := url.Values{
		"label":  {"eval.id=eval_01", "purpose=eval", "purpose=eval"},
		"limit":  {"1"},
		"cursor": {*evalPage.Page.NextCursor},
	}
	evalSecond := serveQuery(t, fixture.handler, "/v1/runs?"+evalNextQuery.Encode())
	var evalNext runPageResponse
	decodeQueryResponse(t, evalSecond, &evalNext)
	if evalSecond.Code != http.StatusOK || len(evalNext.Items) != 1 ||
		evalNext.Items[0].RunID != "run-old" || evalNext.Page.HasMore {
		t.Fatalf("second exact-label Run page = status %d, %+v", evalSecond.Code, evalNext)
	}
	legQuery := url.Values{"label": {"eval.id=eval_01", "eval.leg=a"}}
	leg := serveQuery(t, fixture.handler, "/v1/runs?"+legQuery.Encode())
	var legPage runPageResponse
	decodeQueryResponse(t, leg, &legPage)
	if leg.Code != http.StatusOK || len(legPage.Items) != 1 || legPage.Items[0].RunID != "run-old" {
		t.Fatalf("exact leg Run page = status %d, %+v", leg.Code, legPage)
	}
	equalsQuery := url.Values{"label": {"eval.note=left=right"}}
	equals := serveQuery(t, fixture.handler, "/v1/runs?"+equalsQuery.Encode())
	var equalsPage runPageResponse
	decodeQueryResponse(t, equals, &equalsPage)
	if equals.Code != http.StatusOK || len(equalsPage.Items) != 1 || equalsPage.Items[0].RunID != "run-old" {
		t.Fatalf("selector value containing equals = status %d, %+v", equals.Code, equalsPage)
	}
	contradictionQuery := url.Values{"label": {"eval.leg=a", "eval.leg=b"}}
	contradiction := serveQuery(t, fixture.handler, "/v1/runs?"+contradictionQuery.Encode())
	var empty runPageResponse
	decodeQueryResponse(t, contradiction, &empty)
	if contradiction.Code != http.StatusOK || len(empty.Items) != 0 || empty.Page.HasMore {
		t.Fatalf("contradictory Run label page = status %d, %+v", contradiction.Code, empty)
	}
	mismatchedCursor := serveQuery(
		t, fixture.handler, "/v1/runs?cursor="+url.QueryEscape(*evalPage.Page.NextCursor),
	)
	if mismatchedCursor.Code != http.StatusBadRequest {
		t.Fatalf("label-filter cursor reuse = %d: %s", mismatchedCursor.Code, mismatchedCursor.Body.String())
	}
	overLimit := "/v1/runs?" + strings.TrimPrefix(
		strings.Repeat("&label=purpose%3Deval", runstore.MaxRunMetadataLabels+1), "&",
	)
	for _, target := range []string{
		"/v1/runs?state=unknown",
		"/v1/runs?state=running&cursor=" + url.QueryEscape(*page.Page.NextCursor),
		"/v1/runs?cursor=" + url.QueryEscape(tamperCursor(*page.Page.NextCursor)),
		"/v1/runs?ownerId=user-2",
		"/v1/runs?label=missing-equals",
		"/v1/runs?label=%3Dvalue",
		"/v1/runs?label=purpose%3D",
		"/v1/runs?label=Upper%3Dvalue",
		"/v1/runs?label=contractor.internal%3Dvalue",
		"/v1/runs?label=purpose%3Dbefore%00after",
		overLimit,
	} {
		response := serveQuery(t, fixture.handler, target)
		if response.Code != http.StatusBadRequest {
			t.Errorf("invalid Run query %q = %d: %s", target, response.Code, response.Body.String())
		}
	}
}

func TestRunDetailJoinsImmutableObjectiveTypedPlanInputsAndCursor(t *testing.T) {
	fixture := newHandlerFixture(t)
	snapshot, err := config.Load("../../config/testdata/valid", config.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	workflow, err := snapshot.Workflow("artifact-copy@1")
	if err != nil {
		t.Fatal(err)
	}
	stage := workflow.Stages[workflow.EntryStage]
	stageSnapshot, err := json.Marshal(stage)
	if err != nil {
		t.Fatal(err)
	}
	created := time.Date(2026, 8, 30, 12, 0, 0, 0, time.UTC)
	projectID := "project-detail"
	fixture.runs.runs["run-detail"] = runstore.WorkflowRun{
		RunID: "run-detail", OwnerID: "user-1", WorkflowName: "artifact-copy", WorkflowVersion: "1",
		ProjectID:  &projectID,
		Parameters: map[string]string{"objective": "copy safely"}, State: runstore.RunRunning,
		CreatedAt: created, UpdatedAt: created.Add(time.Minute), StartedAt: timeTestPointer(created.Add(time.Second)),
	}
	user, _ := fixture.artifacts.User("user-1")
	source, err := user.Write(
		t.Context(), contracts.ArtifactRef{Namespace: "projects", Name: "source"},
		artifacts.Payload{MediaType: "text/plain", Data: []byte("source")}, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := fixture.artifacts.ForkInput(t.Context(), "user-1", source.Ref, "run-detail", "source"); err != nil {
		t.Fatal(err)
	}
	sessionID, invocationID := "session-detail", "invocation-detail"
	fixture.runs.executions["run-detail"] = []runstore.StageExecution{{
		StageExecutionID: "stage-detail", RunID: "run-detail", StageName: "copy", Attempt: 1,
		ExecutionConfigVariant: runstore.StageExecutionConfigBase,
		StageSpecSnapshot:      stageSnapshot, State: runstore.StageRunning,
		PlannerSessionID: &sessionID, PlannerInvocationID: &invocationID,
		CreatedAt: created.Add(2 * time.Minute), UpdatedAt: created.Add(3 * time.Minute),
		PlannerStartedAt: timeTestPointer(created.Add(2*time.Minute + time.Second)),
	}}
	fixture.plans.plans["stage-detail"] = planner.PlannerPlanProjection{
		Revision: 1,
		Subtasks: []planner.PlannerSubtask{{
			ID: "0", Objective: "Inspect the exact input", Instructions: "Use the declared tools",
			Status: planner.PlannerSubtaskPending,
		}},
		CurrentSubtaskID: "0",
	}
	fixture.runs.eventCursors["run-detail"] = runstore.WorkflowRunEventCursor{
		Generation: "events-detail", Sequence: 17,
	}
	sourceRevision, targetRevision := "run-output-1", "project-output-1"
	fixture.runs.outputPublications["run-detail"] = []runstore.RunOutputPublication{{
		RunID: "run-detail", ProjectID: projectID, OutputName: "result",
		Status:    runstore.OutputPublicationPublished,
		Source:    contracts.ArtifactRef{Namespace: "outputs", Name: "result", Revision: &sourceRevision},
		Target:    &contracts.ArtifactRef{Namespace: "outputs", Name: "result", Revision: &targetRevision},
		CreatedAt: created.Add(4 * time.Minute),
	}}

	response := serveQuery(t, fixture.handler, "/v1/runs/run-detail")
	if response.Code != http.StatusOK {
		t.Fatalf("Run detail = %d: %s", response.Code, response.Body.String())
	}
	var result runStatusResponse
	decodeQueryResponse(t, response, &result)
	if result.Parameters["objective"] != "copy safely" || result.Inputs["source"].Revision == nil ||
		result.EventCursor == nil || result.EventCursor.Sequence != "17" ||
		result.ActiveStageExecutionID == nil || *result.ActiveStageExecutionID != "stage-detail" ||
		len(result.Attempts) != 1 || result.Attempts[0].Objective != stage.Objective ||
		result.Attempts[0].Plan == nil || result.Attempts[0].Plan.Revision != 1 ||
		result.Attempts[0].ExecutionConfig.Agents["builder"].Origins.ModelPolicy == "" ||
		len(result.OutputPublications) != 1 ||
		result.OutputPublications[0].Status != runstore.OutputPublicationPublished ||
		result.OutputPublications[0].Target == nil {
		t.Fatalf("joined Run detail = %+v", result)
	}
	if strings.Contains(response.Body.String(), stage.Instructions.Text) || strings.Contains(response.Body.String(), "worker-model") ||
		strings.Contains(response.Body.String(), "llmGatewayUrl") {
		t.Fatalf("Run detail leaked private resolved values: %s", response.Body.String())
	}
}

func TestArtifactQueriesPreserveExactHistoryLineageAndRunOwnership(t *testing.T) {
	fixture := newHandlerFixture(t)
	user, _ := fixture.artifacts.User("user-1")
	first, err := user.Write(
		t.Context(), contracts.ArtifactRef{Namespace: "projects", Name: "source"},
		artifacts.Payload{MediaType: "text/plain", Data: []byte("first")}, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	second, err := user.Write(
		t.Context(), contracts.ArtifactRef{Namespace: "projects", Name: "source"},
		artifacts.Payload{MediaType: "text/plain", Data: []byte("second")}, first.Ref.Revision,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := user.Write(
		t.Context(), contracts.ArtifactRef{Namespace: "projects", Name: "other"},
		artifacts.Payload{MediaType: "application/json", Data: []byte(`{}`)}, nil,
	); err != nil {
		t.Fatal(err)
	}
	fixture.runs.runs["run-artifacts"] = queryRun(
		"run-artifacts", "user-1", runstore.RunRunning,
		time.Date(2026, 8, 30, 12, 0, 0, 0, time.UTC),
	)
	if _, err := fixture.artifacts.ForkInput(
		t.Context(), "user-1", first.Ref, "run-artifacts", "source",
	); err != nil {
		t.Fatal(err)
	}

	list := serveQuery(t, fixture.handler, "/v1/artifacts?namespace=projects&limit=1")
	var page artifactPageResponse
	decodeQueryResponse(t, list, &page)
	if list.Code != http.StatusOK || len(page.Items) != 1 || !page.Page.HasMore || page.Page.NextCursor == nil {
		t.Fatalf("Artifact page = status %d, %+v", list.Code, page)
	}
	next := serveQuery(
		t, fixture.handler,
		"/v1/artifacts?namespace=projects&limit=1&cursor="+url.QueryEscape(*page.Page.NextCursor),
	)
	var nextPage artifactPageResponse
	decodeQueryResponse(t, next, &nextPage)
	if next.Code != http.StatusOK || len(nextPage.Items) != 1 || nextPage.Items[0].Ref.Name == page.Items[0].Ref.Name {
		t.Fatalf("second Artifact page = status %d, %+v", next.Code, nextPage)
	}

	metadata := serveQuery(
		t, fixture.handler, "/v1/artifacts/projects/source/metadata?revision="+url.QueryEscape(*first.Ref.Revision),
	)
	var exact artifacts.Metadata
	decodeQueryResponse(t, metadata, &exact)
	if metadata.Code != http.StatusOK || exact.Current || exact.Ref.Revision == nil ||
		*exact.Ref.Revision != *first.Ref.Revision || exact.Size != 5 {
		t.Fatalf("exact Artifact metadata = status %d, %+v", metadata.Code, exact)
	}
	versions := serveQuery(t, fixture.handler, "/v1/artifacts/projects/source/versions?limit=1")
	var history artifactPageResponse
	decodeQueryResponse(t, versions, &history)
	if versions.Code != http.StatusOK || len(history.Items) != 1 || !history.Items[0].Current ||
		history.Items[0].Ref.Revision == nil || *history.Items[0].Ref.Revision != *second.Ref.Revision ||
		!history.Page.HasMore {
		t.Fatalf("Artifact history = status %d, %+v", versions.Code, history)
	}
	lineage := serveQuery(
		t, fixture.handler, "/v1/artifacts/projects/source/lineage?revision="+url.QueryEscape(*first.Ref.Revision),
	)
	var provenance artifactLineagePageResponse
	decodeQueryResponse(t, lineage, &provenance)
	if lineage.Code != http.StatusOK || len(provenance.Items) != 1 ||
		provenance.Items[0].Kind != artifacts.LineageInputFork ||
		provenance.Items[0].SourceScope != artifacts.ScopeUser ||
		provenance.Items[0].TargetScope != artifacts.ScopeRun ||
		strings.Contains(lineage.Body.String(), "user-1") || strings.Contains(lineage.Body.String(), "run-artifacts") {
		t.Fatalf("Artifact lineage = status %d, body %s", lineage.Code, lineage.Body.String())
	}

	runList := serveQuery(t, fixture.handler, "/v1/runs/run-artifacts/artifacts")
	var runPage artifactPageResponse
	decodeQueryResponse(t, runList, &runPage)
	if runList.Code != http.StatusOK || len(runPage.Items) != 1 || runPage.Items[0].Ref.Namespace != "inputs" {
		t.Fatalf("RunScope Artifact page = status %d, %+v", runList.Code, runPage)
	}
	download := serveQuery(t, fixture.handler, "/v1/runs/run-artifacts/artifacts/inputs/source")
	if download.Code != http.StatusOK || download.Body.String() != "first" || download.Header().Get("ETag") == "" {
		t.Fatalf("RunScope download = %d, headers %v, body %q", download.Code, download.Header(), download.Body.String())
	}
	for _, runID := range []string{"lineage-a", "lineage-b"} {
		if _, err := fixture.artifacts.ForkInput(
			t.Context(), "user-1", second.Ref, runID, "source",
		); err != nil {
			t.Fatal(err)
		}
	}
	currentLineage := serveQuery(t, fixture.handler, "/v1/artifacts/projects/source/lineage?limit=1")
	var firstLineagePage artifactLineagePageResponse
	decodeQueryResponse(t, currentLineage, &firstLineagePage)
	if currentLineage.Code != http.StatusOK || len(firstLineagePage.Items) != 1 ||
		!firstLineagePage.Page.HasMore || firstLineagePage.Page.NextCursor == nil {
		t.Fatalf("current lineage page = %d, %+v", currentLineage.Code, firstLineagePage)
	}
	if _, err := user.Write(
		t.Context(), contracts.ArtifactRef{Namespace: "projects", Name: "source"},
		artifacts.Payload{MediaType: "text/plain", Data: []byte("third")}, second.Ref.Revision,
	); err != nil {
		t.Fatal(err)
	}
	continuedLineage := serveQuery(
		t, fixture.handler,
		"/v1/artifacts/projects/source/lineage?limit=1&cursor="+
			url.QueryEscape(*firstLineagePage.Page.NextCursor),
	)
	var secondLineagePage artifactLineagePageResponse
	decodeQueryResponse(t, continuedLineage, &secondLineagePage)
	if continuedLineage.Code != http.StatusOK || len(secondLineagePage.Items) != 1 ||
		secondLineagePage.Items[0].Source.Revision == nil ||
		*secondLineagePage.Items[0].Source.Revision != *second.Ref.Revision {
		t.Fatalf("lineage cursor did not pin selected revision = %d, %+v", continuedLineage.Code, secondLineagePage)
	}

	malformed := serveQuery(t, fixture.handler, "/v1/artifacts/projects/source/metadata?revision=bad+revision")
	if malformed.Code != http.StatusBadRequest {
		t.Fatalf("malformed revision = %d: %s", malformed.Code, malformed.Body.String())
	}
	stored := fixture.runs.runs["run-artifacts"]
	stored.OwnerID = "user-2"
	fixture.runs.runs["run-artifacts"] = stored
	queriesBefore := fixture.repository.queryReads
	foreign := serveQuery(t, fixture.handler, "/v1/runs/run-artifacts/artifacts/inputs/source/metadata")
	if foreign.Code != http.StatusNotFound || fixture.repository.queryReads != queriesBefore {
		t.Fatalf("foreign RunScope query = %d, reads before/after %d/%d", foreign.Code, queriesBefore, fixture.repository.queryReads)
	}
}

func TestQueryRoutesDoNotImplicitlyServeHEAD(t *testing.T) {
	fixture := newHandlerFixture(t)
	for _, target := range []string{
		"/v1/workflows",
		"/v1/workflows/artifact-copy/versions/1",
		"/v1/runs",
		"/v1/artifacts",
		"/v1/artifacts/projects/source/metadata",
		"/v1/artifacts/projects/source/versions",
		"/v1/artifacts/projects/source/lineage",
		"/v1/runs/missing/artifacts",
		"/v1/runs/missing/artifacts/inputs/source",
		"/v1/runs/missing/artifacts/inputs/source/metadata",
		"/v1/runs/missing/artifacts/inputs/source/versions",
		"/v1/runs/missing/artifacts/inputs/source/lineage",
	} {
		request := authenticatedRequest(http.MethodHead, target, bytes.NewReader(nil))
		response := httptest.NewRecorder()
		fixture.handler.ServeHTTP(response, request)
		if response.Code != http.StatusMethodNotAllowed {
			t.Errorf("HEAD %s = %d, want 405", target, response.Code)
		}
	}
}

func serveQuery(t *testing.T, handler http.Handler, target string) *httptest.ResponseRecorder {
	t.Helper()
	request := authenticatedRequest(http.MethodGet, target, bytes.NewReader(nil))
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	return response
}

func decodeQueryResponse(t *testing.T, response *httptest.ResponseRecorder, target any) {
	t.Helper()
	if err := json.Unmarshal(response.Body.Bytes(), target); err != nil {
		t.Fatalf("decode query response %d %q: %v", response.Code, response.Body.String(), err)
	}
}

func queryRun(
	runID, ownerID string, state runstore.WorkflowRunState, createdAt time.Time,
) runstore.WorkflowRun {
	return runstore.WorkflowRun{
		RunID: runID, OwnerID: ownerID, WorkflowName: "artifact-copy", WorkflowVersion: "1",
		Parameters: map[string]string{}, State: state, CreatedAt: createdAt, UpdatedAt: createdAt,
	}
}

func tamperCursor(value string) string {
	if strings.HasSuffix(value, "A") {
		return value[:len(value)-1] + "B"
	}
	return value[:len(value)-1] + "A"
}

func timeTestPointer(value time.Time) *time.Time { return &value }
