package public

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

func TestRunStatusExposesCompletionAsPublicationNotAuditAcceptance(t *testing.T) {
	fixture := newHandlerFixture(t)
	snapshot, err := config.Load("../../config/testdata/valid", config.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	workflow, err := snapshot.Workflow("artifact-copy@1")
	if err != nil {
		t.Fatal(err)
	}
	stage, err := json.Marshal(workflow.Stages[workflow.EntryStage])
	if err != nil {
		t.Fatal(err)
	}
	fixture.runs.runs["run-completion"] = runstore.WorkflowRun{RunID: "run-completion", OwnerID: "user-1", WorkflowName: "artifact-copy", WorkflowVersion: "1", State: runstore.RunSucceeded}
	fixture.runs.executions["run-completion"] = []runstore.StageExecution{{StageExecutionID: "stage-completion", RunID: "run-completion", StageName: "copy", Attempt: 1, ExecutionConfigVariant: runstore.StageExecutionConfigBase, StageSpecSnapshot: stage, State: runstore.StageSucceeded}}
	for _, phase := range []string{"published", "failed"} {
		t.Run(phase, func(t *testing.T) {
			completion := &contracts.WorkerCompletionDiagnostics{Kind: contracts.AuditCheckResultsV1, Phase: phase, AcceptedCount: 2, TotalCount: 2, ReminderCount: 0}
			if phase == "failed" {
				completion.FailureCode = "audit_result_publication_conflict"
			}
			report := contracts.ExecutionReport{ReportID: "private-report-id", Complete: true, Metrics: contracts.ExecutionMetrics{Tools: map[string]contracts.ToolMetrics{}}, ToolCalls: []contracts.ToolCallRecord{}, Errors: []contracts.ExecutionError{}, Completion: completion}
			fixture.metrics.records["stage-completion"] = telemetry.StageMetricsRecord{StageExecutionID: "stage-completion", Metrics: contracts.StageMetrics{Workers: map[string]contracts.ExecutionReport{"checker": report}, Runtime: map[string]contracts.RuntimeReport{}}}
			response := httptest.NewRecorder()
			fixture.handler.ServeHTTP(response, authenticatedRequest(http.MethodGet, "/v1/runs/run-completion", bytes.NewReader(nil)))
			if response.Code != http.StatusOK {
				t.Fatalf("status %d: %s", response.Code, response.Body.String())
			}
			var result runStatusResponse
			if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
				t.Fatal(err)
			}
			diagnostics := result.Attempts[0].Diagnostics
			if diagnostics == nil || len(diagnostics.Items) != 1 || diagnostics.Items[0].Completion == nil || diagnostics.Items[0].Completion.Phase != phase {
				t.Fatal("public view lost completion facts")
			}
			if strings.Contains(response.Body.String(), "private-report-id") {
				t.Fatal("private report identity leaked")
			}
			if phase == "published" && !strings.Contains(diagnostics.Items[0].Message, "acceptance is separate") {
				t.Fatal("technical Run success was presented as Audit acceptance")
			}
		})
	}
}
