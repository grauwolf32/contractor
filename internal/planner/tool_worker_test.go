package planner

import (
	"reflect"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestPassthroughToolWorkflowBindsTargetAndExactReport(t *testing.T) {
	snapshot, err := config.Load("../../configs/scan", config.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	workflow, err := snapshot.Workflow("nuclei-target@1")
	if err != nil {
		t.Fatal(err)
	}
	stage := workflow.Stages["scan"]
	template := stage.Agents["scanner"].Template
	revision := "scan-report-r1"
	report := contracts.ArtifactRef{Namespace: "scanner", Name: "report", Revision: &revision}
	worker := &recordingWorker{result: workerCompletionFromCandidate(contracts.StageContentResult{
		APIVersion: contracts.APIVersion, Outcome: contracts.StageSucceeded, Summary: "Scan completed",
		Artifacts: map[string]contracts.ArtifactRef{"report": report},
	})}
	factory, err := NewPassthroughFactory(&memorySessions{}, worker, &fakeInspector{mediaTypes: map[string]string{
		"scanner/report/scan-report-r1": "application/json",
	}})
	if err != nil {
		t.Fatal(err)
	}
	planner, err := factory.Create(Invocation{
		RunID: "scan-run", StageExecutionID: "scan-stage", Stage: stage,
		Context: StageContext{Parameters: map[string]string{"target": "https://fixture.invalid"}, Artifacts: map[string]*contracts.ArtifactRef{}},
		Workers: map[string]contracts.WorkerHandle{"scanner": {
			AllocationID: "scan-allocation", AgentTemplateRef: template.Ref, WorkerRuntimeRef: template.Runtime,
			AgentCard: map[string]any{"name": "scanner"}, LeaseExpiresAt: time.Now().Add(time.Minute),
		}}, Deadline: time.Now().Add(time.Minute),
	})
	if err != nil {
		t.Fatal(err)
	}
	result, err := planner.Run(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if result.Outcome != contracts.StageSucceeded || !reflect.DeepEqual(result.Artifacts["report"], report) || worker.calls != 1 || worker.binding != "scanner" || worker.request.Parameters["target"] != "https://fixture.invalid" || worker.request.ResultArtifacts["report"] != (contracts.ArtifactRef{Namespace: "scanner", Name: "report"}) {
		t.Fatalf("tool passthrough: result=%+v request=%+v", result, worker.request)
	}
}
