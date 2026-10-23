package resolved

import (
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func laneTestBinding(runtimeID string) ResolvedAgentBinding {
	return ResolvedAgentBinding{Template: contracts.ResolvedAgentTemplate{
		Runtime: contracts.WorkerRuntimeRef{RuntimeID: runtimeID, Version: "1"},
	}}
}

func laneTestStage(plannerID string, runtimes ...string) ResolvedStage {
	stage := ResolvedStage{
		Planner: PlannerRef{PlannerID: plannerID, Version: "1"},
		Agents:  map[string]ResolvedAgentBinding{},
	}
	for index, runtimeID := range runtimes {
		stage.Agents[string(rune('a'+index))] = laneTestBinding(runtimeID)
	}
	return stage
}

// Only a model-free Planner over tool Workers alone makes a Stage `tool`
// (SCH-R175); every other Planner, an ADK Worker or a mix is `model`.
func TestStageLaneClass(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		name  string
		stage ResolvedStage
		want  LaneClass
	}{
		{name: "passthrough tool", stage: laneTestStage("passthrough", "tool"), want: LaneClassTool},
		{name: "passthrough adk", stage: laneTestStage("passthrough", "adk"), want: LaneClassModel},
		{name: "scan-plan tools", stage: laneTestStage("scan-plan", "tool", "tool"), want: LaneClassTool},
		{name: "scan-plan mixed", stage: laneTestStage("scan-plan", "tool", "adk"), want: LaneClassModel},
		{name: "streamline tool", stage: laneTestStage("streamline", "tool"), want: LaneClassModel},
		{name: "streamline adk", stage: laneTestStage("streamline", "adk"), want: LaneClassModel},
		{name: "router tools", stage: laneTestStage("router", "tool", "tool"), want: LaneClassModel},
		{name: "router mixed", stage: laneTestStage("router", "tool", "adk"), want: LaneClassModel},
		{name: "unknown planner", stage: laneTestStage("scan-plan-next", "tool"), want: LaneClassModel},
		{name: "tool runtime of another version", stage: ResolvedStage{
			Planner: PlannerRef{PlannerID: "scan-plan", Version: "1"},
			Agents: map[string]ResolvedAgentBinding{"scanner": {Template: contracts.ResolvedAgentTemplate{
				Runtime: contracts.WorkerRuntimeRef{RuntimeID: "tool", Version: "2"},
			}}},
		}, want: LaneClassModel},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			if got := StageLaneClass(test.stage); got != test.want || !got.Valid() {
				t.Fatalf("StageLaneClass = %q, want %q", got, test.want)
			}
		})
	}
}

// An escalated attempt carries another execution configuration but the same
// Planner and Worker templates, so it keeps its Stage's class.
func TestStageLaneClassIgnoresExecutionConfiguration(t *testing.T) {
	t.Parallel()
	for _, stage := range []ResolvedStage{laneTestStage("scan-plan", "tool"), laneTestStage("streamline", "adk")} {
		escalated := stage
		escalated.ExecutionConfig = ResolvedStageExecutionConfig{
			Planner: &ResolvedConsumerExecutionConfig{},
			Agents:  map[string]ResolvedConsumerExecutionConfig{"a": {}},
		}
		if StageLaneClass(escalated) != StageLaneClass(stage) {
			t.Fatalf("escalated %s Stage changed class", stage.Planner.PlannerID)
		}
	}
}

// A Run's first class is its entry Stage's; credential pinning asks whether
// every Stage has only tool Workers, whatever its Planner.
func TestWorkflowLaneClassAndToolWorkers(t *testing.T) {
	t.Parallel()
	workflow := ResolvedWorkflow{EntryStage: "scan", Stages: map[string]ResolvedStage{
		"scan":   laneTestStage("scan-plan", "tool", "tool"),
		"triage": laneTestStage("streamline", "adk"),
	}}
	if got := EntryLaneClass(workflow); got != LaneClassTool {
		t.Fatalf("scan entry Stage class = %q", got)
	}
	// An Audit counts a scan-then-triage Workflow as model work (AUX-R3).
	if got := WorkflowLaneClass(workflow); got != LaneClassModel {
		t.Fatalf("scan-then-triage Workflow class = %q", got)
	}
	scans := ResolvedWorkflow{EntryStage: "scan", Stages: map[string]ResolvedStage{
		"scan": laneTestStage("scan-plan", "tool", "tool"), "fetch": laneTestStage("passthrough", "tool"),
	}}
	if got := WorkflowLaneClass(scans); got != LaneClassTool {
		t.Fatalf("model-free Workflow class = %q", got)
	}
	if WorkflowToolWorkersOnly(workflow) {
		t.Fatal("a Workflow with an ADK Worker reported tool Workers only")
	}
	workflow.EntryStage = "triage"
	if got := EntryLaneClass(workflow); got != LaneClassModel {
		t.Fatalf("triage entry Stage class = %q", got)
	}
	workflow.Stages["triage"] = laneTestStage("router", "tool")
	if !WorkflowToolWorkersOnly(workflow) || EntryLaneClass(workflow) != LaneClassModel {
		t.Fatal("a router over tool Workers needs no Worker credential but stays in the model pool")
	}
	for _, class := range []LaneClass{"", "scan", "Model"} {
		if class.Valid() {
			t.Fatalf("lane class %q is valid", class)
		}
	}
}
