package resolved

// LaneClass is the Scheduler lane pool that progresses a StageExecution
// (SCH-R175): `tool` for a Stage that calls no model, `model` for every other
// Stage. It is derived from the pinned Stage spec, never from labels or
// Workflow names.
type LaneClass string

const (
	LaneClassModel LaneClass = "model"
	LaneClassTool  LaneClass = "tool"
)

// LaneClasses lists every lane class in the order the Scheduler starts its
// pools.
var LaneClasses = []LaneClass{LaneClassModel, LaneClassTool}

// Valid reports whether c is a known lane class.
func (c LaneClass) Valid() bool {
	return c == LaneClassModel || c == LaneClassTool
}

// StageLaneClass is the one definition of a Stage's lane class: `tool` when
// its Planner is model-free and every Worker it binds is a tool Worker.
// Retries and escalations change only model selections, never the Planner or
// the Worker templates, so every attempt of a Stage has the same class.
func StageLaneClass(stage ResolvedStage) LaneClass {
	if IsModelFreePlanner(stage.Planner) && StageToolWorkersOnly(stage) {
		return LaneClassTool
	}
	return LaneClassModel
}

// EntryLaneClass is the lane class of workflow's first StageExecution, which
// a new Run carries until the Scheduler creates that StageExecution.
func EntryLaneClass(workflow ResolvedWorkflow) LaneClass {
	return StageLaneClass(workflow.Stages[workflow.EntryStage])
}

// WorkflowLaneClass is the class an Audit counts a child Run of workflow
// against (AUX-R3): `tool` when every Stage of workflow is `tool`, otherwise
// `model`, so a Workflow that scans and then calls a model counts as `model`.
func WorkflowLaneClass(workflow ResolvedWorkflow) LaneClass {
	for _, stage := range workflow.Stages {
		if StageLaneClass(stage) != LaneClassTool {
			return LaneClassModel
		}
	}
	return LaneClassTool
}

// StageToolWorkersOnly reports whether every Worker bound in stage is a tool
// Worker, which calls no model.
func StageToolWorkersOnly(stage ResolvedStage) bool {
	for _, binding := range stage.Agents {
		if !binding.Template.IsToolWorker() {
			return false
		}
	}
	return true
}

// WorkflowToolWorkersOnly reports whether every Worker of every Stage of
// workflow is a tool Worker; such a Run needs no LLM credential for its
// Workers.
func WorkflowToolWorkersOnly(workflow ResolvedWorkflow) bool {
	for _, stage := range workflow.Stages {
		if !StageToolWorkersOnly(stage) {
			return false
		}
	}
	return true
}
