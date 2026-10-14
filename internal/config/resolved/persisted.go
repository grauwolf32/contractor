package resolved

import (
	"encoding/json"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/strictjson"
)

// MaxWorkflowSnapshotBytes is LIM-CAT-4: the largest Workflow snapshot a Run
// may store, counted in bytes of the encoding EncodeResolvedWorkflowSnapshot
// produces (CAT-R76).
const MaxWorkflowSnapshotBytes = 2 << 20

// ErrWorkflowSnapshotTooLarge classifies a resolved Workflow whose snapshot
// exceeds MaxWorkflowSnapshotBytes. It is a permanent validation failure.
var ErrWorkflowSnapshotTooLarge = errors.New("Workflow snapshot exceeds its size limit")

// EncodeResolvedWorkflowSnapshot returns the snapshot a Run stores for
// workflow. The catalog, Run creation, the Run store and the Scheduler all
// measure this encoding, never the text PostgreSQL returns for the stored
// jsonb, so a snapshot accepted when its Run is created stays executable
// (CAT-R76).
func EncodeResolvedWorkflowSnapshot(workflow ResolvedWorkflow) ([]byte, error) {
	data, err := json.Marshal(workflow)
	if err != nil {
		return nil, fmt.Errorf("encode Workflow %s@%s snapshot: %w", workflow.Ref.Name, workflow.Ref.Version, err)
	}
	if len(data) > MaxWorkflowSnapshotBytes {
		return nil, fmt.Errorf(
			"%w: Workflow %s@%s snapshot has %d bytes, more than %d",
			ErrWorkflowSnapshotTooLarge, workflow.Ref.Name, workflow.Ref.Version, len(data), MaxWorkflowSnapshotBytes,
		)
	}
	return data, nil
}

// PostgreSQL text-array paths of the Planner of a stored ResolvedStage that
// Eval metrics read. Renaming a tag must rename its constant;
// internal/persistence pins each one against an encoded value.
const (
	StagePlannerIDPath      = "{planner,plannerId}"
	StagePlannerVersionPath = "{planner,version}"
)

// DecodeResolvedWorkflowSnapshot strictly decodes immutable Run authority.
// Every Stage must retain its explicit Worker session mode. Worker session defaults
// are resolved before persistence. Optional maxSubtasks instead retains its
// contract-defined omission meaning, so the pinned selection needs no catalog.
func DecodeResolvedWorkflowSnapshot(data []byte) (ResolvedWorkflow, error) {
	var workflow ResolvedWorkflow
	if err := strictjson.Decode(data, &workflow); err != nil {
		return ResolvedWorkflow{}, err
	}
	// Every reader gets a Workflow that passes the same complete graph check
	// as a catalog load (CAT-R78), so no caller validates it again.
	if err := ValidateWorkflowGraph(workflow); err != nil {
		return ResolvedWorkflow{}, fmt.Errorf("persisted Workflow: %w", err)
	}
	return workflow, nil
}

// DecodeResolvedStageSnapshot strictly decodes the Stage copy retained by
// StageExecution, including its explicit Worker session mode.
func DecodeResolvedStageSnapshot(data []byte) (ResolvedStage, error) {
	var stage ResolvedStage
	if err := strictjson.Decode(data, &stage); err != nil {
		return ResolvedStage{}, err
	}
	if err := stage.MaxSubtasks.Validate(); err != nil {
		return ResolvedStage{}, err
	}
	if err := validateStageAgentNames("persisted Stage", stage.Agents); err != nil {
		return ResolvedStage{}, err
	}
	if err := stage.Session.Validate(); err != nil {
		return ResolvedStage{}, err
	}
	if err := ValidateScanPlanStage(stage); err != nil {
		return ResolvedStage{}, err
	}
	if stage.ScanPlan != nil {
		if err := ValidateStageExecutionConfig("persisted", stage); err != nil {
			return ResolvedStage{}, err
		}
	}
	return stage, nil
}
