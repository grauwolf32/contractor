package scheduler

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"reflect"
	"sort"
	"strings"

	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
)

const maxWorkflowSnapshotBytes = 2 * 1024 * 1024

type executableWorkflow struct {
	workflow  workflowconfig.ResolvedWorkflow
	stageName string
	stage     workflowconfig.ResolvedStage
}

func decodeExecutableWorkflow(run runstore.WorkflowRun) (executableWorkflow, error) {
	if run.WorkflowSchemaVersion != contracts.APIVersion || len(run.WorkflowSnapshot) == 0 ||
		len(run.WorkflowSnapshot) > maxWorkflowSnapshotBytes {
		return executableWorkflow{}, fmt.Errorf("%w: Workflow snapshot version or size is invalid", ErrUnsupportedWorkflow)
	}
	var workflow workflowconfig.ResolvedWorkflow
	decoder := json.NewDecoder(bytes.NewReader(run.WorkflowSnapshot))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&workflow); err != nil {
		return executableWorkflow{}, fmt.Errorf("%w: decode Workflow snapshot: %v", ErrUnsupportedWorkflow, err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		return executableWorkflow{}, fmt.Errorf("%w: Workflow snapshot contains trailing JSON", ErrUnsupportedWorkflow)
	}
	if workflow.Ref.Name != run.WorkflowName || workflow.Ref.Version != run.WorkflowVersion ||
		len(workflow.Stages) != 1 || strings.TrimSpace(workflow.EntryStage) == "" {
		return executableWorkflow{}, fmt.Errorf("%w: only one exact entry Stage is supported", ErrUnsupportedWorkflow)
	}
	stage, ok := workflow.Stages[workflow.EntryStage]
	if !ok || len(stage.Agents) != 1 ||
		stage.Planner.PlannerID+"@"+stage.Planner.Version != planner.PassthroughRef {
		return executableWorkflow{}, fmt.Errorf("%w: MVP requires one passthrough@1 Agent Stage", ErrUnsupportedWorkflow)
	}
	if stage.On.Succeeded.Kind != workflowconfig.TransitionSucceed ||
		stage.On.Failed.Kind != workflowconfig.TransitionFail ||
		stage.On.Interrupted.Kind != workflowconfig.TransitionFail {
		return executableWorkflow{}, fmt.Errorf("%w: MVP requires terminal succeed/fail transitions", ErrUnsupportedWorkflow)
	}
	if strings.TrimSpace(stage.Objective) == "" || strings.TrimSpace(stage.Instructions.Text) == "" {
		return executableWorkflow{}, fmt.Errorf("%w: Stage semantic input is incomplete", ErrUnsupportedWorkflow)
	}
	for logicalName, binding := range stage.Agents {
		if strings.TrimSpace(logicalName) == "" || strings.TrimSpace(binding.Namespace) == "" ||
			binding.Namespace == "inputs" || binding.Namespace == "outputs" ||
			strings.Contains(binding.Namespace, "/") {
			return executableWorkflow{}, fmt.Errorf("%w: Stage Agent binding is invalid", ErrUnsupportedWorkflow)
		}
		if err := binding.Template.Validate(); err != nil {
			return executableWorkflow{}, fmt.Errorf("%w: resolved AgentTemplate is invalid: %v", ErrUnsupportedWorkflow, err)
		}
	}
	for outputName, resultName := range stage.WorkflowOutputs {
		if _, ok := workflow.Outputs[outputName]; !ok {
			return executableWorkflow{}, fmt.Errorf("%w: undeclared Workflow output mapping", ErrUnsupportedWorkflow)
		}
		if _, ok := stage.Result.Artifacts[resultName]; !ok {
			return executableWorkflow{}, fmt.Errorf("%w: undeclared Stage result mapping", ErrUnsupportedWorkflow)
		}
	}
	for outputName, slot := range workflow.Outputs {
		if _, mapped := stage.WorkflowOutputs[outputName]; slot.Required && !mapped {
			return executableWorkflow{}, fmt.Errorf("%w: required output %q is not mapped", ErrUnsupportedWorkflow, outputName)
		}
	}
	return executableWorkflow{workflow: workflow, stageName: workflow.EntryStage, stage: stage}, nil
}

func validatePersistedExecution(
	execution runstore.StageExecution,
	run runstore.WorkflowRun,
	workflow executableWorkflow,
) error {
	if execution.RunID != run.RunID || execution.StageName != workflow.stageName || execution.Attempt != 1 ||
		execution.PreviousExecutionID != nil || execution.StageSpecSchemaVersion != contracts.APIVersion ||
		execution.StageContextSchemaVersion != contracts.APIVersion {
		return fmt.Errorf("persisted StageExecution identity differs from the Workflow snapshot")
	}
	var persisted workflowconfig.ResolvedStage
	decoder := json.NewDecoder(bytes.NewReader(execution.StageSpecSnapshot))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&persisted); err != nil {
		return fmt.Errorf("decode persisted Stage spec: %w", err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		return fmt.Errorf("persisted Stage spec contains trailing JSON")
	}
	if !reflect.DeepEqual(persisted, workflow.stage) {
		return fmt.Errorf("persisted Stage spec differs from the immutable Workflow snapshot")
	}
	if err := execution.StageContext.Validate(); err != nil {
		return err
	}
	if !reflect.DeepEqual(execution.StageContext.Parameters, run.Parameters) ||
		len(execution.StageContext.Artifacts) != len(workflow.stage.Context.Artifacts) {
		return fmt.Errorf("persisted StageContext differs from the immutable Run input")
	}
	for name, declaration := range workflow.stage.Context.Artifacts {
		pinned, ok := execution.StageContext.Artifacts[name]
		if !ok || pinned.Required != declaration.Required {
			return fmt.Errorf("persisted StageContext artifact %q differs from its declaration", name)
		}
		if pinned.Artifact != nil &&
			(pinned.Artifact.Namespace != declaration.Namespace || pinned.Artifact.Name != declaration.Name) {
			return fmt.Errorf("persisted StageContext artifact %q identifies another binding", name)
		}
	}
	return nil
}

func stageSnapshot(stage workflowconfig.ResolvedStage) (json.RawMessage, error) {
	encoded, err := json.Marshal(stage)
	if err != nil {
		return nil, fmt.Errorf("encode resolved Stage snapshot: %w", err)
	}
	return encoded, nil
}

func bindingRequirements(stage workflowconfig.ResolvedStage) []controlplane.BindingRequirement {
	names := make([]string, 0, len(stage.Agents))
	for name := range stage.Agents {
		names = append(names, name)
	}
	sort.Strings(names)
	result := make([]controlplane.BindingRequirement, 0, len(names))
	for _, name := range names {
		binding := stage.Agents[name]
		result = append(result, controlplane.BindingRequirement{
			LogicalAgentName: name,
			Namespace:        binding.Namespace,
			AgentTemplate:    binding.Template,
		})
	}
	return result
}
