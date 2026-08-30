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
		strings.TrimSpace(workflow.EntryStage) == "" {
		return executableWorkflow{}, fmt.Errorf("%w: Workflow identity or entry Stage is invalid", ErrUnsupportedWorkflow)
	}
	if err := workflowconfig.ValidateWorkflowGraph(workflow); err != nil {
		return executableWorkflow{}, fmt.Errorf("%w: invalid Workflow graph: %v", ErrUnsupportedWorkflow, err)
	}
	for stageName, stage := range workflow.Stages {
		plannerRef := stage.Planner.PlannerID + "@" + stage.Planner.Version
		validShape := false
		switch plannerRef {
		case planner.PassthroughRef, planner.StreamlineRef:
			validShape = len(stage.Agents) == 1
		case planner.RouterRef:
			validShape = len(stage.Agents) > 0
		}
		if !validShape {
			return executableWorkflow{}, fmt.Errorf(
				"%w: Stage %q has an unsupported Planner/Agent shape",
				ErrUnsupportedWorkflow, stageName,
			)
		}
		if strings.TrimSpace(stage.Objective) == "" || strings.TrimSpace(stage.Instructions.Text) == "" {
			return executableWorkflow{}, fmt.Errorf("%w: Stage %q semantic input is incomplete", ErrUnsupportedWorkflow, stageName)
		}
		for logicalName, binding := range stage.Agents {
			if strings.TrimSpace(logicalName) == "" || strings.TrimSpace(binding.Namespace) == "" ||
				binding.Namespace == "inputs" || binding.Namespace == "outputs" ||
				strings.Contains(binding.Namespace, "/") {
				return executableWorkflow{}, fmt.Errorf("%w: Stage %q Agent binding is invalid", ErrUnsupportedWorkflow, stageName)
			}
			if err := binding.Template.Validate(); err != nil {
				return executableWorkflow{}, fmt.Errorf("%w: resolved AgentTemplate is invalid: %v", ErrUnsupportedWorkflow, err)
			}
		}
	}
	result := executableWorkflow{workflow: workflow}
	return result.selectStage(workflow.EntryStage)
}

func (w executableWorkflow) selectStage(name string) (executableWorkflow, error) {
	stage, ok := w.workflow.Stages[name]
	if !ok {
		return executableWorkflow{}, fmt.Errorf("%w: Workflow Stage %q does not exist", ErrUnsupportedWorkflow, name)
	}
	w.stageName = name
	w.stage = stage
	return w, nil
}

func validatePersistedExecution(
	execution runstore.StageExecution,
	run runstore.WorkflowRun,
	workflow executableWorkflow,
) error {
	if execution.RunID != run.RunID || execution.StageName != workflow.stageName || execution.Attempt <= 0 ||
		(execution.Attempt == 1) != (execution.PreviousExecutionID == nil) ||
		execution.StageSpecSchemaVersion != contracts.APIVersion ||
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
