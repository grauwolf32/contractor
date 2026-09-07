package scheduler

import (
	"fmt"
	"reflect"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func auditBindingRequirements(run runstore.WorkflowRun, workflow executableWorkflow, context runstore.StageContextSnapshot) ([]controlplane.BindingRequirement, error) {
	bindings, err := bindingRequirements(workflow.stage, run.SkillSnapshot, context)
	if err != nil {
		return nil, err
	}
	completion := run.AuditCompletion
	if completion != nil {
		if run.PublicationMode != runstore.PublicationAuditManaged || run.AuditExecutionID == nil || run.AuditSubmissionKey == nil || completion.Validate() != nil {
			return nil, fmt.Errorf("invalid Run Audit completion authority")
		}
		if completion.Stage == workflow.stageName {
			stage := workflow.stage
			agent, ok := stage.Agents[completion.Agent]
			if !ok || len(stage.Agents) != 1 || stage.Planner.PlannerID != "passthrough" || stage.Planner.Version != "1" || completion.Contract.ValidateAllocation(agent.Namespace, agent.Template) != nil {
				return nil, fmt.Errorf("Audit completion target changed")
			}
			outputFound := false
			for _, result := range stage.Result.Artifacts {
				if result.Required && result.From != nil && result.From.Namespace == completion.Contract.ResultArtifact.Namespace && result.From.Name == completion.Contract.ResultArtifact.Name {
					outputFound = true
				}
			}
			if !outputFound {
				return nil, fmt.Errorf("Audit completion output ownership changed")
			}
			for _, ref := range []contracts.ArtifactRef{completion.Contract.Task, completion.Contract.ExecutionManifest} {
				found := false
				for _, pinned := range context.Artifacts {
					if pinned.Required && pinned.Artifact != nil && reflect.DeepEqual(*pinned.Artifact, ref) {
						found = true
					}
				}
				if !found {
					return nil, fmt.Errorf("Audit completion exact input is absent from Stage context")
				}
			}
			for i := range bindings {
				if bindings[i].LogicalAgentName == completion.Agent {
					bindings[i].CompletionContract = contracts.CloneWorkerCompletionContract(&completion.Contract)
				}
			}
		}
	}
	for _, binding := range bindings {
		if err := contracts.ValidateWorkerCompletionSelection(binding.CompletionContract, binding.Namespace, binding.AgentTemplate); err != nil {
			return nil, err
		}
	}
	return bindings, nil
}

func auditPinnedContextRef(run runstore.WorkflowRun, ref contracts.ArtifactRef) contracts.ArtifactRef {
	if run.AuditCompletion != nil {
		for _, pinned := range []contracts.ArtifactRef{run.AuditCompletion.Contract.Task, run.AuditCompletion.Contract.ExecutionManifest} {
			if pinned.Namespace == ref.Namespace && pinned.Name == ref.Name {
				return cloneArtifactRef(pinned)
			}
		}
	}
	return ref
}
