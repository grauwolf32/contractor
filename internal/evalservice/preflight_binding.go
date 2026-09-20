package evalservice

import (
	"context"
	"errors"
	"slices"
	"sort"

	"github.com/grauwolf32/contractor/internal/agentskills"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
)

const (
	capabilityInputArtifact  = "input.artifact@1"
	capabilityOutputArtifact = "output.artifact@1"
	capabilityWorkflowRun    = "workflow.run@1"
	capabilityAuditRun       = "audit.run@1"
	capabilitySourceArchive  = "source.archive@1"
)

type preflightBinding struct {
	snapshot  BindingSnapshot
	workflows map[string]config.ResolvedWorkflow
	inputs    map[string]config.ArtifactSlot
	standards []auditstandards.ResolvedPackage
}

func (r *Resolver) resolveBinding(ctx context.Context, owner string, v evaldomain.Variant, tracker *preflightCredentials, service *artifacts.Service) (preflightBinding, error) {
	b := preflightBinding{
		snapshot:  BindingSnapshot{Skills: []contracts.RunSkillSnapshot{}, Standards: []auditstandards.ExactPackage{}},
		workflows: map[string]config.ResolvedWorkflow{},
		inputs:    map[string]config.ArtifactSlot{},
	}
	if v.Kind == "workflow" {
		workflow, err := r.Catalog.ResolveRunWorkflow(ctx, v.Selector, v.ExecutionConfig, tracker)
		if err != nil {
			return b, tracker.catalogError(err)
		}
		b.snapshot.Workflow = &workflow
		b.workflows["workflow"] = workflow
		b.inputs = workflow.Inputs
		return b, nil
	}
	patch, err := jsonBytes(v.ExecutionConfig)
	if err != nil {
		return b, err
	}
	if string(patch) != "{}" {
		return b, evaldomain.Failure("eval_not_ready")
	}
	profile, err := r.Catalog.AuditProfile(v.Selector)
	if err != nil {
		return b, errors.Join(evaldomain.Failure("eval_not_ready"), err)
	}
	if !auditservice.ProfileCompatibility(profile).ServerCompatible {
		return b, evaldomain.Failure("eval_not_ready")
	}
	b.snapshot.Audit = &profile
	for name, slot := range profile.Inputs {
		b.inputs[name] = config.ArtifactSlot{Required: slot.Required, MediaTypes: slot.MediaTypes}
	}
	for role, binding := range profile.Workflows {
		b.workflows[role] = binding.Workflow
	}
	catalog, err := auditstandards.NewCatalog(service)
	if err != nil {
		return b, err
	}
	for _, standard := range profile.Standards {
		resolved, err := catalog.Resolve(ctx, owner, auditstandards.Reference{Scheme: standard.Scheme, Version: standard.Version})
		if err != nil {
			return b, preflightDependencyError(err)
		}
		b.snapshot.Standards = append(b.snapshot.Standards, resolved.Source)
		b.standards = append(b.standards, resolved)
	}
	return b, nil
}

func (b *preflightBinding) pinDependencies(ctx context.Context, tx pgx.Tx, service *artifacts.Service, owner string, v evaldomain.Variant, tracker *preflightCredentials, lookup runtimeconfig.TransactionLLMCredentialLookup) error {
	skillRefs := map[string]contracts.ArtifactRef{}
	var skillSets [][]string
	modelFree := b.snapshot.Audit == nil
	for _, workflow := range b.workflows {
		if err := config.ValidateResolvedWorkflowCredentials(ctx, workflow, tracker); err != nil {
			return tracker.catalogError(err)
		}
		refs, err := config.WorkflowSkillRefs(workflow)
		if err != nil {
			return errors.Join(evaldomain.Failure("eval_not_ready"), err)
		}
		for _, ref := range refs {
			skillRefs[ref.Name] = ref
		}
		skillSets = append(skillSets, config.WorkflowSkillSets(workflow)...)
		modelFree = modelFree && workflowModelFree(workflow)
	}
	runtime, err := runtimeconfig.PinRunSnapshot(ctx, tx, v.RuntimeLabels, tracker, lookup, modelFree)
	if err != nil {
		return tracker.dependencyError(err)
	}
	b.snapshot.Runtime = runtime
	if len(skillRefs) == 0 {
		return nil
	}
	refs := make([]contracts.ArtifactRef, 0, len(skillRefs))
	for _, ref := range skillRefs {
		refs = append(refs, ref)
	}
	sort.Slice(refs, func(i, j int) bool { return refs[i].Name < refs[j].Name })
	catalog, err := agentskills.NewCatalog(service)
	if err != nil {
		return err
	}
	b.snapshot.Skills, err = catalog.SelectRunSources(ctx, owner, refs)
	if err != nil {
		return preflightDependencyError(err)
	}
	if err := agentskills.ValidateSelectedLimits(b.snapshot.Skills, skillSets); err != nil {
		return errors.Join(evaldomain.Failure("eval_not_ready"), err)
	}
	for _, skill := range b.snapshot.Skills {
		if skill.Source == nil {
			return evaldomain.Failure("eval_not_ready")
		}
	}
	return nil
}

func workflowModelFree(workflow config.ResolvedWorkflow) bool {
	for _, stage := range workflow.Stages {
		for _, agent := range stage.Agents {
			if !agent.Template.IsToolWorker() {
				return false
			}
		}
	}
	return true
}

func (b preflightBinding) capabilities() []string {
	capabilities := []string{capabilityInputArtifact, capabilityOutputArtifact}
	if b.snapshot.Audit != nil {
		capabilities = append(capabilities, capabilityAuditRun)
	} else {
		capabilities = append(capabilities, capabilityWorkflowRun)
	}
	for _, slot := range b.inputs {
		if slices.Contains(slot.MediaTypes, "application/zip") {
			capabilities = append(capabilities, capabilitySourceArchive)
			break
		}
	}
	slices.Sort(capabilities)
	return capabilities
}
