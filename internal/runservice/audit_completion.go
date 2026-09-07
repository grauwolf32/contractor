package runservice

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// Authority comes from the claim-locked Audit profile, never from caller labels
// or a lookup of the current configuration catalog.
func prepareAuditCompletion(ctx context.Context, service *artifacts.Service, intent auditstore.RunCreationIntent, params AuditCreateParams) (*runstore.AuditCompletionSnapshot, error) {
	var binding config.ResolvedAuditWorkflowBinding
	if err := json.Unmarshal(intent.WorkflowBinding, &binding); err != nil {
		return nil, fmt.Errorf("%w: invalid owned Audit binding", ErrInvalid)
	}
	if binding.WorkerCompletion == nil {
		return nil, nil
	}
	if intent.Execution.Role != auditstore.ExecutionCheck || config.ValidateAuditWorkerCompletion(binding) != nil {
		return nil, fmt.Errorf("%w: invalid Audit completion binding", ErrInvalid)
	}
	raw, err := json.Marshal(binding.Workflow)
	if err != nil {
		return nil, err
	}
	workflow, err := config.DecodeResolvedWorkflowSnapshot(raw)
	if err != nil || !reflect.DeepEqual(workflow, params.Workflow) {
		return nil, fmt.Errorf("%w: completion Workflow differs from owned profile", ErrInvalid)
	}
	selected := binding.WorkerCompletion
	result := &runstore.AuditCompletionSnapshot{Stage: selected.Stage, Agent: selected.Agent}
	stage := workflow.Stages[selected.Stage]
	result.Contract.Kind = selected.Kind
	output := stage.Result.Artifacts[stage.WorkflowOutputs[binding.Outputs["result"]]].From
	result.Contract.ResultArtifact = contracts.ArtifactRef{Namespace: output.Namespace, Name: output.Name}
	var task auditstore.ExactArtifact
	for slot, mapping := range binding.Inputs {
		switch mapping.Source {
		case config.AuditInputFromItemPackage:
			task = params.Inputs[slot]
			result.Contract.Task = contracts.ArtifactRef{Namespace: "inputs", Name: slot}
		case config.AuditInputFromExecutionManifest:
			if !sameExactIdentity(params.Inputs[slot], intent.Execution.Manifest) {
				return nil, fmt.Errorf("%w: completion manifest differs from execution", ErrInvalid)
			}
			result.Contract.ExecutionManifest = contracts.ArtifactRef{Namespace: "inputs", Name: slot}
		}
	}
	if err := validateCompletionMembership(ctx, service, intent, task, params.ExecutionManifest); err != nil {
		return nil, err
	}
	return result, nil
}

func validateCompletionMembership(ctx context.Context, service *artifacts.Service, intent auditstore.RunCreationIntent, task, executionManifest auditstore.ExactArtifact) error {
	invalid := func() error {
		return fmt.Errorf("%w: Audit completion task/manifest membership differs from execution", ErrInvalid)
	}
	if task.MediaType != auditdomain.PackageMediaType || executionManifest.MediaType != auditdomain.JSONMediaType {
		return invalid()
	}
	store, err := service.Project(intent.ProjectID)
	if err != nil {
		return err
	}
	if _, err := verifyExactProjectArtifact(ctx, service, intent.ProjectID, executionManifest); err != nil {
		return err
	}
	manifestData, err := store.Read(ctx, intent.Execution.Manifest.Ref)
	if err != nil {
		return err
	}
	manifest, err := auditdomain.DecodeExecutionManifest(manifestData.Payload.Data)
	if err != nil || auditdomain.ValidateDispatchExecutionManifest(manifest) != nil || len(manifest.Items) != len(intent.Items) || len(manifest.Items) == 0 || len(manifest.Items) > auditstore.MaxCollectionItems {
		return invalid()
	}
	if _, err := verifyExactProjectArtifact(ctx, service, intent.ProjectID, task); err != nil {
		return err
	}
	taskData, err := store.Read(ctx, task.Ref)
	if err != nil {
		return err
	}
	pkg, err := auditdomain.ValidatePackage(taskData.Payload.Data)
	if err != nil {
		return invalid()
	}
	packages := []*auditdomain.Package{pkg}
	if len(manifest.Items) > 1 {
		if pkg.Manifest.Kind != auditdomain.PackageKindTaskSet || len(pkg.Members()) != len(manifest.Items) {
			return invalid()
		}
		packages = make([]*auditdomain.Package, len(manifest.Items))
		for i := range packages {
			member, ok := pkg.MemberByID(fmt.Sprintf("task-%03d", i))
			if !ok {
				return invalid()
			}
			packages[i], err = auditdomain.ValidatePackage(member.Data())
			if err != nil {
				return invalid()
			}
		}
	}
	for i, item := range manifest.Items {
		documentMember, ok := packages[i].MemberByID("task-document")
		if !ok {
			return invalid()
		}
		document, err := auditdomain.DecodeItemTask(documentMember.Data())
		if err != nil || document.ItemKey != item.ItemKey || document.SubjectKey != item.SubjectKey || document.WorkflowRole != intent.Execution.WorkflowRole {
			return invalid()
		}
		owned := intent.Items[i]
		if item.TaskRef == nil || !reflect.DeepEqual(*item.TaskRef, owned.Task.Ref) || item.TaskPackageDigest != owned.Task.Digest ||
			packages[i].Manifest.Kind != auditdomain.PackageKindTask || packages[i].Manifest.PackageID != item.TaskPackageID || packages[i].Digest != item.TaskPackageDigest {
			return invalid()
		}
	}
	return nil
}
