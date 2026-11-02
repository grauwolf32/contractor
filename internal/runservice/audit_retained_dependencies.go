package runservice

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"

	"github.com/grauwolf32/contractor/internal/artifactpolicy"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
)

const auditDependencyInputSchema = "contractor.audit.retained-inputs.v1"

// retainedDependencyInput is the complete trusted mapping a consumer reads.
// Original and retained refs document provenance; only Consumer is a local
// read grant. The materializer never traverses other refs inside payload JSON.
type retainedDependencyInput struct {
	ItemID   string                  `json:"itemId"`
	Name     string                  `json:"name"`
	Source   contracts.ExactArtifact `json:"source"`
	Retained contracts.ExactArtifact `json:"retained"`
	Consumer contracts.ExactArtifact `json:"consumer"`
}

// MaterializeAuditRetainedDependencies runs behind CreateAudit's durable
// initializing fence. One bounded check batch has at most 512 descriptors,
// 1 MiB accepted metadata, and 32 descriptors per item; compatible selection
// keeps different item dependencies in separate children. All forks are
// deterministic Replay forks under the original child intent. The enclosing
// transaction joins all work before it publishes Complete or rolls back.
func MaterializeAuditRetainedDependencies(ctx context.Context, service *artifacts.Service, intent auditstore.RunCreationIntent, run runstore.WorkflowRun, state runstore.AuditInputMaterialization) (runstore.AuditInputMaterialization, error) {
	if intent.Execution.Role != auditstore.ExecutionCheck || run.AuditExecutionID == nil || *run.AuditExecutionID != intent.Execution.ExecutionID || state.Identity == "" {
		return state, fmt.Errorf("%w: retained dependency initialization authority is invalid", ErrInvalid)
	}
	project, err := service.Project(intent.ProjectID)
	if err != nil {
		return state, err
	}
	projectScope, _ := artifacts.ProjectScope(intent.ProjectID)
	runScope, _ := artifacts.RunScope(run.RunID)
	mappings := make([]retainedDependencyInput, 0)
	refs := make([]contracts.ArtifactRef, 0)
	verified := map[string]bool{}
	metadataBytes := 0
	for _, member := range intent.Items {
		taskData, err := project.ReadExact(ctx, member.Task)
		if err != nil {
			return state, err
		}
		pkg, err := auditdomain.ValidatePackage(taskData.Payload.Data)
		if err != nil {
			return state, fmt.Errorf("%w: retained dependency task package is invalid", ErrInvalid)
		}
		documentMember, exists := pkg.MemberByID("task-document")
		if !exists {
			return state, fmt.Errorf("%w: retained dependency task document is missing", ErrInvalid)
		}
		task, err := auditdomain.DecodeItemTask(documentMember.Data())
		if err != nil || task.Dependencies == nil {
			return state, fmt.Errorf("%w: retained dependency task manifest is missing", ErrInvalid)
		}
		manifest := task.Dependencies
		if manifest.OwnerID != intent.OwnerID || manifest.ProjectID != intent.ProjectID || manifest.AuditID != intent.Execution.AuditID {
			return state, fmt.Errorf("%w: foreign retained dependency manifest", ErrInvalid)
		}
		encoded, err := auditdomain.EncodeRetainedDependencies(*manifest)
		if err != nil {
			return state, err
		}
		metadataBytes += len(encoded)
		if len(intent.Items) > 128 || metadataBytes > 1<<20 {
			return state, fmt.Errorf("%w: retained dependency materialization metadata limit", ErrInvalid)
		}
		for _, dependency := range manifest.Dependencies {
			if len(mappings) >= 512 {
				return state, fmt.Errorf("%w: retained dependency materialization count limit", ErrInvalid)
			}
			authorized := slices.ContainsFunc(member.Inputs, func(input auditstore.ExactArtifact) bool { return dependency.Retained.Verify(input) == nil })
			if !authorized {
				return state, fmt.Errorf("%w: undeclared retained dependency input", ErrInvalid)
			}
			sourceStore, sourceScope, kind := project, projectScope, artifacts.LineageInputFork
			if dependency.Kind == "prepare-output" {
				sourceScope, _ = artifacts.AuditScope(intent.Execution.AuditID)
				sourceStore, _ = service.Scoped(sourceScope)
				kind = artifacts.LineageAuditInput
			}
			// ReadExact verifies payload bytes, not only the stored hash, before a
			// child can execute. Consumer reads use the same normal Artifact bounds.

			if !verified[dependency.Retained.Ref.Key()] {
				if _, err := sourceStore.ReadExact(ctx, dependency.Retained); err != nil {
					if errors.Is(err, contracts.ErrExactArtifactMismatch) {
						return state, fmt.Errorf("%w: accepted retained dependency content differs", ErrInvalid)
					}
					return state, err
				}
				verified[dependency.Retained.Ref.Key()] = true
			}

			target := contracts.ArtifactRef{Namespace: artifactpolicy.AuditContextNamespace, Name: auditdomain.DeterministicID("retained-input", member.ItemID, dependency.Name)}
			fork, err := service.Fork(ctx, artifacts.Fork{Kind: kind, Source: sourceScope, SourceRef: dependency.Retained.Ref, Target: runScope, TargetRef: target, Frozen: true, Replay: true})
			if err != nil {
				return state, err
			}
			consumer := dependency.Retained.Clone()
			consumer.Ref = fork.TargetRef
			mappings = append(mappings, retainedDependencyInput{ItemID: member.ItemID, Name: dependency.Name, Source: dependency.Source, Retained: dependency.Retained, Consumer: consumer})
			refs = append(refs, consumer.Ref)
		}
	}
	if len(mappings) == 0 {
		return state, fmt.Errorf("%w: retained dependency initialization has no accepted inputs", ErrInvalid)
	}
	payload, err := json.Marshal(struct {
		Schema   string                    `json:"schema"`
		Mappings []retainedDependencyInput `json:"mappings"`
	}{auditDependencyInputSchema, mappings})
	if err != nil {
		return state, err
	}
	contextArtifact, err := service.CreateFrozen(ctx, runScope, contracts.ArtifactRef{Namespace: artifactpolicy.AuditContextNamespace, Name: "retained-dependencies"}, artifacts.Payload{MediaType: auditdomain.JSONMediaType, Data: payload})
	if err != nil {
		return state, err
	}
	refs = append(refs, contextArtifact.Ref)
	// The durable state stores only the explicit local ref set. The mapping
	// document is itself one of those refs; no large JSON copy is needed here.
	state.Checkpoint = json.RawMessage(`{"schema":"contractor.audit.retained-inputs.v1"}`)
	state.Refs = refs
	state.Complete = true
	return state, nil
}
