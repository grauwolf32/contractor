package evalservice

import (
	"context"
	"encoding/json"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5"
)

type memberResources struct{ operations *executionOperations }

func (d *memberResources) binding(ctx context.Context, e evalstore.Experiment, m evalstore.Member) (BindingSnapshot, error) {
	resource, err := evalstore.NewPostgresStore(d.operations.pool, d.operations.storage).PlanResource(ctx, e.OwnerID, e.ID, "bindings/"+m.VariantID+".json")
	if err != nil {
		return BindingSnapshot{}, err
	}
	var wrapper struct {
		Settings struct {
			Snapshot BindingSnapshot `json:"snapshot"`
		} `json:"settings"`
	}
	if err = json.Unmarshal(resource.Document.Bytes(), &wrapper); err != nil {
		return BindingSnapshot{}, err
	}
	return wrapper.Settings.Snapshot, nil
}

func (d *memberResources) inputs(ctx context.Context, e evalstore.Experiment, m evalstore.Member, c evalstore.Claim, projectID string, workflow *resolved.ResolvedWorkflow) (map[string]contracts.ArtifactRef, error) {
	op, err := prepareOperation(ctx, d.operations, e, m, c, "inputs", func() (inputCopyRequest, error) {
		inputs, err := MapInputs(evaldomain.Case{Inputs: m.Recipe.Case.Inputs}, m.Recipe.Variant)
		return inputCopyRequest{Inputs: inputs, ProjectID: projectID}, err
	})
	if err != nil {
		return nil, err
	}
	if op.State == "succeeded" {
		var refs map[string]contracts.ArtifactRef
		err = json.Unmarshal(op.Response, &refs)
		return refs, err
	}
	var request inputCopyRequest
	if err = json.Unmarshal(op.Request, &request); err != nil {
		return nil, err
	}
	refs := map[string]contracts.ArtifactRef{}
	err = d.operations.tx(ctx, func(s *evalstore.Store, tx pgx.Tx) error {
		recovery := s.RecoverMember(scope(e), e.ID, m.MemberID, c)
		if err := recovery.LockMember(ctx); err != nil {
			return err
		}
		service := artifacts.NewService(d.operations.storage.Repository(tx))
		target, err := service.Project(request.ProjectID)
		if err != nil {
			return err
		}
		for slot, ref := range request.Inputs {
			if err = verifyArtifact(ctx, tx, service, e.OwnerID, ref); err != nil {
				return err
			}
			if workflow != nil && workflow.Inputs[slot].SourceScope == "user" {
				if ref.Scope != "user" {
					return evaldomain.Failure("eval_pin_mismatch")
				}
				// The ordinary Run service forks this exact owner source. There is
				// no Eval-owned UserScope copy to retain or clean up, and replay
				// never substitutes the source binding's mutable current revision.
				refs[slot] = contracts.ArtifactRef{Namespace: ref.Namespace, Name: ref.Name, Revision: &ref.Revision}
				continue
			}
			source, err := artifactScope(service, ref)
			if err != nil {
				return err
			}
			read, err := source.Read(ctx, contracts.ArtifactRef{Namespace: ref.Namespace, Name: ref.Name, Revision: &ref.Revision})
			if err != nil {
				return err
			}
			if evaldomain.Digest(read.Payload.Data) != ref.SHA256 {
				return evaldomain.Failure("eval_pin_mismatch")
			}
			written, err := target.Write(ctx, contracts.ArtifactRef{Namespace: m.SubmissionKey, Name: slot}, read.Payload, nil)
			if err != nil {
				return err
			}
			refs[slot] = written.Ref
		}
		response, err := jsonBytes(refs)
		if err != nil {
			return err
		}
		return recovery.ResolveSuboperation(ctx, "inputs", response, false)
	})
	return refs, err
}

func memberCase(m evalstore.Member) evaldomain.Case {
	return evaldomain.Case{
		ID:       m.Recipe.Case.ID,
		Task:     m.Recipe.Case.Task,
		Inputs:   m.Recipe.Case.Inputs,
		Requires: m.Recipe.Case.Requires,
		Outputs:  m.Recipe.Case.Outputs,
	}
}

func (d *memberResources) auditProject(ctx context.Context, e evalstore.Experiment, m evalstore.Member, c evalstore.Claim) (string, error) {
	op, err := prepareOperation(ctx, d.operations, e, m, c, "project-create", func() (projectCreationRequest, error) {
		return projectCreationRequest{ProjectID: "project-" + m.SubmissionKey}, nil
	})
	if err != nil {
		return "", err
	}
	var request projectCreationRequest
	if err = json.Unmarshal(op.Request, &request); err != nil {
		return "", err
	}
	id := request.ProjectID
	if op.State == "succeeded" {
		return id, nil
	}
	err = d.operations.tx(ctx, func(s *evalstore.Store, tx pgx.Tx) error {
		recovery := s.RecoverMember(scope(e), e.ID, m.MemberID, c)
		if err := recovery.LockMember(ctx); err != nil {
			return err
		}
		_, _, err := projectstore.NewPostgresStore(tx).Create(ctx, projectstore.CreateParams{
			OwnerID:        e.OwnerID,
			ProjectID:      id,
			Kind:           projectstore.KindProject,
			Name:           "Eval " + m.MemberID,
			IdempotencyKey: op.Key,
			RequestDigest:  evaldomain.Digest(op.Request),
		})
		if err != nil {
			return err
		}
		if err = recovery.RegisterExecutionProject(ctx, id, op.Key); err != nil {
			return err
		}
		raw, _ := jsonBytes(executionCreated{ID: id})
		return recovery.ResolveSuboperation(ctx, op.Kind, raw, false)
	})
	return id, err
}
