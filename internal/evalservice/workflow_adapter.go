package evalservice

import (
	"context"
	"encoding/json"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
	"github.com/grauwolf32/contractor/internal/runservice"
	"github.com/grauwolf32/contractor/internal/runstore"
)

type workflowAdapter struct {
	operations *executionOperations
	resources  *memberResources
	Runs       RunCreator
	Notifier   RunNotifier
}

type runRequest struct {
	ID, ProjectID, Workflow, WorkflowSHA256, RuntimeSHA256, SkillsSHA256 string
	Inputs                                                               map[string]contracts.ArtifactRef
	Parameters                                                           map[string]string
}

func (d *workflowAdapter) Create(ctx context.Context, e evalstore.Experiment, m evalstore.Member, c evalstore.Claim) error {
	op, err := prepareOperation(ctx, d.operations, e, m, c, "run-create", func() (runRequest, error) {
		inputs, err := d.resources.inputs(ctx, e, m, c, e.ProjectID)
		if err != nil {
			return runRequest{}, err
		}
		snapshot, err := d.resources.binding(ctx, e, m)
		if err != nil {
			return runRequest{}, err
		}
		w, err := hashJSON(snapshot.Workflow)
		if err != nil {
			return runRequest{}, err
		}
		r, err := hashJSON(snapshot.Runtime)
		if err != nil {
			return runRequest{}, err
		}
		skills, err := hashJSON(snapshot.Skills)
		if err != nil {
			return runRequest{}, err
		}
		return runRequest{
			ID:             "run-" + m.SubmissionKey,
			ProjectID:      e.ProjectID,
			Workflow:       m.Recipe.Variant.Selector,
			WorkflowSHA256: w,
			RuntimeSHA256:  r,
			SkillsSHA256:   skills,
			Inputs:         inputs,
			Parameters:     MapParameters(memberCase(m), m.Recipe.Variant),
		}, nil
	})
	if err != nil {
		return err
	}
	if op.State == "rejected" {
		return nil
	}
	var request runRequest
	if err = json.Unmarshal(op.Request, &request); err != nil {
		return err
	}
	result, err := d.Runs.CreatePublic(ctx, runservice.PublicCreateParams{
		OwnerID:                e.OwnerID,
		ProjectID:              &request.ProjectID,
		Workflow:               request.Workflow,
		ExecutionConfig:        m.Recipe.Variant.ExecutionConfig,
		RuntimeLabels:          m.Recipe.Variant.RuntimeLabels,
		Inputs:                 request.Inputs,
		Parameters:             request.Parameters,
		IdempotencyKey:         op.Key,
		RequestDigest:          evaldomain.Digest(op.Request),
		NewRunID:               func() (string, error) { return request.ID, nil },
		ExpectedWorkflowSHA256: request.WorkflowSHA256,
		ExpectedRuntimeSHA256:  request.RuntimeSHA256,
		ExpectedSkillsSHA256:   request.SkillsSHA256,
	})
	if err != nil {
		return d.operations.failDefinite(ctx, e, m, c, op, err)
	}
	if err = resolveOperation(ctx, d.operations, e, m, c, op, executionCreated{ID: result.Run.RunID}, false, result.Run.RunID); err == nil && d.Notifier != nil {
		d.Notifier.Wake()
	}
	return err
}
func (d *workflowAdapter) Cancel(ctx context.Context, e evalstore.Experiment, m evalstore.Member, c evalstore.Claim, sub evalstore.Submission) error {
	if sub.ExecutionID == nil {
		return evaldomain.Failure("eval_member_conflict")
	}
	op, err := prepareOperation(ctx, d.operations, e, m, c, "cancel", func() (cancellationRequest, error) {
		return cancellationRequest{ID: *sub.ExecutionID, Kind: m.ExecutionKind, RequestedAt: time.Now().UTC()}, nil
	})
	if err != nil {
		return err
	}
	if op.State == "succeeded" {
		return nil
	}
	var request cancellationRequest
	if err = json.Unmarshal(op.Request, &request); err != nil {
		return err
	}
	reason := "Evaluation experiment stopped."
	owner := e.OwnerID
	run, err := runstore.NewPostgresStore(d.operations.pool).RequestRunCancellation(ctx, *sub.ExecutionID, runstore.WorkflowRunCancellation{Code: runstore.CancellationUserRequested, RequestedAt: request.RequestedAt, RequestedBy: &owner, Reason: &reason})
	if err != nil {
		return err
	}
	if d.Notifier != nil && run.State == runstore.RunCancelling {
		d.Notifier.Cancel(run.RunID)
	}
	return resolveOperation(ctx, d.operations, e, m, c, op, cancellationAccepted{ID: *sub.ExecutionID, Requested: true}, false, "")
}
