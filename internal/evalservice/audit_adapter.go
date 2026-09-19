package evalservice

import (
	"context"
	"encoding/json"
	"strconv"
	"time"

	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
)

type auditAdapter struct {
	operations *executionOperations
	resources  *memberResources
	Audits     AuditExecutor
}

type auditRequest struct {
	ID, ProjectID, ProfileSHA256 string
	Profile                      auditservice.ProfileSelector
	Inputs                       map[string]contracts.ArtifactRef
	Scope                        auditservice.Scope
}
type startRequest struct {
	AuditID                                      string
	Revision                                     uint64
	RuntimeSHA256, SkillsSHA256, StandardsSHA256 string
}

func (d *auditAdapter) Create(ctx context.Context, e evalstore.Experiment, m evalstore.Member, c evalstore.Claim) error {
	op, err := prepareOperation(ctx, d.operations, e, m, c, "audit-create", func() (auditRequest, error) {
		projectID, err := d.resources.auditProject(ctx, e, m, c)
		if err != nil {
			return auditRequest{}, err
		}
		inputs, err := d.resources.inputs(ctx, e, m, c, projectID)
		if err != nil {
			return auditRequest{}, err
		}
		snapshot, err := d.resources.binding(ctx, e, m)
		if err != nil {
			return auditRequest{}, err
		}
		if snapshot.Audit == nil {
			return auditRequest{}, evaldomain.Failure("eval_pin_mismatch")
		}
		digest, err := hashJSON(snapshot.Audit)
		if err != nil {
			return auditRequest{}, err
		}
		params := MapParameters(memberCase(m), m.Recipe.Variant)
		objective := m.Recipe.Case.Task.Objective
		if value, ok := params["objective"]; ok {
			objective = value
		}
		return auditRequest{
			ID:            "audit-" + m.SubmissionKey,
			ProjectID:     projectID,
			ProfileSHA256: digest,
			Profile:       auditservice.ProfileSelector{Name: snapshot.Audit.Ref.Name, Version: snapshot.Audit.Ref.Version},
			Inputs:        inputs,
			Scope:         auditservice.Scope{Objective: objective, Target: params["target"], AuthorizationScope: params["authorizationScope"]},
		}, nil
	})
	if err != nil {
		return err
	}
	if op.State == "rejected" {
		return nil
	}
	var request auditRequest
	if err = json.Unmarshal(op.Request, &request); err != nil {
		return err
	}
	audit, _, err := d.Audits.CreateDraft(ctx, auditservice.CreateDraftParams{
		OwnerID:               e.OwnerID,
		ProjectID:             request.ProjectID,
		AuditID:               request.ID,
		Profile:               request.Profile,
		Inputs:                request.Inputs,
		RuntimeLabels:         m.Recipe.Variant.RuntimeLabels,
		Scope:                 request.Scope,
		IdempotencyKey:        op.Key,
		RequestDigest:         evaldomain.Digest(op.Request),
		ExpectedProfileSHA256: request.ProfileSHA256,
	})
	if err != nil {
		return d.operations.failDefinite(ctx, e, m, c, op, err)
	}
	if err = resolveOperation(ctx, d.operations, e, m, c, op, executionCreated{ID: audit.AuditID}, false, ""); err != nil {
		return err
	}
	// The Audit exists, but no managed execution receipt is confirmed until its
	// separate start operation succeeds. Cancelling an unstarted draft uses the
	// ordinary deletion path and later its authoritative deletion tombstone.
	if e.State == "cancelling" && audit.State == auditstore.AuditDraft {
		return d.cancelDraft(ctx, e, m, c, audit)
	}
	if audit.State == auditstore.AuditDeleting {
		return nil // The ordinary Audit controller will retain a deletion tombstone.
	}
	start, err := prepareOperation(ctx, d.operations, e, m, c, "audit-start", func() (startRequest, error) {
		snapshot, err := d.resources.binding(ctx, e, m)
		if err != nil {
			return startRequest{}, err
		}
		runtime, err := hashJSON(snapshot.Runtime)
		if err != nil {
			return startRequest{}, err
		}
		skills, err := hashJSON(snapshot.Skills)
		if err != nil {
			return startRequest{}, err
		}
		standards, err := hashJSON(snapshot.Standards)
		return startRequest{audit.AuditID, audit.Revision, runtime, skills, standards}, err
	})
	if err != nil {
		return err
	}
	if start.State == "rejected" {
		return d.cancelDraft(ctx, e, m, c, audit)
	}
	var sr startRequest
	if err = json.Unmarshal(start.Request, &sr); err != nil {
		return err
	}
	result, err := d.Audits.Start(ctx, auditservice.StartParams{
		OwnerID:                 e.OwnerID,
		AuditID:                 sr.AuditID,
		ExpectedRevision:        sr.Revision,
		IdempotencyKey:          start.Key,
		RequestDigest:           evaldomain.Digest(start.Request),
		ExpectedRuntimeSHA256:   sr.RuntimeSHA256,
		ExpectedSkillsSHA256:    sr.SkillsSHA256,
		ExpectedStandardsSHA256: sr.StandardsSHA256,
	})
	if err != nil {
		return d.operations.failDefinite(ctx, e, m, c, start, err)
	}
	return resolveOperation(ctx, d.operations, e, m, c, start, executionCreated{ID: result.Audit.AuditID}, false, result.Audit.AuditID)
}
func (d *auditAdapter) cancelDraft(ctx context.Context, e evalstore.Experiment, m evalstore.Member, c evalstore.Claim, a auditstore.Audit) error {
	op, err := prepareOperation(ctx, d.operations, e, m, c, "cancel", func() (draftDeletionRequest, error) {
		return draftDeletionRequest{ID: a.AuditID, Kind: "audit", DeleteDraft: true}, nil
	})
	if err != nil {
		return err
	}
	if op.State == "succeeded" {
		return nil
	}
	result, err := d.Audits.Delete(ctx, auditservice.MutationParams{
		OwnerID:          e.OwnerID,
		AuditID:          a.AuditID,
		ExpectedRevision: a.Revision,
		IdempotencyKey:   op.Key + "-" + strconv.FormatUint(a.Revision, 10),
		RequestDigest:    evaldomain.Digest(append(op.Request, []byte(strconv.FormatUint(a.Revision, 10))...)),
	})
	if err != nil {
		return err
	}
	return resolveOperation(ctx, d.operations, e, m, c, op, draftDeletionAccepted{ID: result.Audit.AuditID, DeleteDraft: true}, false, "")
}
func (d *auditAdapter) Cancel(ctx context.Context, e evalstore.Experiment, m evalstore.Member, c evalstore.Claim, sub evalstore.Submission) error {
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

	audit, err := d.Audits.Get(ctx, e.OwnerID, *sub.ExecutionID)
	if err != nil {
		return err
	}
	if audit.State != auditstore.AuditCancelling && audit.State != auditstore.AuditCompleted && audit.State != auditstore.AuditCancelled && audit.State != auditstore.AuditFailed && audit.State != auditstore.AuditDeleting {
		// Each CAS attempt has a deterministic revision-scoped domain key. The
		// immutable outer intent is cancellation of this exact owned Audit.
		revision := strconv.FormatUint(audit.Revision, 10)
		_, err = d.Audits.Cancel(ctx, auditservice.MutationParams{
			OwnerID:          e.OwnerID,
			AuditID:          audit.AuditID,
			ExpectedRevision: audit.Revision,
			IdempotencyKey:   op.Key + "-" + revision,
			RequestDigest:    evaldomain.Digest(append(op.Request, []byte(revision)...)),
		})
		if err != nil {
			return err
		}
	}
	return resolveOperation(ctx, d.operations, e, m, c, op, cancellationAccepted{ID: *sub.ExecutionID, Requested: true}, false, "")
}
