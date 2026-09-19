package evalservice

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
	pg "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runservice"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type RunCreator interface {
	CreatePublic(context.Context, runservice.PublicCreateParams) (runservice.CreateResult, error)
}
type AuditExecutor interface {
	CreateDraft(context.Context, auditservice.CreateDraftParams) (auditstore.Audit, bool, error)
	Start(context.Context, auditservice.StartParams) (auditservice.StartedAudit, error)
	Get(context.Context, string, string) (auditstore.Audit, error)
	Cancel(context.Context, auditservice.MutationParams) (auditservice.MutationResult, error)
	Delete(context.Context, auditservice.MutationParams) (auditservice.MutationResult, error)
}
type RunNotifier interface {
	Wake()
	Cancel(string)
}
type Driver struct {
	Pool     *pgxpool.Pool
	Runs     RunCreator
	Audits   AuditExecutor
	Notifier RunNotifier
}

func (d *Driver) tx(ctx context.Context, fn func(*evalstore.Store, pgx.Tx) error) error {
	return pg.InTx(ctx, d.Pool, pgx.TxOptions{}, func(tx pgx.Tx) error { return fn(evalstore.NewTxStore(tx), tx) })
}
func notFound(err error) bool {
	var e *evaldomain.Error
	return errors.As(err, &e) && e.Code == "eval_not_found"
}
func (d *Driver) operation(ctx context.Context, e evalstore.Experiment, m evalstore.Member, c evalstore.Claim, kind string, build func() (any, error)) (evalstore.Suboperation, error) {
	store := evalstore.NewPostgresStore(d.Pool)
	op, err := store.Suboperation(ctx, e.OwnerID, e.ID, m.MemberID, kind)
	if err == nil {
		return op, nil
	}
	if !notFound(err) {
		return op, err
	}
	value, err := build()
	if err != nil {
		return op, err
	}
	raw, err := jsonBytes(value)
	if err != nil {
		return op, err
	}
	op = evalstore.Suboperation{Kind: kind, Key: m.SubmissionKey + "-" + kind, Request: raw, State: "intent"}
	err = d.tx(ctx, func(s *evalstore.Store, _ pgx.Tx) error {
		return s.PutSuboperation(ctx, scope(e), e.ID, m.MemberID, c, op)
	})
	return op, err
}
func (d *Driver) resolve(ctx context.Context, e evalstore.Experiment, m evalstore.Member, c evalstore.Claim, op evalstore.Suboperation, response any, rejected bool, executionID string) error {
	raw, err := jsonBytes(response)
	if err != nil {
		return err
	}
	return d.tx(ctx, func(s *evalstore.Store, _ pgx.Tx) error {
		if executionID != "" {
			if err := s.BindExecution(ctx, scope(e), e.ID, m.MemberID, executionID, c); err != nil {
				return err
			}
		}
		return s.ResolveSuboperation(ctx, scope(e), e.ID, m.MemberID, op.Kind, c, raw, rejected)
	})
}
func (d *Driver) failDefinite(ctx context.Context, e evalstore.Experiment, m evalstore.Member, c evalstore.Claim, op evalstore.Suboperation, err error) error {
	// Only local validation/fence failures establish non-acceptance. Timeouts and
	// unknown database outcomes retain the original intent for idempotent replay.
	if errors.Is(err, runservice.ErrInvalid) || errors.Is(err, runservice.ErrPinnedSelectionChanged) || errors.Is(err, auditservice.ErrInvalid) || errors.Is(err, auditservice.ErrPinnedSelectionChanged) || errors.Is(err, auditservice.ErrProfileNotFound) || errors.Is(err, auditservice.ErrUnsupported) || errors.Is(err, projectstore.ErrDeleting) || errors.Is(err, auditstore.ErrProjectDeleting) {
		return d.resolve(ctx, e, m, c, op, map[string]string{"reason": "eval_pin_mismatch"}, true, "")
	}
	return err
}
func (d *Driver) Reconcile(ctx context.Context, e evalstore.Experiment, m evalstore.Member, c evalstore.Claim) error {
	if d.Pool == nil || d.Runs == nil || d.Audits == nil {
		return errors.New("eval execution dependencies are incomplete")
	}
	store := evalstore.NewPostgresStore(d.Pool)
	sub, err := store.Submission(ctx, e.OwnerID, e.ID, m.MemberID)
	if err != nil {
		return err
	}
	if sub.State == "terminal" || sub.State == "rejected" {
		return nil
	}
	tombstone, tombErr := store.Tombstone(ctx, e.OwnerID, e.ID, m.MemberID)
	if tombErr == nil {
		return d.tx(ctx, func(s *evalstore.Store, _ pgx.Tx) error {
			if !tombstone.NeverStarted {
				if err := s.BindTombstone(ctx, scope(e), e.ID, m.MemberID, c); err != nil {
					return err
				}
			}
			// Recover any lost acknowledgement without calling a deleted execution's
			// creation API again. The tombstone came from its ordinary deletion path.
			for _, kind := range []string{"run-create", "audit-create", "audit-start", "cancel"} {
				op, err := s.Suboperation(ctx, e.OwnerID, e.ID, m.MemberID, kind)
				if notFound(err) {
					continue
				}
				if err != nil {
					return err
				}
				if op.State == "intent" {
					response, _ := jsonBytes(map[string]any{"id": tombstone.ID, "deleted": true})
					if err = s.ResolveSuboperation(ctx, scope(e), e.ID, m.MemberID, kind, c, response, tombstone.NeverStarted && kind == "audit-start"); err != nil {
						return err
					}
				}
			}
			return s.Settle(ctx, scope(e), e.ID, m.MemberID, c)
		})
	}
	if !notFound(tombErr) {
		return tombErr
	}
	if sub.State == "intent" {
		kind := "run-create"
		if m.ExecutionKind == "audit" {
			kind = "audit-create"
		}
		op, opErr := store.Suboperation(ctx, e.OwnerID, e.ID, m.MemberID, kind)
		if e.State == "cancelling" && notFound(opErr) {
			return d.tx(ctx, func(s *evalstore.Store, _ pgx.Tx) error {
				for _, kind := range []string{"inputs", "project-create"} {
					op, err := s.Suboperation(ctx, e.OwnerID, e.ID, m.MemberID, kind)
					if notFound(err) {
						continue
					}
					if err != nil {
						return err
					}
					if op.State == "intent" {
						if err = s.ResolveSuboperation(ctx, scope(e), e.ID, m.MemberID, kind, c, []byte(`{"reason":"cancelled_before_submission"}`), true); err != nil {
							return err
						}
					}
				}
				return s.Settle(ctx, scope(e), e.ID, m.MemberID, c)
			})
		}
		if opErr == nil && op.State == "rejected" {
			return d.tx(ctx, func(s *evalstore.Store, _ pgx.Tx) error { return s.Settle(ctx, scope(e), e.ID, m.MemberID, c) })
		}
		if opErr != nil && !notFound(opErr) {
			return opErr
		}
		if m.ExecutionKind == "run" {
			if err = d.createRun(ctx, e, m, c); err != nil {
				return d.rejectPreparation(ctx, e, m, c, kind, err)
			}
		} else {
			if err = d.createAudit(ctx, e, m, c); err != nil {
				return d.rejectPreparation(ctx, e, m, c, kind, err)
			}
		}
	}
	sub, err = store.Submission(ctx, e.OwnerID, e.ID, m.MemberID)
	if err != nil {
		return err
	}
	if sub.State != "accepted" {
		return nil
	}
	if e.State == "cancelling" {
		if err = d.cancel(ctx, e, m, c, sub); err != nil {
			return err
		}
	}
	tokens, err := d.knownTokens(ctx, e, m, sub)
	if err != nil {
		return err
	}
	return d.tx(ctx, func(s *evalstore.Store, _ pgx.Tx) error {
		if err := s.ObserveTokens(ctx, scope(e), e.ID, m.MemberID, c, tokens); err != nil {
			return err
		}
		err := s.Settle(ctx, scope(e), e.ID, m.MemberID, c)
		if errors.Is(err, evalstore.ErrDrain) {
			return nil
		}
		return err
	})
}

// Only a local validation failure before any creation intent establishes that
// no remote effect can exist. Once an execution intent exists, replay owns it.
func (d *Driver) rejectPreparation(ctx context.Context, e evalstore.Experiment, m evalstore.Member, c evalstore.Claim, kind string, cause error) error {
	var domain *evaldomain.Error
	if !errors.As(cause, &domain) || (domain.Code != "eval_pin_mismatch" && domain.Code != "eval_evidence_unavailable" && domain.Code != "eval_not_found" && domain.Code != "eval_invalid") {
		return cause
	}
	return d.tx(ctx, func(s *evalstore.Store, _ pgx.Tx) error {
		if err := s.LockMemberRecovery(ctx, scope(e), e.ID, m.MemberID, c); err != nil {
			return err
		}
		if _, err := s.Suboperation(ctx, e.OwnerID, e.ID, m.MemberID, kind); !notFound(err) {
			return cause
		}
		for _, k := range []string{"inputs", "project-create"} {
			op, err := s.Suboperation(ctx, e.OwnerID, e.ID, m.MemberID, k)
			if notFound(err) {
				continue
			}
			if err != nil {
				return err
			}
			if op.State == "intent" {
				if err = s.ResolveSuboperation(ctx, scope(e), e.ID, m.MemberID, k, c, diagnostic(domain.Code), true); err != nil {
					return err
				}
			}
		}
		raw, _ := jsonBytes(map[string]any{"rejectedBeforeCreation": true, "code": domain.Code})
		if err := s.PutSuboperation(ctx, scope(e), e.ID, m.MemberID, c, evalstore.Suboperation{Kind: kind, Key: m.SubmissionKey + "-" + kind, Request: raw, State: "intent"}); err != nil {
			return err
		}
		if err := s.ResolveSuboperation(ctx, scope(e), e.ID, m.MemberID, kind, c, diagnostic(domain.Code), true); err != nil {
			return err
		}
		return s.Settle(ctx, scope(e), e.ID, m.MemberID, c)
	})
}
func (d *Driver) binding(ctx context.Context, e evalstore.Experiment, m evalstore.Member) (BindingSnapshot, error) {
	resource, err := evalstore.NewPostgresStore(d.Pool).PlanResource(ctx, e.OwnerID, e.ID, "bindings/"+m.VariantID+".json")
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
func (d *Driver) inputs(ctx context.Context, e evalstore.Experiment, m evalstore.Member, c evalstore.Claim, projectID string) (map[string]contracts.ArtifactRef, error) {
	op, err := d.operation(ctx, e, m, c, "inputs", func() (any, error) {
		inputs, err := MapInputs(evaldomain.Case{Inputs: m.Recipe.Case.Inputs}, m.Recipe.Variant)
		return map[string]any{"inputs": inputs, "projectId": projectID}, err
	})
	if err != nil {
		return nil, err
	}
	if op.State == "succeeded" {
		var refs map[string]contracts.ArtifactRef
		err = json.Unmarshal(op.Response, &refs)
		return refs, err
	}
	var request struct {
		Inputs    map[string]evaldomain.Artifact `json:"inputs"`
		ProjectID string                         `json:"projectId"`
	}
	if err = json.Unmarshal(op.Request, &request); err != nil {
		return nil, err
	}
	refs := map[string]contracts.ArtifactRef{}
	err = d.tx(ctx, func(s *evalstore.Store, tx pgx.Tx) error {
		if err := s.LockMemberRecovery(ctx, scope(e), e.ID, m.MemberID, c); err != nil {
			return err
		}
		service := artifacts.NewService(artifacts.NewPostgresRepository(tx))
		target, err := service.Project(request.ProjectID)
		if err != nil {
			return err
		}
		for slot, ref := range request.Inputs {
			if err = verifyArtifact(ctx, tx, service, e.OwnerID, ref); err != nil {
				return err
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
		return s.ResolveSuboperation(ctx, scope(e), e.ID, m.MemberID, "inputs", c, response, false)
	})
	return refs, err
}
func memberCase(m evalstore.Member) evaldomain.Case {
	return evaldomain.Case{ID: m.Recipe.Case.ID, Task: m.Recipe.Case.Task, Inputs: m.Recipe.Case.Inputs, Requires: m.Recipe.Case.Requires, Outputs: m.Recipe.Case.Outputs}
}

type runRequest struct {
	ID, ProjectID, Workflow, WorkflowSHA256, RuntimeSHA256, SkillsSHA256 string
	Inputs                                                               map[string]contracts.ArtifactRef
	Parameters                                                           map[string]string
}

func (d *Driver) createRun(ctx context.Context, e evalstore.Experiment, m evalstore.Member, c evalstore.Claim) error {
	op, err := d.operation(ctx, e, m, c, "run-create", func() (any, error) {
		inputs, err := d.inputs(ctx, e, m, c, e.ProjectID)
		if err != nil {
			return nil, err
		}
		snapshot, err := d.binding(ctx, e, m)
		if err != nil {
			return nil, err
		}
		w, err := hashJSON(snapshot.Workflow)
		if err != nil {
			return nil, err
		}
		r, err := hashJSON(snapshot.Runtime)
		if err != nil {
			return nil, err
		}
		skills, err := hashJSON(snapshot.Skills)
		if err != nil {
			return nil, err
		}
		return runRequest{ID: "run-" + m.SubmissionKey, ProjectID: e.ProjectID, Workflow: m.Recipe.Variant.Selector, WorkflowSHA256: w, RuntimeSHA256: r, SkillsSHA256: skills, Inputs: inputs, Parameters: MapParameters(memberCase(m), m.Recipe.Variant)}, nil
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
	result, err := d.Runs.CreatePublic(ctx, runservice.PublicCreateParams{OwnerID: e.OwnerID, ProjectID: &request.ProjectID, Workflow: request.Workflow, ExecutionConfig: m.Recipe.Variant.ExecutionConfig, RuntimeLabels: m.Recipe.Variant.RuntimeLabels, Inputs: request.Inputs, Parameters: request.Parameters, IdempotencyKey: op.Key, RequestDigest: evaldomain.Digest(op.Request), NewRunID: func() (string, error) { return request.ID, nil }, ExpectedWorkflowSHA256: request.WorkflowSHA256, ExpectedRuntimeSHA256: request.RuntimeSHA256, ExpectedSkillsSHA256: request.SkillsSHA256})
	if err != nil {
		return d.failDefinite(ctx, e, m, c, op, err)
	}
	if err = d.resolve(ctx, e, m, c, op, map[string]string{"id": result.Run.RunID}, false, result.Run.RunID); err == nil && d.Notifier != nil {
		d.Notifier.Wake()
	}
	return err
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

func (d *Driver) auditProject(ctx context.Context, e evalstore.Experiment, m evalstore.Member, c evalstore.Claim) (string, error) {
	op, err := d.operation(ctx, e, m, c, "project-create", func() (any, error) { return map[string]string{"projectId": "project-" + m.SubmissionKey}, nil })
	if err != nil {
		return "", err
	}
	var request map[string]string
	if err = json.Unmarshal(op.Request, &request); err != nil {
		return "", err
	}
	id := request["projectId"]
	if op.State == "succeeded" {
		return id, nil
	}
	err = d.tx(ctx, func(s *evalstore.Store, tx pgx.Tx) error {
		if err := s.LockMemberRecovery(ctx, scope(e), e.ID, m.MemberID, c); err != nil {
			return err
		}
		_, _, err := projectstore.NewPostgresStore(tx).Create(ctx, projectstore.CreateParams{OwnerID: e.OwnerID, ProjectID: id, Kind: projectstore.KindProject, Name: "Eval " + m.MemberID, IdempotencyKey: op.Key, RequestDigest: evaldomain.Digest(op.Request)})
		if err != nil {
			return err
		}
		if err = s.RegisterExecutionProject(ctx, scope(e), e.ID, m.MemberID, id, op.Key, c); err != nil {
			return err
		}
		raw, _ := jsonBytes(map[string]string{"id": id})
		return s.ResolveSuboperation(ctx, scope(e), e.ID, m.MemberID, op.Kind, c, raw, false)
	})
	return id, err
}
func (d *Driver) createAudit(ctx context.Context, e evalstore.Experiment, m evalstore.Member, c evalstore.Claim) error {
	op, err := d.operation(ctx, e, m, c, "audit-create", func() (any, error) {
		projectID, err := d.auditProject(ctx, e, m, c)
		if err != nil {
			return nil, err
		}
		inputs, err := d.inputs(ctx, e, m, c, projectID)
		if err != nil {
			return nil, err
		}
		snapshot, err := d.binding(ctx, e, m)
		if err != nil {
			return nil, err
		}
		if snapshot.Audit == nil {
			return nil, evaldomain.Failure("eval_pin_mismatch")
		}
		digest, err := hashJSON(snapshot.Audit)
		if err != nil {
			return nil, err
		}
		params := MapParameters(memberCase(m), m.Recipe.Variant)
		objective := m.Recipe.Case.Task.Objective
		if value, ok := params["objective"]; ok {
			objective = value
		}
		return auditRequest{ID: "audit-" + m.SubmissionKey, ProjectID: projectID, ProfileSHA256: digest, Profile: auditservice.ProfileSelector{Name: snapshot.Audit.Ref.Name, Version: snapshot.Audit.Ref.Version}, Inputs: inputs, Scope: auditservice.Scope{Objective: objective, Target: params["target"], AuthorizationScope: params["authorizationScope"]}}, nil
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
	audit, _, err := d.Audits.CreateDraft(ctx, auditservice.CreateDraftParams{OwnerID: e.OwnerID, ProjectID: request.ProjectID, AuditID: request.ID, Profile: request.Profile, Inputs: request.Inputs, RuntimeLabels: m.Recipe.Variant.RuntimeLabels, Scope: request.Scope, IdempotencyKey: op.Key, RequestDigest: evaldomain.Digest(op.Request), ExpectedProfileSHA256: request.ProfileSHA256})
	if err != nil {
		return d.failDefinite(ctx, e, m, c, op, err)
	}
	if err = d.resolve(ctx, e, m, c, op, map[string]string{"id": audit.AuditID}, false, ""); err != nil {
		return err
	}
	// The Audit exists, but no managed execution receipt is confirmed until its
	// separate start operation succeeds. Cancelling an unstarted draft uses the
	// ordinary deletion path and later its authoritative deletion tombstone.
	if e.State == "cancelling" && audit.State == auditstore.AuditDraft {
		return d.cancelAuditDraft(ctx, e, m, c, audit)
	}
	if audit.State == auditstore.AuditDeleting {
		return nil // The ordinary Audit controller will retain a deletion tombstone.
	}
	start, err := d.operation(ctx, e, m, c, "audit-start", func() (any, error) {
		snapshot, err := d.binding(ctx, e, m)
		if err != nil {
			return nil, err
		}
		runtime, err := hashJSON(snapshot.Runtime)
		if err != nil {
			return nil, err
		}
		skills, err := hashJSON(snapshot.Skills)
		if err != nil {
			return nil, err
		}
		standards, err := hashJSON(snapshot.Standards)
		return startRequest{audit.AuditID, audit.Revision, runtime, skills, standards}, err
	})
	if err != nil {
		return err
	}
	if start.State == "rejected" {
		return d.cancelAuditDraft(ctx, e, m, c, audit)
	}
	var sr startRequest
	if err = json.Unmarshal(start.Request, &sr); err != nil {
		return err
	}
	result, err := d.Audits.Start(ctx, auditservice.StartParams{OwnerID: e.OwnerID, AuditID: sr.AuditID, ExpectedRevision: sr.Revision, IdempotencyKey: start.Key, RequestDigest: evaldomain.Digest(start.Request), ExpectedRuntimeSHA256: sr.RuntimeSHA256, ExpectedSkillsSHA256: sr.SkillsSHA256, ExpectedStandardsSHA256: sr.StandardsSHA256})
	if err != nil {
		return d.failDefinite(ctx, e, m, c, start, err)
	}
	return d.resolve(ctx, e, m, c, start, map[string]string{"id": result.Audit.AuditID}, false, result.Audit.AuditID)
}
func (d *Driver) cancelAuditDraft(ctx context.Context, e evalstore.Experiment, m evalstore.Member, c evalstore.Claim, a auditstore.Audit) error {
	op, err := d.operation(ctx, e, m, c, "cancel", func() (any, error) { return map[string]any{"id": a.AuditID, "kind": "audit", "deleteDraft": true}, nil })
	if err != nil {
		return err
	}
	if op.State == "succeeded" {
		return nil
	}
	result, err := d.Audits.Delete(ctx, auditservice.MutationParams{OwnerID: e.OwnerID, AuditID: a.AuditID, ExpectedRevision: a.Revision, IdempotencyKey: op.Key + "-" + strconv.FormatUint(a.Revision, 10), RequestDigest: evaldomain.Digest(append(op.Request, []byte(strconv.FormatUint(a.Revision, 10))...))})
	if err != nil {
		return err
	}
	return d.resolve(ctx, e, m, c, op, map[string]any{"id": result.Audit.AuditID, "deleteDraft": true}, false, "")
}
func (d *Driver) cancel(ctx context.Context, e evalstore.Experiment, m evalstore.Member, c evalstore.Claim, sub evalstore.Submission) error {
	if sub.ExecutionID == nil {
		return evaldomain.Failure("eval_member_conflict")
	}
	op, err := d.operation(ctx, e, m, c, "cancel", func() (any, error) {
		return map[string]any{"id": *sub.ExecutionID, "kind": m.ExecutionKind, "requestedAt": time.Now().UTC().Format(time.RFC3339Nano)}, nil
	})
	if err != nil {
		return err
	}
	if op.State == "succeeded" {
		return nil
	}
	if m.ExecutionKind == "run" {
		var request struct {
			RequestedAt time.Time `json:"requestedAt"`
		}
		if err = json.Unmarshal(op.Request, &request); err != nil {
			return err
		}
		reason := "Evaluation experiment stopped."
		owner := e.OwnerID
		run, err := runstore.NewPostgresStore(d.Pool).RequestRunCancellation(ctx, *sub.ExecutionID, runstore.WorkflowRunCancellation{Code: runstore.CancellationUserRequested, RequestedAt: request.RequestedAt, RequestedBy: &owner, Reason: &reason})
		if err != nil {
			return err
		}
		if d.Notifier != nil && run.State == runstore.RunCancelling {
			d.Notifier.Cancel(run.RunID)
		}
	} else {
		audit, err := d.Audits.Get(ctx, e.OwnerID, *sub.ExecutionID)
		if err != nil {
			return err
		}
		if audit.State != auditstore.AuditCancelling && audit.State != auditstore.AuditCompleted && audit.State != auditstore.AuditCancelled && audit.State != auditstore.AuditFailed && audit.State != auditstore.AuditDeleting {
			// Each CAS attempt has a deterministic revision-scoped domain key. The
			// immutable outer intent is cancellation of this exact owned Audit.
			revision := strconv.FormatUint(audit.Revision, 10)
			_, err = d.Audits.Cancel(ctx, auditservice.MutationParams{OwnerID: e.OwnerID, AuditID: audit.AuditID, ExpectedRevision: audit.Revision, IdempotencyKey: op.Key + "-" + revision, RequestDigest: evaldomain.Digest(append(op.Request, []byte(revision)...))})
			if err != nil {
				return err
			}
		}
	}
	return d.resolve(ctx, e, m, c, op, map[string]any{"id": *sub.ExecutionID, "requested": true}, false, "")
}

// This bounded lower bound reads each StageMetrics row once, including all
// owned Audit roles. It never adds a parent aggregate to child totals. Full
// completeness/provenance normalization belongs to the collection read model.
func (d *Driver) knownTokens(ctx context.Context, e evalstore.Experiment, m evalstore.Member, sub evalstore.Submission) (int64, error) {
	if sub.ExecutionID == nil {
		return 0, nil
	}
	var total int64
	owned := `SELECT run_id FROM workflow_runs WHERE run_id=$1 AND owner_id=$2`
	if m.ExecutionKind == "audit" {
		owned = `SELECT x.run_id FROM audit_executions x JOIN audits a USING(audit_id) WHERE x.audit_id=$1 AND a.owner_id=$2 AND x.run_id IS NOT NULL ORDER BY x.execution_id LIMIT 10000`
	}
	err := d.Pool.QueryRow(ctx, `WITH owned AS (`+owned+`) SELECT COALESCE(sum(known_tokens),0) FROM (
 SELECT COALESCE((metrics.summary->>'totalTokens')::bigint,0) AS known_tokens
 FROM stage_metrics metrics JOIN stage_executions stage USING(stage_execution_id)
 JOIN owned ON owned.run_id=stage.run_id
 ORDER BY metrics.stage_execution_id LIMIT 10000) observed`, *sub.ExecutionID, e.OwnerID).Scan(&total)
	if err != nil {
		return 0, fmt.Errorf("read eval observed usage: %w", err)
	}
	return total, nil
}
