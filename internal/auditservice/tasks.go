package auditservice

import (
	"context"
	"maps"
	"sort"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

type TaskCreationParams struct {
	OwnerID          string
	AuditID          string
	ExpectedRevision uint64
	IdempotencyKey   string
	RequestDigest    string
	Work             []auditcontract.PreparedSelection
}

type TaskCreationResult struct {
	Acceptance auditstore.TaskAcceptance
	Replayed   bool
}

// TaskCreationRequestDigest binds a public key to the exact accepted request,
// including its Audit revision. All callers use this owner-defined identity.
func TaskCreationRequestDigest(auditID string, revision uint64, work []auditcontract.PreparedSelection) (string, error) {
	return contentdigest.CanonicalJSON(map[string]any{"audit_id": auditID, "revision": revision, "work": work})
}

// CreateTasks accepts explicit already available work. It submits no Run;
// paused intake uses exactly the same mutation as initial/prepared work.
func (s *Service) CreateTasks(ctx context.Context, p TaskCreationParams) (TaskCreationResult, error) {
	if !contracts.ValidResourceID(p.AuditID) || p.OwnerID == "" || p.ExpectedRevision == 0 || !contracts.ValidIdempotencyKey(p.IdempotencyKey) || !contentdigest.Valid(p.RequestDigest) {
		return TaskCreationResult{}, ErrInvalid
	}
	if prior, found, err := s.owners().LookupTaskAcceptance(ctx, p.OwnerID, p.AuditID, p.IdempotencyKey, p.RequestDigest); err != nil || found {
		return TaskCreationResult{Acceptance: prior, Replayed: found}, err
	}
	audit, err := s.reads().Get(ctx, p.OwnerID, p.AuditID)
	if err != nil {
		return TaskCreationResult{}, err
	}
	if audit.State != auditstore.AuditActive && audit.State != auditstore.AuditPaused || audit.Revision != p.ExpectedRevision {
		return TaskCreationResult{}, auditstore.ErrPrecondition
	}
	pinned, err := auditprofile.DecodePinnedProgram(audit.ProfileSnapshot)
	if err != nil {
		return TaskCreationResult{}, err
	}
	baseline, err := DecodeBaseline(audit.BaselineSnapshot)
	if err != nil {
		return TaskCreationResult{}, err
	}
	params := auditstore.AcceptTasksParams{OwnerID: p.OwnerID, AuditID: p.AuditID, ExpectedRevision: p.ExpectedRevision, IdempotencyKey: p.IdempotencyKey, RequestDigest: p.RequestDigest}
	return s.acceptCommonWork(ctx, audit, pinned, baseline, p.Work, params, func(item auditcontract.PreparedSelection) auditstore.TaskSource {
		return auditstore.TaskSource{Kind: "manual", Key: item.Key}
	})
}

func (s *Service) Tasks(ctx context.Context, params auditstore.ListTasksParams) (auditstore.TaskPage, error) {
	return s.reads().ListTasks(ctx, params)
}

func (s *Service) Task(ctx context.Context, ownerID, auditID, taskID string) (auditstore.Task, error) {
	return s.reads().GetTask(ctx, ownerID, auditID, taskID)
}

// acceptCommonWork is the shared owner path: resolve a finite selection,
// retain its exact material through ArtifactStore, then commit Task records
// and intake progress through the existing keyed mutation family.
func (s *Service) acceptCommonWork(ctx context.Context, audit auditstore.Audit, pinned auditprofile.PinnedProgram, baseline BaselineSnapshot, work []auditcontract.PreparedSelection, params auditstore.AcceptTasksParams, source func(auditcontract.PreparedSelection) auditstore.TaskSource) (TaskCreationResult, error) {
	if len(work) == 0 || len(work) > auditcontract.MaxMutationRecords {
		return TaskCreationResult{}, ErrInvalid
	}
	selected := make([]auditstore.AcceptedTask, 0, len(work))
	materials := map[string]auditcontract.Material{}
	seen := map[string]bool{}
	used := 1
	for _, item := range work {
		if !contracts.ValidName(item.Key) || seen[item.Key] || item.Inputs == nil {
			return TaskCreationResult{}, ErrInvalid
		}
		seen[item.Key] = true
		binding, allowed := pinned.Program.Workflows[item.Workflow]
		workflow, closure := pinned.Workflows[item.Workflow]
		if !allowed || !closure {
			return TaskCreationResult{}, ErrInvalid
		}
		inputs := map[string]auditcontract.Material{}
		for slot, mapping := range binding.Inputs {
			if mapping.Source != "audit-input" {
				continue
			}
			if input, found := baseline.Inputs[mapping.Name]; found {
				inputs[slot] = auditcontract.Material{ScopeKind: "project", ScopeID: audit.ProjectID, Ref: input.Ref.Clone(), Digest: input.Digest, MediaType: input.MediaType, SizeBytes: input.SizeBytes}
			}
		}
		for slot, material := range item.Inputs {
			inputs[slot] = material
		}
		if err := auditprofile.ValidateTaskInputs(binding, workflow, inputs); err != nil {
			return TaskCreationResult{}, err
		}
		for slot, contract := range workflow.Inputs {
			mapping := binding.Inputs[slot]
			if contract.Required && (mapping.Source == "audit-input" || mapping.Source == "task-input") {
				if _, available := inputs[slot]; !available {
					return TaskCreationResult{}, ErrInvalid
				}
			}
		}
		origin := source(item)
		sourceKey, err := contentdigest.CanonicalJSON(origin)
		if err != nil {
			return TaskCreationResult{}, err
		}
		document := auditcontract.Task{Schema: auditcontract.TaskSchema, AuditID: audit.AuditID, TaskID: auditdomain.DeterministicID("task", audit.AuditID, sourceKey), Workflow: item.Workflow, Inputs: inputs, Parameters: maps.Clone(binding.Parameters), Objective: item.Objective}
		data, err := strictjson.Canonical(document)
		used += len(data) + 1
		if err != nil || used > auditcontract.MaxMutationBytes {
			return TaskCreationResult{}, ErrInvalid
		}
		frozen, err := auditcontract.DecodeTask(data)
		if err != nil {
			return TaskCreationResult{}, err
		}
		for _, material := range frozen.Inputs {
			key, err := contentdigest.CanonicalJSON(material)
			if err != nil {
				return TaskCreationResult{}, err
			}
			materials[key] = material
		}
		if len(materials) > auditcontract.MaxMutationRecords {
			return TaskCreationResult{}, ErrInvalid
		}
		selected = append(selected, auditstore.AcceptedTask{Document: frozen, Source: origin})
	}
	keys := make([]string, 0, len(materials))
	for key := range materials {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	retained := map[string]auditcontract.Material{}
	preparedExecution := ""
	if params.PreparedProgress != nil {
		preparedExecution = params.PreparedProgress.ExecutionID
	}
	for _, key := range keys {
		material, err := s.retainTaskInput(ctx, audit, params.Claim, params.ExpectedRevision, materials[key], preparedExecution)
		if err != nil {
			return TaskCreationResult{}, err
		}
		retained[key] = material
	}
	for index := range selected {
		for slot, material := range selected[index].Document.Inputs {
			key, err := contentdigest.CanonicalJSON(material)
			if err != nil {
				return TaskCreationResult{}, err
			}
			selected[index].Document.Inputs[slot] = retained[key]
		}
	}
	params.Tasks = selected
	accepted, replayed, err := s.owners().AcceptTasks(ctx, params)
	if err != nil {
		return TaskCreationResult{}, err
	}
	return TaskCreationResult{Acceptance: accepted, Replayed: replayed}, nil
}
