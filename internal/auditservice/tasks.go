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
	Assertions       map[string]HypothesisContent
	Attention        []string
	OwnerID          string
	AuditID          string
	ExpectedRevision uint64
	IdempotencyKey   string
	RequestDigest    string
	Hypotheses       map[string]string
	Work             []auditcontract.PreparedSelection
}

type TaskCreationResult struct {
	Acceptance auditstore.TaskAcceptance
	Replayed   bool
}

// TaskCreationRequestDigest binds a public key to the exact accepted request,
// including its Audit revision. All callers use this owner-defined identity.
func TaskCreationRequestDigest(p TaskCreationParams) (string, error) {
	request := map[string]any{"audit_id": p.AuditID, "revision": p.ExpectedRevision, "work": p.Work}
	if len(p.Hypotheses) > 0 {
		request["hypotheses"] = p.Hypotheses
	}
	if len(p.Attention) > 0 {
		request["attention"] = p.Attention
		request["hypotheses"] = p.Hypotheses
	}
	if len(p.Assertions) > 0 {
		request["assertions"] = p.Assertions
	}
	return contentdigest.CanonicalJSON(request)
}

// CreateTasks accepts explicit already available work. It submits no Run;
// paused intake uses exactly the same mutation as initial/prepared work.
func (s *Service) CreateTasks(ctx context.Context, p TaskCreationParams) (TaskCreationResult, error) {
	if len(p.Work) == 0 {
		return TaskCreationResult{}, ErrInvalid
	}

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
	pinned, err := executionProfile(audit)
	if err != nil {
		return TaskCreationResult{}, err
	}
	baseline, err := DecodeBaseline(audit.BaselineSnapshot)
	if err != nil {
		return TaskCreationResult{}, err
	}
	for _, item := range p.Work {
		for slot := range item.Inputs {
			mapping := pinned.Program.Workflows[item.Workflow].Inputs[slot]
			if mapping.Source == "prior-results" || mapping.Source == "task-context" {
				return TaskCreationResult{}, ErrInvalid
			}
		}
	}
	params := auditstore.AcceptTasksParams{Attention: p.Attention, Hypotheses: p.Hypotheses, OwnerID: p.OwnerID, AuditID: p.AuditID, ExpectedRevision: p.ExpectedRevision, IdempotencyKey: p.IdempotencyKey, RequestDigest: p.RequestDigest}
	seeds := map[string]auditcontract.Hypothesis{}
	for key, content := range p.Assertions {
		if len(p.Assertions)+len(p.Work) > auditcontract.MaxMutationRecords {
			return TaskCreationResult{}, ErrInvalid
		}
		found := false
		for _, work := range p.Work {
			found = found || work.Key == key
		}
		if !found {
			return TaskCreationResult{}, ErrInvalid
		}
		h := auditcontract.Hypothesis{Schema: auditcontract.HypothesisSchema, ID: auditdomain.DeterministicID("hypothesis", audit.AuditID, p.IdempotencyKey, content.Key), Revision: 1, Statement: content.Statement, Scope: content.Scope, Grounds: content.Grounds, Origin: auditcontract.HypothesisOrigin{Kind: "manual", AuditID: audit.AuditID}}
		fixed := false
		if id := p.Hypotheses[key]; id != "" {
			prior, err := s.reads().CurrentHypothesis(ctx, audit.OwnerID, audit.AuditID, id)
			if err != nil {
				return TaskCreationResult{}, err
			}
			if content.Key != prior.SourceKey {
				return TaskCreationResult{}, ErrInvalid
			}
			h.ID = prior.Document.ID
			h.Revision = prior.Document.Revision + 1
			h.Origin = prior.Document.Origin
			h.Sources = prior.Document.Sources
			fixed = prior.Fixed
		}
		data, err := strictjson.Canonical(h)
		if err != nil {
			return TaskCreationResult{}, ErrInvalid
		}
		h, err = auditcontract.DecodeHypothesis(data)
		if err != nil {
			return TaskCreationResult{}, err
		}
		seeds[key] = h
		params.HypothesisWrites = append(params.HypothesisWrites, auditstore.HypothesisWrite{Document: h, SourceKey: content.Key, Fixed: fixed})
	}
	params.AssertionSnapshots = seeds
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
	if len(work) == 0 && params.PreparedProgress == nil || len(work) > auditcontract.MaxMutationRecords {
		return TaskCreationResult{}, ErrInvalid
	}
	for key := range params.Hypotheses {
		found := false
		for _, w := range work {
			found = found || w.Key == key
		}
		if !found {
			return TaskCreationResult{}, ErrInvalid
		}
	}
	attention := map[string]bool{}
	for _, key := range params.Attention {
		if attention[key] {
			return TaskCreationResult{}, ErrInvalid
		}
		attention[key] = true
	}
	if len(attention) > len(work) {
		return TaskCreationResult{}, ErrInvalid
	}
	for key := range attention {
		found := false
		for _, w := range work {
			found = found || w.Key == key
		}
		if !found {
			return TaskCreationResult{}, ErrInvalid
		}
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
		for slot, mapping := range binding.Inputs {
			if mapping.Source == "task-results" {
				if _, exists := inputs[slot]; !exists {
					empty, err := s.retainResultSet(ctx, audit, params.Claim, nil)
					if err != nil {
						return TaskCreationResult{}, err
					}
					inputs[slot] = empty
				}
			}
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
		accepted := auditstore.AcceptedTask{Attention: attention[item.Key], Document: frozen, Source: origin}

		if id := params.Hypotheses[item.Key]; id != "" {
			h, err := s.reads().CurrentHypothesis(ctx, audit.OwnerID, audit.AuditID, id)
			if err != nil {
				return TaskCreationResult{}, err
			}
			accepted.Hypothesis = &h.Document
			accepted.Document.Hypothesis = &auditcontract.HypothesisRef{ID: h.Document.ID, Revision: h.Document.Revision}
		}

		if h, found := params.AssertionSnapshots[item.Key]; found {
			accepted.Hypothesis = &h
			accepted.Document.Hypothesis = &auditcontract.HypothesisRef{ID: h.ID, Revision: h.Revision}
		}
		if origin.Kind == "initial" {
			for _, initial := range pinned.Program.InitialWork {
				if initial.Key == origin.Key && initial.Hypothesis != "" {
					h := auditdomain.InitialHypothesis(audit.AuditID, initial.Hypothesis, pinned.Program.Hypotheses[initial.Hypothesis])
					if baseline.InitialAssertions != nil {
						if _, found := baseline.InitialAssertions.Hypotheses[initial.Hypothesis]; found {
							h.Origin.Source = &baseline.InitialAssertions.Source
						}
					}
					accepted.Hypothesis = &h
					accepted.Document.Hypothesis = &auditcontract.HypothesisRef{ID: h.ID, Revision: h.Revision}
					break
				}
			}
		}
		selected = append(selected, accepted)
	}
	for _, write := range params.HypothesisWrites {
		data, err := strictjson.Canonical(write.Document)
		used += len(data) + 1
		if err != nil || used > auditcontract.MaxMutationBytes {
			return TaskCreationResult{}, ErrInvalid
		}
		for _, material := range write.Document.Sources {
			key, err := contentdigest.CanonicalJSON(material)
			if err != nil {
				return TaskCreationResult{}, err
			}
			materials[key] = material
		}
	}
	if len(materials) > auditcontract.MaxMutationRecords {
		return TaskCreationResult{}, ErrInvalid
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
	retainedHypotheses := map[string]auditcontract.Hypothesis{}
	for i := range params.HypothesisWrites {
		h := params.HypothesisWrites[i].Document
		for j, m := range h.Sources {
			key, err := contentdigest.CanonicalJSON(m)
			if err != nil {
				return TaskCreationResult{}, err
			}
			h.Sources[j] = retained[key]
		}
		data, err := strictjson.Canonical(h)
		if err != nil {
			return TaskCreationResult{}, err
		}
		h, err = auditcontract.DecodeHypothesis(data)
		if err != nil {
			return TaskCreationResult{}, err
		}
		params.HypothesisWrites[i].Document = h
		retainedHypotheses[h.ID] = h
	}
	for i := range selected {
		if h := selected[i].Hypothesis; h != nil {
			if retained, exists := retainedHypotheses[h.ID]; exists {
				selected[i].Hypothesis = &retained
			}
		}
	}
	params.Tasks = selected
	accepted, replayed, err := s.owners().AcceptTasks(ctx, params)
	if err != nil {
		return TaskCreationResult{}, err
	}
	return TaskCreationResult{Acceptance: accepted, Replayed: replayed}, nil
}
