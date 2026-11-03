package auditservice

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strconv"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"github.com/grauwolf32/contractor/internal/projectstore"
)

// AcceptInitialTasks uses the same acceptance mutation as explicit API work.
// Its only extra state is an offset into the immutable authored initial list.
func (s *Service) AcceptInitialTasks(ctx context.Context, claim auditstore.ControllerClaim, snapshot auditstore.ReconcileSnapshot) (auditstore.TaskIntakeResult, error) {
	audit := snapshot.Audit
	if claim.AuditID != audit.AuditID {
		return auditstore.TaskIntakeResult{}, ErrInvalid
	}
	pinned, err := auditprofile.DecodePinnedProgram(audit.ProfileSnapshot)
	if err != nil {
		return auditstore.TaskIntakeResult{}, err
	}
	if audit.InitialWorkOffset >= len(pinned.Program.InitialWork) {
		return auditstore.TaskIntakeResult{Complete: true}, nil
	}
	if audit.State != auditstore.AuditActive && audit.State != auditstore.AuditPaused {
		return auditstore.TaskIntakeResult{Complete: true}, nil
	}
	baseline, err := DecodeBaseline(audit.BaselineSnapshot)
	if err != nil {
		return auditstore.TaskIntakeResult{}, err
	}
	work := make([]auditcontract.PreparedSelection, 0, auditcontract.MaxMutationRecords)
	for _, initial := range pinned.Program.InitialWork[audit.InitialWorkOffset:min(len(pinned.Program.InitialWork), audit.InitialWorkOffset+auditcontract.MaxMutationRecords)] {
		if initial.Hypothesis != "" {
			return auditstore.TaskIntakeResult{}, fmt.Errorf("%w: initial hypothesis snapshots require their accepting owner", ErrInvalid)
		}
		inputs := map[string]auditcontract.Material{}
		for slot, selected := range initial.Inputs {
			input, exists := baseline.Inputs[selected]
			if !exists {
				return auditstore.TaskIntakeResult{}, ErrInvalid
			}
			inputs[slot] = auditcontract.Material{ScopeKind: "project", ScopeID: audit.ProjectID, Ref: input.Ref, Digest: input.Digest, MediaType: input.MediaType, SizeBytes: input.SizeBytes}
		}
		work = append(work, auditcontract.PreparedSelection{Key: initial.Key, Workflow: initial.Workflow, Objective: initial.Objective, Inputs: inputs})
	}
	prefix, err := boundedTaskWork(audit, pinned, baseline, work)
	if err != nil {
		return auditstore.TaskIntakeResult{}, err
	}
	if len(prefix) == 0 {
		return auditstore.TaskIntakeResult{Complete: true, GapCode: "task-budget-exhausted"}, nil
	}
	progress := &auditstore.InitialWorkProgress{From: audit.InitialWorkOffset, To: audit.InitialWorkOffset + len(prefix)}
	digest, err := contentdigest.CanonicalJSON(map[string]any{"profile": pinned.Ref.Digest, "from": progress.From, "to": progress.To, "work": prefix})
	if err != nil {
		return auditstore.TaskIntakeResult{}, err
	}
	params := auditstore.AcceptTasksParams{OwnerID: audit.OwnerID, AuditID: audit.AuditID, Claim: &claim, InitialProgress: progress, IdempotencyKey: auditdomain.DeterministicID("initial", audit.AuditID, strconv.Itoa(progress.From)), RequestDigest: digest}
	result, err := s.acceptCommonWork(ctx, audit, pinned, baseline, prefix, params, func(item auditcontract.PreparedSelection) auditstore.TaskSource {
		return auditstore.TaskSource{Kind: "initial", Key: item.Key}
	})
	return auditstore.TaskIntakeResult{Changed: err == nil && !result.Replayed, Complete: err == nil && progress.To == len(pinned.Program.InitialWork)}, err
}

// AcceptPreparedWork accepts only a declared logical work output of a pinned
// ordinary producer. Source metadata is verified before common validation;
// the source Run remains collecting until this cursor is drained or closed.
func (s *Service) AcceptPreparedWork(ctx context.Context, claim auditstore.ControllerClaim, snapshot auditstore.ReconcileSnapshot, execution auditstore.Execution, source auditcontract.Material, data []byte) (auditstore.TaskIntakeResult, error) {
	audit := snapshot.Audit
	if claim.AuditID != audit.AuditID || execution.AuditID != audit.AuditID || execution.RunID == nil || source.ScopeKind != "run" || source.ScopeID != *execution.RunID || source.Digest != contentdigest.Bytes(data) || source.SizeBytes != int64(len(data)) || len(data) > auditcontract.MaxPreparedWorkBytes {
		return auditstore.TaskIntakeResult{}, ErrInvalid
	}
	pinned, err := auditprofile.DecodePinnedProgram(audit.ProfileSnapshot)
	if err != nil {
		return auditstore.TaskIntakeResult{}, err
	}
	binding, exists := pinned.Program.Workflows[execution.WorkflowRole]
	if !exists || source.Ref.Namespace != "outputs" || source.Ref.Name != binding.Outputs["work"] {
		return auditstore.TaskIntakeResult{}, ErrInvalid
	}
	run, err := artifacts.NewService(s.storage.Repository(s.pool)).Run(*execution.RunID)
	if err != nil {
		return auditstore.TaskIntakeResult{}, err
	}
	metadata, err := run.Metadata(ctx, source.Ref)
	if err != nil {
		return auditstore.TaskIntakeResult{}, err
	}
	if !metadata.Frozen || exactTaskMaterial(source).Verify(metadata.Exact()) != nil {
		return auditstore.TaskIntakeResult{}, ErrInvalid
	}
	canonical, err := auditcontract.Decode(data, nil)
	if err != nil {
		return auditstore.TaskIntakeResult{}, err
	}
	var prepared auditcontract.PreparedWork
	if strictjson.Decode(canonical, &prepared) != nil || prepared.Schema != auditcontract.PreparedWorkSchema {
		return auditstore.TaskIntakeResult{}, ErrInvalid
	}
	materials := map[string]auditcontract.Material{}
	for _, item := range prepared.Work {
		for _, material := range item.Inputs {
			if material.ScopeKind != "run" || material.ScopeID != *execution.RunID {
				return auditstore.TaskIntakeResult{}, ErrInvalid
			}
			keyBytes, err := strictjson.Canonical(material.Ref)
			if err != nil || len(keyBytes) > auditstore.MaxArtifactRefBytes {
				return auditstore.TaskIntakeResult{}, ErrInvalid
			}
			key := string(keyBytes)
			if prior, exists := materials[key]; exists {
				before, _ := strictjson.Canonical(prior)
				after, _ := strictjson.Canonical(material)
				if string(before) != string(after) {
					return auditstore.TaskIntakeResult{}, ErrInvalid
				}
			}
			materials[key] = material
		}
	}
	if len(materials) > auditcontract.MaxPreparedMaterials {
		return auditstore.TaskIntakeResult{}, ErrInvalid
	}
	keys := slices.Sorted(maps.Keys(materials))
	refs := make([]contracts.ArtifactRef, 0, len(keys))
	for _, key := range keys {
		refs = append(refs, materials[key].Ref)
	}
	trusted := auditcontract.PreparedContext{Schema: auditcontract.PreparedContextSchema, Program: pinned.Program, ProducerRunID: *execution.RunID, Materials: []auditcontract.Material{}}
	for start := 0; start < len(refs); start += artifacts.MaxExactReadBatchSize {
		end := min(start+artifacts.MaxExactReadBatchSize, len(refs))
		metadata, err := run.DescribeExact(ctx, refs[start:end])
		if err != nil {
			return auditstore.TaskIntakeResult{}, err
		}
		if len(metadata) != end-start {
			return auditstore.TaskIntakeResult{}, ErrInvalid
		}
		for index, exact := range metadata {
			selected := materials[keys[start+index]]
			if exactTaskMaterial(selected).Verify(exact) != nil {
				return auditstore.TaskIntakeResult{}, ErrInvalid
			}
			trusted.Materials = append(trusted.Materials, selected)
		}
	}
	contextData, err := strictjson.Canonical(trusted)
	if err != nil {
		return auditstore.TaskIntakeResult{}, err
	}
	prepared, _, err = auditcontract.DecodePreparedWork(data, contextData)
	if err != nil {
		return auditstore.TaskIntakeResult{}, err
	}
	for _, item := range prepared.Work {
		binding := pinned.Program.Workflows[item.Workflow]
		if err := auditprofile.ValidateTaskInputs(binding, pinned.Workflows[item.Workflow], item.Inputs); err != nil {
			return auditstore.TaskIntakeResult{}, err
		}
	}
	cursor, err := s.owners().RecordPreparedSource(ctx, auditstore.PreparedSourceParams{Claim: claim, ExecutionID: execution.ExecutionID, Source: source, Count: len(prepared.Work)})
	if err != nil {
		return auditstore.TaskIntakeResult{}, err
	}
	if !cursor.Accepting || cursor.Offset == cursor.Count {
		return auditstore.TaskIntakeResult{Complete: true}, nil
	}
	current, err := s.reads().Get(ctx, audit.OwnerID, audit.AuditID)
	if err != nil {
		return auditstore.TaskIntakeResult{}, err
	}
	baseline, err := DecodeBaseline(current.BaselineSnapshot)
	if err != nil {
		return auditstore.TaskIntakeResult{}, err
	}
	prefix, err := boundedTaskWork(current, pinned, baseline, prepared.Work[cursor.Offset:])
	if err != nil {
		return auditstore.TaskIntakeResult{}, err
	}
	if len(prefix) == 0 {
		return auditstore.TaskIntakeResult{Complete: true, GapCode: "task-budget-exhausted"}, nil
	}
	progress := &auditstore.PreparedWorkProgress{ExecutionID: execution.ExecutionID, From: cursor.Offset, To: cursor.Offset + len(prefix)}
	digest, err := contentdigest.CanonicalJSON(map[string]any{"source": source, "from": progress.From, "to": progress.To, "work": prefix})
	if err != nil {
		return auditstore.TaskIntakeResult{}, err
	}
	params := auditstore.AcceptTasksParams{OwnerID: audit.OwnerID, AuditID: audit.AuditID, Claim: &claim, PreparedProgress: progress, IdempotencyKey: auditdomain.DeterministicID("prepared", execution.ExecutionID, strconv.Itoa(progress.From)), RequestDigest: digest}
	accepted, err := s.acceptCommonWork(ctx, current, pinned, baseline, prefix, params, func(item auditcontract.PreparedSelection) auditstore.TaskSource {
		return auditstore.TaskSource{Kind: "prepared", Key: item.Key, ExecutionID: execution.ExecutionID, Output: &source}
	})
	if errors.Is(err, projectstore.ErrDeleting) {
		return auditstore.TaskIntakeResult{Complete: true, GapCode: "parent-deleting"}, nil
	}
	if errors.Is(err, auditstore.ErrTaskBudgetExhausted) {
		return auditstore.TaskIntakeResult{Complete: true, GapCode: "task-budget-exhausted"}, nil
	}
	if errors.Is(err, auditstore.ErrPrecondition) {
		latest, readErr := s.reads().Get(ctx, audit.OwnerID, audit.AuditID)
		if readErr != nil {
			return auditstore.TaskIntakeResult{}, readErr
		}
		if latest.State != auditstore.AuditActive && latest.State != auditstore.AuditPaused {
			return auditstore.TaskIntakeResult{Complete: true, GapCode: "parent-closed"}, nil
		}
	}
	return auditstore.TaskIntakeResult{Complete: err == nil && progress.To == cursor.Count, Changed: err == nil && !accepted.Replayed}, err
}

// The estimate reserves 8 KiB per exact descriptor and per source/identity
// envelope. Actual retained refs have the existing 4 KiB reference bound;
// parameters/objectives are measured canonically. No payload or new identity
// is written to choose this conservative finite prefix.
func boundedTaskWork(audit auditstore.Audit, pinned auditprofile.PinnedProgram, baseline BaselineSnapshot, work []auditcontract.PreparedSelection) ([]auditcontract.PreparedSelection, error) {
	maximum := min(auditcontract.MaxMutationRecords, auditstore.RemainingTaskCapacity(audit), len(work))
	used := 1
	count := 0
	for _, item := range work[:maximum] {
		binding, exists := pinned.Program.Workflows[item.Workflow]
		if !exists {
			return nil, ErrInvalid
		}
		inputs := map[string]bool{}
		for slot, mapping := range binding.Inputs {
			if mapping.Source == "audit-input" {
				if _, exists := baseline.Inputs[mapping.Name]; exists {
					inputs[slot] = true
				}
			}
		}
		for slot := range item.Inputs {
			inputs[slot] = true
		}
		base, err := strictjson.Canonical(map[string]any{"workflow": item.Workflow, "parameters": binding.Parameters, "objective": item.Objective})
		if err != nil {
			return nil, err
		}
		cost := len(base) + (len(inputs)+1)*(8<<10)
		if cost > auditcontract.MaxMutationBytes-used {
			break
		}
		used += cost
		count++
	}
	if maximum > 0 && count == 0 {
		return nil, ErrInvalid
	}
	return work[:count], nil
}
