package auditservice

import (
	"context"
	"reflect"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
)

type JudgmentStaleError struct {
	Generations []auditcontract.JudgmentGeneration
}

func (*JudgmentStaleError) Error() string { return "Check judgment generation changed" }
func judgmentKey(group string, grant controlplane.AllocationGrant, input auditcontract.JudgmentSubmission) string {
	return "judgment/" + group + "/" + auditdomain.DeterministicID("receipt", grant.AllocationID, input.InvocationID, input.SubmissionID)
}

// RecordJudgments uses one normalized collector/publisher for both owners.
// The immutable ordinary Artifact receipt is the acknowledgement, not Run success.
func (s *Service) RecordJudgments(ctx context.Context, grant controlplane.AllocationGrant, input auditcontract.JudgmentSubmission) (auditcontract.JudgmentReceipt, error) {
	if input.Validate() != nil || grant.ReadPolicy != controlplane.ReadCurrentRun {
		return auditcontract.JudgmentReceipt{}, ErrInvalid
	}
	run, err := runstore.NewPostgresStore(s.pool, s.storage).GetRun(ctx, grant.RunID)
	if err != nil {
		return auditcontract.JudgmentReceipt{}, err
	}
	owner, err := loadResultOwner(ctx, s.pool, s.storage, run)
	if err != nil {
		return auditcontract.JudgmentReceipt{}, err
	}
	if saved, found, err := replayJudgment(ctx, owner, grant, input); found || err != nil {
		return saved, err
	}
	if grant.WriteFenced || grant.Lost || grant.WritePolicy != controlplane.WriteInputsAndIntermediates {
		return auditcontract.JudgmentReceipt{}, auditstore.ErrPrecondition
	}
	var receipt auditcontract.JudgmentReceipt
	err = postgres.InTxWithRetry(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		o, err := loadResultOwner(ctx, tx, s.storage, run)
		if err != nil {
			return err
		}
		if err = o.lock(ctx, tx, s.storage); err != nil {
			return err
		}
		tool := "submit_result"
		if input.Mode == "batch" {
			tool = "submit_results"
		}
		if err = checkResultGrant(ctx, runstore.NewPostgresStore(tx, s.storage), run, grant, tool); err != nil {
			return err
		}
		if saved, found, err := replayJudgment(ctx, o, grant, input); found || err != nil {
			receipt = saved
			return err
		}
		if input.Mode == "batch" && len(input.Items) != len(o.members) {
			return ErrInvalid
		}
		for _, update := range input.Items {
			if _, ok := o.members[update.Conclusion.ItemKey]; !ok {
				return ErrInvalid
			}
		}
		latest, err := loadJudgments(ctx, o, input.Items)
		if err != nil {
			return err
		}
		normalized, err := normalizeJudgments(ctx, o, input.Items)
		if err != nil {
			return err
		}
		normalized, generations, err := assignJudgmentGenerations(input, o, latest, normalized)
		if err != nil {
			return err
		}
		data, err := strictjson.Canonical(normalized)
		if err != nil {
			return err
		}
		if err = o.admit(ctx, "judgment", 0, int64(len(data))); err != nil {
			return err
		}
		metadata := judgmentMetadata(o, grant, input)
		metadata.Generations = generations
		provenance, err := strictjson.Canonical(metadata)
		if err != nil || metadata.Validate() != nil {
			return ErrInvalid
		}
		ref := contracts.ArtifactRef{Namespace: o.namespace, Name: auditdomain.DeterministicID("judgment", grant.AllocationID, input.InvocationID, input.SubmissionID)}
		payload := artifacts.Payload{MediaType: "application/json", Data: data}
		written, err := o.artifacts.CreateFrozen(ctx, o.scope, ref, payload)
		if err != nil {
			return err
		}
		material := o.material(written.Ref, payload)
		link := auditstore.ArtifactLink{Scope: o.scope.Kind(), LogicalKey: judgmentKey(o.group, grant, input), Artifact: material.Exact(), SourceProvenance: provenance}
		if err = o.retain(ctx, "judgment", []auditstore.ArtifactLink{link}, 0, 0); err != nil {
			return err
		}
		receipt = auditcontract.JudgmentReceipt{APIVersion: contracts.APIVersion, Material: material, Generations: generations}
		return nil
	})
	return receipt, err
}

func judgmentMetadata(o *resultOwner, grant controlplane.AllocationGrant, input auditcontract.JudgmentSubmission) auditcontract.JudgmentMutation {
	request, _ := strictjson.Canonical(input)
	v := auditcontract.JudgmentMutation{Schema: "contractor.audit.result-receipt.v1", RequestDigest: auditdomain.DigestBytes(request), RunID: o.run.RunID, AllocationID: grant.AllocationID, StageExecutionID: grant.StageExecutionID, LogicalAgentName: grant.LogicalAgentName, InvocationID: input.InvocationID, SubmissionID: input.SubmissionID}
	if o.audit != nil {
		v.AuditID = o.audit.AuditID
		v.ExecutionID = o.group
	} else {
		v.Finding = o.run.AuditCompletion.Finding
	}
	return v
}
func replayJudgment(ctx context.Context, o *resultOwner, grant controlplane.AllocationGrant, input auditcontract.JudgmentSubmission) (auditcontract.JudgmentReceipt, bool, error) {
	link, found, err := o.readLink(ctx, judgmentKey(o.group, grant, input))
	if !found || err != nil {
		return auditcontract.JudgmentReceipt{}, found, err
	}
	store, _ := o.artifacts.Scoped(o.scope)
	read, err := store.ReadExact(ctx, link.Artifact)
	if err != nil {
		return auditcontract.JudgmentReceipt{}, false, err
	}
	metadata, _, err := auditcontract.DecodeJudgment(read.Payload.Data, link.SourceProvenance)
	expected := judgmentMetadata(o, grant, input)
	expected.Generations = metadata.Generations
	if err != nil || !reflect.DeepEqual(metadata, expected) {
		return auditcontract.JudgmentReceipt{}, false, auditstore.ErrConflict
	}
	return auditcontract.JudgmentReceipt{APIVersion: contracts.APIVersion, Material: o.material(read.Ref, read.Payload), Generations: metadata.Generations, Replayed: true}, true, nil
}
func loadJudgments(ctx context.Context, o *resultOwner, updates []auditcontract.JudgmentUpdate) (map[string]auditcontract.AcceptedJudgment, error) {
	links, err := o.list(ctx, "judgment")
	if err != nil {
		return nil, err
	}
	store, _ := o.artifacts.Scoped(o.scope)
	records := []auditcontract.StoredJudgment{}
	wanted := map[string]bool{}
	for _, update := range updates {
		wanted[update.Conclusion.ItemKey] = true
	}
	var size int64
	for _, link := range links {
		var metadata auditcontract.JudgmentMutation
		if strictjson.Decode(link.SourceProvenance, &metadata) != nil || metadata.Validate() != nil || metadata.RunID != o.run.RunID {
			return nil, ErrInvalid
		}
		needed := false
		for _, generation := range metadata.Generations {
			member, ok := o.members[generation.ItemKey]
			if !ok || generation.MemberID() != member.id {
				return nil, ErrInvalid
			}
			needed = needed || wanted[generation.ItemKey]
		}
		if !needed {
			continue
		}
		size += link.Artifact.SizeBytes
		if size > auditcontract.MaximumDocumentBytes {
			return nil, ErrInvalid
		}
		read, err := store.ReadExact(ctx, link.Artifact)
		if err != nil {
			return nil, err
		}
		records = append(records, auditcontract.StoredJudgment{Material: o.material(read.Ref, read.Payload), Data: read.Payload.Data, Provenance: link.SourceProvenance})
	}
	if o.audit != nil {
		return auditcontract.LatestJudgments(records, o.audit.AuditID, o.group, o.run.RunID)
	}
	return auditcontract.LatestFindingJudgments(records, o.scope.ID(), o.run.RunID, *o.run.AuditCompletion.Finding)
}
func normalizeJudgments(ctx context.Context, o *resultOwner, updates []auditcontract.JudgmentUpdate) (auditcontract.ResultDocument, error) {
	links, err := o.list(ctx, "evidence")
	if err != nil {
		return auditcontract.ResultDocument{}, err
	}
	store, _ := o.artifacts.Scoped(o.scope)
	allowed := map[string][]auditcontract.Material{}
	wanted := map[string]bool{}
	for _, update := range updates {
		if len(update.Conclusion.Evidence) > 0 {
			wanted[o.members[update.Conclusion.ItemKey].id] = true
		}
	}
	var size int64
	for _, link := range links {
		var provenance struct {
			AuditID     string `json:"audit_id"`
			TaskID      string `json:"task_id"`
			RunID       string `json:"run_id"`
			ExecutionID string `json:"execution_id"`
			SourceCount int    `json:"sourceCount"`
			SourceBytes int64  `json:"sourceBytes"`
		}
		if strictjson.Decode(link.SourceProvenance, &provenance) != nil || provenance.RunID != o.run.RunID || provenance.SourceCount < 1 || provenance.SourceCount > 256 || provenance.SourceBytes < 0 {
			return auditcontract.ResultDocument{}, ErrInvalid
		}
		identity := provenance.TaskID
		if o.audit == nil {
			identity = o.run.AuditCompletion.Finding.FindingID
		} else if provenance.AuditID != o.audit.AuditID || provenance.ExecutionID != o.group {
			return auditcontract.ResultDocument{}, ErrInvalid
		}
		if !wanted[identity] {
			continue
		}
		size += link.Artifact.SizeBytes
		if size > auditcontract.MaximumDocumentBytes {
			return auditcontract.ResultDocument{}, ErrInvalid
		}
		read, err := store.ReadExact(ctx, link.Artifact)
		if err != nil {
			return auditcontract.ResultDocument{}, err
		}
		var record auditcontract.EvidenceRecord
		if strictjson.Decode(read.Payload.Data, &record) != nil || record.Validate() != nil || !o.ownsEvidence(record) || record.Origin.RunID != o.run.RunID {
			return auditcontract.ResultDocument{}, ErrInvalid
		}
		identity = record.Origin.TaskID
		if o.audit != nil {
			if record.Origin.AuditID != o.audit.AuditID || record.Origin.ExecutionID != o.group {
				return auditcontract.ResultDocument{}, ErrInvalid
			}
		} else {
			if !reflect.DeepEqual(record.Origin.Finding, o.run.AuditCompletion.Finding) {
				return auditcontract.ResultDocument{}, ErrInvalid
			}
			identity = record.Origin.Finding.FindingID
		}
		allowed[identity] = append(allowed[identity], record.Sources...)
		allowed[identity] = append(allowed[identity], o.material(read.Ref, read.Payload))
	}
	contextItems := []any{}
	items := []auditcontract.Conclusion{}
	for _, update := range updates {
		member := o.members[update.Conclusion.ItemKey]
		evidence := []auditcontract.Material{}
		seen := map[string]bool{}
		for _, material := range allowed[member.id] {
			key, _ := strictjson.Canonical(material)
			if !seen[string(key)] {
				evidence = append(evidence, material)
				seen[string(key)] = true
			}
		}
		item := map[string]any{"item_key": update.Conclusion.ItemKey, "objective": member.objective, "evidence": evidence}
		if o.audit != nil {
			item["task_id"] = member.id
			if member.hypothesis != nil {
				item["hypothesis"] = member.hypothesis
			}
		} else {
			item["finding_id"] = member.id
		}
		contextItems = append(contextItems, item)
		items = append(items, update.Conclusion)
	}
	digest, err := strictjson.Canonical(o.run.WorkflowSnapshot)
	if err != nil {
		return auditcontract.ResultDocument{}, err
	}
	context := map[string]any{"schema": "contractor.audit.context.v1", "run_id": o.run.RunID, "workflow_ref": o.run.WorkflowName + "@" + o.run.WorkflowVersion, "workflow_digest": auditdomain.DigestBytes(digest), "items": contextItems}
	if o.audit != nil {
		context["audit_id"] = o.audit.AuditID
	} else {
		context["project_id"] = o.scope.ID()
		context["finding"] = o.run.AuditCompletion.Finding
	}
	trusted, _ := strictjson.Canonical(context)
	raw, _ := strictjson.Canonical(map[string]any{"schema": "contractor.audit.conclusions.v1", "items": items})
	data, err := auditcontract.Decode(raw, trusted)
	var result auditcontract.ResultDocument
	if err != nil || strictjson.Decode(data, &result) != nil {
		return result, ErrInvalid
	}
	return result, nil
}

// The same correction algorithm applies to real Task or Finding membership.
func assignJudgmentGenerations(input auditcontract.JudgmentSubmission, o *resultOwner, latest map[string]auditcontract.AcceptedJudgment, normalized auditcontract.ResultDocument) (auditcontract.ResultDocument, []auditcontract.JudgmentGeneration, error) {
	generations := []auditcontract.JudgmentGeneration{}
	stale := false
	for index, update := range input.Items {
		key := update.Conclusion.ItemKey
		member := o.members[key]
		prior, exists := latest[key]
		current := uint64(0)
		if exists {
			if prior.Generation.MemberID() != member.id {
				return auditcontract.ResultDocument{}, nil, ErrInvalid
			}
			current = prior.Generation.Generation
		}
		same := exists && reflect.DeepEqual(prior.Item.Conclusion, update.Conclusion)
		if same {
			normalized.Items[index] = prior.Item
		} else {
			if input.Mode == "batch" && exists {
				return auditcontract.ResultDocument{}, nil, auditstore.ErrConflict
			}
			if current != update.BaseGeneration {
				stale = true
			}
			if current == auditcontract.MaximumResultGeneration {
				return auditcontract.ResultDocument{}, nil, auditstore.ErrEvidenceBudgetExhausted
			}
		}
		generation := auditcontract.JudgmentGeneration{ItemKey: key, Generation: current}
		if o.audit != nil {
			generation.TaskID = member.id
		} else {
			generation.FindingID = member.id
		}
		if !same {
			generation.Generation++
		}
		generations = append(generations, generation)
	}
	if stale {
		for i := range generations {
			generations[i].Generation = latest[generations[i].ItemKey].Generation.Generation
		}
		return auditcontract.ResultDocument{}, nil, &JudgmentStaleError{Generations: generations}
	}
	return normalized, generations, nil
}
