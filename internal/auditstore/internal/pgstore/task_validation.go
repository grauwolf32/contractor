package pgstore

import (
	"bytes"
	"errors"
	"maps"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

type validatedTask struct {
	Attention   bool                      `json:"attention"`
	Hypothesis  *auditcontract.Hypothesis `json:"hypothesis_snapshot,omitempty"`
	Document    auditcontract.Task        `json:"document"`
	Source      auditstore.TaskSource     `json:"source"`
	SourceKey   string                    `json:"source_key"`
	BatchDigest string                    `json:"batch_digest"`
	Ordinal     int                       `json:"ordinal"`
	Iteration   int                       `json:"iteration"`
}

func validTaskSource(source auditstore.TaskSource) bool {
	if !contracts.ValidName(source.Key) {
		return false
	}
	if len(source.ObservationTaskIDs) > contracts.MaxSelectedMaterials || source.Kind != "continuation" && len(source.ObservationTaskIDs) > 0 {
		return false
	}
	seen := map[string]bool{}
	for _, id := range source.ObservationTaskIDs {
		if !contracts.ValidResourceID(id) || seen[id] {
			return false
		}
		seen[id] = true
	}
	switch source.Kind {
	case "manual", "initial":
		return source.ExecutionID == "" && source.Output == nil && source.ThroughOrdinal == 0
	case "continuation":
		return source.ExecutionID == "" && source.Output == nil && source.ThroughOrdinal > 0 && source.ThroughOrdinal <= auditprofile.MaxAuditItemsTotal
	case "prepared":
		if source.ThroughOrdinal != 0 {
			return false
		}
		if !contracts.ValidResourceID(source.ExecutionID) || source.Output == nil || source.Output.ScopeKind != "run" || !contracts.ValidResourceID(source.Output.ScopeID) {
			return false
		}
		output := source.Output
		exact := contracts.ExactArtifact{Ref: output.Ref, Digest: output.Digest, MediaType: output.MediaType, SizeBytes: output.SizeBytes}
		return exact.HasMetadata() && exact.Validate() == nil
	default:
		return false
	}
}

func taskBatchDigest(document auditcontract.Task) (string, error) {
	return contentdigest.CanonicalJSON(struct {
		Workflow   string                            `json:"workflow"`
		Inputs     map[string]auditcontract.Material `json:"inputs"`
		Parameters map[string]string                 `json:"parameters"`
	}{document.Workflow, document.Inputs, document.Parameters})
}

// validateAcceptedTasks freezes caller-owned maps before a transaction starts.
// A slice bounds both primary Task records and distinct exact input references;
// metadata verification can therefore use at most four ordinary 32-ref reads.
func validateAcceptedTasks(params auditstore.AcceptTasksParams) ([]validatedTask, error) {
	if !contracts.ValidResourceID(params.AuditID) || params.OwnerID == "" || len(params.Tasks) == 0 && params.PreparedProgress == nil || len(params.Tasks) > auditcontract.MaxMutationRecords || validateIdempotency(params.IdempotencyKey, params.RequestDigest) != nil {
		return nil, auditstore.ErrInvalid
	}
	if params.Claim == nil && (params.ExpectedRevision == 0 || params.InitialProgress != nil || params.PreparedProgress != nil) || params.Claim != nil && (validateClaimIdentity(*params.Claim) != nil || params.Claim.AuditID != params.AuditID || params.ExpectedRevision != 0) || params.InitialProgress != nil && params.PreparedProgress != nil {
		return nil, auditstore.ErrInvalid
	}
	result := make([]validatedTask, 0, len(params.Tasks))
	keys, ids, materials := map[string]bool{}, map[string]bool{}, map[string]bool{}
	used := 1
	for _, accepted := range params.Tasks {
		data, err := strictjson.Canonical(accepted.Document)
		if err != nil || len(data) > auditcontract.MaxMutationBytes {
			return nil, auditstore.ErrInvalid
		}
		document, err := auditcontract.DecodeTask(data)
		if err != nil || document.AuditID != params.AuditID || ids[document.TaskID] || !validTaskSource(accepted.Source) {
			return nil, auditstore.ErrInvalid
		}
		ids[document.TaskID] = true
		sourceBytes, err := strictjson.Canonical(accepted.Source)
		if err != nil || len(sourceBytes) > 4096 {
			return nil, auditstore.ErrInvalid
		}
		var source auditstore.TaskSource
		if strictjson.Decode(sourceBytes, &source) != nil {
			return nil, auditstore.ErrInvalid
		}
		if params.Claim == nil && source.Kind != "manual" || source.Kind == "initial" && params.InitialProgress == nil || source.Kind == "continuation" && (params.ContinuationRevision == 0 || params.Claim == nil) || source.Kind == "prepared" && (params.PreparedProgress == nil || params.PreparedProgress.ExecutionID != source.ExecutionID) || params.InitialProgress != nil && source.Kind != "initial" || params.PreparedProgress != nil && source.Kind != "prepared" {
			return nil, auditstore.ErrInvalid
		}
		sourceKey := contentdigest.Bytes(sourceBytes)
		if keys[sourceKey] {
			return nil, auditstore.ErrInvalid
		}
		keys[sourceKey] = true
		for _, material := range document.Inputs {
			if material.ScopeKind != "audit" || material.ScopeID != params.AuditID {
				return nil, auditstore.ErrInvalid
			}
			ref, err := strictjson.Canonical(material.Ref)
			if err != nil || len(ref) > auditstore.MaxArtifactRefBytes {
				return nil, auditstore.ErrInvalid
			}
			materials[string(ref)] = true
		}
		if len(materials) > auditcontract.MaxMutationRecords {
			return nil, auditstore.ErrInvalid
		}
		batch, err := taskBatchDigest(document)
		if err != nil {
			return nil, err
		}
		if accepted.Attention && source.Kind != "manual" {
			return nil, auditstore.ErrInvalid
		}
		row := validatedTask{Attention: accepted.Attention, Hypothesis: accepted.Hypothesis, Document: document, Source: source, SourceKey: sourceKey, BatchDigest: batch}
		if document.Hypothesis != nil {
			if accepted.Hypothesis == nil || accepted.Hypothesis.ID != document.Hypothesis.ID || accepted.Hypothesis.Revision != document.Hypothesis.Revision {
				return nil, auditstore.ErrInvalid
			}
			snapshot, err := strictjson.Canonical(accepted.Hypothesis)
			if err != nil {
				return nil, auditstore.ErrInvalid
			}
			frozen, err := auditcontract.DecodeHypothesis(snapshot)
			if err != nil {
				return nil, auditstore.ErrInvalid
			}
			row.Hypothesis = &frozen
		} else if accepted.Hypothesis != nil {
			return nil, auditstore.ErrInvalid
		}
		encoded, err := strictjson.Canonical(row)
		used += len(encoded) + 1
		if err != nil || used > auditcontract.MaxMutationBytes {
			return nil, auditstore.ErrInvalid
		}
		result = append(result, row)
	}
	return result, nil
}

func validateTaskBinding(pinned auditprofile.PinnedProgram, document auditcontract.Task) error {
	binding, exists := pinned.Program.Workflows[document.Workflow]
	workflow, closureExists := pinned.Workflows[document.Workflow]
	if !exists || !closureExists || !maps.Equal(binding.Parameters, document.Parameters) {
		return auditstore.ErrInvalid
	}
	if document.Hypothesis != nil {
		context := false
		for _, m := range binding.Inputs {
			context = context || m.Source == "task-context"
		}
		if !context || binding.Outputs["result"] == "" {
			return auditstore.ErrInvalid
		}
	}
	if err := auditprofile.ValidateTaskInputs(binding, workflow, document.Inputs); err != nil {
		return err
	}
	for name, slot := range workflow.Inputs {
		source := binding.Inputs[name].Source
		if slot.Required && (source == "audit-input" || source == "task-input") {
			if _, selected := document.Inputs[name]; !selected {
				return auditstore.ErrInvalid
			}
		}
	}
	return nil
}

func decodeAuditProgram(audit auditstore.Audit) (auditprofile.PinnedProgram, error) {
	pinned, err := auditprofile.DecodePinnedProgram(audit.ProfileSnapshot)
	if err != nil {
		return auditprofile.PinnedProgram{}, err
	}
	limits := pinned.Program.Limits
	if audit.Profile.Name != pinned.Ref.Name || audit.Profile.Version != pinned.Ref.Version || audit.Profile.Digest != pinned.Ref.Digest || validateLimits(audit.Limits) != nil || audit.Limits.MaxRounds > limits.MaxIterations || audit.Limits.BatchSize > limits.BatchSize || audit.Limits.MaxItemsPerRound > limits.MaxTasksPerIteration || audit.Limits.MaxItemsTotal > limits.MaxTasks || audit.Limits.MaxSubmittedRuns > limits.MaxSubmittedRuns || audit.Limits.MaxItemRunAttempts > limits.MaxAttempts || audit.Limits.MaxEvidenceBytes > limits.MaxEvidenceBytes {
		return auditprofile.PinnedProgram{}, errStoredTask
	}
	if len(audit.BaselineSnapshot) > 0 {
		pinned.Program, err = auditdomain.ExecutionProgram(pinned.Program, audit.BaselineSnapshot)
		if err != nil {
			return auditprofile.PinnedProgram{}, err
		}
	}
	return pinned, nil
}

func sameAcceptedTask(prior auditstore.Task, selected validatedTask) bool {
	a, aErr := strictjson.Canonical(prior.Document)
	b, bErr := strictjson.Canonical(selected.Document)
	return aErr == nil && bErr == nil && bytes.Equal(a, b) && prior.SourceKey == selected.SourceKey && prior.BatchDigest == selected.BatchDigest && prior.Attention == selected.Attention
}

var errStoredTask = errors.New("stored Audit task is invalid")
