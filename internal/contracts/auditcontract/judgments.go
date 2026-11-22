package auditcontract

import (
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	jsonschema "github.com/santhosh-tekuri/jsonschema/v6"
	"strings"
	"sync"
	"unicode/utf8"
)

const MaximumResultGeneration = uint64(contracts.MaxExactInteger)

// Conclusion is normalized Runtime input. Source membership and origin are
// validated/derived by the allocation-bound accepting owner, never the model.
type Conclusion struct {
	ItemKey          string     `json:"item_key"`
	Rationale        string     `json:"rationale"`
	TestedConditions []string   `json:"tested_conditions"`
	Limitations      []string   `json:"limitations"`
	Evidence         []Material `json:"evidence"`
	Verdict          *string    `json:"verdict,omitempty"`
}

type JudgmentUpdate struct {
	BaseGeneration uint64     `json:"baseGeneration" strictjson:"required"`
	Conclusion     Conclusion `json:"conclusion"`
}

type JudgmentSubmission struct {
	APIVersion   string           `json:"apiVersion"`
	InvocationID string           `json:"invocationId"`
	SubmissionID string           `json:"submissionId"`
	Mode         string           `json:"mode"`
	Items        []JudgmentUpdate `json:"items"`
}

func (v JudgmentSubmission) Validate() error {
	if v.APIVersion != contracts.APIVersion || !contracts.ValidResourceID(v.InvocationID) || len(v.InvocationID) > 128 || !contracts.ValidResourceID(v.SubmissionID) || len(v.SubmissionID) > 128 || len(v.Items) == 0 || len(v.Items) > 64 || v.Mode != "single" && v.Mode != "batch" || v.Mode == "single" && len(v.Items) != 1 {
		return errInvalid
	}
	seen := map[string]bool{}
	items := make([]Conclusion, 0, len(v.Items))
	for _, update := range v.Items {
		if _, ok := ExecutionItemOrdinal(update.Conclusion.ItemKey); !ok || seen[update.Conclusion.ItemKey] || update.BaseGeneration > MaximumResultGeneration {
			return errInvalid
		}
		seen[update.Conclusion.ItemKey] = true
		items = append(items, update.Conclusion)
	}
	data, err := strictjson.Canonical(map[string]any{"schema": "contractor.audit.conclusions.v1", "items": items})
	if err != nil {
		return errInvalid
	}
	return ValidateResultShape("contractor.audit.conclusions.v1", data)
}

type JudgmentGeneration struct {
	ItemKey    string `json:"itemKey"`
	Generation uint64 `json:"generation"`
	TaskID     string `json:"taskId"`
}

type JudgmentReceipt struct {
	APIVersion  string               `json:"apiVersion"`
	Material    Material             `json:"material"`
	Generations []JudgmentGeneration `json:"generations"`
	Replayed    bool                 `json:"replayed"`
}

// JudgmentMutation is the provenance of an existing immutable Artifact link.
// Its binding is the receipt; it introduces no independent journal or identity.
type JudgmentMutation struct {
	Schema           string               `json:"schema"`
	RequestDigest    string               `json:"request_digest"`
	AuditID          string               `json:"audit_id"`
	ExecutionID      string               `json:"execution_id"`
	RunID            string               `json:"run_id"`
	AllocationID     string               `json:"allocation_id"`
	StageExecutionID string               `json:"stage_execution_id"`
	LogicalAgentName string               `json:"logical_agent_name"`
	InvocationID     string               `json:"invocation_id"`
	SubmissionID     string               `json:"submission_id"`
	Generations      []JudgmentGeneration `json:"generations"`
}

func (v JudgmentMutation) Validate() error {
	if v.Schema != "contractor.audit.result-receipt.v1" || !contentdigest.Valid(v.RequestDigest) || len(v.Generations) == 0 || len(v.Generations) > 64 {
		return errInvalid
	}
	for _, id := range []string{v.AuditID, v.ExecutionID, v.RunID, v.AllocationID, v.StageExecutionID, v.InvocationID, v.SubmissionID} {
		if !contracts.ValidResourceID(id) {
			return errInvalid
		}
	}
	if contracts.ValidateArtifactName(v.LogicalAgentName) != nil || len(v.InvocationID) > 128 || len(v.SubmissionID) > 128 {
		return errInvalid
	}
	keys, tasks := map[string]bool{}, map[string]bool{}
	for _, item := range v.Generations {
		if _, ok := ExecutionItemOrdinal(item.ItemKey); !ok || keys[item.ItemKey] || tasks[item.TaskID] || !contracts.ValidResourceID(item.TaskID) || item.Generation < 1 || item.Generation > MaximumResultGeneration {
			return errInvalid
		}
		keys[item.ItemKey], tasks[item.TaskID] = true, true
	}
	return nil
}

type ResultStop struct {
	Source  string `json:"source"`
	Code    string `json:"code"`
	Message string `json:"message"`
}

// TaskStop is system-authored owner metadata, never a Worker conclusion or a
// security assessment. Existing Artifact links already retain its exact sources.
type TaskStop struct {
	Author   string     `json:"author"`
	Meaning  string     `json:"meaning"`
	Verdict  *string    `json:"verdict,omitempty"`
	Reason   ResultStop `json:"reason"`
	Evidence []Material `json:"evidence"`
}

var taskStopSchema = sync.OnceValues(func() (*jsonschema.Schema, error) {
	c, err := auditSchemaCompiler()
	if err != nil {
		return nil, err
	}
	return c.Compile("urn:contractor:audit:v1#/$defs/task_stop")
})

func (v TaskStop) Validate(auditID string, checking bool) error {
	if checking != (v.Verdict != nil) || !utf8.ValidString(v.Reason.Message) || strings.ContainsRune(v.Reason.Message, 0) || len(v.Reason.Message) > 4096 {
		return errInvalid
	}
	data, err := strictjson.Canonical(v)
	if err != nil || len(data) > 1<<20 {
		return errInvalid
	}
	var value any
	if strictjson.Decode(data, &value) != nil {
		return errInvalid
	}
	schema, err := taskStopSchema()
	if err != nil || schema.Validate(value) != nil {
		return errInvalid
	}
	for _, source := range v.Evidence {
		if source.ScopeKind != "audit" || source.ScopeID != auditID {
			return errInvalid
		}
	}
	return nil
}

// ResultItem is the retained model judgment. System interruption is separate
// TaskStop owner metadata and cannot enter ordinary Worker completion.
type ResultItem struct {
	Conclusion
	Author          string       `json:"author"`
	Origin          ResultOrigin `json:"origin"`
	FindingReceipts []string     `json:"finding_receipts,omitempty"`
}
type ResultOrigin struct {
	AuditID    string         `json:"audit_id"`
	RunID      string         `json:"run_id"`
	TaskID     string         `json:"task_id"`
	Hypothesis *HypothesisRef `json:"hypothesis,omitempty"`
}
type ResultDocument struct {
	Schema string       `json:"schema"`
	Items  []ResultItem `json:"items"`
}

// DecodeJudgment checks strict retained bytes and their receipt metadata.
// The caller still verifies the current execution/allocation membership.
func DecodeJudgment(data, provenance []byte) (JudgmentMutation, ResultDocument, error) {
	var receipt JudgmentMutation
	var doc ResultDocument
	if strictjson.Decode(provenance, &receipt) != nil || receipt.Validate() != nil || ValidateResultShape("contractor.audit.result.v1", data) != nil || strictjson.Decode(data, &doc) != nil || len(doc.Items) != len(receipt.Generations) {
		return receipt, doc, errInvalid
	}
	members := map[string]JudgmentGeneration{}
	for _, member := range receipt.Generations {
		members[member.ItemKey] = member
	}
	for _, item := range doc.Items {
		member, ok := members[item.ItemKey]
		if !ok || item.Author != "model" || item.Origin.AuditID != receipt.AuditID || item.Origin.RunID != receipt.RunID || item.Origin.TaskID != member.TaskID {
			return receipt, doc, errInvalid
		}
		delete(members, item.ItemKey)
	}
	return receipt, doc, nil
}

// StoredJudgment is one acknowledged ordinary Artifact receipt, read by the
// existing owner. The aggregate reader is bounded before materializing bytes.
type StoredJudgment struct {
	Material   Material
	Provenance []byte
	Data       []byte
}

type AcceptedJudgment struct {
	Material   Material
	Generation JudgmentGeneration
	Item       ResultItem
}

// LatestJudgments folds exact receipt generations without inspecting current
// bindings or inventing another semantic revision during terminal collection.
func LatestJudgments(records []StoredJudgment, auditID, executionID, runID string) (map[string]AcceptedJudgment, error) {
	if len(records) > 256 {
		return nil, errInvalid
	}
	result := map[string]AcceptedJudgment{}
	var bytes int64
	for _, record := range records {
		bytes += record.Material.SizeBytes
		if bytes > MaximumDocumentBytes || record.Material.ScopeKind != "audit" || record.Material.ScopeID != auditID || record.Material.Ref.ValidateExact() != nil || int64(len(record.Data)) != record.Material.SizeBytes || record.Material.Digest != contentdigest.Bytes(record.Data) || record.Material.MediaType != "application/json" {
			return nil, errInvalid
		}
		metadata, doc, err := DecodeJudgment(record.Data, record.Provenance)
		if err != nil || metadata.AuditID != auditID || metadata.ExecutionID != executionID || metadata.RunID != runID {
			return nil, errInvalid
		}
		generations := map[string]JudgmentGeneration{}
		for _, generation := range metadata.Generations {
			generations[generation.ItemKey] = generation
		}
		for _, item := range doc.Items {
			generation := generations[item.ItemKey]
			prior, ok := result[item.ItemKey]
			if ok && prior.Generation.TaskID != generation.TaskID {
				return nil, errInvalid
			}
			if ok && prior.Generation.Generation == generation.Generation {
				a, _ := strictjson.Canonical(prior.Item)
				b, _ := strictjson.Canonical(item)
				if string(a) != string(b) {
					return nil, errInvalid
				}
			}
			if !ok || prior.Generation.Generation < generation.Generation {
				result[item.ItemKey] = AcceptedJudgment{Material: record.Material, Generation: generation, Item: item}
			}
		}
	}
	return result, nil
}
