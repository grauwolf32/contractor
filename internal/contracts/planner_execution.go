package contracts

import (
	"bytes"
	"encoding/json"
	"strings"
	"unicode/utf8"
)

// PlannerInvocation carries model input and tool declarations only. Tool effects,
// plan identity, persistence and dispatch remain behind the execution host.
type PlannerInvocation struct {
	Input      string             `json:"input" strictjson:"required"`
	Tools      []PlannerTool      `json:"tools" strictjson:"required"`
	Compaction *PlannerCompaction `json:"compaction,omitempty"`
}

type PlannerCompaction struct {
	Instructions     string              `json:"instructions" strictjson:"required"`
	Threshold        int64               `json:"threshold" strictjson:"required"`
	KeepRecentGroups int                 `json:"keepRecentGroups" strictjson:"required"`
	MaxCompactions   int                 `json:"maxCompactions" strictjson:"required"`
	Policy           ResolvedModelPolicy `json:"policy" strictjson:"required"`
}

type PlannerTool struct {
	Name        string          `json:"name" strictjson:"required"`
	Description string          `json:"description" strictjson:"required"`
	Parameters  json.RawMessage `json:"parameters" strictjson:"required"`
}

const (
	MaxPlannerOperationBytes        = 2 << 20
	MaxPlannerInvocationBytes       = 512 * 1024
	MaxPlannerTools                 = 64
	MaxPlannerOperations            = 5*MaxWorkerModelCalls + 1
	MaxPlannerKeepRecentGroups      = 16
	MaxPlannerCompactions           = 16
	MaxPlannerCompactorInstructions = 8000
)

type PlannerOperation struct {
	APIVersion string          `json:"apiVersion" strictjson:"required"`
	Sequence   int             `json:"sequence" strictjson:"required"`
	Name       string          `json:"name" strictjson:"required"`
	Arguments  json.RawMessage `json:"arguments" strictjson:"required"`
}

type PlannerOperationResult struct {
	APIVersion string            `json:"apiVersion" strictjson:"required"`
	Result     json.RawMessage   `json:"result" strictjson:"required"`
	Stop       bool              `json:"stop" strictjson:"required"`
	Escalate   bool              `json:"escalate" strictjson:"required"`
	Failure    *TerminationError `json:"failure,omitempty"`
}

func (p PlannerInvocation) Validate() error {
	if p.Compaction != nil {
		c := p.Compaction
		if strings.TrimSpace(c.Instructions) == "" || utf8.RuneCountInString(c.Instructions) > MaxPlannerCompactorInstructions || c.Threshold < 1 || c.Threshold > MaxExactInteger || c.KeepRecentGroups < 1 || c.KeepRecentGroups > MaxPlannerKeepRecentGroups || c.MaxCompactions < 1 || c.MaxCompactions > MaxPlannerCompactions {
			return Invalidf("Planner compaction configuration is invalid")
		}
		if err := c.Policy.ValidateForCompactor(); err != nil {
			return err
		}
	}
	if strings.TrimSpace(p.Input) == "" || len(p.Tools) == 0 || len(p.Tools) > MaxPlannerTools {
		return Invalidf("Planner input and bounded tool declarations are required")
	}
	seen := map[string]bool{}
	for _, t := range p.Tools {
		if !ValidName(t.Name) || seen[t.Name] || strings.TrimSpace(t.Description) == "" || !plannerJSONObject(t.Parameters) {
			return Invalidf("Planner tool declaration is invalid")
		}
		seen[t.Name] = true
	}
	return nil
}

func (p PlannerOperation) Validate() error {
	if ValidateAPIVersion(p.APIVersion) != nil || p.Sequence < 1 || p.Sequence > MaxPlannerOperations || !ValidName(p.Name) || !plannerJSONObject(p.Arguments) {
		return Invalidf("Planner operation is invalid")
	}
	if size, err := ResultJSONSize(p); err != nil || size > MaxPlannerOperationBytes {
		return Invalidf("Planner operation exceeds its bound")
	}
	return nil
}

func (p PlannerOperationResult) Validate() error {
	if ValidateAPIVersion(p.APIVersion) != nil || !plannerJSONObject(p.Result) || (p.Escalate && !p.Stop) {
		return Invalidf("Planner operation result is invalid")
	}
	if p.Failure != nil {
		if err := p.Failure.Validate(); err != nil {
			return err
		}
	}
	if size, err := ResultJSONSize(p); err != nil || size > MaxPlannerOperationBytes {
		return Invalidf("Planner operation result exceeds its bound")
	}
	return nil
}

func plannerJSONObject(raw json.RawMessage) bool {
	raw = bytes.TrimSpace(raw)
	return len(raw) > 0 && raw[0] == '{' && json.Valid(raw)
}

func (policy ResolvedModelPolicy) ValidateForPlanner() error {
	if err := policy.Validate(); err != nil {
		return err
	}
	if policy.MaxOutputTokens <= 0 {
		return Invalidf("Planner modelPolicy requires maxOutputTokens")
	}
	if policy.MaxModelCalls <= 0 {
		return Invalidf("Planner modelPolicy requires maxModelCalls")
	}
	if policy.MaxWorkerCalls <= 0 {
		return Invalidf("Planner modelPolicy requires maxWorkerCalls")
	}
	if policy.MaxTotalTokens <= 0 {
		return Invalidf("Planner modelPolicy requires maxTotalTokens")
	}
	if policy.MaxToolCalls != 0 {
		return Invalidf("Planner modelPolicy must omit maxToolCalls")
	}
	return nil
}

func (r WorkerRuntimeRef) IsPlanner() bool { return r.RuntimeID == "planner" && r.Version == "1" }
