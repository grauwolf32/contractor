package contracts

// Workflow-selected Worker context management (WFL-R90..WFL-R94) and its
// effective allocation branch (WRK-R184).

import (
	"encoding/json"
	"math"
	"strings"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/strictjson"
)

// WorkerContextStrategy names one closed context-management strategy. A
// Workflow binding selects template@1 or compact@1; an AllocationSpec carries
// the single effective branch none, terminal@1 or compact@1.
type WorkerContextStrategy string

const (
	ContextStrategyTemplate WorkerContextStrategy = "template@1"
	ContextStrategyCompact  WorkerContextStrategy = "compact@1"
	ContextStrategyNone     WorkerContextStrategy = "none"
	ContextStrategyTerminal WorkerContextStrategy = "terminal@1"
)

// Compaction bounds, LIM-WFL-9 to LIM-WFL-12.
const (
	MinWorkerCompactions                   = 1
	MaxWorkerCompactionsPerInvocation      = 16
	DefaultWorkerCompactions               = 8
	MinWorkerCompactionRecentGroups        = 1
	MaxWorkerCompactionRecentGroups        = 16
	DefaultWorkerCompactionRecentGroups    = 4
	DefaultWorkerCompactionContextRatio    = 0.75
	MaxWorkerContextInstructionsCharacters = 8000
)

// WorkerCompaction is the closed compact@1 policy. Every member is explicit in
// a resolved binding and in AllocationSpec; authoring defaults are applied by
// configuration resolution only.
type WorkerCompaction struct {
	ModelPolicy        ResolvedModelPolicy  `json:"modelPolicy" strictjson:"required"`
	Instructions       ResolvedInstructions `json:"instructions" strictjson:"required"`
	IncludeReasoning   bool                 `json:"includeReasoning" strictjson:"required"`
	ContextWindowRatio float64              `json:"contextWindowRatio" strictjson:"required"`
	KeepRecentGroups   int                  `json:"keepRecentGroups" strictjson:"required"`
	MaxCompactions     int                  `json:"maxCompactions" strictjson:"required"`
}

// WorkerContextManagement is one flat tagged union: strategy alone, or
// compact@1 with every WorkerCompaction member promoted beside it. The
// embedded policy is nil for every other strategy, so encoding/json omits it.
type WorkerContextManagement struct {
	Strategy WorkerContextStrategy `json:"strategy" strictjson:"required"`
	*WorkerCompaction
}

type compactContextManagementWire struct {
	Strategy WorkerContextStrategy `json:"strategy" strictjson:"required"`
	WorkerCompaction
}

func (c *WorkerContextManagement) UnmarshalJSON(data []byte) error {
	var fields map[string]json.RawMessage
	if err := strictjson.DecodeNonNull(data, &fields); err != nil {
		return err
	}
	raw, ok := fields["strategy"]
	if !ok {
		return Invalidf("contextManagement.strategy is required")
	}
	var strategy WorkerContextStrategy
	if err := strictjson.DecodeNonNull(raw, &strategy); err != nil {
		return err
	}
	if strategy != ContextStrategyCompact {
		if len(fields) != 1 {
			return Invalidf("contextManagement %q has no other members", strategy)
		}
		*c = WorkerContextManagement{Strategy: strategy}
		return nil
	}
	var wire compactContextManagementWire
	if err := strictjson.DecodeNonNull(data, &wire); err != nil {
		return err
	}
	compaction := wire.WorkerCompaction
	*c = WorkerContextManagement{Strategy: strategy, WorkerCompaction: &compaction}
	return nil
}

// IsCompact reports whether c selects the continuing compact@1 strategy.
func (c WorkerContextManagement) IsCompact() bool {
	return c.Strategy == ContextStrategyCompact
}

// Clone returns a copy that shares no pointers with c.
func (c WorkerContextManagement) Clone() WorkerContextManagement {
	if c.WorkerCompaction == nil {
		return c
	}
	compaction := *c.WorkerCompaction
	compaction.ModelPolicy = c.ModelPolicy.Clone()
	return WorkerContextManagement{Strategy: c.Strategy, WorkerCompaction: &compaction}
}

// ValidateBinding checks a resolved Workflow binding selection: explicit
// template@1, or a complete compact@1 policy (WFL-R90).
func (c WorkerContextManagement) ValidateBinding() error {
	switch c.Strategy {
	case ContextStrategyTemplate:
		if c.WorkerCompaction != nil {
			return Invalidf("contextManagement template@1 has no compaction members")
		}
		return nil
	case ContextStrategyCompact:
		return c.validateCompaction()
	default:
		return Invalidf("contextManagement.strategy must be template@1 or compact@1")
	}
}

// EffectiveContextManagement projects a validated binding selection to the
// one branch an allocation executes: the template summarizer (terminal@1),
// none, or the binding's compact@1 policy, which wholly replaces the template
// summarizer without changing the template.
func EffectiveContextManagement(binding WorkerContextManagement, template ResolvedAgentTemplate) WorkerContextManagement {
	switch {
	case binding.IsCompact():
		return binding.Clone()
	case template.Summarizer != nil:
		return WorkerContextManagement{Strategy: ContextStrategyTerminal}
	default:
		return WorkerContextManagement{Strategy: ContextStrategyNone}
	}
}

// ValidateAllocation checks the effective branch against the allocation it
// travels in (WRK-R184).
func (c WorkerContextManagement) ValidateAllocation(spec AllocationContext) error {
	switch c.Strategy {
	case ContextStrategyNone:
		if c.WorkerCompaction != nil || spec.Template.Summarizer != nil {
			return Invalidf("contextManagement none requires a template without summarizer")
		}
	case ContextStrategyTerminal:
		if c.WorkerCompaction != nil || spec.Template.Summarizer == nil || spec.Template.IsToolWorker() {
			return Invalidf("contextManagement terminal@1 requires the template summarizer")
		}
	case ContextStrategyCompact:
		if err := c.validateCompaction(); err != nil {
			return err
		}
		if err := ValidateCompactionTarget(spec.Template, spec.SessionMode, spec.CompletionContract != nil); err != nil {
			return err
		}
		if spec.SummarizerModelPolicy != nil {
			return Invalidf("compact@1 forbids summarizerModelPolicy")
		}
		if err := ValidateCompactedWorkerPolicy(spec.WorkerPolicy); err != nil {
			return err
		}
	default:
		return Invalidf("contextManagement.strategy must be none, terminal@1 or compact@1")
	}
	if spec.SummarizerModelPolicy != nil && c.Strategy != ContextStrategyTerminal {
		return Invalidf("summarizerModelPolicy requires terminal@1 context management")
	}
	return nil
}

// AllocationContext is the part of an allocation the effective context branch
// depends on.
type AllocationContext struct {
	Template              ResolvedAgentTemplate
	SessionMode           WorkerSessionMode
	CompletionContract    *WorkerCompletionContract
	SummarizerModelPolicy *ResolvedModelPolicy
	WorkerPolicy          ResolvedModelPolicy
}

// The audit-results@2 Toolset is selected only by Audit completion Workers
// (AWC-R18), so its selection identifies one in a Workflow binding.
const (
	AuditResultsToolsetID      = "audit-results"
	AuditResultsToolsetVersion = "2"
)

// SelectsAuditCompletion reports whether t selects audit-results@2.
func (t ResolvedAgentTemplate) SelectsAuditCompletion() bool {
	for _, toolset := range t.Toolsets {
		if toolset.Ref == (ToolsetRef{ToolsetID: AuditResultsToolsetID, Version: AuditResultsToolsetVersion}) {
			return true
		}
	}
	return false
}

// ValidateCompactionTarget rejects the Worker kinds compact@1 does not
// support yet: model-free Workers, shared sessions (until V417-004) and Audit
// completion (until V417-005).
func ValidateCompactionTarget(template ResolvedAgentTemplate, session WorkerSessionMode, auditCompletion bool) error {
	if template.IsToolWorker() {
		return Invalidf("compact@1 requires a modeled ADK Worker")
	}
	if session != WorkerSessionIsolated {
		return Invalidf("compact@1 supports only isolated Worker sessions")
	}
	if auditCompletion || template.SelectsAuditCompletion() {
		return Invalidf("compact@1 does not support Audit completion Workers")
	}
	return nil
}

// ValidateCompactedWorkerPolicy requires the effective main Worker policy
// metadata the compaction boundary is derived from (WRK-R185).
func ValidateCompactedWorkerPolicy(policy ResolvedModelPolicy) error {
	if policy.ContextWindowTokens <= 0 || policy.MaxOutputTokens <= 0 {
		return Invalidf("compact@1 Worker modelPolicy requires contextWindowTokens and maxOutputTokens")
	}
	return nil
}

// ValidateForWorkerCompactor checks the tool-free compactor consumer: one
// call per cycle with explicit context, output and finite token limits.
func (p ResolvedModelPolicy) ValidateForWorkerCompactor() error {
	if err := validateModelPolicy(p); err != nil {
		return err
	}
	if p.ContextWindowTokens <= 0 || p.MaxOutputTokens <= 0 || p.MaxTotalTokens <= 0 {
		return Invalidf("Worker compactor modelPolicy requires contextWindowTokens, maxOutputTokens and maxTotalTokens")
	}
	if p.MaxModelCalls != 1 {
		return Invalidf("Worker compactor modelPolicy requires maxModelCalls=1")
	}
	if p.MaxToolCalls != 0 || p.MaxWorkerCalls != 0 {
		return Invalidf("Worker compactor modelPolicy must omit maxToolCalls and maxWorkerCalls")
	}
	return nil
}

func (c WorkerContextManagement) validateCompaction() error {
	compaction := c.WorkerCompaction
	if compaction == nil {
		return Invalidf("contextManagement compact@1 requires its compaction policy")
	}
	if err := compaction.ModelPolicy.ValidateForWorkerCompactor(); err != nil {
		return err
	}
	instructions := compaction.Instructions
	if strings.TrimSpace(instructions.Ref) == "" || strings.TrimSpace(instructions.Text) == "" {
		return Invalidf("compact@1 instructions ref and text must not be empty")
	}
	if !utf8.ValidString(instructions.Text) || utf8.RuneCountInString(instructions.Text) > MaxWorkerContextInstructionsCharacters {
		return Invalidf("compact@1 instructions must contain at most %d Unicode characters", MaxWorkerContextInstructionsCharacters)
	}
	if err := ValidateDigest("contextManagement.instructions.digest", instructions.Digest); err != nil {
		return err
	}
	ratio := compaction.ContextWindowRatio
	if ratio <= 0 || ratio >= 1 || math.IsNaN(ratio) || math.IsInf(ratio, 0) {
		return Invalidf("compact@1 contextWindowRatio must be finite and between 0 and 1")
	}
	if compaction.KeepRecentGroups < MinWorkerCompactionRecentGroups || compaction.KeepRecentGroups > MaxWorkerCompactionRecentGroups {
		return Invalidf("compact@1 keepRecentGroups must be %d..%d", MinWorkerCompactionRecentGroups, MaxWorkerCompactionRecentGroups)
	}
	if compaction.MaxCompactions < MinWorkerCompactions || compaction.MaxCompactions > MaxWorkerCompactionsPerInvocation {
		return Invalidf("compact@1 maxCompactions must be %d..%d", MinWorkerCompactions, MaxWorkerCompactionsPerInvocation)
	}
	return nil
}
