package resolved

import (
	"bytes"
	"fmt"
	"math"
	"strings"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/strictjson"
)

// Planner context-management strategies of a Stage (PLN-R125). The union is
// closed: a Stage either keeps its whole Planner conversation (none) or
// compacts it with compact@1.
const (
	PlannerContextNone    = "none"
	PlannerContextCompact = "compact@1"
)

// Frozen compact@1 bounds (LIM-PLN-10 to LIM-PLN-12): the defaults an
// omitted field takes and the inclusive ranges an authored value must fit.
const (
	DefaultPlannerContextWindowRatio = 0.75
	DefaultPlannerKeepRecentGroups   = 4
	MaxPlannerKeepRecentGroups       = 16
	DefaultPlannerMaxCompactions     = 8
	MaxPlannerCompactions            = 16
	maxPlannerCompactorInstructions  = 8000
)

// PlannerContextManagement is the resolved compact@1 branch of a Stage's
// plannerContextManagement. A Stage that keeps its whole conversation has
// none: its ResolvedStage holds nil and its snapshot omits the member, so
// the encoding of a Stage without compaction is unchanged (PLN-R125). Every
// compact@1 default is explicit. The compactor ModelPolicy is not part of
// this value: it is a Planner execution selection
// (ResolvedConsumerExecutionConfig.CompactorModelPolicy), resolved with the
// ordinary executionConfig precedence.
type PlannerContextManagement struct {
	Strategy           string                         `json:"strategy" strictjson:"required"`
	Instructions       contracts.ResolvedInstructions `json:"instructions" strictjson:"required"`
	ContextWindowRatio float64                        `json:"contextWindowRatio" strictjson:"required"`
	KeepRecentGroups   int                            `json:"keepRecentGroups" strictjson:"required"`
	MaxCompactions     int                            `json:"maxCompactions" strictjson:"required"`
}

// UnmarshalJSON strictly decodes a persisted compact@1 branch and validates
// it, so a snapshot cannot carry an explicit none, a partial branch or a
// value outside its frozen bounds.
func (p *PlannerContextManagement) UnmarshalJSON(data []byte) error {
	if bytes.Equal(bytes.TrimSpace(data), []byte("null")) {
		return contracts.Invalidf("plannerContextManagement must not be null")
	}
	type plain PlannerContextManagement
	var value plain
	if err := strictjson.Decode(data, &value); err != nil {
		return err
	}
	decoded := PlannerContextManagement(value)
	if err := decoded.Validate(); err != nil {
		return err
	}
	*p = decoded
	return nil
}

// Validate checks the shape and frozen bounds of the compact@1 branch alone.
// ValidateStageExecutionConfig checks it against its Stage and Planner model
// selection.
func (p PlannerContextManagement) Validate() error {
	if p.Strategy != PlannerContextCompact {
		return contracts.Invalidf("plannerContextManagement strategy must be %s", PlannerContextCompact)
	}
	instructions := p.Instructions
	if strings.TrimSpace(instructions.Ref) == "" || strings.TrimSpace(instructions.Text) == "" {
		return contracts.Invalidf("plannerContextManagement instructions ref and text must not be empty")
	}
	if !utf8.ValidString(instructions.Text) ||
		utf8.RuneCountInString(instructions.Text) > maxPlannerCompactorInstructions {
		return contracts.Invalidf(
			"plannerContextManagement instructions must contain at most %d Unicode characters",
			maxPlannerCompactorInstructions,
		)
	}
	if err := contracts.ValidateDigest("plannerContextManagement.instructions.digest", instructions.Digest); err != nil {
		return err
	}
	if math.IsNaN(p.ContextWindowRatio) || math.IsInf(p.ContextWindowRatio, 0) ||
		p.ContextWindowRatio <= 0 || p.ContextWindowRatio >= 1 {
		return contracts.Invalidf("plannerContextManagement contextWindowRatio must be finite and between 0 and 1")
	}
	if p.KeepRecentGroups < 1 || p.KeepRecentGroups > MaxPlannerKeepRecentGroups {
		return contracts.Invalidf(
			"plannerContextManagement keepRecentGroups must be between 1 and %d", MaxPlannerKeepRecentGroups,
		)
	}
	if p.MaxCompactions < 1 || p.MaxCompactions > MaxPlannerCompactions {
		return contracts.Invalidf(
			"plannerContextManagement maxCompactions must be between 1 and %d", MaxPlannerCompactions,
		)
	}
	return nil
}

// Clone returns a copy of p; it holds no shared references.
func (p *PlannerContextManagement) Clone() *PlannerContextManagement {
	if p == nil {
		return nil
	}
	value := *p
	return &value
}

// PlannerCompactsContext reports whether stage selects compact@1.
func PlannerCompactsContext(stage ResolvedStage) bool {
	return stage.PlannerContextManagement != nil
}

// ValidatePlannerCompactorModelPolicy checks the tool-free compactor policy of
// compact@1 (TPL-R117): an exact context window, one bounded response below
// it, exactly one model call per compaction and a finite total, and no tool
// or Worker budget.
func ValidatePlannerCompactorModelPolicy(policy contracts.ResolvedModelPolicy) error {
	if err := policy.Validate(); err != nil {
		return err
	}
	if policy.ContextWindowTokens <= 0 {
		return contracts.Invalidf("Planner compactor modelPolicy requires contextWindowTokens")
	}
	if policy.MaxOutputTokens <= 0 || policy.MaxOutputTokens >= policy.ContextWindowTokens {
		return contracts.Invalidf("Planner compactor modelPolicy requires maxOutputTokens below contextWindowTokens")
	}
	if policy.MaxModelCalls != 1 {
		return contracts.Invalidf("Planner compactor modelPolicy requires maxModelCalls 1")
	}
	if policy.MaxTotalTokens <= 0 {
		return contracts.Invalidf("Planner compactor modelPolicy requires maxTotalTokens")
	}
	if policy.MaxToolCalls != 0 || policy.MaxWorkerCalls != 0 {
		return contracts.Invalidf("Planner compactor modelPolicy must omit maxToolCalls and maxWorkerCalls")
	}
	return nil
}

// validatePlannerContextManagement checks a Stage's context-management
// selection against its Planner and Planner model selection: compact@1
// requires a modeled Streamline or Router Planner whose main ModelPolicy pins
// a context window, and exactly its pinned compactor policy.
func validatePlannerContextManagement(stage ResolvedStage) error {
	compaction := stage.PlannerContextManagement
	selection := stage.ExecutionConfig.Planner
	if compaction == nil {
		if selection != nil && (selection.CompactorModelPolicy != nil || selection.Origins.CompactorModelPolicy != "") {
			return fmt.Errorf("compactorModelPolicy requires plannerContextManagement %s", PlannerContextCompact)
		}
		return nil
	}
	if err := compaction.Validate(); err != nil {
		return err
	}
	ref := stage.Planner.Selector()
	if stage.Operation != nil || (ref != StreamlinePlanner && ref != RouterPlanner) {
		return fmt.Errorf("plannerContextManagement %s requires %s or %s", PlannerContextCompact, StreamlinePlanner, RouterPlanner)
	}
	if selection == nil {
		return fmt.Errorf("plannerContextManagement %s requires a Planner execution selection", PlannerContextCompact)
	}
	if selection.ModelPolicy.ContextWindowTokens <= 0 {
		return fmt.Errorf("plannerContextManagement %s requires a Planner modelPolicy with contextWindowTokens", PlannerContextCompact)
	}
	if selection.CompactorModelPolicy == nil {
		return fmt.Errorf("plannerContextManagement %s requires a compactor modelPolicy", PlannerContextCompact)
	}
	if err := ValidatePlannerCompactorModelPolicy(*selection.CompactorModelPolicy); err != nil {
		return err
	}
	if strings.TrimSpace(selection.Origins.CompactorModelPolicy) == "" {
		return fmt.Errorf("compactorModelPolicy origin is required")
	}
	return nil
}
