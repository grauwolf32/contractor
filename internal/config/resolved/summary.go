package resolved

import (
	"encoding/json"
	"fmt"

	"github.com/grauwolf32/contractor/internal/artifactpolicy"
	"github.com/grauwolf32/contractor/internal/strictjson"
)

const MaxSummaryContinuationDeadlineSeconds = 24 * 60 * 60

// Strategy escalation has no authored exhaustion edge: a source cannot be
// summarized twice. Ordinary model escalation retains its required then.
func (e *EscalateTransition) UnmarshalJSON(data []byte) error {
	type plain EscalateTransition
	var decoded plain
	if err := strictjson.Decode(data, &decoded); err != nil {
		return err
	}
	var fields map[string]json.RawMessage
	if err := strictjson.Decode(data, &fields); err != nil {
		return err
	}
	if _, strategy := fields["planner"]; strategy {
		if decoded.Planner == nil {
			return fmt.Errorf("escalation planner must not be null")
		}
		if _, then := fields["then"]; then {
			return fmt.Errorf("summary strategy escalation has no then")
		}
	} else if _, deadline := fields["deadlineSeconds"]; deadline {
		return fmt.Errorf("deadlineSeconds requires a summary strategy Planner")
	}
	*e = EscalateTransition(decoded)
	return nil
}

// NamedTransition carries one authored outcome and its typed action.
type NamedTransition struct {
	Outcome string
	Action  TransitionAction
}

// Actions is the complete declared outcome set in deterministic order.
func (t StageTransitions) Actions() []NamedTransition {
	result := []NamedTransition{
		{Outcome: "succeeded", Action: t.Succeeded},
		{Outcome: "failed", Action: t.Failed},
		{Outcome: "interrupted", Action: t.Interrupted},
	}
	if t.Summarized != nil {
		result = append(result, NamedTransition{Outcome: "summarized", Action: *t.Summarized})
	}
	return result
}

func escalationStage(stage ResolvedStage, outcome string, escalation EscalateTransition) (ResolvedStage, error) {
	variant := stage
	variant.ExecutionConfig = cloneStageExecutionConfig(stage.ExecutionConfig)
	if outcome != "summarized" {
		if escalation.Planner != nil || escalation.DeadlineSeconds != 0 {
			return ResolvedStage{}, fmt.Errorf("Planner strategy and deadline overrides require on.summarized")
		}
		return variant, nil
	}
	if stage.Planner != (PlannerRef{PlannerID: "passthrough", Version: "1"}) || len(stage.Agents) != 1 {
		return ResolvedStage{}, fmt.Errorf("on.summarized requires passthrough@1 with one Worker")
	}
	if stage.On.Failed.Kind == TransitionEscalate || stage.On.Interrupted.Kind == TransitionEscalate {
		return ResolvedStage{}, fmt.Errorf("summary strategy escalation cannot share a Stage with model-only escalation")
	}
	for _, binding := range stage.Agents {
		if binding.Template.Summarizer == nil || binding.Template.IsToolWorker() {
			return ResolvedStage{}, fmt.Errorf("on.summarized requires an ADK Worker with a summarizer")
		}
		if artifactpolicy.IsPurposeReservedNamespace(binding.Namespace) {
			return ResolvedStage{}, fmt.Errorf("summary escalation requires a writable ordinary Agent Namespace")
		}
	}
	if escalation.Planner == nil || *escalation.Planner != (PlannerRef{PlannerID: "streamline", Version: "1"}) || escalation.MaxAttempts != 1 {
		return ResolvedStage{}, fmt.Errorf("on.summarized requires one escalation to streamline@1")
	}
	if escalation.Then != (TransitionAction{}) {
		return ResolvedStage{}, fmt.Errorf("on.summarized escalation has no then")
	}
	if escalation.DeadlineSeconds < 1 || escalation.DeadlineSeconds > MaxSummaryContinuationDeadlineSeconds {
		return ResolvedStage{}, fmt.Errorf("summary continuation deadlineSeconds must be 1..%d", MaxSummaryContinuationDeadlineSeconds)
	}
	if len(escalation.ExecutionConfig.Override.Agents) != 0 || escalation.ExecutionConfig.Override.Planner == nil {
		return ResolvedStage{}, fmt.Errorf("summary escalation config must select only the Planner")
	}
	selection := escalation.ExecutionConfig.Override.Planner
	if selection.ModelPolicy == nil || selection.LLMGateway == nil {
		return ResolvedStage{}, fmt.Errorf("summary escalation requires an explicit Planner modelPolicy and llmGateway")
	}
	variant.Planner = *escalation.Planner
	variant.ExecutionConfig.Planner = &ResolvedConsumerExecutionConfig{}
	return variant, nil
}
