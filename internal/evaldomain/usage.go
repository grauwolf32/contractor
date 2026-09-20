package evaldomain

import (
	"reflect"
	"sort"
	"time"
)

// AttemptObservation is one authoritative cumulative snapshot, not a delta.
// It deliberately has no Audit parent aggregate counter: only leaf Run stage
// executions contribute tokens/calls. Service code verifies associations first.
type AttemptObservation struct {
	RunID            string           `json:"runId"`
	StageExecutionID string           `json:"stageExecutionId"`
	Metrics          map[string]int64 `json:"metrics"`
	ReportsComplete  bool             `json:"reportsComplete"`
	Truncated        bool             `json:"truncated"`
	SourceSHA256     string           `json:"sourceSha256"`
}

type UsageObservation struct {
	MemberID           string               `json:"memberId"`
	Parent             ExecutionRef         `json:"parent"`
	Executions         []ExecutionRef       `json:"executions"`
	InventoryComplete  bool                 `json:"inventoryComplete"`
	Terminal           bool                 `json:"terminal"`
	StartedAt          *string              `json:"startedAt"`
	FinishedAt         *string              `json:"finishedAt"`
	Attempts           []AttemptObservation `json:"attempts"`
	Missing            []string             `json:"missing"`
	ParentSourceSHA256 string               `json:"parentSourceSha256"`
}

// NormalizeUsage is a pure format reducer. It performs no polling, state writes,
// discovery or scheduling; callers pass a single verified inventory observation.
func NormalizeUsage(in UsageObservation) (Usage, error) {
	if !memberPattern.MatchString(in.MemberID) || len(in.Executions) > 1024 || len(in.Attempts) > MaxMembers {
		return Usage{}, Failure("eval_invalid")
	}
	kind := "workflow"
	if in.Parent.Kind == "audit" {
		kind = "audit"
	} else if in.Parent.Kind != "run" {
		return Usage{}, Failure("eval_invalid")
	}
	owned := map[string]bool{}
	refs := map[ExecutionRef]bool{}
	foundParent := false
	for _, ref := range in.Executions {
		if ref.ID == "" || ref.Kind != "run" && ref.Kind != "audit" {
			return Usage{}, Failure("eval_invalid")
		}
		if ref.Kind == "audit" && ref != in.Parent {
			return Usage{}, Failure("eval_member_conflict")
		}
		if ref == in.Parent {
			foundParent = true
		}
		if ref.Kind == "run" {
			owned[ref.ID] = true
		}
		refs[ref] = true
	}
	if !foundParent {
		return Usage{}, Failure("eval_member_conflict")
	}
	if kind == "workflow" && (len(refs) != 1 || !owned[in.Parent.ID]) {
		return Usage{}, Failure("eval_member_conflict")
	}
	executions := make([]ExecutionRef, 0, len(refs))
	executions = append(executions, in.Parent)
	for ref := range refs {
		if ref != in.Parent {
			executions = append(executions, ref)
		}
	}
	sort.Slice(executions[1:], func(i, j int) bool {
		a, b := executions[i+1], executions[j+1]
		if a.Kind != b.Kind {
			return a.Kind < b.Kind
		}
		return a.ID < b.ID
	})
	missing := append([]string{}, in.Missing...)
	if !in.InventoryComplete {
		missing = append(missing, "execution inventory incomplete")
	}
	if !in.Terminal {
		missing = append(missing, "execution nonterminal")
	}
	seen := map[[2]string]AttemptObservation{}
	observedRuns := map[string]bool{}
	for _, attempt := range in.Attempts {
		if !owned[attempt.RunID] || attempt.StageExecutionID == "" || !digestPattern.MatchString(attempt.SourceSHA256) {
			return Usage{}, Failure("eval_member_conflict")
		}
		key := [2]string{attempt.RunID, attempt.StageExecutionID}
		if prior, ok := seen[key]; ok {
			if !reflect.DeepEqual(prior, attempt) {
				return Usage{}, Failure("eval_member_conflict")
			}
			continue
		}
		for _, value := range attempt.Metrics {
			if value < 0 {
				return Usage{}, Failure("eval_invalid")
			}
		}
		seen[key] = attempt
		observedRuns[attempt.RunID] = true
	}
	for run := range owned {
		if !observedRuns[run] {
			missing = append(missing, "owned Run metrics unavailable")
		}
	}
	if len(owned) == 0 {
		missing = append(missing, "leaf execution metrics unavailable")
	}
	metric := func(name, unit string) Measure {
		gaps := append([]string{}, missing...)
		sources := []string{}
		var total int64
		count := 0
		for _, attempt := range seen {
			value, ok := attempt.Metrics[name]
			if !ok {
				gaps = append(gaps, "counter unavailable")
				continue
			}
			if value > 0 && total > int64(^uint64(0)>>1)-value {
				gaps = append(gaps, "counter overflow")
				continue
			}
			total += value
			count++
			sources = append(sources, attempt.SourceSHA256)
			if !attempt.ReportsComplete || attempt.Truncated {
				gaps = append(gaps, "incomplete or truncated execution reports")
			}
		}
		gaps = uniqueStrings(gaps)
		sources = uniqueStrings(sources)
		m := Measure{
			Unit:         unit,
			Completeness: "unavailable",
			SourceRefs:   sources,
			Scope:        MeasureScope{MemberID: in.MemberID, Kind: kind, Executions: executions, Missing: gaps},
		}
		if count > 0 && (len(gaps) == 0 || total > 0) {
			value := float64(total)
			m.Value = &value
			m.Completeness = "complete"
			if len(gaps) > 0 {
				m.Completeness = "partial"
			}
		}
		if m.Value == nil && len(m.Scope.Missing) == 0 {
			m.Scope.Missing = []string{"counter unavailable"}
		}
		return m
	}
	out := Usage{
		InputTokens:       metric("inputTokens", "tokens"),
		OutputTokens:      metric("outputTokens", "tokens"),
		TotalTokens:       metric("totalTokens", "tokens"),
		CachedInputTokens: metric("cachedInputTokens", "tokens"),
		ModelCalls:        metric("modelCalls", "calls"),
		ToolCalls:         metric("toolCalls", "calls"),
		ToolFailures:      metric("toolFailures", "calls"),
	}
	out.WallMS = Measure{
		Unit:         "milliseconds",
		Completeness: "unavailable",
		SourceRefs:   []string{},
		Scope: MeasureScope{
			MemberID:   in.MemberID,
			Kind:       kind,
			Executions: executions,
			Missing:    []string{"parent timestamps unavailable"},
		},
	}
	if in.Terminal && in.StartedAt != nil && in.FinishedAt != nil && digestPattern.MatchString(in.ParentSourceSHA256) {
		start, e1 := time.Parse(time.RFC3339Nano, *in.StartedAt)
		end, e2 := time.Parse(time.RFC3339Nano, *in.FinishedAt)
		if e1 != nil || e2 != nil || end.Before(start) {
			return Usage{}, Failure("eval_invalid")
		}
		value := float64(end.Sub(start).Milliseconds())
		out.WallMS.Value = &value
		out.WallMS.Completeness = "complete"
		out.WallMS.SourceRefs = []string{in.ParentSourceSHA256}
		out.WallMS.Scope.Missing = []string{}
		out.WallMS.Scope.Interval = &Interval{Start: *in.StartedAt, End: *in.FinishedAt}
	}
	return out, nil
}

func uniqueStrings(values []string) []string {
	out := make([]string, 0, len(values))
	seen := map[string]bool{}
	for _, v := range values {
		if !seen[v] {
			out = append(out, v)
			seen[v] = true
		}
	}
	sort.Strings(out)
	return out
}
