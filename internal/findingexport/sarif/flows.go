package sarif

import (
	"strconv"
	"strings"

	"github.com/grauwolf32/contractor/internal/findinganalysis"
	"github.com/grauwolf32/contractor/internal/findingexport"
)

type locatedStep struct {
	step          findinganalysis.Step
	originalOrder int
}

func (b *builder) flows(receipt findingexport.Receipt, locations map[string]projectedLocation, warn func(string, string)) ([]any, map[string]int, error) {
	output := []any{}
	counts := map[string]int{}
	if receipt.Analysis == nil {
		return output, counts, nil
	}
	flows := map[string]findinganalysis.Flow{}
	for _, flow := range receipt.Analysis.Analysis.Flows {
		flows[flow.ID] = flow
	}
	for _, id := range sortedKeys(flows) {
		flow := flows[id]
		gaps := map[string]bool{}
		edges := map[string]findinganalysis.Edge{}
		for _, gap := range flow.Gaps {
			gaps[gap.AfterStepID+"\x00"+gap.BeforeStepID] = true
		}
		for _, edge := range flow.Edges {
			if edge.Basis != "producer-asserted" {
				return nil, nil, findingexport.ErrInvalid
			}
			edges[edge.From+"\x00"+edge.To] = edge
		}
		segments := [][]locatedStep{}
		current := []locatedStep{}
		flush := func() {
			if len(current) > 0 {
				segments = append(segments, current)
				current = nil
			}
		}
		incomplete := flow.Completeness != "complete"
		for i, step := range flow.Steps {
			location, exists := locations[step.LocationID]
			// A file-level location remains related evidence, not an invented
			// source point that can bridge a code-flow segment.
			if !exists || location.native.Region == nil {
				flush()
				incomplete = true
				warn("flow_coordinates_missing", "analysis.flows")
				continue
			}
			if i > 0 && gaps[flow.Steps[i-1].ID+"\x00"+step.ID] {
				flush()
				incomplete = true
			}
			current = append(current, locatedStep{step: step, originalOrder: i + 1})
		}
		flush()
		segmentCount := 0
		for _, segment := range segments {
			if len(segment) < 2 {
				continue
			}
			segmentCount++
			text := flow.Kind + ": " + flow.Description + " [Producer-asserted relation.]"
			if incomplete {
				text = "Partial " + text + " (located segment)"
				warn("flow_partial", "analysis.flows")
			}
			codeFlow := object{"message": object{"text": text}, "properties": object{"contractor": object{"flow_id": flow.ID, "kind": flow.Kind, "claimed_completeness": flow.Completeness, "partial_projection": incomplete, "segment": segmentCount}}}
			if err := b.budget.add(codeFlow); err != nil {
				return nil, nil, err
			}
			thread := object{"id": flow.ID + "/" + strconv.Itoa(segmentCount)}
			if err := b.budget.add(thread); err != nil {
				return nil, nil, err
			}
			points := make([]any, 0, len(segment))
			for i, located := range segment {
				step := located.step
				message := step.Message
				for _, role := range step.Roles {
					if role == "sanitizer" || role == "validation" {
						message += " [Producer claim; this does not establish safety.]"
						break
					}
				}
				if step.ValueState == "clean" || step.ValueState == "validated" {
					message += " [Claimed value state: " + step.ValueState + ".]"
				}
				details := object{"step_id": step.ID, "native_order": located.originalOrder, "native_step": step, "basis": "producer-asserted"}
				if i > 0 {
					edge, exists := edges[segment[i-1].step.ID+"\x00"+step.ID]
					if !exists {
						return nil, nil, findingexport.ErrInvalid
					}
					details["edge_from_previous"] = edge
				}
				point := object{"location": locations[step.LocationID].value, "executionOrder": i + 1, "kinds": step.Roles, "properties": object{"contractor": details}}
				// A step message belongs to its location in SARIF.
				projected := object{}
				for k, v := range locations[step.LocationID].value {
					projected[k] = v
				}
				projected["message"] = object{"text": message}
				point["location"] = projected
				state := object{}
				if step.Value != "" {
					state["value"] = object{"text": step.Value}
				}
				if step.ValueState != "" {
					state["valueState"] = object{"text": step.ValueState}
				}
				if len(state) > 0 {
					point["state"] = state
				}
				if err := b.budget.add(point); err != nil {
					return nil, nil, err
				}
				points = append(points, point)
			}
			thread["locations"] = points
			codeFlow["threadFlows"] = []any{thread}
			output = append(output, codeFlow)
		}
		counts[id] = segmentCount
		if incomplete || len(flow.Gaps) > 0 {
			warn("flow_partial", "analysis.flows")
		}
		if flow.Kind == "data-flow" && flow.Completeness == "complete" && incomplete {
			return nil, nil, findingexport.ErrInvalid
		}
	}
	return output, counts, nil
}

func flowSummary(counts map[string]int) string {
	if len(counts) == 0 {
		return ""
	}
	parts := make([]string, 0, len(counts))
	for _, id := range sortedKeys(counts) {
		parts = append(parts, id+": "+strconv.Itoa(counts[id])+" located segment(s)")
	}
	return "\nFlow projection: " + strings.Join(parts, "; ") + ". Completeness describes the retained claim, not whole-program coverage."
}
