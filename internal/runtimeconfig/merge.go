package runtimeconfig

import (
	"reflect"
	"sort"
)

// MergeSameLayer merges independent leaves, rejects ambiguity between atomic
// blocks, and never uses label order as precedence. The lexically first field
// path is returned when more than one conflict exists.
func MergeSameLayer(entries []LayerEntry) (Spec, error) {
	ordered := append([]LayerEntry(nil), entries...)
	sort.Slice(ordered, func(i, j int) bool {
		if ordered[i].Ref.String() != ordered[j].Ref.String() {
			return ordered[i].Ref.String() < ordered[j].Ref.String()
		}
		return ordered[i].Label < ordered[j].Label
	})

	var result Spec
	conflicts := make([]*MergeConflictError, 0)
	mergeField := func(path string, destination any, values []mergeValue) {
		if len(values) == 0 {
			return
		}
		first := values[0]
		conflict := false
		for _, candidate := range values[1:] {
			if !reflect.DeepEqual(first.value, candidate.value) {
				conflict = true
			}
		}
		if conflict {
			refs := make([]Ref, 0, len(values))
			seen := make(map[Ref]struct{}, len(values))
			for _, candidate := range values {
				if _, exists := seen[candidate.ref]; !exists {
					refs = append(refs, candidate.ref)
					seen[candidate.ref] = struct{}{}
				}
			}
			sort.Slice(refs, func(i, j int) bool { return refs[i].String() < refs[j].String() })
			conflicts = append(conflicts, &MergeConflictError{Path: path, Refs: refs})
			return
		}
		reflect.ValueOf(destination).Elem().Set(reflect.ValueOf(first.value))
	}

	gatewayValues := collect(ordered, func(s Spec) (any, bool) { return s.Worker.LLMGateway.Gateway, s.Worker.LLMGateway.Gateway.Present })
	credentialValues := collect(ordered, func(s Spec) (any, bool) {
		return s.Worker.LLMGateway.Credential, s.Worker.LLMGateway.Credential.Present
	})
	workerTelemetry := collect(ordered, func(s Spec) (any, bool) { return s.Worker.Telemetry, s.Worker.Telemetry.Present })
	workerProxy := collect(ordered, func(s Spec) (any, bool) { return s.Worker.HTTPProxy, s.Worker.HTTPProxy.Present })
	plannerTelemetry := collect(ordered, func(s Spec) (any, bool) { return s.Planner.Telemetry, s.Planner.Telemetry.Present })

	mergeField("worker.llmGateway.gateway", &result.Worker.LLMGateway.Gateway, gatewayValues)
	mergeField("worker.llmGateway.credential", &result.Worker.LLMGateway.Credential, credentialValues)
	mergeField("worker.telemetry", &result.Worker.Telemetry, workerTelemetry)
	mergeField("worker.httpProxy", &result.Worker.HTTPProxy, workerProxy)
	mergeField("planner.telemetry", &result.Planner.Telemetry, plannerTelemetry)
	result.Worker.LLMGateway.Present = result.Worker.LLMGateway.Gateway.Present || result.Worker.LLMGateway.Credential.Present

	if len(conflicts) != 0 {
		sort.Slice(conflicts, func(i, j int) bool { return conflicts[i].Path < conflicts[j].Path })
		return Spec{}, conflicts[0]
	}
	return result, nil
}

type mergeValue struct {
	ref   Ref
	value any
}

func collect(entries []LayerEntry, selectValue func(Spec) (any, bool)) []mergeValue {
	result := make([]mergeValue, 0)
	for _, entry := range entries {
		value, present := selectValue(entry.Spec)
		if present {
			result = append(result, mergeValue{ref: entry.Ref, value: value})
		}
	}
	return result
}
