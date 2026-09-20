package runtimeconfig

import (
	"reflect"
	"sort"

	"github.com/grauwolf32/contractor/internal/contracts"
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
	conflicts := []*MergeConflictError{
		mergeField(ordered, "worker.llmGateway.gateway", &result.Worker.LLMGateway.Gateway,
			func(s Spec) (Field[contracts.LLMGatewayConfigRef], bool) {
				return s.Worker.LLMGateway.Gateway, s.Worker.LLMGateway.Gateway.Present
			}),
		mergeField(ordered, "worker.llmGateway.credential", &result.Worker.LLMGateway.Credential,
			func(s Spec) (Field[string], bool) {
				return s.Worker.LLMGateway.Credential, s.Worker.LLMGateway.Credential.Present
			}),
		mergeField(ordered, "worker.telemetry", &result.Worker.Telemetry,
			func(s Spec) (AtomicPatch[TelemetryConfig], bool) {
				return s.Worker.Telemetry, s.Worker.Telemetry.Present
			}),
		mergeField(ordered, "worker.httpProxy", &result.Worker.HTTPProxy,
			func(s Spec) (AtomicPatch[HTTPProxyConfig], bool) {
				return s.Worker.HTTPProxy, s.Worker.HTTPProxy.Present
			}),
		mergeField(ordered, "worker.caido", &result.Worker.Caido,
			func(s Spec) (AtomicPatch[CaidoConfig], bool) {
				return s.Worker.Caido, s.Worker.Caido.Present
			}),
		mergeField(ordered, "planner.telemetry", &result.Planner.Telemetry,
			func(s Spec) (AtomicPatch[TelemetryConfig], bool) {
				return s.Planner.Telemetry, s.Planner.Telemetry.Present
			}),
	}
	result.Worker.LLMGateway.Present = result.Worker.LLMGateway.Gateway.Present || result.Worker.LLMGateway.Credential.Present

	var firstConflict *MergeConflictError
	for _, conflict := range conflicts {
		if conflict != nil && (firstConflict == nil || conflict.Path < firstConflict.Path) {
			firstConflict = conflict
		}
	}
	if firstConflict != nil {
		return Spec{}, firstConflict
	}
	return result, nil
}

// The selector and destination share T, so assigning a field with another
// field's type is rejected by the compiler. Atomic values still use deep
// equality: equal telemetry pointers and proxy target slices may be distinct.
func mergeField[T any](
	entries []LayerEntry,
	path string,
	destination *T,
	selectValue func(Spec) (T, bool),
) *MergeConflictError {
	var first T
	present, conflict := false, false
	refs := make([]Ref, 0)
	seen := make(map[Ref]struct{})
	for _, entry := range entries {
		value, selected := selectValue(entry.Spec)
		if !selected {
			continue
		}
		if _, exists := seen[entry.Ref]; !exists {
			refs = append(refs, entry.Ref)
			seen[entry.Ref] = struct{}{}
		}
		if !present {
			first, present = value, true
		} else if !reflect.DeepEqual(first, value) {
			conflict = true
		}
	}
	if !conflict {
		if present {
			*destination = first
		}
		return nil
	}
	sort.Slice(refs, func(i, j int) bool { return refs[i].String() < refs[j].String() })
	return &MergeConflictError{Path: path, Refs: refs}
}
