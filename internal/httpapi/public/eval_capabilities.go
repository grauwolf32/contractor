package public

import (
	"encoding/json"
	"fmt"
	"sort"
	"sync"

	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/evaldomain"
)

type capabilityBinding struct {
	Kind      string  `json:"kind"`
	Selector  string  `json:"selector"`
	Available bool    `json:"available"`
	Reason    *string `json:"reason"`
}

type capabilityPage struct {
	Items             []capabilityBinding
	HasMore           bool
	Snapshot, LastKey string
}

type capabilityIndex struct {
	rows   []capabilityBinding
	keys   []string
	digest string
}

// capabilitiesReader retains only the latest immutable catalog's three light
// binding indexes. Profile compatibility continues to use AuditService's rule.
// Building happens outside the lock; a page copies only its requested rows.
type capabilitiesReader struct {
	mu     sync.Mutex
	cached *capabilitySnapshot
}

type capabilitySnapshot struct {
	source  *config.Snapshot
	once    sync.Once
	indexes map[string]*capabilityIndex
	err     error
}

func (r *capabilitiesReader) bindings(current *config.Snapshot, kind, after string, limit int, expectedSnapshot string) (capabilityPage, error) {
	if kind != "" && kind != "workflow" && kind != "audit" || limit < 1 || limit > evaldomain.MaxPageSize {
		return capabilityPage{}, evaldomain.Failure("eval_invalid")
	}
	if current == nil {
		return capabilityPage{}, fmt.Errorf("Eval capability catalog is unavailable")
	}
	r.mu.Lock()
	if r.cached == nil || r.cached.source != current {
		r.cached = &capabilitySnapshot{source: current}
	}
	cached := r.cached
	r.mu.Unlock()
	cached.once.Do(func() { cached.indexes, cached.err = buildCapabilityIndexes(current) })
	if cached.err != nil {
		return capabilityPage{}, cached.err
	}
	index := cached.indexes[kind]
	if expectedSnapshot != "" && expectedSnapshot != index.digest {
		return capabilityPage{}, evaldomain.Failure("eval_view_changed")
	}
	start := sort.Search(len(index.keys), func(i int) bool { return index.keys[i] > after })
	end := min(start+limit, len(index.rows))
	items := make([]capabilityBinding, end-start)
	copy(items, index.rows[start:end])
	for i := range items {
		if items[i].Reason != nil {
			reason := *items[i].Reason
			items[i].Reason = &reason
		}
	}
	result := capabilityPage{Items: items, HasMore: end < len(index.rows), Snapshot: index.digest}
	if end > start {
		result.LastKey = index.keys[end-1]
	}
	return result, nil
}

func buildCapabilityIndexes(snapshot *config.Snapshot) (map[string]*capabilityIndex, error) {
	bindings := []capabilityBinding{}
	for _, workflow := range snapshot.Workflows() {
		bindings = append(bindings, capabilityBinding{Kind: "workflow", Selector: workflow.Ref.Name + "@" + workflow.Ref.Version, Available: true})
	}
	for _, profile := range snapshot.AuditProfiles() {
		compatible := auditservice.ProfileCompatibility(profile).ServerCompatible
		var reason *string
		if !compatible {
			message := "This AuditProfile requires unsupported Server capabilities."
			reason = &message
		}
		bindings = append(bindings, capabilityBinding{Kind: "audit", Selector: profile.Ref.Name + "@" + profile.Ref.Version, Available: compatible, Reason: reason})
	}
	sort.Slice(bindings, func(i, j int) bool {
		return bindings[i].Kind+":"+bindings[i].Selector < bindings[j].Kind+":"+bindings[j].Selector
	})
	indexes := make(map[string]*capabilityIndex, 3)
	for _, kind := range []string{"", "workflow", "audit"} {
		index := &capabilityIndex{rows: []capabilityBinding{}}
		for _, binding := range bindings {
			if kind == "" || binding.Kind == kind {
				index.rows = append(index.rows, binding)
				index.keys = append(index.keys, binding.Kind+":"+binding.Selector)
			}
		}
		raw, err := json.Marshal(index.rows)
		if err != nil {
			return nil, fmt.Errorf("encode Eval capabilities: %w", err)
		}
		index.digest = evaldomain.Digest(raw)
		indexes[kind] = index
	}
	return indexes, nil
}
