package auditdomain

import (
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"maps"
	"sort"
)

// InitialHypothesis assigns a stable opaque identity to an authored seed. The
// source alias reconciles intake; it is not an owner prefix in the public ID.
func InitialHypothesis(auditID, alias string, seed auditcontract.HypothesisSeed) auditcontract.Hypothesis {
	return auditcontract.Hypothesis{Schema: auditcontract.HypothesisSchema,
		ID: DeterministicID("hypothesis", auditID, alias), Revision: 1,
		Statement: seed.Statement, Scope: maps.Clone(seed.Scope), Grounds: seed.Grounds,
		Origin: auditcontract.HypothesisOrigin{Kind: "checklist", AuditID: auditID}}
}

// InitialHypothesisKeys preserves declared source order, then any unassigned
// authored seeds. It has no dependence on result history or model ranking.
func InitialHypothesisKeys(program auditcontract.Program) []string {
	keys := []string{}
	if program.SeedSource != nil && program.SeedSource.Standard != nil {
		for key := range program.Hypotheses {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		return keys
	}
	seen := map[string]bool{}
	for _, work := range program.InitialWork {
		if work.Hypothesis != "" && !seen[work.Hypothesis] {
			keys = append(keys, work.Hypothesis)
			seen[work.Hypothesis] = true
		}
	}
	remaining := []string{}
	for key := range program.Hypotheses {
		if !seen[key] {
			remaining = append(remaining, key)
		}
	}
	sort.Strings(remaining)
	return append(keys, remaining...)
}
