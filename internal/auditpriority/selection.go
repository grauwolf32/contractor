package auditpriority

import (
	"reflect"
	"sort"

	"github.com/grauwolf32/contractor/internal/contracts"
)

// BindVerdict validates the assigned item and evidence whitelist then detaches
// the model response. Evidence IDs must be obtained from the retained context
// identified by cycle.ContextDigest; this pure helper cannot authenticate refs.
func BindVerdict(cycle CycleBinding, candidate Candidate, verdict Verdict, contextEvidenceIDs []string) (BoundVerdict, error) {
	if err := cycle.Validate(); err != nil {
		return BoundVerdict{}, err
	}
	id, err := CandidateID(cycle.InventoryDigest, candidate.ItemKey, candidate.ItemVersion)
	if err != nil || id != candidate.ID {
		return BoundVerdict{}, invalid(CodeInvalidPool)
	}
	if err := ValidateVerdict(verdict, candidate.ItemKey, contextEvidenceIDs); err != nil {
		return BoundVerdict{}, err
	}
	return BoundVerdict{Cycle: cycle, CandidateID: candidate.ID, Verdict: cloneVerdict(verdict)}, nil
}

// Select calculates a proposal over the entire remaining pool. All verdicts
// must match its exact semantic binding and every candidate appears once. It
// does not inspect or grant receipt, journal, budget or execution authority.
func Select(pool Pool, cycle CycleBinding, verdicts []BoundVerdict, contextEvidenceIDs []string) (Selection, error) {
	empty := Selection{}
	if err := pool.Validate(); err != nil {
		return empty, err
	}
	if err := cycle.Validate(); err != nil {
		return empty, err
	}
	poolDigest, err := PoolDigest(pool)
	if err != nil || cycle.InventoryDigest != pool.InventoryDigest || cycle.PoolDigest != poolDigest {
		return empty, invalid(CodeInvalidBinding)
	}
	if !validEvidenceIDs(contextEvidenceIDs) {
		return empty, invalid(CodeInvalidBinding)
	}
	if len(verdicts) != len(pool.Candidates) {
		return empty, invalid(CodeIncompleteRanking)
	}
	byID := make(map[string]BoundVerdict, len(verdicts))
	for _, verdict := range verdicts {
		if verdict.Cycle != cycle {
			return empty, invalid(CodeInvalidBinding)
		}
		if _, exists := byID[verdict.CandidateID]; exists {
			return empty, invalid(CodeIncompleteRanking)
		}
		byID[verdict.CandidateID] = verdict
	}
	result := Selection{Schema: SelectionSchema, Cycle: cycle, Rows: make([]SelectionRow, 0, len(pool.Candidates))}
	for _, candidate := range pool.Candidates {
		verdict, exists := byID[candidate.ID]
		if !exists {
			return empty, invalid(CodeIncompleteRanking)
		}
		if err := ValidateVerdict(verdict.Verdict, candidate.ItemKey, contextEvidenceIDs); err != nil {
			return empty, err
		}
		result.Rows = append(result.Rows, SelectionRow{Candidate: candidate, Verdict: cloneVerdict(verdict.Verdict)})
	}
	sort.Slice(result.Rows, func(i, j int) bool {
		left, right := result.Rows[i], result.Rows[j]
		if left.Verdict.Priority != right.Verdict.Priority {
			return priorityOrder(left.Verdict.Priority) < priorityOrder(right.Verdict.Priority)
		}
		return left.Candidate.ID < right.Candidate.ID
	})
	result.SelectedCount = min(cycle.TopN, len(result.Rows))
	result.DeferredCount = len(result.Rows) - result.SelectedCount
	for i := range result.Rows {
		result.Rows[i].Rank = i + 1
		result.Rows[i].Selected = i < result.SelectedCount
		if !result.Rows[i].Selected {
			result.Rows[i].Code = CodeDeferredTopN
		}
	}
	if len(result.Rows) == 0 {
		result.Code = CodeNoRemainingCandidates
	} else if len(result.Rows) < cycle.TopN {
		result.Code = CodeFewerCandidates
	}
	return result, nil
}

func priorityOrder(value Priority) int {
	switch value {
	case PriorityCritical:
		return 0
	case PriorityHigh:
		return 1
	case PriorityMedium:
		return 2
	case PriorityLow:
		return 3
	default:
		return 4 // Invalid values are rejected before sorting.
	}
}

// Validate checks the proposal's internal identities, complete membership,
// order, counts and codes. It cannot prove its externally supplied context or
// model provenance. A trusted caller must use Select with the retained context
// whitelist and verify the accepted Run-output receipt before admission.
func (s Selection) Validate() error {
	if s.Schema != SelectionSchema || s.Rows == nil || len(s.Rows) > MaxCandidates {
		return invalid(CodeInvalidSelection)
	}
	items := make([]ItemIdentity, 0, len(s.Rows))
	verdicts := make([]BoundVerdict, 0, len(s.Rows))
	evidenceSet := make(map[string]bool)
	for _, row := range s.Rows {
		items = append(items, ItemIdentity{Key: row.Candidate.ItemKey, Version: row.Candidate.ItemVersion})
		verdicts = append(verdicts, BoundVerdict{Cycle: s.Cycle, CandidateID: row.Candidate.ID, Verdict: row.Verdict})
		// Bound each programmatic row before processing caller-owned slices.
		if len(row.Verdict.EvidenceIDs) > MaxEvidenceIDs {
			return invalid(CodeInvalidSelection)
		}
		for _, id := range row.Verdict.EvidenceIDs {
			evidenceSet[id] = true
			if len(evidenceSet) > MaxContextEvidence {
				return invalid(CodeInvalidSelection)
			}
		}
	}
	pool, err := NewPool(s.Cycle.InventoryDigest, items)
	if err != nil {
		return invalid(CodeInvalidSelection)
	}
	evidence := make([]string, 0, len(evidenceSet))
	for id := range evidenceSet {
		evidence = append(evidence, id)
	}
	expected, err := Select(pool, s.Cycle, verdicts, evidence)
	if err != nil || !reflect.DeepEqual(expected, s) {
		return invalid(CodeInvalidSelection)
	}
	return nil
}

func MarshalSelection(selection Selection) ([]byte, error) {
	if err := selection.Validate(); err != nil {
		return nil, err
	}
	data, err := contracts.MarshalPrivateCanonical(selection)
	if err != nil || len(data) > MaxSelectionBytes {
		return nil, invalid(CodeInvalidSelection)
	}
	return data, nil
}

func SelectionDigest(selection Selection) (string, error) {
	data, err := MarshalSelection(selection)
	if err != nil {
		return "", err
	}
	return digestBytes(data), nil
}
