package evalservice

import (
	"fmt"
	"sort"

	"github.com/grauwolf32/contractor/internal/evaldomain"
)

type effectiveTask struct {
	Task       evaldomain.Task   `json:"task"`
	Parameters map[string]string `json:"parameters"`
	Inputs     map[string]string `json:"inputs"`
	Outputs    map[string]string `json:"outputs"`
}

func (b *planBuilder) buildVariants(cases []evaldomain.Case) ([]variantReference, map[string]map[string]Pin, error) {
	source, err := hashJSON(b.bundle.Inputs)
	if err != nil {
		return nil, nil, err
	}
	checks, err := hashJSON(struct {
		Checks  []evaldomain.Check
		Private []evaldomain.PrivateCheck
	}{b.draft.Checks, b.dataset.PrivateChecks})
	if err != nil {
		return nil, nil, err
	}
	refs := make([]variantReference, 0, len(b.draft.Variants))
	allPins := map[string]map[string]Pin{}
	for _, variant := range b.draft.Variants {
		resolved, ok := b.preflight[variant.ID]
		if !ok {
			return nil, nil, evaldomain.Failure("eval_not_ready")
		}
		pins := map[string]Pin{}
		for key, value := range resolved.Pins {
			pins[key] = value
		}
		tasks := make([]effectiveTask, 0, len(cases))
		for _, c := range cases {
			tasks = append(tasks, effectiveTask{Task: c.Task, Parameters: MapParameters(c, variant), Inputs: variant.InputMapping, Outputs: variant.OutputMapping})
		}
		taskDigest, err := hashJSON(tasks)
		if err != nil {
			return nil, nil, err
		}
		pins["source"], pins["tasks"] = observedPin(source), observedPin(taskDigest)
		pins["scorers"], pins["expected"] = observedPin(checks), observedPin(checks)
		dimensions := append(append([]string{}, b.draft.Comparison.RequiredEqual...), b.draft.Comparison.AllowedDifferences...)
		for _, dimension := range dimensions {
			if _, ok := pins[dimension]; !ok {
				pins[dimension] = Pin{Origin: "unavailable"}
			}
		}
		allPins[variant.ID] = pins
		doc, err := bindingDocument(variant, resolved)
		if err != nil {
			return nil, nil, err
		}
		path := "bindings/" + variant.ID + ".json"
		b.bundle.Resources = append(b.bundle.Resources, Resource{Path: path, Kind: doc.Kind(), Document: doc})
		refs = append(refs, variantReference{ID: variant.ID, Binding: documentRef{Resource: path, SHA256: doc.Digest()}})
	}
	for _, dimension := range b.draft.Comparison.RequiredEqual {
		a, c := allPins[b.draft.Comparison.Baseline][dimension], allPins[b.draft.Comparison.Candidate][dimension]
		if a.Origin != "observed" || c.Origin != "observed" || a.Value == nil || c.Value == nil || *a.Value != *c.Value {
			return nil, nil, evaldomain.Failure("eval_pin_mismatch")
		}
	}
	return refs, allPins, nil
}

func (b *planBuilder) buildMembers(experimentID string, cases []evaldomain.Case, caseRefs []documentRef, variants []variantReference) ([]planMember, []string, error) {
	size := len(cases) * b.draft.Repetitions * len(variants)
	members := make([]planMember, 0, size)
	order := make([]string, 0, size)
	for ci, c := range cases {
		for sample := 1; sample <= b.draft.Repetitions; sample++ {
			for _, variant := range variants {
				id, err := evaldomain.MemberID(experimentID, b.draft.Dataset.ID, c.ID, sample, variant.ID)
				if err != nil {
					return nil, nil, err
				}
				eligibility, ok := b.preflight[variant.ID].Cases[c.ID]
				if !ok {
					return nil, nil, evaldomain.Failure("eval_not_ready")
				}
				members = append(members, planMember{
					PublicMember: evaldomain.PublicMember{
						MemberID:      id,
						SuiteID:       b.draft.Dataset.ID,
						CaseID:        c.ID,
						Sample:        sample,
						VariantID:     variant.ID,
						CaseSHA256:    caseRefs[ci].SHA256,
						BindingSHA256: variant.Binding.SHA256,
						Eligibility:   eligibility.State,
					},
					Reason: eligibility.Reason,
				})
				order = append(order, id)
				b.bundle.Cases[id] = c
			}
		}
	}
	if b.draft.Order.Kind == "seeded_shuffle" {
		// Version-independent permutation is retained explicitly in the frozen plan.
		sort.Slice(order, func(i, j int) bool {
			left := evaldomain.Digest([]byte(fmt.Sprintf("%d:%s", *b.draft.Order.Seed, order[i])))
			right := evaldomain.Digest([]byte(fmt.Sprintf("%d:%s", *b.draft.Order.Seed, order[j])))
			return left < right
		})
	}
	return members, order, nil
}
