package evalservice

import (
	"encoding/json"
	"fmt"
	"sort"
	"time"

	"github.com/grauwolf32/contractor/internal/evaldomain"
)

// Preflight is populated by read-only catalog/artifact resolution. Unsupported
// members remain in the matrix. No arbitrary client-supplied capability becomes
// a verified pin merely by appearing in a dataset or a binding.
type Preflight struct {
	Pins         map[string]Pin
	Capabilities []string
	Cases        map[string]Eligibility
	Snapshot     json.RawMessage
}
type Pin struct {
	Value  any    `json:"value"`
	Origin string `json:"origin"`
}
type Eligibility struct {
	State  string
	Reason *string
}
type Resource struct {
	Path, Kind string
	Document   evaldomain.Frozen `json:"-"`
}
type PlanBundle struct {
	Private   map[string]evaldomain.Frozen   `json:"-"`
	Plan      evaldomain.Frozen              `json:"-"`
	Setup     json.RawMessage                `json:"-"`
	Resources []Resource                     `json:"-"`
	Inputs    map[string]evaldomain.Artifact `json:"-"`
	Cases     map[string]evaldomain.Case     `json:"-"`
}
type documentRef struct {
	Resource string `json:"resource"`
	SHA256   string `json:"sha256"`
}
type blobRef struct {
	Resource  string `json:"resource"`
	SHA256    string `json:"sha256"`
	MediaType string `json:"media_type"`
	SizeBytes int64  `json:"size_bytes"`
}

func jsonBytes(value any) ([]byte, error) {
	b, err := json.Marshal(value)
	if err != nil {
		return nil, evaldomain.Failure("eval_invalid")
	}
	return b, nil
}
func hashJSON(value any) (string, error) {
	b, err := jsonBytes(value)
	if err != nil {
		return "", err
	}
	return evaldomain.Digest(b), nil
}

// BuildPlan materializes portable documents only. The caller persists the whole
// bundle atomically after verifying the same draft revision and claim. The
// frozen execution order is authoritative; no client has to reproduce a PRNG.
func BuildPlan(portableID string, createdAt time.Time, draft evaldomain.Draft, dataset evaldomain.DatasetInput, preflight map[string]Preflight) (PlanBundle, error) {
	out := PlanBundle{Private: map[string]evaldomain.Frozen{}, Inputs: map[string]evaldomain.Artifact{}, Cases: map[string]evaldomain.Case{}, Resources: []Resource{}}
	draftBytes, err := jsonBytes(draft)
	if err != nil {
		return out, err
	}
	if err = evaldomain.Validate("Draft", draftBytes); err != nil {
		return out, err
	}
	var clonedDraft evaldomain.Draft
	if err = json.Unmarshal(draftBytes, &clonedDraft); err != nil {
		return out, err
	}
	draft = clonedDraft
	dataBytes, err := jsonBytes(dataset)
	if err != nil {
		return out, err
	}
	if err = evaldomain.Validate("DatasetInput", dataBytes); err != nil {
		return out, err
	}
	if dataset.DatasetID != draft.Dataset.ID {
		return out, evaldomain.Failure("eval_member_conflict")
	}
	add := func(path, kind string, value any) (documentRef, error) {
		b, err := jsonBytes(value)
		if err != nil {
			return documentRef{}, err
		}
		doc, err := evaldomain.Freeze(kind, b)
		if err != nil {
			return documentRef{}, err
		}
		out.Resources = append(out.Resources, Resource{path, kind, doc})
		return documentRef{path, doc.Digest()}, nil
	}
	byCase := map[string]evaldomain.Case{}
	for _, c := range dataset.Cases {
		byCase[c.ID] = c
	}
	selected := make([]evaldomain.Case, 0, len(draft.CaseIDs))
	caseRefs := make([]documentRef, 0, len(draft.CaseIDs))
	checks := make([]any, 0, len(draft.Checks))
	private := map[string]evaldomain.PrivateCheck{}
	for _, c := range dataset.PrivateChecks {
		private[c.ID+"@"+c.Revision] = c
	}
	for i := range draft.Checks {
		check := &draft.Checks[i]
		if check.Evaluator == "human-review@1" {
			if _, ok := private[check.ID+"@"+check.RubricRevision]; !ok {
				return out, evaldomain.Failure("eval_not_ready")
			}
			if check.ImplementationSHA256 != "" && check.ImplementationSHA256 != HumanPolicySHA256() {
				return out, evaldomain.Failure("eval_pin_mismatch")
			}
			check.ImplementationSHA256 = HumanPolicySHA256()
			raw, err := jsonBytes(private[check.ID+"@"+check.RubricRevision])
			if err != nil {
				return out, err
			}
			doc, err := evaldomain.Freeze("PrivateCheck", raw)
			if err != nil {
				return out, err
			}
			out.Private["private/"+check.ID+".json"] = doc
		} else {
			// The collection task extends this fixed registry with native checks.
			// A supplied hash never authorizes arbitrary evaluator implementation.
			return out, evaldomain.Failure("eval_not_ready")
		}
		checks = append(checks, map[string]any{"id": check.ID, "scorer": check.Evaluator, "implementation_sha256": check.ImplementationSHA256, "parameters": check.Parameters, "ground_truth_role": nil, "required": check.Required, "allow_not_applicable": check.AllowNotApplicable})
		if check.Evaluator == "human-review@1" {
			checks[len(checks)-1].(map[string]any)["ground_truth_role"] = check.ID
		}
		if check.Parameters == nil {
			checks[len(checks)-1].(map[string]any)["parameters"] = map[string]string{}
		}
	}
	for _, id := range draft.CaseIDs {
		c, ok := byCase[id]
		if !ok {
			return out, evaldomain.Failure("eval_not_found")
		}
		selected = append(selected, c)
		inputs := map[string]blobRef{}
		for name, ref := range c.Inputs {
			path := "inputs/" + id + "/" + name
			out.Inputs[path] = ref
			inputs[name] = blobRef{path, ref.SHA256, ref.MediaType, ref.SizeBytes}
		}
		outputs := map[string]any{}
		for name, v := range c.Outputs {
			outputs[name] = map[string]any{"media_types": v.MediaTypes, "required": v.Required}
		}
		// Private check material remains in the retained dataset revision. Portable
		// provenance links its exact digest; it is never copied into an input blob.
		truth := map[string]any{}
		for _, check := range draft.Checks {
			if check.Evaluator == "human-review@1" {
				v := private[check.ID+"@"+check.RubricRevision]
				b, err := jsonBytes(v)
				if err != nil {
					return out, err
				}
				truth[check.ID] = map[string]any{"resource": "private/" + check.ID + ".json", "sha256": evaldomain.Digest(b), "media_type": "application/json", "size_bytes": len(b)}
			}
		}
		ref, err := add("cases/"+id+".json", "playground.case/v2", map[string]any{"schema_version": "playground.case/v2", "id": id, "task": c.Task, "inputs": inputs, "requires": c.Requires, "outputs": outputs, "evaluation": map[string]any{"ground_truth": truth, "assertions": map[string]any{}}, "provenance": map[string]any{"sources": []any{}, "datasets": []any{map[string]any{"id": dataset.DatasetID, "revision": draft.Dataset.Revision, "sha256": nil}}}})
		if err != nil {
			return out, err
		}
		caseRefs = append(caseRefs, ref)
	}
	suiteRef, err := add("suite.json", "playground.suite/v2", map[string]any{"schema_version": "playground.suite/v2", "id": draft.Dataset.ID, "cases": caseRefs, "scoring": map[string]any{"decision": "all_required_pass", "checks": checks}, "provenance": map[string]any{"sources": []any{}}})
	if err != nil {
		return out, err
	}
	sourcePin, err := hashJSON(out.Inputs)
	if err != nil {
		return out, err
	}
	checkPin, err := hashJSON(struct {
		Checks  []evaldomain.Check
		Private []evaldomain.PrivateCheck
	}{draft.Checks, dataset.PrivateChecks})
	if err != nil {
		return out, err
	}
	allPins := map[string]map[string]Pin{}
	variants := make([]any, 0, 2)
	bindingRefs := map[string]documentRef{}
	for _, variant := range draft.Variants {
		resolution, ok := preflight[variant.ID]
		if !ok {
			return out, evaldomain.Failure("eval_not_ready")
		}
		pins := map[string]Pin{}
		for key, value := range resolution.Pins {
			pins[key] = value
		}
		pins["source"] = Pin{sourcePin, "observed"}
		effectiveTasks := make([]any, 0, len(selected))
		for _, c := range selected {
			effectiveTasks = append(effectiveTasks, map[string]any{"task": c.Task, "parameters": MapParameters(c, variant), "inputs": variant.InputMapping, "outputs": variant.OutputMapping})
		}
		taskPin, err := hashJSON(effectiveTasks)
		if err != nil {
			return out, err
		}
		pins["tasks"] = Pin{taskPin, "observed"}
		pins["scorers"] = Pin{checkPin, "observed"}
		pins["expected"] = Pin{checkPin, "observed"}
		for _, dimension := range append(append([]string{}, draft.Comparison.RequiredEqual...), draft.Comparison.AllowedDifferences...) {
			if _, ok := pins[dimension]; !ok {
				pins[dimension] = Pin{nil, "unavailable"}
			}
		}
		allPins[variant.ID] = pins
		document, err := bindingDocument(variant, resolution)
		if err != nil {
			return out, err
		}
		path := "bindings/" + variant.ID + ".json"
		out.Resources = append(out.Resources, Resource{path, document.Kind(), document})
		ref := documentRef{path, document.Digest()}
		bindingRefs[variant.ID] = ref
		variants = append(variants, map[string]any{"id": variant.ID, "binding": ref})
	}
	for _, dimension := range draft.Comparison.RequiredEqual {
		a, b := allPins[draft.Comparison.Baseline][dimension], allPins[draft.Comparison.Candidate][dimension]
		if a.Origin != "observed" || b.Origin != "observed" || a.Value == nil || b.Value == nil {
			return out, evaldomain.Failure("eval_pin_mismatch")
		}
		av, err := jsonBytes(a.Value)
		if err != nil {
			return out, err
		}
		bv, err := jsonBytes(b.Value)
		if err != nil {
			return out, err
		}
		if string(av) != string(bv) {
			return out, evaldomain.Failure("eval_pin_mismatch")
		}
	}
	comparison, extensions := evaldomain.PortableComparison(draft.Comparison)
	budgets := map[string]any{"max_members": draft.Budgets.MaxMembers, "max_in_flight": draft.Budgets.MaxInFlight, "wall_ms": draft.Budgets.WallMS, "max_observed_total_tokens": draft.Budgets.MaxObservedTotalTokens}
	publication := map[string]any{"provider": "contractor@1", "connection": "managed", "enabled": true}
	experimentRef, err := add("experiment.json", "playground.experiment/v1", map[string]any{"schema_version": "playground.experiment/v1", "id": portableID, "suites": []documentRef{suiteRef}, "variants": variants, "repetitions": draft.Repetitions, "order": map[string]any{"kind": draft.Order.Kind, "seed": draft.Order.Seed}, "budgets": budgets, "comparison": comparison, "publication": publication, "extensions": extensions})
	if err != nil {
		return out, err
	}
	members := make([]any, 0, len(selected)*draft.Repetitions*2)
	order := make([]string, 0, len(selected)*draft.Repetitions*2)
	for ci, c := range selected {
		for sample := 1; sample <= draft.Repetitions; sample++ {
			for _, v := range draft.Variants {
				mid, err := evaldomain.MemberID(portableID, draft.Dataset.ID, c.ID, sample, v.ID)
				if err != nil {
					return out, err
				}
				eligibility, ok := preflight[v.ID].Cases[c.ID]
				if !ok {
					return out, evaldomain.Failure("eval_not_ready")
				}
				members = append(members, map[string]any{"member_id": mid, "suite_id": draft.Dataset.ID, "case_id": c.ID, "sample": sample, "variant_id": v.ID, "case_sha256": caseRefs[ci].SHA256, "binding_sha256": bindingRefs[v.ID].SHA256, "eligibility": eligibility.State, "reason": eligibility.Reason})
				order = append(order, mid)
				out.Cases[mid] = c
			}
		}
	}
	if draft.Order.Kind == "seeded_shuffle" {
		sort.Slice(order, func(i, j int) bool {
			// Version-independent seeded permutation, retained explicitly in the plan.
			left := evaldomain.Digest([]byte(fmt.Sprintf("%d:%s", *draft.Order.Seed, order[i])))
			right := evaldomain.Digest([]byte(fmt.Sprintf("%d:%s", *draft.Order.Seed, order[j])))
			return left < right
		})
	}
	raw, err := jsonBytes(map[string]any{"schema_version": "playground.plan/v1", "experiment_id": portableID, "created_at": createdAt.UTC().Format(time.RFC3339Nano), "experiment_ref": experimentRef, "suites": []any{map[string]any{"id": draft.Dataset.ID, "ref": suiteRef}}, "variants": variants, "pins": allPins, "members": members, "execution_order": order, "budgets": budgets, "comparison": comparison, "publication": publication})
	if err != nil {
		return out, err
	}
	out.Plan, err = evaldomain.Freeze("playground.plan/v1", raw)
	if err != nil {
		return out, err
	}
	out.Setup, err = jsonBytes(map[string]any{"dataset": draft.Dataset, "caseIds": draft.CaseIDs, "repetitions": draft.Repetitions, "variants": draft.Variants, "checks": draft.Checks, "comparison": draft.Comparison, "budgets": draft.Budgets})
	if err == nil {
		err = evaldomain.Validate("ExperimentSetup", out.Setup)
	}
	return out, err
}
