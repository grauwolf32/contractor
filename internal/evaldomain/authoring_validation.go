package evaldomain

import (
	"encoding/json"
	"reflect"
)

func decodeValue[T any](value any) (T, error) {
	var out T
	raw, err := json.Marshal(value)
	if err == nil {
		err = json.Unmarshal(raw, &out)
	}
	if err != nil {
		return out, Failure("eval_invalid")
	}
	return out, nil
}
func typedCheck[T any](value any, validate func(T) error) error {
	out, err := decodeValue[T](value)
	if err != nil {
		return err
	}
	return validate(out)
}
func uniqueRows(rows []any, key string) bool {
	seen := map[any]bool{}
	for _, row := range rows {
		v := row.(map[string]any)[key]
		if seen[v] {
			return false
		}
		seen[v] = true
	}
	return true
}
func asObject(value any) map[string]any { return value.(map[string]any) }
func asRows(value any) []any            { return value.([]any) }
func numeric(value any) float64 {
	switch n := value.(type) {
	case json.Number:
		f, _ := n.Float64()
		return f
	case float64:
		return n
	default:
		return 0
	}
}

func validateSemantics(kind string, v map[string]any) error {
	switch kind {
	case "CreateExperiment":
		if v["controlMode"] == "server" {
			return typedCheck(v["draft"], validateDraft)
		}
		return typedCheck(v["registration"], validateRegistration)
	case "DraftUpdate":
		return typedCheck(v["draft"], validateDraft)
	case "Draft":
		return typedCheck(v, validateDraft)
	case "ExternalRegistration":
		return typedCheck(v, validateRegistration)
	case "DatasetInput":
		d, err := decodeValue[DatasetInput](v)
		if err != nil {
			return err
		}
		seen := map[string]bool{}
		for _, c := range d.Cases {
			if seen[c.ID] {
				return Failure("eval_invalid")
			}
			seen[c.ID] = true
			if err := validateCase(c); err != nil {
				return err
			}
		}
		checks := map[string]bool{}
		for _, c := range d.PrivateChecks {
			key := c.ID + "@" + c.Revision
			if checks[key] {
				return Failure("eval_invalid")
			}
			checks[key] = true
		}
	case "Case":
		return typedCheck(v, validateCase)
	case "PublicPlan":
		return typedCheck(v, validatePublicPlan)
	case "Usage":
		return typedCheck(v, func(u Usage) error { return validateUsage(u, "") })
	case "Measure":
		return typedCheck(v, func(m Measure) error { return validateMeasure(m, "") })
	case "ResultInput":
		return validateResult(v)
	case "AssessmentInput":
		if !uniqueRows(asRows(v["checks"]), "id") {
			return Failure("eval_invalid")
		}
	case "SelectionInput":
		if !uniqueRows(asRows(v["selections"]), "memberId") {
			return Failure("eval_member_conflict")
		}
	case "Summary":
		return validateSummary(v)
	case "Pair":
		return validatePair(v)
	case "Chart":
		return validateChart(v)
	case "ExecutionPage":
		if v["inventoryComplete"] == true && len(asRows(v["gaps"])) != 0 {
			return Failure("eval_invalid")
		}
	case "playground.plan/v1":
		return validatePortablePlan(v)
	}
	return validateNested(v)
}

func validateCase(c Case) error {
	for _, a := range c.Inputs {
		if a.Scope == "run" {
			return Failure("eval_invalid")
		}
	}
	return nil
}
func validateVariants(variants []Variant, comparison Comparison, checks []Check, budgets Budgets) error {
	if len(variants) != 2 || variants[0].ID == variants[1].ID || variants[0].Kind != variants[1].Kind {
		return Failure("eval_invalid")
	}
	if comparison.Baseline == comparison.Candidate {
		return Failure("eval_invalid")
	}
	ids := map[string]bool{variants[0].ID: true, variants[1].ID: true}
	if !ids[comparison.Baseline] || !ids[comparison.Candidate] || budgets.MaxInFlight > budgets.MaxMembers {
		return Failure("eval_invalid")
	}
	equal := map[string]bool{}
	for _, pin := range comparison.RequiredEqual {
		equal[pin] = true
	}
	for _, pin := range comparison.AllowedDifferences {
		if equal[pin] {
			return Failure("eval_pin_mismatch")
		}
	}
	seen, required := map[string]bool{}, false
	for _, check := range checks {
		if seen[check.ID] {
			return Failure("eval_invalid")
		}
		seen[check.ID] = true
		required = required || check.Required
		if check.Evaluator == "human-review@1" && check.RubricRevision == "" {
			return Failure("eval_invalid")
		}
	}
	if !required {
		return Failure("eval_invalid")
	}
	return nil
}
func validateDraft(d Draft) error {
	if err := validateVariants(d.Variants, d.Comparison, d.Checks, d.Budgets); err != nil {
		return err
	}
	if len(d.CaseIDs)*2*d.Repetitions > d.Budgets.MaxMembers {
		return Failure("eval_limit_exceeded")
	}
	return nil
}
func validatePublicPlan(p PublicPlan) error {
	seen := map[string]bool{}
	for _, m := range p.Members {
		id, err := MemberID(p.ExperimentID, m.SuiteID, m.CaseID, m.Sample, m.VariantID)
		if err != nil || id != m.MemberID || seen[id] {
			return Failure("eval_member_conflict")
		}
		seen[id] = true
	}
	return nil
}
func validateRegistration(r ExternalRegistration) error {
	if r.SourcePlanSHA256 != r.Manifest.SourceRecordSHA256 {
		return Failure("eval_pin_mismatch")
	}
	if err := validateVariants(r.Variants, r.Comparison, r.Checks, r.Budgets); err != nil {
		return err
	}
	if err := validatePublicPlan(r.Manifest); err != nil {
		return err
	}
	if len(r.Manifest.Members) > r.Budgets.MaxMembers {
		return Failure("eval_limit_exceeded")
	}
	if len(r.Recipes) != len(r.Manifest.Members) {
		return Failure("eval_invalid")
	}
	members := map[string]PublicMember{}
	pairs := map[string][]PublicMember{}
	variants := map[string]bool{}
	for _, v := range r.Variants {
		variants[v.ID] = true
	}
	for _, m := range r.Manifest.Members {
		if !variants[m.VariantID] {
			return Failure("eval_invalid")
		}
		members[m.MemberID] = m
		pair, _ := PairID(r.Manifest.ExperimentID, m.SuiteID, m.CaseID, m.Sample)
		pairs[pair] = append(pairs[pair], m)
	}

	bindingByVariant := map[string]string{}
	hashByCase := map[string]string{}
	samplesByCase := map[string]map[int]bool{}
	maxSample := 0
	for _, m := range r.Manifest.Members {
		if prior, ok := bindingByVariant[m.VariantID]; ok && prior != m.BindingSHA256 {
			return Failure("eval_pin_mismatch")
		}
		bindingByVariant[m.VariantID] = m.BindingSHA256
		key := m.SuiteID + "/" + m.CaseID
		if prior, ok := hashByCase[key]; ok && prior != m.CaseSHA256 {
			return Failure("eval_pin_mismatch")
		}
		hashByCase[key] = m.CaseSHA256
		if samplesByCase[key] == nil {
			samplesByCase[key] = map[int]bool{}
		}
		samplesByCase[key][m.Sample] = true
		if m.Sample > maxSample {
			maxSample = m.Sample
		}
	}
	for _, samples := range samplesByCase {
		if len(samples) != maxSample {
			return Failure("eval_member_conflict")
		}
	}
	for _, pair := range pairs {
		if len(pair) != 2 || pair[0].CaseSHA256 != pair[1].CaseSHA256 {
			return Failure("eval_pin_mismatch")
		}
	}
	seen := map[string]bool{}
	caseRecipes := map[string]Case{}
	for _, recipe := range r.Recipes {
		m, ok := members[recipe.MemberID]
		if !ok || seen[recipe.MemberID] || recipe.Case.ID != m.CaseID {
			return Failure("eval_member_conflict")
		}
		seen[recipe.MemberID] = true
		if err := validateCase(recipe.Case); err != nil {
			return err
		}
		key := m.SuiteID + "/" + m.CaseID
		if prior, ok := caseRecipes[key]; ok && !reflect.DeepEqual(prior, recipe.Case) {
			return Failure("eval_pin_mismatch")
		}
		caseRecipes[key] = recipe.Case
	}
	return nil
}
