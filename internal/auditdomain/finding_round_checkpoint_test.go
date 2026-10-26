package auditdomain

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

func TestFindingRoundCheckpointStaysConstantAcross4096TypedMembers(t *testing.T) {
	shape := testFindingRoundShape()
	shape.Paged = true
	budget, err := NewFindingRoundBudget(MaximumItems, shape)
	if err != nil {
		t.Fatal(err)
	}
	plan := acceptedPlanFixture(t)
	source := FindingCheckSourceDocument{Schema: FindingCheckSourceSchema, Source: plan.Source, Check: ProposedCheck{Method: plan.Plan.Method, Objective: plan.Plan.Objective}, Limitations: []string{}}
	for i := range MaximumItems {
		source.Source.ReceiptID = fmt.Sprintf("receipt-%04d", i)
		plan.Source = source.Source
		plan.ItemKey = FindingCheckItemKey(source.Source.ReceiptID, 0)
		// Distinct routes must be measured independently, including long role names.
		if i%2 == 0 {
			plan.Route.WorkflowRole = "role-a"
		} else {
			plan.Route.WorkflowRole = "role-" + strings.Repeat("x", 80)
		}
		encoded, err := EncodeFindingCheckSourceDocument(source)
		if err != nil {
			t.Fatal(err)
		}
		options := FindingCheckInventoryOptions{Common: shape.Inventory, Check: FindingCheckOptions{WorkflowRole: plan.Route.WorkflowRole, ApprovalRequirement: ApprovalNone, AcceptedPlan: &plan, ExecutionInputs: []ExactInput{}}}
		result, err := budget.AdmitCheck(encoded, options)
		if err != nil || !result.Admitted {
			t.Fatalf("candidate%d rejected: %+v,%v", i, result, err)
		}
		if i == 127 || i == 2047 || i == 4095 {
			state, err := budget.Checkpoint()
			if err != nil {
				t.Fatal(err)
			}
			data, err := EncodeFindingRoundBudgetState(state)
			if err != nil || len(data) > MaximumFindingRoundBudgetStateBytes {
				t.Fatalf("checkpoint %d bytes: %v", len(data), err)
			}
			decoded, err := DecodeFindingRoundBudgetState(data)
			if err != nil {
				t.Fatal(err)
			}
			budget, err = ResumeFindingRoundBudget(MaximumItems, shape, decoded)
			if err != nil {
				t.Fatal(err)
			}
		}
	}
	if budget.Checks() != MaximumItems || !budget.Full() {
		t.Fatalf("count=%d full=%t", budget.Checks(), budget.Full())
	}
	state, err := budget.Checkpoint()
	if err != nil {
		t.Fatal(err)
	}
	data, _ := EncodeFindingRoundBudgetState(state)
	var raw map[string]json.RawMessage
	_ = json.Unmarshal(data, &raw)
	delete(raw, "full")
	missing, _ := json.Marshal(raw)
	if _, err = DecodeFindingRoundBudgetState(missing); err == nil {
		t.Fatal("missing false/full accepted")
	}
	shape.Inventory.Round++
	if _, err = ResumeFindingRoundBudget(MaximumItems, shape, state); err == nil {
		t.Fatal("changed generation shape accepted")
	}
}

func TestFindingFragmentLimitationsMatchAdmissionBounds(t *testing.T) {
	for _, values := range [][]string{nil, {"same", "same"}, {strings.Repeat("x", 513)}} {
		if err := ValidateFindingLimitations(values); err == nil {
			t.Fatal("invalid limitations accepted")
		}
	}
	if err := ValidateFindingLimitations([]string{strings.Repeat("x", 512)}); err != nil {
		t.Fatal(err)
	}
}
