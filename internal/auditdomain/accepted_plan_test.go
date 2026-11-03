package auditdomain

import (
	"encoding/json"
	"testing"

	artifactpolicy "github.com/grauwolf32/contractor/internal/artifacts/policy"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func acceptedPlanFixture(t *testing.T) AcceptedPlan {
	t.Helper()
	ordinal := 0
	revision := "r1"
	original := contracts.ExactArtifact{Ref: contracts.ArtifactRef{Namespace: "findings", Name: "proposal", Revision: &revision}, Digest: DigestBytes([]byte("proposal")), MediaType: JSONMediaType, SizeBytes: 0}
	retained := original
	retained.Ref.Namespace = artifactpolicy.FindingMaterialsNamespace
	plan := CheckPlan{Schema: contracts.FindingCheckPlanSchema, Method: "verify", Phase: "verify", Objective: "Кириллица <>& \"\n\u2028\u2029"}
	data, err := EncodeCheckPlan(plan)
	if err != nil {
		t.Fatal(err)
	}
	digest := DigestBytes([]byte("pin"))
	return AcceptedPlan{Schema: AcceptedPlanSchema, OwnerID: "owner", ProjectID: "project", AuditID: "audit", ItemKey: FindingCheckItemKey("receipt", 0), BaselineDigest: digest, ProfileDigest: digest, Source: AcceptedPlanSource{Kind: "finding-receipt", ReceiptID: "receipt", ProposedCheckOrdinal: &ordinal, RunID: "producer", Proposal: original, RetainedProposal: retained}, Plan: plan, PlanDigest: DigestBytes(data), Route: AcceptedPlanRoute{MappingID: "named-route", MappingDigest: digest, WorkflowRole: "renamed-check", BindingDigest: digest, WorkflowDigest: digest, SkillsDigest: digest, RequiredCapabilities: []contracts.CheckCapabilityRequirement{}}, Authorization: AcceptedPlanAuthorization{ScopeDigest: digest, ActiveChecks: contracts.ActiveChecksProhibited, ApprovalRequirement: ApprovalNone, LimitsDigest: digest}}
}

func TestAcceptedPlanExactBindingAndFragmentFactory(t *testing.T) {
	plan := acceptedPlanFixture(t)
	data, err := EncodeAcceptedPlan(plan, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = DecodeAcceptedPlan(data, nil); err != nil {
		t.Fatal(err)
	}
	var value map[string]json.RawMessage
	if err = json.Unmarshal(data, &value); err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{"proposed_check_ordinal", "sizeBytes"} {
		var source map[string]json.RawMessage
		_ = json.Unmarshal(value["source"], &source)
		if field == "sizeBytes" {
			var exact map[string]json.RawMessage
			_ = json.Unmarshal(source["proposal"], &exact)
			delete(exact, field)
			source["proposal"], _ = json.Marshal(exact)
		} else {
			delete(source, field)
		}
		body := make(map[string]json.RawMessage, len(value))
		for k, v := range value {
			body[k] = v
		}
		body["source"], _ = json.Marshal(source)
		bad, _ := json.Marshal(body)
		if _, err = DecodeAcceptedPlan(bad, nil); err == nil {
			t.Fatalf("missing %s accepted", field)
		}
	}
	source := FindingCheckSourceDocument{Schema: FindingCheckSourceSchema, Source: plan.Source, Check: ProposedCheck{Objective: plan.Plan.Objective, Method: plan.Plan.Method}, Limitations: []string{"bounded-source"}}
	encoded, err := EncodeFindingCheckSourceDocument(source)
	if err != nil {
		t.Fatal(err)
	}
	rev := "source-r1"
	common := InventoryOptions{Round: 2, ProfileMode: "custom-checklist", WorkflowRole: plan.Route.WorkflowRole, SourceInputName: "proposal_inventory", SourceRef: contracts.ArtifactRef{Namespace: ArtifactNamespace("audit"), Name: "selected-source", Revision: &rev}, ApprovalRequirement: ApprovalNone}
	inventory, err := BuildFindingCheckInventory(encoded, FindingCheckInventoryOptions{Common: common, Check: FindingCheckOptions{WorkflowRole: plan.Route.WorkflowRole, ApprovalRequirement: ApprovalNone, AcceptedPlan: &plan}})
	if err != nil {
		t.Fatal(err)
	}
	if inventory.SourceContentDigest != DigestBytes(encoded) || inventory.Tasks[0].Document.AcceptedPlan == nil || inventory.Tasks[0].Document.Finding.ProposedCheckOrdinal != 0 {
		t.Fatal("exact fragment source drifted")
	}
	inventory.Tasks[0].Document.AcceptedPlan = nil
	if err = ValidateInventory(inventory); err == nil {
		t.Fatal("task/package plan mismatch accepted")
	}
}

func TestItemAuthorityMembersRejectExplicitNull(t *testing.T) {
	inventory, err := BuildChecklistInventory([]byte(`{"schema":"contractor.audit.checklist.v1","items":[{"key":"item-1","version":"1","statement":"Check","applicability":"always","allowed_methods":["static"],"required_evidence":[],"review_policy":"automatic"}]}`), JSONMediaType, testInventoryOptions("check"))
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"accepted_plan", "dependencies"} {
		encoded, _ := EncodeItemTask(inventory.Tasks[0].Document)
		var task map[string]any
		_ = json.Unmarshal(encoded, &task)
		task[name] = nil
		data, _ := json.Marshal(task)
		if _, err = DecodeItemTask(data); err == nil {
			t.Fatalf("task accepted null %s", name)
		}
		manifest, _ := EncodeExecutionManifest(inventory.ExecutionManifest)
		var execution map[string]any
		_ = json.Unmarshal(manifest, &execution)
		execution["items"].([]any)[0].(map[string]any)[name] = nil
		data, _ = json.Marshal(execution)
		if _, err = DecodeExecutionManifest(data); err == nil {
			t.Fatalf("execution accepted null %s", name)
		}
	}
}
