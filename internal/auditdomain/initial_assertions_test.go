package auditdomain

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
)

func TestChecklistAssertionsPreserveSourceContractAndRejectMalformedSlices(t *testing.T) {
	source := `{"schema":"contractor.audit.assertions.v1","items":[{"key":"one","version":"1","statement":"Reader A may access B's record","scope":{"actor":"A"},"grounds":"An explicit suspicion","applicability":"selected routes","allowed_methods":["static"],"required_evidence":["Exact inspected source"]}]}`
	plan, err := ChecklistAssertions([]byte(source), "application/json", "check")
	if err != nil || len(plan.Hypotheses) != 1 || len(plan.Work) != 1 {
		t.Fatalf("plan=%+v error=%v", plan, err)
	}
	seed := plan.Hypotheses["one"]
	if seed.Scope["source_version"] != "1" || seed.Scope["applicability"] != "selected routes" || seed.Scope["required_evidence"] != `["Exact inspected source"]` || plan.Work[0].Hypothesis != "one" {
		t.Fatalf("lost source contract: %+v", plan)
	}
	yaml := `schema: contractor.audit.assertions.v1
items:
- key: one
  version: "1"
  statement: Reader A may access B's record
  scope: {actor: A}
  grounds: An explicit suspicion
  applicability: selected routes
  allowed_methods: [static]
  required_evidence: [Exact inspected source]
`
	fromYAML, err := ChecklistAssertions([]byte(yaml), "application/yaml", "check")
	if err != nil {
		t.Fatal(err)
	}
	a, _ := json.Marshal(plan)
	b, _ := json.Marshal(fromYAML)
	if string(a) != string(b) {
		t.Fatal("JSON/YAML source rule drift")
	}
	for _, invalid := range []string{strings.Replace(source, `"version":"1"`, `"version":null`, 1), strings.Replace(source, `"key":"one"`, `"key":"one","key":"two"`, 1), strings.Replace(source, `"grounds":"An explicit suspicion"`, `"grounds":" "`, 1), strings.Replace(source, `"allowed_methods":["static"]`, `"allowed_methods":[null]`, 1), strings.Replace(source, `"schema":`, `"Schema":`, 1), strings.Replace(source, `"statement":"Reader A may access B's record"`, `"statement":"`+strings.Repeat("a", 32768)+`"`, 1)} {
		if got, err := ChecklistAssertions([]byte(invalid), "application/json", "check"); err == nil || len(got.Work) != 0 {
			t.Fatal("malformed source returned authoritative work")
		}
	}
}

func TestStandardAssertionsRetainUnmappedRequirementsAndSufficiency(t *testing.T) {
	contract := auditstandards.EvidenceContract{ID: "source", Version: "1", Assessments: []string{"inconclusive", "violated"}, EvidenceKinds: []string{"artifact"}, MinimumEvidence: 1, MaximumEvidence: 8, HumanReview: "never", RationaleRequired: true}
	pkg := auditstandards.Package{Document: auditstandards.Document{Standard: auditstandards.Metadata{Scheme: "example", Version: "1", Title: "Example"}, EvidenceContracts: []auditstandards.EvidenceContract{contract}, Entries: []auditstandards.Entry{{ID: "EX-1", Statement: "A scoped property", Title: "Source property", Applicability: auditstandards.Applicability{Mode: "always"}, AllowedMethods: []string{"source-analysis"}, EvidenceContract: contract.Reference()}, {ID: "EX-2", Statement: "An operational requirement", Title: "Operational property", Applicability: auditstandards.Applicability{Mode: "manual"}, AllowedMethods: []string{"manual"}, EvidenceContract: contract.Reference()}}, Mappings: []auditstandards.Mapping{{Key: "source-check", EntryIDs: []string{"EX-1"}, WorkflowRole: "check", Method: "source-analysis", Objective: "Inspect the exact source paths"}}}}
	plan, err := StandardAssertions(pkg, auditcontract.SeedSource{Workflow: "check", Standard: &auditcontract.SeedStandard{Scheme: "example", Version: "1"}})
	if err != nil || len(plan.Hypotheses) != 2 || len(plan.Work) != 1 {
		t.Fatalf("plan=%+v error=%v", plan, err)
	}
	first, second := plan.Hypotheses["standard-001"], plan.Hypotheses["standard-002"]
	if first.Scope["entry_id"] != "EX-1" || first.Scope["standard_version"] != "1" || !strings.Contains(first.Scope["evidence_contract"], `"minimumEvidence":1`) || !strings.Contains(first.Scope["standard_mapping"], "Inspect the exact source paths") || second.Scope["coverage_gap"] == "" {
		t.Fatal("standard membership or sufficiency was dropped")
	}
	last := pkg.Document.Entries[0]
	last.ID = "EX-3"
	pkg.Document.Entries = append(pkg.Document.Entries, last)
	pkg.Document.Mappings[0].EntryIDs = append(pkg.Document.Mappings[0].EntryIDs, "EX-3")
	expanded, err := StandardAssertions(pkg, auditcontract.SeedSource{Workflow: "check", Standard: &auditcontract.SeedStandard{Scheme: "example", Version: "1"}})
	if err != nil || len(expanded.Hypotheses) != 3 || expanded.Hypotheses["standard-003"].Scope["entry_id"] != "EX-3" {
		t.Fatal("an unmapped middle requirement changed source identity", err)
	}
	if _, err := StandardAssertions(pkg, auditcontract.SeedSource{Workflow: "check", EntryIDs: []string{"EX-absent"}}); err == nil {
		t.Fatal("unknown selected requirement accepted")
	}
}
