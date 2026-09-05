package auditdomain

import (
	"testing"
)

func TestChecklistInventoryIsAtomicDeterministicAndComplete(t *testing.T) {
	source := []byte(`
schema: contractor.audit.checklist.v1
items:
  - key: check-b
    version: "1"
    statement: Verify B.
    applicability: always
    allowed_methods: [review, static]
    required_evidence: [trace, source]
    review_policy: automatic
  - key: check-a
    version: "2"
    statement: Verify A.
    applicability: when-api
    allowed_methods: [static]
    required_evidence: []
    review_policy: manual
`)
	options := testInventoryOptions("check")
	options.Scope = map[string]string{"target": "service"}
	inventory, err := BuildChecklistInventory(source, "application/yaml", options)
	if err != nil {
		t.Fatal(err)
	}
	if len(inventory.Worklist.Items) != 2 || len(inventory.Tasks) != 2 || len(inventory.Coverage.Rows) != 2 || len(inventory.ExecutionManifest.Items) != 2 {
		t.Fatalf("incomplete inventory: %+v", inventory)
	}
	if inventory.Worklist.Items[0].ItemKey != "check-a" || inventory.Worklist.Items[0].Ordinal != 0 || inventory.Worklist.Items[0].ApprovalRequirement != ApprovalHumanReview || inventory.Worklist.Items[1].ItemKey != "check-b" {
		t.Fatalf("unexpected deterministic order/policy: %+v", inventory.Worklist.Items)
	}
	for index, row := range inventory.Coverage.Rows {
		if row.Status != "not-tested" || row.ItemKey != inventory.Worklist.Items[index].ItemKey {
			t.Fatalf("coverage row %d = %+v", index, row)
		}
		validated, err := ValidatePackage(inventory.Tasks[index].Package)
		if err != nil || validated.Digest != inventory.Tasks[index].PackageDigest {
			t.Fatalf("task package %d = %+v, %v", index, validated, err)
		}
		taskMember, ok := validated.MemberByID("task-document")
		if !ok {
			t.Fatal("task document is absent")
		}
		decoded, err := DecodeItemTask(taskMember.Data())
		if err != nil || decoded.ItemKey != row.ItemKey || decoded.SourceContentDigest != inventory.SourceContentDigest {
			t.Fatalf("task document = %+v, %v", decoded, err)
		}
	}
	*options.SourceRef.Revision = "caller-mutated"
	if *inventory.Tasks[0].Document.SourceRef.Revision != "source-revision-1" ||
		*inventory.ExecutionManifest.Items[0].Inputs[0].Ref.Revision != "source-revision-1" || ValidateInventory(inventory) != nil {
		t.Fatal("inventory retained caller-owned ArtifactRef revision storage")
	}
	*options.SourceRef.Revision = "source-revision-1"

	again, err := BuildChecklistInventory(source, "application/yaml", options)
	if err != nil || again.CanonicalInventoryDigest != inventory.CanonicalInventoryDigest || again.Tasks[0].PackageDigest != inventory.Tasks[0].PackageDigest {
		t.Fatalf("inventory is not repeatable: %+v, %v", again, err)
	}
	changedRevision := "source-revision-2"
	changedOptions := options
	changedOptions.SourceRef.Revision = &changedRevision
	changed, err := BuildChecklistInventory(source, "application/yaml", changedOptions)
	if err != nil {
		t.Fatal(err)
	}
	if changed.CanonicalInventoryDigest != inventory.CanonicalInventoryDigest || changed.Tasks[0].Item.TaskPackageID == inventory.Tasks[0].Item.TaskPackageID || changed.Tasks[0].PackageDigest == inventory.Tasks[0].PackageDigest {
		t.Fatal("exact source ref did not affect task provenance independently of inventory identity")
	}
}

func TestChecklistRejectsDuplicateOrMalformedItemWithoutPartialResult(t *testing.T) {
	options := testInventoryOptions("check")
	tests := [][]byte{
		[]byte("schema: contractor.audit.checklist.v1\nschema: contractor.audit.checklist.v1\nitems: []\n"),
		[]byte("schema: contractor.audit.checklist.v1\nitems:\n- key: same\n  version: '1'\n  statement: One\n  applicability: always\n  allowed_methods: []\n  required_evidence: []\n  review_policy: automatic\n- key: same\n  version: '1'\n  statement: Two\n  applicability: always\n  allowed_methods: []\n  required_evidence: []\n  review_policy: automatic\n"),
		[]byte("schema: contractor.audit.checklist.v1\nitems:\n- key: x\n  version: '1'\n  statement: One\n  applicability: always\n  allowed_methods: []\n  required_evidence: []\n  review_policy: unknown\n"),
		[]byte("schema: contractor.audit.checklist.v1\nitems:\n- key: x\n  version: '1'\n  statement: One\n  applicability: always\n  allowed_methods: [static, static]\n  required_evidence: []\n  review_policy: automatic\n"),
	}
	for index, source := range tests {
		inventory, err := BuildChecklistInventory(source, "application/yaml", options)
		if err == nil || len(inventory.Tasks) != 0 || len(inventory.Worklist.Items) != 0 {
			t.Fatalf("case %d accepted partial inventory: %+v, %v", index, inventory, err)
		}
	}
}
