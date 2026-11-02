package auditdomain

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestSharedAuthoredChecklists(t *testing.T) {
	var cases []struct {
		Name     string          `json:"name"`
		Document json.RawMessage `json:"document"`
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "audit-checklist-authoring.json"), &cases); err != nil {
		t.Fatal(err)
	}
	if len(cases) == 0 {
		t.Fatal("empty authoring fixture gate")
	}
	for _, test := range cases {
		t.Run(test.Name, func(t *testing.T) {
			var doc ChecklistDocument
			if err := json.Unmarshal(test.Document, &doc); err != nil {
				t.Fatal(err)
			}
			inventory, err := BuildChecklistInventory(test.Document, "application/json", testInventoryOptions("check"))
			if err != nil {
				t.Fatal(err)
			}
			if len(inventory.Worklist.Items) != len(doc.Items) {
				t.Fatal("authored entry lost")
			}
			for index, item := range inventory.Worklist.Items {
				if item.ItemKey != doc.Items[index].Key {
					t.Fatal("stable author key lost")
				}
			}
		})
	}
	// The inline composer ceiling still follows the ordinary complete inventory path.
	var doc ChecklistDocument
	if err := json.Unmarshal(cases[0].Document, &doc); err != nil {
		t.Fatal(err)
	}
	first := doc.Items[0]
	doc.Items = nil
	for i := 1; i <= 128; i++ {
		entry := first
		entry.Key = fmt.Sprintf("check-%04d", i)
		doc.Items = append(doc.Items, entry)
	}
	data, err := json.Marshal(doc)
	if err != nil {
		t.Fatal(err)
	}
	inventory, err := BuildChecklistInventory(data, "application/json", testInventoryOptions("check"))
	if err != nil || len(inventory.Tasks) != 128 {
		t.Fatalf("128 authored tasks: %d, %v", len(inventory.Tasks), err)
	}
}
