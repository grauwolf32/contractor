package contracts_test

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestSharedMaterialSetDescriptorsAndCumulativeLimits(t *testing.T) {
	var fixture struct {
		Limits struct{ Materials, DescriptorBytes int }
		Base   map[string]any
		Cases  []struct {
			Name   string
			Count  int
			Valid  bool
			Change map[string]any
		}
	}
	if e := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "material-set-cases.json"), &fixture); e != nil {
		t.Fatal(e)
	}
	if fixture.Limits.Materials != contracts.MaxSelectedMaterials || fixture.Limits.DescriptorBytes != contracts.MaxMaterialDescriptorBytes {
		t.Fatal("shared limits drift")
	}
	for _, c := range fixture.Cases {
		t.Run(c.Name, func(t *testing.T) {
			members := []any{}
			for i := range c.Count {
				m := map[string]any{}
				for k, v := range fixture.Base {
					m[k] = v
				}
				ref := map[string]any{"namespace": "reports", "name": fmt.Sprint("report-", i), "revision": "rev-1"}
				if c.Name == "duplicates" {
					ref["name"] = "same"
				}
				if c.Name == "missing-revision" {
					delete(ref, "revision")
				}
				m["ref"] = ref
				for k, v := range c.Change {
					m[k] = v
				}
				members = append(members, m)
			}
			data := map[string]any{"schema": contracts.MaterialSetSchema, "materials": members}
			if c.Name == "null-members" {
				data["materials"] = nil
			}
			raw, _ := json.Marshal(data)
			if c.Name == "raw-byte-overflow" {
				raw = append(raw, []byte(strings.Repeat(" ", contracts.MaxMaterialDescriptorBytes))...)
			}
			got, e := contracts.DecodeMaterialSet(raw)
			if (e == nil) != c.Valid {
				t.Fatalf("accepted=%v want=%v: %v", e == nil, c.Valid, e)
			}
			if c.Valid {
				budget := &contracts.MaterialBudget{}
				for _, m := range got.Materials {
					if e := budget.Add(m); e != nil {
						t.Fatal(e)
					}
					if e := budget.Add(m); e != nil {
						t.Fatal("duplicate charged twice", e)
					}
				}
				if len(got.Materials) == 32 {
					extra := got.Materials[0]
					extra.Ref.Name = "distinct-singleton"
					if e := budget.Add(extra); e == nil {
						t.Fatal("singleton escaped cumulative budget")
					}
				}
			}
		})
	}
}
