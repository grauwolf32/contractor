package auditcontract_test

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestSharedPreparedWorkDocumentBounds(t *testing.T) {
	var fixture struct {
		Cases  []fixtureCase  `json:"cases"`
		Limits map[string]int `json:"limits"`
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "audit-common", "prepared-work.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Limits["work"] != auditcontract.MaxPreparedWork || fixture.Limits["metadata_bytes"] != auditcontract.MaxPreparedWorkBytes || fixture.Limits["materials"] != auditcontract.MaxPreparedMaterials {
		t.Fatal("shared prepared-work bounds drifted")
	}
	var list auditcontract.PreparedWork
	if err := json.Unmarshal([]byte(fixture.Cases[0].Document), &list); err != nil {
		t.Fatal(err)
	}
	item := list.Work[0]
	for count := auditcontract.MaxPreparedWork; count <= auditcontract.MaxPreparedWork+1; count++ {
		list.Work = nil
		for i := 0; i < count; i++ {
			next := item
			next.Key = fmt.Sprintf("operation-%d", i)
			list.Work = append(list.Work, next)
		}
		data, _ := json.Marshal(list)
		result, err := auditcontract.Decode(data, []byte(fixture.Cases[0].Context))
		if count == auditcontract.MaxPreparedWork && err != nil {
			t.Fatal(err)
		}
		if count > auditcontract.MaxPreparedWork && (err == nil || len(result) != 0) {
			t.Fatal("over-bound list returned partial output")
		}
	}
	list.Work = []auditcontract.PreparedSelection{item}
	list.Gaps = []string{}
	for i := 0; i < 17; i++ {
		list.Gaps = append(list.Gaps, fmt.Sprintf("%02d", i)+strings.Repeat("x", 60000))
	}
	list.Gaps = append(list.Gaps, "tail")
	data, _ := json.Marshal(list)
	remaining := auditcontract.MaxPreparedWorkBytes - len(data)
	if remaining < 0 || remaining > 65000 {
		t.Fatal("boundary fixture no longer fits")
	}
	list.Gaps[len(list.Gaps)-1] += strings.Repeat("x", remaining)
	data, _ = json.Marshal(list)
	canonical, err := auditcontract.Decode(data, nil)
	if err != nil || len(canonical) != auditcontract.MaxPreparedWorkBytes {
		t.Fatalf("canonical ceiling: %d %v", len(canonical), err)
	}
	list.Gaps[len(list.Gaps)-1] += "x"
	data, _ = json.Marshal(list)
	if result, err := auditcontract.Decode(data, nil); err == nil || len(result) != 0 {
		t.Fatal("metadata overflow returned output")
	}
	list.Gaps = []string{}
	data, _ = json.Marshal(list)
	padded := append(data, []byte(strings.Repeat(" ", auditcontract.MaxPreparedWorkBytes-len(data)))...)
	if _, err := auditcontract.Decode(padded, nil); err != nil {
		t.Fatal("raw ceiling", err)
	}
	if _, err := auditcontract.Decode(append(padded, ' '), nil); err == nil {
		t.Fatal("raw metadata overflow accepted")
	}
}
