package auditcontract_test

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

type fixtureCase struct {
	Name     string          `json:"name"`
	Document string          `json:"document"`
	Context  string          `json:"context"`
	Valid    bool            `json:"valid"`
	Expected json.RawMessage `json:"expected"`
}

func TestSharedAuditCommonDocuments(t *testing.T) {
	for _, file := range []string{"programs.json", "results.json", "controls.json", "proposals.json", "tasks.json"} {
		var fixtures struct {
			Cases []fixtureCase `json:"cases"`
		}
		if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "audit-common", file), &fixtures); err != nil {
			t.Fatal(err)
		}
		if len(fixtures.Cases) == 0 {
			t.Fatal("empty fixture selection")
		}
		for _, c := range fixtures.Cases {
			t.Run(c.Name, func(t *testing.T) {
				actual, err := auditcontract.Decode([]byte(c.Document), []byte(c.Context))
				if (err == nil) != c.Valid {
					t.Fatalf("valid=%v error=%v", c.Valid, err)
				}
				if !c.Valid && len(actual) != 0 {
					t.Fatal("partial result on rejection")
				}
				if err != nil && strings.Contains(err.Error(), "canary") {
					t.Fatal("diagnostic exposed input")
				}
				if len(c.Expected) > 0 && err == nil {
					contractstest.AssertSemanticJSONEqual(t, actual, c.Expected)
				}
			})
		}
	}
}

func TestSharedAuditCommonByteAndMutationBounds(t *testing.T) {
	var fixture struct {
		Hypothesis      map[string]any `json:"hypothesis"`
		MutationRecords int            `json:"mutation_records"`
		MutationBytes   int            `json:"mutation_bytes"`
		HypothesisBytes int            `json:"hypothesis_bytes"`
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "audit-common", "boundaries.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	encode := func(v any) []byte {
		b, e := json.Marshal(v)
		if e != nil {
			t.Fatal(e)
		}
		return b
	}
	mutation := map[string]any{"schema": "contractor.audit.mutation.v1"}
	records := make([]any, fixture.MutationRecords)
	for i := range records {
		item := map[string]any{}
		for k, v := range fixture.Hypothesis {
			item[k] = v
		}
		item["id"] = fmt.Sprintf("hypothesis-%d", i)
		records[i] = item
	}
	mutation["records"] = records
	data := encode(mutation)
	if _, err := auditcontract.Decode(data, nil); err != nil {
		t.Fatal("at record limit", err)
	}
	mutation["records"] = append(records, fixture.Hypothesis)
	if _, err := auditcontract.Decode(encode(mutation), nil); err == nil {
		t.Fatal("over record limit accepted")
	}
	exact := append(data, bytes.Repeat([]byte(" "), fixture.MutationBytes-len(data))...)
	if _, err := auditcontract.Decode(exact, nil); err != nil {
		t.Fatal("at document byte limit", err)
	}
	if _, err := auditcontract.Decode(append(exact, ' '), nil); err == nil {
		t.Fatal("over document byte limit accepted")
	}
	h := encode(fixture.Hypothesis)
	exact = append(h, bytes.Repeat([]byte(" "), fixture.HypothesisBytes-len(h))...)
	if _, err := auditcontract.Decode(exact, nil); err != nil {
		t.Fatal("at hypothesis byte limit", err)
	}
	if _, err := auditcontract.Decode(append(exact, ' '), nil); err == nil {
		t.Fatal("over hypothesis byte limit accepted")
	}
	for _, bad := range [][]byte{append([]byte(`{"schema":"`), 0xff), []byte(strings.Repeat("[", 100) + strings.Repeat("]", 100))} {
		if _, err := auditcontract.Decode(bad, nil); err == nil {
			t.Fatal("malformed input accepted")
		}
	}
}

func TestSharedAuditCommonTextUsesTheWireWhitespaceRule(t *testing.T) {
	var wire struct {
		Blank []string `json:"blank"`
		Text  []string `json:"text"`
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "blank-text-cases.json"), &wire); err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Cases []fixtureCase `json:"cases"`
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "audit-common", "results.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	if len(wire.Blank) == 0 || len(wire.Text) == 0 || len(fixture.Cases) == 0 {
		t.Fatal("empty whitespace selection")
	}
	var document map[string]any
	if err := json.Unmarshal([]byte(fixture.Cases[0].Document), &document); err != nil {
		t.Fatal(err)
	}
	for _, group := range []struct {
		values []string
		valid  bool
	}{{wire.Blank, false}, {wire.Text, true}} {
		for _, text := range group.values {
			document["items"].([]any)[0].(map[string]any)["rationale"] = text
			data, err := json.Marshal(document)
			if err != nil {
				t.Fatal(err)
			}
			_, err = auditcontract.Decode(data, []byte(fixture.Cases[0].Context))
			// NUL is nonblank in the shared classification but forbidden in
			// every persisted common document independently of whitespace.
			want := group.valid && !strings.ContainsRune(text, '\x00')
			if (err == nil) != want {
				t.Fatalf("wire whitespace classification drift: valid=%v error=%v", want, err)
			}
		}
	}
}
