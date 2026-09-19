package evalservice

import (
	"bytes"
	"encoding/json"
	"os"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/evaldomain"
)

func readFixture(t *testing.T, name string, out any) {
	t.Helper()
	b, err := os.ReadFile("../../api/testdata/evals/valid/" + name + ".json")
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(b, out); err != nil {
		t.Fatal(err)
	}
}
func planInputs(t *testing.T) (evaldomain.Draft, evaldomain.DatasetInput, map[string]Preflight) {
	t.Helper()
	var d evaldomain.Draft
	var data evaldomain.DatasetInput
	readFixture(t, "draft", &d)
	readFixture(t, "dataset", &data)
	pre := map[string]Preflight{}
	for _, v := range d.Variants {
		cases := map[string]Eligibility{}
		for _, id := range d.CaseIDs {
			cases[id] = Eligibility{State: "eligible"}
		}
		pre[v.ID] = Preflight{Pins: map[string]Pin{}, Capabilities: []string{}, Cases: cases, Snapshot: json.RawMessage(`{}`)}
	}
	return d, data, pre
}
func TestPlanBuildsPrivatePortableClosureAndStableMatrix(t *testing.T) {
	draft, data, pre := planInputs(t)
	original, _ := json.Marshal(draft)
	now := time.Date(2026, 9, 19, 10, 0, 0, 0, time.UTC)
	b, err := BuildPlan("native-1", now, draft, data, pre)
	if err != nil {
		t.Fatal(err)
	}
	after, _ := json.Marshal(draft)
	if !bytes.Equal(original, after) {
		t.Fatal("preparation mutated its draft")
	}
	if len(b.Cases) != 8 {
		t.Fatal(len(b.Cases))
	}
	manifest, err := evaldomain.PublicPlanProjection(b.Plan)
	if err != nil {
		t.Fatal(err)
	}
	safe, _ := json.Marshal(manifest)
	if bytes.Contains(safe, []byte("PRIVATE_")) {
		t.Fatal("private truth leaked")
	}
	var plan map[string]any
	if err = json.Unmarshal(b.Plan.Bytes(), &plan); err != nil {
		t.Fatal(err)
	}
	resources := map[string]string{}
	for _, r := range b.Resources {
		resources[r.Path] = r.Document.Digest()
	}
	for path, doc := range b.Private {
		resources[path] = doc.Digest()
	}
	for path, ref := range b.Inputs {
		resources[path] = ref.SHA256
	}
	var walk func(any)
	walk = func(v any) {
		switch x := v.(type) {
		case map[string]any:
			if path, ok := x["resource"].(string); ok {
				if resources[path] != x["sha256"] {
					t.Fatalf("unreachable or mismatched resource %s", path)
				}
			}
			for _, child := range x {
				walk(child)
			}
		case []any:
			for _, child := range x {
				walk(child)
			}
		}
	}
	walk(plan)
	for _, r := range b.Resources {
		var v any
		if err = json.Unmarshal(r.Document.Bytes(), &v); err != nil {
			t.Fatal(err)
		}
		walk(v)
	}
	repeat, err := BuildPlan("native-1", now, draft, data, pre)
	if err != nil || !bytes.Equal(repeat.Plan.Bytes(), b.Plan.Bytes()) {
		t.Fatal("non-deterministic plan", err)
	}
	for mid, c := range b.Cases {
		visible, err := evaldomain.ExecutionProjection(c)
		if err != nil {
			t.Fatal(err)
		}
		raw, _ := json.Marshal(visible)
		if bytes.Contains(raw, []byte("PRIVATE_")) {
			t.Fatal("private recipe", mid)
		}
	}
}
func TestPlanPinsAndUnsupportedMembersRemainExplicit(t *testing.T) {
	d, data, pre := planInputs(t)
	reason := "Required capability unavailable."
	pre["b"].Cases[d.CaseIDs[0]] = Eligibility{"unsupported", &reason}
	b, err := BuildPlan("native-1", time.Now(), d, data, pre)
	if err != nil {
		t.Fatal(err)
	}
	m, err := evaldomain.PublicPlanProjection(b.Plan)
	if err != nil {
		t.Fatal(err)
	}
	unsupported := 0
	for _, v := range m.Members {
		if v.Eligibility == "unsupported" {
			unsupported++
		}
	}
	if unsupported != 2 || len(m.Members) != 8 {
		t.Fatal("denominator changed")
	}
	d.Comparison.RequiredEqual = append(d.Comparison.RequiredEqual, "model-revision")
	if _, err = BuildPlan("native-1", time.Now(), d, data, pre); err == nil {
		t.Fatal("unknown required pin accepted")
	}
	pre["a"].Pins["model-revision"] = Pin{Value: stringPointer("claimed"), Origin: "operator_supplied"}
	pre["b"].Pins["model-revision"] = Pin{Value: stringPointer("claimed"), Origin: "operator_supplied"}
	if _, err = BuildPlan("native-1", time.Now(), d, data, pre); err == nil {
		t.Fatal("producer claim treated as observation")
	}
}
func TestPlanSeededOrderIsRetainedAndRepeatable(t *testing.T) {
	d, data, pre := planInputs(t)
	seed := int64(23)
	d.Order = evaldomain.Order{Kind: "seeded_shuffle", Seed: &seed}
	now := time.Now()
	a, err := BuildPlan("native-1", now, d, data, pre)
	if err != nil {
		t.Fatal(err)
	}
	b, err := BuildPlan("native-1", now, d, data, pre)
	if err != nil || !bytes.Equal(a.Plan.Bytes(), b.Plan.Bytes()) {
		t.Fatal(err)
	}
}

func stringPointer(value string) *string { return &value }
