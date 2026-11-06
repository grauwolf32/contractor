package public

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/grauwolf32/contractor/internal/evaldomain"
)

// experimentFields returns the members of GET /v1/eval-experiments/{id}.
func (h *evalAPIHarness) experimentFields(t *testing.T, id string) map[string]json.RawMessage {
	t.Helper()
	return apiDecode[map[string]json.RawMessage](t, h.request(t, "GET", "/v1/eval-experiments/"+id, nil, "", "", 200))
}

// storedPlanView reads experiment id's frozen plan directly: its identity and
// setup, and the readiness its document yields under the setup comparison.
func (h *evalAPIHarness) storedPlanView(t *testing.T, id string) (identity string, setup []byte, readiness *evaldomain.Readiness) {
	t.Helper()
	var kind, digest string
	var document []byte
	if err := h.pool.QueryRow(t.Context(), `SELECT plan_kind, plan_document, plan_document_sha256, plan_setup, plan_sha256 FROM eval_experiments WHERE experiment_id=$1`, id).
		Scan(&kind, &document, &digest, &setup, &identity); err != nil {
		t.Fatal(err)
	}
	if kind != "playground.plan/v1" {
		return identity, setup, nil
	}
	var parsed struct {
		Comparison evaldomain.Comparison `json:"comparison"`
	}
	if err := json.Unmarshal(setup, &parsed); err != nil {
		t.Fatal(err)
	}
	derived, err := evaldomain.NativeReadiness(evaldomain.Restore(kind, document, digest), parsed.Comparison)
	if err != nil {
		t.Fatal(err)
	}
	return identity, setup, &derived
}

func requireFields(t *testing.T, state string, fields map[string]json.RawMessage, present, absent []string) {
	t.Helper()
	for _, name := range present {
		if value, ok := fields[name]; !ok || string(value) == "null" {
			t.Fatalf("%s experiment lacks %s: %v", state, name, fields)
		}
	}
	for _, name := range absent {
		if value, ok := fields[name]; ok && string(value) != "null" {
			t.Fatalf("%s experiment has %s %s", state, name, value)
		}
	}
}

func requireJSONEqual(t *testing.T, name string, got, want []byte) {
	t.Helper()
	var a, b any
	if err := json.Unmarshal(got, &a); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(want, &b); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(a, b) {
		t.Fatalf("%s:\n got %s\nwant %s", name, got, want)
	}
}

// The public Experiment keeps its fields in every plan state while Get reads
// the stored readiness instead of the plan document: a draft and a preparing
// experiment show their draft, a native plan its identity, setup and the
// readiness its document yields, an external registration no readiness, and
// a deleted unprepared draft its authoring setup. A stored readiness that
// fails strict decoding or the closed response schema is never returned.
func TestSharedEvalPostgresExperimentViewFieldsAcrossPlanStates(t *testing.T) {
	h := newEvalAPIHarness(t)
	draft, _ := h.dataset(t, "workflow")
	create := func(name, key string) string {
		t.Helper()
		return apiDecode[struct {
			ID string `json:"experimentId"`
		}](t, h.request(t, "POST", "/v1/projects/evaluation/eval-experiments", evaldomain.CreateExperiment{Name: name, ControlMode: "server", Draft: &draft}, key, "", 201)).ID
	}

	native := create("Native readiness", "native")
	requireFields(t, "draft", h.experimentFields(t, native), []string{"draft"}, []string{"setup", "readiness", "planSha256"})
	h.command(t, native, "prepare")
	preparing := h.experimentFields(t, native)
	if string(preparing["state"]) != `"preparing"` {
		t.Fatalf("prepare accepted in state %s", preparing["state"])
	}
	requireFields(t, "preparing", preparing, []string{"draft"}, []string{"setup", "readiness", "planSha256"})
	h.tick(t)
	ready := h.experimentFields(t, native)
	requireFields(t, "ready", ready, []string{"setup", "readiness", "planSha256"}, []string{"draft"})
	identity, setup, readiness := h.storedPlanView(t, native)
	want, err := json.Marshal(readiness)
	if err != nil {
		t.Fatal(err)
	}
	requireJSONEqual(t, "readiness", ready["readiness"], want)
	requireJSONEqual(t, "setup", ready["setup"], setup)
	requireJSONEqual(t, "planSha256", ready["planSha256"], []byte(`"`+identity+`"`))

	deleted := create("Deleted draft", "deleted")
	h.request(t, "DELETE", "/v1/eval-experiments/"+deleted, map[string]any{}, "delete", `"1"`, 202)
	deleting := h.experimentFields(t, deleted)
	requireFields(t, "deleted draft", deleting, []string{"setup", "deletionRequestedAt"}, []string{"draft", "readiness", "planSha256"})
	authoring, err := json.Marshal(draft.Setup())
	if err != nil {
		t.Fatal(err)
	}
	requireJSONEqual(t, "deleted draft setup", deleting["setup"], authoring)

	producer := newEvalAPIHarness(t)
	external, _, _ := producer.registerExternal(t, "workflow")
	registered := producer.experimentFields(t, external.ID)
	requireFields(t, "external", registered, []string{"setup", "planSha256"}, []string{"draft", "readiness"})
	identity, setup, readiness = producer.storedPlanView(t, external.ID)
	if readiness != nil {
		t.Fatal("an external registration derived native readiness")
	}
	requireJSONEqual(t, "external setup", registered["setup"], setup)
	requireJSONEqual(t, "external planSha256", registered["planSha256"], []byte(`"`+identity+`"`))

	corrupt := func(value string, status int) {
		t.Helper()
		for _, statement := range []string{
			`ALTER TABLE eval_experiments DISABLE TRIGGER eval_plans_immutable; ALTER TABLE eval_experiments DISABLE TRIGGER eval_experiments_protect`,
			`UPDATE eval_experiments SET plan_readiness=convert_to('` + value + `','UTF8') WHERE experiment_id='` + native + `'`,
			`ALTER TABLE eval_experiments ENABLE TRIGGER eval_plans_immutable; ALTER TABLE eval_experiments ENABLE TRIGGER eval_experiments_protect`,
		} {
			if _, err := h.pool.Exec(t.Context(), statement); err != nil {
				t.Fatal(err)
			}
		}
		h.request(t, "GET", "/v1/eval-experiments/"+native, nil, "", "", status)
	}
	corrupt(`{"pins":[],"arms":[],"probe":true}`, 500)
	corrupt(`{"pins":[{"dimension":"source","requiredEqual":true,"baselineOrigin":"observed","candidateOrigin":"observed","status":"maybe"}],"arms":[]}`, 422)
}
