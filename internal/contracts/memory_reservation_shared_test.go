package contracts

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

// TestSharedWorkerResultReservedBindings decodes a WorkerCompletion whose
// result names each binding of
// api/testdata/v1alpha1/memory-reservation-cases.json: the wire refuses it
// exactly when the binding is resultReserved, as the Runtime's WorkerResult
// model does (runtime/tests/test_shared_memory_reservation.py).
func TestSharedWorkerResultReservedBindings(t *testing.T) {
	t.Parallel()
	var cases struct {
		Description               string   `json:"description"`
		MemoryPrefix              string   `json:"memoryPrefix"`
		PurposeReservedNamespaces []string `json:"purposeReservedNamespaces"`
		Bindings                  []struct {
			Namespace      string `json:"namespace"`
			Name           string `json:"name"`
			MemoryReserved *bool  `json:"memoryReserved"`
			ResultReserved *bool  `json:"resultReserved"`
		} `json:"bindings"`
	}
	decoder := json.NewDecoder(bytes.NewReader(contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "memory-reservation-cases.json")))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&cases); err != nil {
		t.Fatal(err)
	}
	if len(cases.Bindings) == 0 {
		t.Fatal("shared Memory reservation cases are empty")
	}
	for _, binding := range cases.Bindings {
		if binding.ResultReserved == nil {
			t.Fatalf("binding %s/%s needs a resultReserved verdict", binding.Namespace, binding.Name)
		}
		var completion map[string]any
		if err := json.Unmarshal(contractstest.ReadFixture(t, "valid", "worker-completion-success.json"), &completion); err != nil {
			t.Fatal(err)
		}
		completion["result"].(map[string]any)["artifacts"] = map[string]any{
			"report": map[string]any{"namespace": binding.Namespace, "name": binding.Name, "revision": "r1"},
		}
		data, err := json.Marshal(completion)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := DecodePrivateStrict[WorkerCompletion](data); (err != nil) != *binding.ResultReserved {
			t.Errorf("Worker result naming %s/%s refused = %t, want %t: %v", binding.Namespace, binding.Name, err != nil, *binding.ResultReserved, err)
		}
	}
}
