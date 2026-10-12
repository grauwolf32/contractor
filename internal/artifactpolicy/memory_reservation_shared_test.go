package artifactpolicy

import (
	"bytes"
	"encoding/json"
	"slices"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

// memoryReservationBinding is one binding of
// api/testdata/v1alpha1/memory-reservation-cases.json.
type memoryReservationBinding struct {
	Namespace      string `json:"namespace"`
	Name           string `json:"name"`
	MemoryReserved *bool  `json:"memoryReserved"`
	ResultReserved *bool  `json:"resultReserved"`
}

// TestSharedMemoryReservationCases checks the reserved-binding rule of
// api/testdata/v1alpha1/memory-reservation-cases.json, which the Runtime
// applies too (runtime/tests/test_shared_memory_reservation.py).
func TestSharedMemoryReservationCases(t *testing.T) {
	t.Parallel()
	var cases struct {
		Description               string                     `json:"description"`
		MemoryPrefix              string                     `json:"memoryPrefix"`
		PurposeReservedNamespaces []string                   `json:"purposeReservedNamespaces"`
		Bindings                  []memoryReservationBinding `json:"bindings"`
	}
	decoder := json.NewDecoder(bytes.NewReader(contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "memory-reservation-cases.json")))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&cases); err != nil {
		t.Fatal(err)
	}
	if cases.MemoryPrefix != MemoryArtifactPrefix || len(cases.Bindings) == 0 {
		t.Fatalf("shared Memory reservation cases are incomplete: %+v", cases)
	}
	purpose := slices.Sorted(slices.Values(PurposeReservedNamespaces()))
	if !slices.Equal(purpose, cases.PurposeReservedNamespaces) {
		t.Fatalf("PurposeReservedNamespaces() = %q, want the shared %q", purpose, cases.PurposeReservedNamespaces)
	}
	for _, binding := range cases.Bindings {
		if binding.MemoryReserved == nil || binding.ResultReserved == nil {
			t.Fatalf("binding %s/%s needs both verdicts", binding.Namespace, binding.Name)
		}
		if got := IsReservedMemoryBinding(binding.Namespace, binding.Name); got != *binding.MemoryReserved {
			t.Errorf("IsReservedMemoryBinding(%q, %q) = %t, want %t", binding.Namespace, binding.Name, got, *binding.MemoryReserved)
		}
		result := IsPurposeReservedNamespace(binding.Namespace) || IsReservedMemoryBinding(binding.Namespace, binding.Name)
		if result != *binding.ResultReserved {
			t.Errorf("result binding %s/%s reserved = %t, want %t", binding.Namespace, binding.Name, result, *binding.ResultReserved)
		}
	}
}
