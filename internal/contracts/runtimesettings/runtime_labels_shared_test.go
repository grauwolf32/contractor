package runtimesettings

import (
	"encoding/json"
	"slices"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

// runtimeLabelSetCases is api/testdata/v1alpha1/runtime-label-set-cases.json.
type runtimeLabelSetCases struct {
	Description   string `json:"description"`
	MaxLabels     int    `json:"maxLabels"`
	MaxLabelBytes int    `json:"maxLabelBytes"`
	Cases         []struct {
		Name      string   `json:"name"`
		Labels    []string `json:"labels"`
		Valid     bool     `json:"valid"`
		Canonical []string `json:"canonical"`
	} `json:"cases"`
}

// TestSharedRuntimeLabelSetCases holds the one Go definition of a Runtime
// label set, and the provenance arrays of Runtime settings that carry one, to
// the cases the browser and the Runtime are held to as well.
func TestSharedRuntimeLabelSetCases(t *testing.T) {
	t.Parallel()
	var fixture runtimeLabelSetCases
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "runtime-label-set-cases.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.MaxLabels != MaxRuntimeLabels || fixture.MaxLabelBytes != maxRuntimeLabelBytes || len(fixture.Cases) == 0 {
		t.Fatalf("fixture bounds = %d labels of %d bytes, want %d and %d", fixture.MaxLabels, fixture.MaxLabelBytes, MaxRuntimeLabels, maxRuntimeLabelBytes)
	}
	config := RuntimeConfigRef{Name: "contractor-empty", Version: "1", Digest: "sha256:80a1754c01f8443c29fdc8f650a2254b2461694819918b204a55a7ad3425dc5f"}
	for _, test := range fixture.Cases {
		canonical, err := CanonicalRuntimeLabels(test.Labels)
		if (err == nil) != test.Valid || test.Valid && !slices.Equal(canonical, test.Canonical) {
			t.Errorf("%s: CanonicalRuntimeLabels = (%q, %v), want valid %t and %q", test.Name, canonical, err, test.Valid, test.Canonical)
		}
		stored := test.Valid && slices.Equal(test.Labels, test.Canonical)
		if err := ValidateCanonicalRuntimeLabels(test.Labels); (err == nil) != stored {
			t.Errorf("%s: ValidateCanonicalRuntimeLabels = %v, want accepted %t", test.Name, err, stored)
		}
		bindings := make([]RuntimeLabelBindingProvenance, 0, len(test.Labels))
		for _, label := range test.Labels {
			bindings = append(bindings, RuntimeLabelBindingProvenance{Label: label, BindingRevision: 1, Config: config})
		}
		if err := validateProvenanceBindings("runLabels", bindings); (err == nil) != stored {
			t.Errorf("%s: provenance bindings accepted = %v, want %t", test.Name, err, stored)
		}
	}
}
