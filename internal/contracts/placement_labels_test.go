package contracts

import (
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestSharedPlacementLabels(t *testing.T) {
	t.Parallel()
	var fixture struct {
		Description   string `json:"description"`
		MaxLabels     int    `json:"maxLabels"`
		MaxLabelBytes int    `json:"maxLabelBytes"`
		Cases         []struct {
			Name   string   `json:"name"`
			Labels []string `json:"labels"`
			Valid  bool     `json:"valid"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "placement-label-cases.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	if len(fixture.Cases) == 0 || fixture.MaxLabels != 32 || fixture.MaxLabelBytes != 63 {
		t.Fatal("missing placement cases or changed shared bounds")
	}
	for _, test := range fixture.Cases {
		t.Run(test.Name, func(t *testing.T) {
			if err := ValidatePlacementLabels(test.Labels); (err == nil) != test.Valid {
				t.Fatalf("accepted = %t, want %t: %v", err == nil, test.Valid, err)
			}
		})
	}
}
