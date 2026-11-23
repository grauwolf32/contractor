package contracts

import (
	"encoding/json"
	"slices"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestSharedPlacementLabels(t *testing.T) {
	t.Parallel()
	var fixture struct {
		Description   string `json:"description"`
		MaxLabels     int    `json:"maxLabels"`
		MaxLabelBytes int    `json:"maxLabelBytes"`
		Matches       []struct {
			Name     string   `json:"name"`
			Required []string `json:"required"`
			Actual   []string `json:"actual"`
			Matches  bool     `json:"matches"`
		} `json:"matches"`
		Cases []struct {
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
	if len(fixture.Matches) == 0 {
		t.Fatal("missing shared matching cases")
	}
	for _, test := range fixture.Matches {
		if got := PlacementLabelsMatch(test.Required, test.Actual); got != test.Matches {
			t.Errorf("%s: matched = %t, want %t", test.Name, got, test.Matches)
		}
	}
	for _, test := range fixture.Cases {
		t.Run(test.Name, func(t *testing.T) {
			if err := ValidatePlacementLabels(test.Labels); (err == nil) != test.Valid {
				t.Fatalf("accepted = %t, want %t: %v", err == nil, test.Valid, err)
			}
		})
	}
}

func TestPlacementNormalizationDetachesAndRejectsContradictions(t *testing.T) {
	input := []string{"cloud/aws", "cloud"}
	normalized, err := NormalizePlacementLabels(input)
	if err != nil || !slices.Equal(normalized, []string{"cloud", "cloud/aws"}) {
		t.Fatalf("normalized = %v, %v", normalized, err)
	}
	input[0] = "local/lab"
	if normalized[1] != "cloud/aws" {
		t.Fatal("normalized labels alias caller input")
	}
	for _, invalid := range [][]string{{"cloud", "local"}, {"cloud/aws", "cloud/hetzner"}, {"cloud", "cloud"}} {
		if _, err := NormalizePlacementLabels(invalid); err == nil || PlacementLabelsMatch(invalid, []string{"cloud/aws"}) {
			t.Fatalf("accepted inconsistent requirement %v", invalid)
		}
	}
}
