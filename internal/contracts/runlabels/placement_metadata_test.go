package runlabels

import (
	"encoding/json"
	"fmt"
	"slices"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestSharedPlacementMetadataExtraction(t *testing.T) {
	var fixture struct {
		MaxAuditLabels int `json:"maxAuditLabels"`
		Cases          []struct {
			Name     string            `json:"name"`
			Labels   RunMetadataLabels `json:"labels"`
			Required []string          `json:"required"`
			Valid    bool              `json:"valid"`
		} `json:"metadataCases"`
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "placement-label-cases.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	if len(fixture.Cases) == 0 {
		t.Fatal("missing metadata placement cases")
	}
	if fixture.MaxAuditLabels != MaxAuditLabels {
		t.Fatalf("Audit label bound=%d, fixture=%d", MaxAuditLabels, fixture.MaxAuditLabels)
	}
	for _, test := range fixture.Cases {
		t.Run(test.Name, func(t *testing.T) {
			got, err := test.Labels.PlacementRequirements()
			if (err == nil) != test.Valid || test.Valid && !slices.Equal(got, test.Required) {
				t.Fatalf("required=%v err=%v want=%v valid=%t", got, err, test.Required, test.Valid)
			}
		})
	}
}

func TestAuditLabelsReserveChildProvenanceBudget(t *testing.T) {
	labels := map[string]string{}
	for index := 0; index < MaxAuditLabels; index++ {
		labels[fmt.Sprintf("tag%d", index)] = "value"
	}
	normalized, err := NormalizeAuditLabels(labels)
	if err != nil {
		t.Fatal(err)
	}
	normalized["tag0"] = "changed"
	if labels["tag0"] != "value" {
		t.Fatal("Audit labels alias caller map")
	}
	labels["extra"] = "value"
	if _, err := NormalizeAuditLabels(labels); err == nil {
		t.Fatal("too many inherited labels accepted")
	}
	if _, err := NormalizeAuditLabels(map[string]string{"audit.role": "check"}); err == nil {
		t.Fatal("reserved child provenance accepted")
	}
}
