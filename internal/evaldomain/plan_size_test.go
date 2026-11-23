package evaldomain

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestSharedPlanSizeEstimateMatchesSharedCases(t *testing.T) {
	t.Parallel()
	data := contractstest.ReadFile(t, "api", "testdata", "evals", "plan-size-cases.json")
	var document struct {
		MaxDocumentBytes int `json:"maxDocumentBytes"`
		Cases            []struct {
			Name    string          `json:"name"`
			Draft   json.RawMessage `json:"draft"`
			Padding *struct {
				Unit   string `json:"unit"`
				Repeat int    `json:"repeat"`
			} `json:"padding"`
			EstimatedBytes int  `json:"estimatedBytes"`
			WithinLimit    bool `json:"withinLimit"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(data, &document); err != nil {
		t.Fatal(err)
	}
	if document.MaxDocumentBytes != MaxDocumentBytes {
		t.Fatalf("shared limit = %d, want %d", document.MaxDocumentBytes, MaxDocumentBytes)
	}
	for _, test := range document.Cases {
		t.Run(test.Name, func(t *testing.T) {
			var draft Draft
			decoder := json.NewDecoder(bytes.NewReader(test.Draft))
			decoder.DisallowUnknownFields()
			if err := decoder.Decode(&draft); err != nil {
				t.Fatal(err)
			}
			if test.Padding != nil {
				if draft.Variants[0].Parameters == nil {
					draft.Variants[0].Parameters = map[string]string{}
				}
				draft.Variants[0].Parameters["padding"] = strings.Repeat(test.Padding.Unit, test.Padding.Repeat)
			}
			estimated, err := estimatedPlanBytes(draft)
			if err != nil || estimated != test.EstimatedBytes {
				t.Fatalf("estimatedPlanBytes = %d, %v; want %d", estimated, err, test.EstimatedBytes)
			}
			// Every shared draft is otherwise valid, so only the plan size decides.
			if err := validateDraft(draft); (err == nil) != test.WithinLimit {
				t.Fatalf("validateDraft = %v, want within limit %v", err, test.WithinLimit)
			}
		})
	}
}
