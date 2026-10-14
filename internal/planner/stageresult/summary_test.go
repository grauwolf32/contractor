package stageresult

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestSharedStageSummaryContracts(t *testing.T) {
	data := contractstest.ReadFile(t, "api", "testdata", "stage-summary-cases.json")
	var cases []struct {
		Name   string
		Valid  bool
		Result json.RawMessage
	}
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	for _, tc := range cases {
		t.Run(tc.Name, func(t *testing.T) {
			var value StageContentResult
			err := json.Unmarshal(tc.Result, &value)
			if err == nil {
				err = value.Validate()
			}
			if (err == nil) != tc.Valid {
				t.Fatalf("valid=%v: %v", tc.Valid, err)
			}
		})
	}
}

func TestSummaryContinuationBoundsAndClone(t *testing.T) {
	revision := "r1"
	value := SummaryContinuation{SourceStageExecutionID: "source", Summary: strings.Repeat("ж", MaxStageResultSummaryBytes/2),
		SummaryArtifact: contracts.ArtifactRef{Namespace: "builder", Name: "summary.source", Revision: &revision},
		Artifacts:       boundedStageResult("partial", MaxStageResultArtifacts).Artifacts}
	if err := value.Validate(); err != nil {
		t.Fatalf("exact bounds rejected: %v", err)
	}
	clone := value.Clone()
	*clone.SummaryArtifact.Revision = "changed"
	if *value.SummaryArtifact.Revision != "r1" {
		t.Fatal("clone shares summary revision")
	}
	value.Summary += "x"
	if err := value.Validate(); err == nil {
		t.Fatal("summary overflow accepted")
	}
	value.Summary = "partial"
	value.Artifacts["overflow"] = value.SummaryArtifact
	if err := value.Validate(); err == nil {
		t.Fatal("saved reference overflow accepted")
	}
}
