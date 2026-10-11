package contracts

import (
	"fmt"
	"strings"
	"testing"
	"time"
)

func boundedStageResult(summary string, artifacts int) StageContentResult {
	result := StageContentResult{
		APIVersion: APIVersion, Outcome: StageSucceeded, Summary: summary,
		Artifacts: map[string]ArtifactRef{},
	}
	for index := range artifacts {
		revision := "r1"
		result.Artifacts[fmt.Sprintf("slot%03d", index)] = ArtifactRef{Namespace: "builder", Name: "result", Revision: &revision}
	}
	return result
}

// The StageResult bounds are counted once, in compact UTF-8 JSON (WRK-R37):
// optional HTML escaping must not consume budget, required escaping does.
func TestStageContentResultValidateOwnsItsBounds(t *testing.T) {
	for _, test := range []struct {
		name   string
		result StageContentResult
		valid  bool
	}{
		{"HTML summary at the summary bound", boundedStageResult(strings.Repeat("<", MaxStageResultSummaryBytes), 0), true},
		{"summary over its bound", boundedStageResult(strings.Repeat("<", MaxStageResultSummaryBytes+1), 0), false},
		{"multi-byte summary at the byte bound", boundedStageResult(strings.Repeat("ж", MaxStageResultSummaryBytes/2), 0), true},
		{"required escaping over the encoded bound", boundedStageResult("x"+strings.Repeat("\x01", 65000), 0), false},
		{"artifacts at their bound", boundedStageResult("done", MaxStageResultArtifacts), true},
		{"artifacts over their bound", boundedStageResult("done", MaxStageResultArtifacts+1), false},
		{"null artifacts", StageContentResult{APIVersion: APIVersion, Outcome: StageSucceeded, Summary: "done"}, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			if err := test.result.Validate(); (err == nil) != test.valid {
				t.Fatalf("valid=%v: %v", test.valid, err)
			}
		})
	}
	encoded, err := ResultJSONSize(boundedStageResult(strings.Repeat("<", MaxStageResultSummaryBytes), 0))
	if err != nil || encoded > MaxStageResultBytes {
		t.Fatalf("compact size of the HTML summary = %d, %v", encoded, err)
	}
}

func TestStageContentResultRejectsAbsentArtifactsOnDecode(t *testing.T) {
	for _, input := range []string{
		`{"apiVersion":"contractor/v1alpha1","outcome":"succeeded","summary":"done"}`,
		`{"apiVersion":"contractor/v1alpha1","outcome":"succeeded","summary":"done","artifacts":null}`,
	} {
		if _, err := DecodeStrict[StageContentResult]([]byte(input)); err == nil {
			t.Fatalf("StageContentResult without artifacts accepted: %s", input)
		}
	}
}

func TestStageContentRequestValidateCountsCompactUTF8(t *testing.T) {
	deadline := time.Date(2026, 10, 11, 12, 0, 0, 0, time.UTC)
	request := func(instructions string) StageContentRequest {
		return StageContentRequest{
			APIVersion: APIVersion, SubtaskID: "0", Objective: "check", Instructions: instructions,
			Parameters: map[string]string{}, Artifacts: map[string]ArtifactRef{}, Deadline: &deadline,
		}
	}
	// 64 Ki '<' encode to 384 KiB with HTML escaping but 64 KiB compactly.
	if err := request(strings.Repeat("<", 64*1024)).Validate(); err != nil {
		t.Fatalf("HTML instructions within the compact bound rejected: %v", err)
	}
	if err := request(strings.Repeat("x", MaxStageRequestBytes)).Validate(); err == nil {
		t.Fatal("request over its encoded bound accepted")
	}
}
