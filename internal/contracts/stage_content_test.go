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

// The schema requires parameters and artifacts, so a nil map, which encodes to
// null, is invalid both when the Planner builds a request and on decode.
func TestStageContentRequestRequiresParametersAndArtifacts(t *testing.T) {
	valid := StageContentRequest{
		APIVersion: APIVersion, SubtaskID: "0", Objective: "check", Instructions: "inspect",
		Parameters: map[string]string{}, Artifacts: map[string]ArtifactRef{},
	}
	if err := valid.Validate(); err != nil {
		t.Fatalf("empty maps rejected: %v", err)
	}
	withoutParameters, withoutArtifacts := valid, valid
	withoutParameters.Parameters = nil
	withoutArtifacts.Artifacts = nil
	for name, request := range map[string]StageContentRequest{
		"nil parameters": withoutParameters, "nil artifacts": withoutArtifacts,
	} {
		if err := request.Validate(); err == nil {
			t.Errorf("%s accepted", name)
		}
	}
	const prefix = `{"apiVersion":"contractor/v1alpha1","subtaskId":"0","objective":"check","instructions":"inspect"`
	for _, input := range []string{
		prefix + `,"artifacts":{}}`,
		prefix + `,"parameters":{}}`,
		prefix + `,"parameters":null,"artifacts":{}}`,
		prefix + `,"parameters":{},"artifacts":null}`,
	} {
		if _, err := DecodePrivateStrict[StageContentRequest]([]byte(input)); err == nil {
			t.Errorf("StageContentRequest accepted: %s", input)
		}
	}
	if _, err := DecodePrivateStrict[StageContentRequest]([]byte(prefix + `,"parameters":{},"artifacts":{}}`)); err != nil {
		t.Fatalf("valid StageContentRequest rejected: %v", err)
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
