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

// The schema requires parameters, artifacts and the deadline, so a nil map,
// which encodes to null, and a zero deadline are invalid both when the
// Planner builds a request and on decode (WRK-R6).
func TestStageContentRequestRequiresMapsAndDeadline(t *testing.T) {
	valid := StageContentRequest{
		APIVersion: APIVersion, SubtaskID: "0", Objective: "check", Instructions: "inspect",
		Parameters: map[string]string{}, Artifacts: map[string]ArtifactRef{},
		Deadline: time.Date(2026, 10, 11, 12, 0, 0, 0, time.UTC),
	}
	if err := valid.Validate(); err != nil {
		t.Fatalf("empty maps rejected: %v", err)
	}
	withoutParameters, withoutArtifacts, withoutDeadline := valid, valid, valid
	withoutParameters.Parameters = nil
	withoutArtifacts.Artifacts = nil
	withoutDeadline.Deadline = time.Time{}
	for name, request := range map[string]StageContentRequest{
		"nil parameters": withoutParameters, "nil artifacts": withoutArtifacts, "zero deadline": withoutDeadline,
	} {
		if err := request.Validate(); err == nil {
			t.Errorf("%s accepted", name)
		}
	}
	const prefix = `{"apiVersion":"contractor/v1alpha1","subtaskId":"0","objective":"check","instructions":"inspect"`
	for _, input := range []string{
		prefix + `,"artifacts":{},"deadline":"2026-10-11T12:00:00Z"}`,
		prefix + `,"parameters":{},"deadline":"2026-10-11T12:00:00Z"}`,
		prefix + `,"parameters":null,"artifacts":{},"deadline":"2026-10-11T12:00:00Z"}`,
		prefix + `,"parameters":{},"artifacts":null,"deadline":"2026-10-11T12:00:00Z"}`,
		prefix + `,"parameters":{},"artifacts":{}}`,
		prefix + `,"parameters":{},"artifacts":{},"deadline":null}`,
		prefix + `,"parameters":{},"artifacts":{},"deadline":"0001-01-01T00:00:00Z"}`,
		prefix + `,"parameters":{},"artifacts":{},"deadline":"2026-10-11T12:00:00"}`,
	} {
		if _, err := DecodePrivateStrict[StageContentRequest]([]byte(input)); err == nil {
			t.Errorf("StageContentRequest accepted: %s", input)
		}
	}
	decoded, err := DecodePrivateStrict[StageContentRequest]([]byte(prefix + `,"parameters":{},"artifacts":{},"deadline":"2026-10-11T14:00:00+02:00"}`))
	if err != nil {
		t.Fatalf("valid StageContentRequest rejected: %v", err)
	}
	if !decoded.Deadline.Equal(valid.Deadline) {
		t.Fatalf("deadline = %s, want %s", decoded.Deadline, valid.Deadline)
	}
}

// StageContentResult is the Planner's result (PLN-R1), not a private wire
// message, so its outcome and error pairing and exact result refs are held
// here rather than by shared fixtures.
func TestStageContentResultPairsOutcomeWithErrorAndExactRefs(t *testing.T) {
	revision := "r2"
	failure := &TerminationError{Code: "invalid_input", Message: "The source cannot produce the output", Retryable: false}
	for _, test := range []struct {
		name   string
		result StageContentResult
		valid  bool
	}{
		{"succeeded with an exact artifact", StageContentResult{
			APIVersion: APIVersion, Outcome: StageSucceeded, Summary: "written",
			Artifacts: map[string]ArtifactRef{"copied": {Namespace: "builder", Name: "copied", Revision: &revision}},
		}, true},
		{"failed with its error", StageContentResult{
			APIVersion: APIVersion, Outcome: StageFailed, Summary: "invalid input",
			Artifacts: map[string]ArtifactRef{}, Error: failure,
		}, true},
		{"succeeded with an error", StageContentResult{
			APIVersion: APIVersion, Outcome: StageSucceeded, Summary: "contradictory",
			Artifacts: map[string]ArtifactRef{}, Error: failure,
		}, false},
		{"failed without an error", StageContentResult{
			APIVersion: APIVersion, Outcome: StageFailed, Summary: "no cause", Artifacts: map[string]ArtifactRef{},
		}, false},
		{"unknown outcome", StageContentResult{
			APIVersion: APIVersion, Outcome: "interrupted", Summary: "unknown", Artifacts: map[string]ArtifactRef{},
		}, false},
		{"versionless artifact", StageContentResult{
			APIVersion: APIVersion, Outcome: StageSucceeded, Summary: "can fall forward",
			Artifacts: map[string]ArtifactRef{"result": {Namespace: "builder", Name: "result"}},
		}, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			if err := test.result.Validate(); (err == nil) != test.valid {
				t.Fatalf("valid=%v: %v", test.valid, err)
			}
		})
	}
}

func TestStageContentRequestValidateCountsCompactUTF8(t *testing.T) {
	deadline := time.Date(2026, 10, 11, 12, 0, 0, 0, time.UTC)
	request := func(instructions string) StageContentRequest {
		return StageContentRequest{
			APIVersion: APIVersion, SubtaskID: "0", Objective: "check", Instructions: instructions,
			Parameters: map[string]string{}, Artifacts: map[string]ArtifactRef{}, Deadline: deadline,
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
