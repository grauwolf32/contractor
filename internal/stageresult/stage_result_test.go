package stageresult

import (
	"fmt"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func boundedStageResult(summary string, artifacts int) StageContentResult {
	result := StageContentResult{
		APIVersion: contracts.APIVersion, Outcome: StageSucceeded, Summary: summary,
		Artifacts: map[string]contracts.ArtifactRef{},
	}
	for index := range artifacts {
		revision := "r1"
		result.Artifacts[fmt.Sprintf("slot%03d", index)] = contracts.ArtifactRef{Namespace: "builder", Name: "result", Revision: &revision}
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
		{"null artifacts", StageContentResult{APIVersion: contracts.APIVersion, Outcome: StageSucceeded, Summary: "done"}, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			if err := test.result.Validate(); (err == nil) != test.valid {
				t.Fatalf("valid=%v: %v", test.valid, err)
			}
		})
	}
	encoded, err := contracts.ResultJSONSize(boundedStageResult(strings.Repeat("<", MaxStageResultSummaryBytes), 0))
	if err != nil || encoded > MaxStageResultBytes {
		t.Fatalf("compact size of the HTML summary = %d, %v", encoded, err)
	}
}

func TestStageContentResultRejectsAbsentArtifactsOnDecode(t *testing.T) {
	for _, input := range []string{
		`{"apiVersion":"contractor/v1alpha1","outcome":"succeeded","summary":"done"}`,
		`{"apiVersion":"contractor/v1alpha1","outcome":"succeeded","summary":"done","artifacts":null}`,
	} {
		if _, err := contracts.DecodePrivateStrict[StageContentResult]([]byte(input)); err == nil {
			t.Fatalf("StageContentResult without artifacts accepted: %s", input)
		}
	}
}

// StageContentResult is the Planner's result (PLN-R1), not a private wire
// message, so its outcome and error pairing and exact result refs are held
// here rather than by shared fixtures.
func TestStageContentResultPairsOutcomeWithErrorAndExactRefs(t *testing.T) {
	revision := "r2"
	failure := &contracts.TerminationError{Code: "invalid_input", Message: "The source cannot produce the output", Retryable: false}
	for _, test := range []struct {
		name   string
		result StageContentResult
		valid  bool
	}{
		{"succeeded with an exact artifact", StageContentResult{
			APIVersion: contracts.APIVersion, Outcome: StageSucceeded, Summary: "written",
			Artifacts: map[string]contracts.ArtifactRef{"copied": {Namespace: "builder", Name: "copied", Revision: &revision}},
		}, true},
		{"failed with its error", StageContentResult{
			APIVersion: contracts.APIVersion, Outcome: StageFailed, Summary: "invalid input",
			Artifacts: map[string]contracts.ArtifactRef{}, Error: failure,
		}, true},
		{"succeeded with an error", StageContentResult{
			APIVersion: contracts.APIVersion, Outcome: StageSucceeded, Summary: "contradictory",
			Artifacts: map[string]contracts.ArtifactRef{}, Error: failure,
		}, false},
		{"failed without an error", StageContentResult{
			APIVersion: contracts.APIVersion, Outcome: StageFailed, Summary: "no cause", Artifacts: map[string]contracts.ArtifactRef{},
		}, false},
		{"unknown outcome", StageContentResult{
			APIVersion: contracts.APIVersion, Outcome: "interrupted", Summary: "unknown", Artifacts: map[string]contracts.ArtifactRef{},
		}, false},
		{"versionless artifact", StageContentResult{
			APIVersion: contracts.APIVersion, Outcome: StageSucceeded, Summary: "can fall forward",
			Artifacts: map[string]contracts.ArtifactRef{"result": {Namespace: "builder", Name: "result"}},
		}, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			if err := test.result.Validate(); (err == nil) != test.valid {
				t.Fatalf("valid=%v: %v", test.valid, err)
			}
		})
	}
}

func TestStageContentResultCloneDoesNotAliasItsSource(t *testing.T) {
	revision := "r1"
	source := StageContentResult{
		APIVersion: contracts.APIVersion, Outcome: StageFailed, Summary: "failed",
		Artifacts: map[string]contracts.ArtifactRef{"partial": {Namespace: "builder", Name: "partial", Revision: &revision}},
		Error:     &contracts.TerminationError{Code: "worker_failed", Message: "The Worker failed"},
	}
	clone := source.Clone()
	*clone.Artifacts["partial"].Revision = "changed"
	delete(clone.Artifacts, "partial")
	clone.Error.Code = "changed"
	if revision != "r1" || len(source.Artifacts) != 1 || source.Error.Code != "worker_failed" {
		t.Fatalf("a clone aliases its source: %+v", source)
	}
}
