package streamline

import (
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestStreamlineWorkerResultSizeUsesCompactUTF8(t *testing.T) {
	for _, test := range []struct {
		name, text string
		valid      bool
	}{
		{"HTML", strings.Repeat("<&>", 21000), true},
		{"required control escaping", "x" + strings.Repeat("\x01", 65000), false},
	} {
		t.Run(test.name, func(t *testing.T) {
			result := contracts.WorkerResult{Result: test.text, Artifacts: map[string]contracts.ArtifactRef{}}
			failure := (&streamlinePlanner{}).validateWorkerResult(t.Context(), result)
			if (failure == nil) != test.valid {
				t.Fatalf("valid=%v: %v", test.valid, failure)
			}
			if failure != nil && failure.Failure.Code != "invalid_worker_result" {
				t.Fatalf("unexpected error: %v", failure)
			}
		})
	}
}
