package planner

import (
	"strings"
	"testing"

	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestPassthroughResultSizeUsesCompactUTF8(t *testing.T) {
	for _, test := range []struct {
		name, text string
		valid      bool
	}{
		{"HTML", strings.Repeat("<&>", 21000), true},
		{"required control escaping", "x" + strings.Repeat("\x01", 65000), false},
	} {
		t.Run(test.name, func(t *testing.T) {
			candidate := contracts.StageContentResult{
				APIVersion: contracts.APIVersion, Outcome: contracts.StageSucceeded,
				Summary: test.text, Artifacts: map[string]contracts.ArtifactRef{},
			}
			sessions := &memorySessions{}
			worker := &recordingWorker{result: workerCompletionFromCandidate(candidate)}
			factory, err := NewPassthroughFactory(sessions, worker, &fakeInspector{})
			if err != nil {
				t.Fatal(err)
			}
			invocation := testInvocation()
			invocation.Stage.Result.Artifacts = map[string]workflowconfig.ArtifactSlot{}
			instance, err := factory.Create(invocation)
			if err != nil {
				t.Fatal(err)
			}
			got, err := instance.Run(t.Context())
			if (err == nil) != test.valid {
				t.Fatalf("valid=%v: %v", test.valid, err)
			}
			if test.valid && (got.Summary != test.text || worker.calls != 1 || sessions.completeCalls != 1) {
				t.Fatal("accepted result changed text or completion accounting")
			}
			if !test.valid && FailureFrom(err).Code != "invalid_worker_result" {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}
