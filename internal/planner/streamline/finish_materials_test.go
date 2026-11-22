package streamline

import (
	"github.com/grauwolf32/contractor/internal/contracts"
	"testing"
)

func TestFinishUsesAcceptedMaterialsAndRejectsAmbiguousOwners(t *testing.T) {
	state := newExecutionState(Limits{})
	p := &streamlinePlanner{invocation: testInvocation("builder")}
	ref := exactRef("builder", "report", "r1")
	selections := map[string]materialSelection{"report": {Namespace: "builder", Name: "report"}}
	if _, err := p.finishArtifacts(state, selections); err == nil {
		t.Fatal("unknown material accepted")
	}
	for _, revision := range []string{"r1", "r2"} {
		ref = exactRef("builder", "report", revision)
		if !state.selectWorkerArtifacts("builder", map[string]contracts.ArtifactRef{"report": ref}) {
			t.Fatal("admission refused")
		}
		result, err := p.finishArtifacts(state, selections)
		if err != nil || *result["report"].Revision != revision {
			t.Fatalf("resolved %v, %v", result, err)
		}
	}
	// An independent Worker cannot silently decide the version by completion order.
	state.selectWorkerArtifacts("other", map[string]contracts.ArtifactRef{"report": exactRef("builder", "report", "different")})
	if _, err := p.finishArtifacts(state, selections); err == nil {
		t.Fatal("ambiguous material accepted")
	}
	state.selectWorkerArtifacts("other", nil)
	if _, err := p.finishArtifacts(state, selections); err != nil {
		t.Fatal(err)
	}
	// Input-only selections must also be unambiguous.
	state = newExecutionState(Limits{})
	p.invocation.Context.Artifacts["one"] = pointerRef(ref)
	p.invocation.Context.Artifacts["two"] = pointerRef(exactRef("builder", "report", "old"))
	if _, err := p.finishArtifacts(state, selections); err == nil {
		t.Fatal("ambiguous inputs accepted")
	}
}
