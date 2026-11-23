package runtimeconfig

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestSharedRunSnapshotCredentialSetsUseByteOrder(t *testing.T) {
	var fixture struct {
		Cases []struct {
			Name  string   `json:"name"`
			IDs   []string `json:"ids"`
			Valid bool     `json:"valid"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "runtime-snapshot-credential-sets.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	for _, c := range fixture.Cases {
		t.Run(c.Name, func(t *testing.T) {
			for _, llm := range []bool{false, true} {
				snapshot := builtInRunSnapshot()
				if llm {
					snapshot.LLMCredentialIDs = c.IDs
				} else {
					snapshot.RuntimeCredentialIDs = c.IDs
				}
				if err := snapshot.Validate(); (err == nil) != c.Valid {
					t.Fatalf("llm=%t: valid=%t, error=%v", llm, c.Valid, err)
				}
			}
		})
	}
}

// builtInRunSnapshot pins the default label to the built-in RuntimeConfig,
// as runtimeconfigtest.BuiltInRunSnapshot does for other packages' tests.
func builtInRunSnapshot() RunSnapshot {
	return RunSnapshot{
		Default: PinnedLabel{
			Label: DefaultLabel, BindingRevision: 1,
			Config: Ref{Name: BuiltInName, Version: BuiltInVersion, Digest: BuiltInDigest},
		},
		Labels: []PinnedLabel{}, LLMCredentialIDs: []string{}, RuntimeCredentialIDs: []string{},
	}
}

func TestRunSnapshotValidationKeepsSafeExactPins(t *testing.T) {
	snapshot := builtInRunSnapshot()
	snapshot.Labels = []PinnedLabel{{
		Label: "debug", Explicit: true, BindingRevision: 3,
		Config: Ref{Name: "debug", Version: "2", Digest: "sha256:" + strings.Repeat("a", 64)},
	}}
	snapshot.LLMCredentialIDs = []string{"worker-route"}
	snapshot.RuntimeCredentialIDs = []string{"otel-auth"}
	if err := snapshot.Validate(); err != nil {
		t.Fatal(err)
	}
	invalid := snapshot
	invalid.Labels = append(invalid.Labels, invalid.Labels[0])
	if err := invalid.Validate(); err == nil {
		t.Fatal("duplicate pinned Runtime label accepted")
	}
}
