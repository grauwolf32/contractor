package runtimeconfig

import (
	"strings"
	"testing"
)

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

func TestRunSnapshotCloneAndValidationKeepSafeExactPins(t *testing.T) {
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
	clone := snapshot.Clone()
	clone.Labels[0].Label = "changed"
	clone.LLMCredentialIDs[0] = "changed-route"
	if snapshot.Labels[0].Label != "debug" || snapshot.LLMCredentialIDs[0] != "worker-route" {
		t.Fatal("Run RuntimeConfig snapshot clone aliases source slices")
	}
	invalid := snapshot.Clone()
	invalid.Labels = append(invalid.Labels, invalid.Labels[0])
	if err := invalid.Validate(); err == nil {
		t.Fatal("duplicate pinned Runtime label accepted")
	}
}
