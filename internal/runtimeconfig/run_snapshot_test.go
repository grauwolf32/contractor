package runtimeconfig

import (
	"strings"
	"testing"
)

func TestNormalizeRunLabelsCanonicalizesOnlyOrder(t *testing.T) {
	labels, err := NormalizeRunLabels([]string{"trace", "caido", "debug"})
	if err != nil || strings.Join(labels, ",") != "caido,debug,trace" {
		t.Fatalf("canonical Run labels = (%v, %v)", labels, err)
	}
	for _, invalidLabels := range [][]string{
		{"debug", "debug"}, {DefaultLabel}, {"Bad"}, {""},
	} {
		if _, err := NormalizeRunLabels(invalidLabels); err == nil {
			t.Fatalf("invalid Run labels accepted: %v", invalidLabels)
		}
	}
	tooMany := make([]string, MaximumRunLabels+1)
	for index := range tooMany {
		tooMany[index] = "label_" + strings.Repeat("a", index+1)
	}
	if _, err := NormalizeRunLabels(tooMany); err == nil {
		t.Fatal("oversized Run label set accepted")
	}
}

func TestRunSnapshotCloneAndValidationKeepSafeExactPins(t *testing.T) {
	snapshot := BuiltInRunSnapshot()
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
