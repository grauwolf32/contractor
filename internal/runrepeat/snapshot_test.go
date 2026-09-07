package runrepeat

import (
	"errors"
	"testing"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestSnapshotRoundTripPreservesExactInputsAndExecutionPatch(t *testing.T) {
	var patch config.ExecutionConfigPatch
	if err := patch.UnmarshalJSON([]byte(`{"workers":{"credential":null,"modelPolicy":"strong@1"}}`)); err != nil {
		t.Fatal(err)
	}
	revision := "revision-1"
	projectID := "project-1"
	encoded, err := Encode(Snapshot{
		Workflow: config.WorkflowRef{Name: "review", Version: "1"}, ProjectID: &projectID,
		Inputs: map[string]contracts.ArtifactRef{
			"source": {Namespace: "sources", Name: "service", Revision: &revision},
		},
		ExecutionConfig: patch,
	})
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := Decode(encoded)
	if err != nil {
		t.Fatal(err)
	}
	reencoded, err := Encode(decoded)
	if err != nil {
		t.Fatal(err)
	}
	if string(encoded) != string(reencoded) {
		t.Fatalf("non-canonical round trip\nfirst:  %s\nsecond: %s", encoded, reencoded)
	}
}

func TestSnapshotRejectsVersionlessAndUnknownData(t *testing.T) {
	_, err := Encode(Snapshot{
		Workflow: config.WorkflowRef{Name: "review", Version: "1"},
		Inputs: map[string]contracts.ArtifactRef{
			"source": {Namespace: "sources", Name: "service"},
		},
	})
	if !errors.Is(err, ErrInvalidSnapshot) {
		t.Fatalf("versionless input error = %v", err)
	}
	if _, err := Decode([]byte(`{"schemaVersion":"contractor.run-repeat-request/v1","workflow":{"name":"review","version":"1"},"inputs":{},"executionConfig":{},"extra":true}`)); !errors.Is(err, ErrInvalidSnapshot) {
		t.Fatalf("unknown field error = %v", err)
	}
}
