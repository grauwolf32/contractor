package config

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
)

const snapshotLimitInstruction = "instructions/copy-planner.md"

// TestCatalogRejectsAWorkflowWhoseSnapshotExceedsItsBound is the regression
// for a 2.3 MB instruction that the catalog accepted and the Scheduler then
// rejected after the Run was accepted: the bound now fails the load (CAT-R76).
func TestCatalogRejectsAWorkflowWhoseSnapshotExceedsItsBound(t *testing.T) {
	root := copyCoreFixture(t)
	writeFile(t, filepath.Join(root, snapshotLimitInstruction), []byte(strings.Repeat("a", 2_300_000)))

	_, err := LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "managed"))
	if !errors.Is(err, resolved.ErrWorkflowSnapshotTooLarge) || !strings.Contains(err.Error(), "artifact-copy@1") {
		t.Fatalf("load with a 2.3 MB instruction error = %v, want ErrWorkflowSnapshotTooLarge for artifact-copy@1", err)
	}
}

// TestWorkflowSnapshotBoundAppliesToTheSnapshotARunStores pads a Workflow to
// exactly MaxWorkflowSnapshotBytes. The catalog and a Run with no model defaults
// accept it; captured deployment defaults that enlarge it fail before any
// credential lookup, so the Run cannot be accepted and fail later.
func TestWorkflowSnapshotBoundAppliesToTheSnapshotARunStores(t *testing.T) {
	root := copyCoreFixture(t)
	replaceFile(t, filepath.Join(root, "workflows/artifact_copy.yaml"), "test-worker@1", "worker@1")
	data, err := os.ReadFile(filepath.Join(root, "model-policies/test-worker.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	writeFile(t, filepath.Join(root, "model-policies/worker.yaml"), []byte(strings.Replace(string(data), "name: test-worker", "name: worker", 1)))
	snapshot := loadWithSnapshotLimitInstruction(t, root, "a")
	gateway, err := snapshot.LLMGateway("local-litellm@1")
	if err != nil {
		t.Fatal(err)
	}
	lookup := &countingCredentialLookup{lookup: metadataLookup{
		"development-worker": credentialMetadata("development-worker", gateway.Ref),
	}}
	base := encodedRunWorkflowSize(t, snapshot, lookup)

	// The 1-byte instruction is ASCII and embedded once, so padding it adds
	// exactly its new length to the encoding.
	padded := loadWithSnapshotLimitInstruction(t, root, strings.Repeat("a", 1+resolved.MaxWorkflowSnapshotBytes-base))
	workflow, err := padded.Workflow("artifact-copy@1")
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := resolved.EncodeResolvedWorkflowSnapshot(workflow)
	if err != nil || len(encoded) != resolved.MaxWorkflowSnapshotBytes {
		t.Fatalf("catalog snapshot at the bound = (%d bytes, %v), want exactly %d", len(encoded), err, resolved.MaxWorkflowSnapshotBytes)
	}
	if size := encodedRunWorkflowSize(t, padded, lookup); size != resolved.MaxWorkflowSnapshotBytes {
		t.Fatalf("Run snapshot without model defaults = %d bytes, want exactly %d", size, resolved.MaxWorkflowSnapshotBytes)
	}

	policy, err := padded.ModelPolicy("test-domain-worker@1")
	if err != nil {
		t.Fatal(err)
	}
	defaults := map[string]contracts.ResolvedModelPolicy{"worker": policy}
	lookups := lookup.calls.Load()
	_, err = padded.resolveRunWorkflow(t.Context(), "artifact-copy@1", lookup, defaults)
	if !errors.Is(err, resolved.ErrWorkflowSnapshotTooLarge) || errors.Is(err, resolved.ErrCredentialLookup) {
		t.Fatalf("enlarging model default error = %v, want ErrWorkflowSnapshotTooLarge", err)
	}
	if lookup.calls.Load() != lookups {
		t.Fatal("an oversized Run snapshot consulted the credential store before it was rejected")
	}
	if _, err := resolved.EncodeResolvedWorkflowSnapshot(resolved.ResolvedWorkflow{
		Ref:        resolved.WorkflowRef{Name: "unbounded", Version: "1"},
		EntryStage: strings.Repeat("s", resolved.MaxWorkflowSnapshotBytes),
	}); !errors.Is(err, resolved.ErrWorkflowSnapshotTooLarge) {
		t.Fatalf("EncodeResolvedWorkflowSnapshot over the bound error = %v", err)
	}
}

func loadWithSnapshotLimitInstruction(t *testing.T, root, text string) *Snapshot {
	t.Helper()
	writeFile(t, filepath.Join(root, snapshotLimitInstruction), []byte(text))
	return mustLoad(t, root)
}

func encodedRunWorkflowSize(t *testing.T, snapshot *Snapshot, lookup resolved.CredentialLookup) int {
	t.Helper()
	workflow, err := snapshot.ResolveRunWorkflow(t.Context(), "artifact-copy@1", lookup)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := resolved.EncodeResolvedWorkflowSnapshot(workflow)
	if err != nil {
		t.Fatal(err)
	}
	return len(encoded)
}

type countingCredentialLookup struct {
	lookup resolved.CredentialLookup
	calls  atomic.Int64
}

func (l *countingCredentialLookup) LookupLLMCredential(ctx context.Context, id string) (resolved.CredentialMetadata, error) {
	l.calls.Add(1)
	return l.lookup.LookupLLMCredential(ctx, id)
}
