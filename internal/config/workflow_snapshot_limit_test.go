package config

import (
	"context"
	"errors"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
)

const snapshotLimitInstruction = "instructions/copy-planner.md"

// TestCatalogRejectsAWorkflowWhoseSnapshotExceedsItsBound is the regression
// for a 2.3 MB instruction that the catalog accepted and the Scheduler then
// rejected after the Run was accepted: the bound now fails the load (CAT-R76).
func TestCatalogRejectsAWorkflowWhoseSnapshotExceedsItsBound(t *testing.T) {
	root := copyCoreFixture(t)
	writeFile(t, filepath.Join(root, snapshotLimitInstruction), []byte(strings.Repeat("a", 2_300_000)))

	_, err := Load(root)
	if !errors.Is(err, ErrWorkflowSnapshotTooLarge) || !strings.Contains(err.Error(), "artifact-copy@1") {
		t.Fatalf("load with a 2.3 MB instruction error = %v, want ErrWorkflowSnapshotTooLarge for artifact-copy@1", err)
	}
}

// TestWorkflowSnapshotBoundAppliesToTheSnapshotARunStores pads a Workflow to
// exactly MaxWorkflowSnapshotBytes. The catalog and an override-free Run
// accept it; a Run override that enlarges the snapshot is rejected before
// any credential lookup, so it can never be accepted and fail later.
func TestWorkflowSnapshotBoundAppliesToTheSnapshotARunStores(t *testing.T) {
	root := copyCoreFixture(t)
	snapshot := loadWithSnapshotLimitInstruction(t, root, "a")
	gateway, err := snapshot.LLMGateway("local-litellm@1")
	if err != nil {
		t.Fatal(err)
	}
	lookup := &countingCredentialLookup{lookup: metadataLookup{
		"development-worker": credentialMetadata("development-worker", gateway.Ref),
	}}
	// The override selects a policy whose resolved selection encodes longer
	// than the Workflow default, so it enlarges every snapshot by the same delta.
	enlarging := decodeExecutionConfigPatch(t, `{"workers":{"modelPolicy":"test-domain-worker@1"}}`)
	base := encodedRunWorkflowSize(t, snapshot, ExecutionConfigPatch{}, lookup)
	enlarged := encodedRunWorkflowSize(t, snapshot, enlarging, lookup)
	if enlarged <= base {
		t.Fatalf("override snapshot = %d bytes, base = %d bytes; the override must enlarge it", enlarged, base)
	}

	// The 1-byte instruction is ASCII and embedded once, so padding it adds
	// exactly its new length to the encoding.
	padded := loadWithSnapshotLimitInstruction(t, root, strings.Repeat("a", 1+MaxWorkflowSnapshotBytes-base))
	workflow, err := padded.Workflow("artifact-copy@1")
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := EncodeResolvedWorkflowSnapshot(workflow)
	if err != nil || len(encoded) != MaxWorkflowSnapshotBytes {
		t.Fatalf("catalog snapshot at the bound = (%d bytes, %v), want exactly %d", len(encoded), err, MaxWorkflowSnapshotBytes)
	}
	if size := encodedRunWorkflowSize(t, padded, ExecutionConfigPatch{}, lookup); size != MaxWorkflowSnapshotBytes {
		t.Fatalf("override-free Run snapshot = %d bytes, want exactly %d", size, MaxWorkflowSnapshotBytes)
	}

	lookups := lookup.calls.Load()
	_, err = padded.ResolveRunWorkflow(t.Context(), "artifact-copy@1", enlarging, lookup)
	if !errors.Is(err, ErrWorkflowSnapshotTooLarge) || errors.Is(err, ErrCredentialLookup) {
		t.Fatalf("enlarging Run override error = %v, want ErrWorkflowSnapshotTooLarge", err)
	}
	if lookup.calls.Load() != lookups {
		t.Fatal("an oversized Run snapshot consulted the credential store before it was rejected")
	}
	if _, err := EncodeResolvedWorkflowSnapshot(ResolvedWorkflow{
		Ref:        WorkflowRef{Name: "unbounded", Version: "1"},
		EntryStage: strings.Repeat("s", MaxWorkflowSnapshotBytes),
	}); !errors.Is(err, ErrWorkflowSnapshotTooLarge) {
		t.Fatalf("EncodeResolvedWorkflowSnapshot over the bound error = %v", err)
	}
}

func loadWithSnapshotLimitInstruction(t *testing.T, root, text string) *Snapshot {
	t.Helper()
	writeFile(t, filepath.Join(root, snapshotLimitInstruction), []byte(text))
	return mustLoad(t, root)
}

func encodedRunWorkflowSize(t *testing.T, snapshot *Snapshot, patch ExecutionConfigPatch, lookup CredentialLookup) int {
	t.Helper()
	workflow, err := snapshot.ResolveRunWorkflow(t.Context(), "artifact-copy@1", patch, lookup)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := EncodeResolvedWorkflowSnapshot(workflow)
	if err != nil {
		t.Fatal(err)
	}
	return len(encoded)
}

type countingCredentialLookup struct {
	lookup CredentialLookup
	calls  atomic.Int64
}

func (l *countingCredentialLookup) LookupLLMCredential(ctx context.Context, id string) (CredentialMetadata, error) {
	l.calls.Add(1)
	return l.lookup.LookupLLMCredential(ctx, id)
}
