package config

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

type memoryCatalogEntry struct {
	Retired string `json:"retired"`
	Active  string `json:"active"`
}
type memoryCatalog struct {
	SchemaVersion int                  `json:"schema_version"`
	Templates     []memoryCatalogEntry `json:"templates"`
	Workflows     []memoryCatalogEntry `json:"workflows"`
	AuditProfiles []memoryCatalogEntry `json:"audit_profiles"`
}

func repositoryMemoryCatalog(t *testing.T) memoryCatalog {
	t.Helper()
	var catalog memoryCatalog
	if err := json.Unmarshal(readFile(t, filepath.Join(repositoryConfigRoot, "memory-catalog.json")), &catalog); err != nil {
		t.Fatal(err)
	}
	return catalog
}

func TestRetiredWorkflowAndRebasedProfileKeepPinnedSnapshots(t *testing.T) {
	t.Parallel()
	root := filepath.Join(t.TempDir(), "configs")
	if err := os.CopyFS(root, os.DirFS(filepath.Join("..", "..", "testdata", "configs"))); err != nil {
		t.Fatal(err)
	}
	pinned := mustLoad(t, root, MVPDescriptors())
	workflow, err := pinned.Workflow("artifact-copy@1")
	if err != nil {
		t.Fatal(err)
	}
	profile, err := pinned.AuditProfile("source-checklist@1")
	if err != nil {
		t.Fatal(err)
	}
	workflowJSON, err := json.Marshal(workflow)
	if err != nil {
		t.Fatal(err)
	}
	profileJSON, err := json.Marshal(profile)
	if err != nil {
		t.Fatal(err)
	}

	// Retire the Workflow and rebase the profile onto new Worker instructions
	// while keeping its selector.
	if err := os.Remove(filepath.Join(root, "workflows", "artifact_copy.yaml")); err != nil {
		t.Fatal(err)
	}
	appendFile(t, filepath.Join(root, "instructions", "audit-source-checker-worker.md"), "\nRecord every checked file.\n")
	current := mustLoad(t, root, MVPDescriptors())
	if _, err := current.ResolveRunWorkflow(t.Context(), "artifact-copy@1", ExecutionConfigPatch{}, nil); err == nil {
		t.Fatal("new Run silently resolved retired Workflow")
	}
	rebased, err := current.AuditProfile("source-checklist@1")
	if err != nil || rebased.Ref.Digest == profile.Ref.Digest {
		t.Fatalf("rebased profile kept the pinned digest: %v", err)
	}

	decodedWorkflow, err := DecodeResolvedWorkflowSnapshot(workflowJSON)
	if err != nil || !reflect.DeepEqual(workflow, decodedWorkflow) {
		t.Fatalf("pinned Workflow changed after catalog retirement: %v", err)
	}
	decodedProfile, err := DecodeResolvedAuditProfileSnapshot(profileJSON)
	if err != nil || !reflect.DeepEqual(profile, decodedProfile) {
		t.Fatalf("pinned AuditProfile changed after catalog rebase: %v", err)
	}
}
