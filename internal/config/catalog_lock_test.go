package config

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCatalogLockRejectsEveryShippedContentChange(t *testing.T) {
	for _, change := range []struct{ name, file, from, to string }{
		{"policy", "model-policies/test-worker.yaml", "temperature: 0.1", "temperature: 0.6"},
		{"workflow", "workflows/artifact_copy.yaml", "objective: Copy the source artifact into the declared result artifact", "objective: Copy source into result"},
		{"template", "agent-templates/artifact_builder.yaml", "description: Reads a declared input and writes the requested result", "description: Copy the declared source precisely"},
		{"gateway", "llm-gateways/local_litellm.yaml", "4000", "4100"},
		{"audit", "audit-profiles/common_tasks.yaml", "Inspect selected source", "Inspect source carefully"},
		{"instructions", "instructions/artifact-builder.md", "", "\nAdditional instruction.\n"},
	} {
		t.Run(change.name, func(t *testing.T) {
			root := copyCoreFixture(t, "audit-common-catalog")
			managed := filepath.Join(t.TempDir(), "managed")
			snapshot, err := UpdateCatalogLock(root, managed)
			if err != nil {
				t.Fatal(err)
			}
			if err := ValidateCatalogLock(root, snapshot, true); err != nil {
				t.Fatal(err)
			}
			manager := newTestManager(t, root, managed, ManagerOptions{})
			before := manager.Snapshot()
			lock := readFile(t, filepath.Join(root, catalogLockName))
			if change.from == "" {
				writeFile(t, filepath.Join(root, change.file), append(readFile(t, filepath.Join(root, change.file)), []byte(change.to)...))
			} else {
				replaceFile(t, filepath.Join(root, change.file), change.from, change.to)
			}
			if _, err := LoadUnionReadOnly(root, managed); err == nil || !strings.Contains(err.Error(), "catalog lock drift") {
				t.Fatalf("offline drift: %v", err)
			}
			if _, err := NewManager(ManagerOptions{OperatorRoot: root, ManagedRoot: managed}); err == nil {
				t.Fatal("startup accepted drift")
			}
			if _, err := UpdateCatalogLock(root, managed); err == nil {
				t.Fatal("generator rebound a locked identity")
			}
			if !bytes.Equal(lock, readFile(t, filepath.Join(root, catalogLockName))) {
				t.Fatal("rejected generator changed lock")
			}
			if _, err := manager.Publish(t.Context(), PublicationRequest{Kind: ConfigurationModelPolicies, Name: "extra", Version: "1", IdempotencyKey: "publish-extra", ModelPolicy: &ModelPolicyPublication{Model: "extra-model"}}); err == nil {
				t.Fatal("reload accepted drift")
			}
			if manager.Snapshot() != before {
				t.Fatal("rejected reload replaced current snapshot")
			}
		})
	}
}

func TestCatalogLockExtendsNewVersionsAndRetainsRetiredIdentities(t *testing.T) {
	root := copyCoreFixture(t)
	managed := filepath.Join(t.TempDir(), "managed")
	if _, err := UpdateCatalogLock(root, managed); err != nil {
		t.Fatal(err)
	}
	lock := readFile(t, filepath.Join(root, catalogLockName))
	if _, err := UpdateCatalogLock(root, managed); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(lock, readFile(t, filepath.Join(root, catalogLockName))) {
		t.Fatal("lock is not deterministic")
	}
	extra := filepath.Join(root, "model-policies/extra.yaml")
	data := bytes.ReplaceAll(readFile(t, filepath.Join(root, "model-policies/test-worker.yaml")), []byte("name: test-worker"), []byte("name: extra"))
	writeFile(t, extra, data)
	if _, err := LoadUnionReadOnly(root, managed); err == nil {
		t.Fatal("unlocked identity admitted")
	}
	if _, err := UpdateCatalogLock(root, managed); err != nil {
		t.Fatal(err)
	}
	extended := readFile(t, filepath.Join(root, catalogLockName))
	if bytes.Equal(lock, extended) {
		t.Fatal("new identity was not locked")
	}
	if err := os.Remove(extra); err != nil {
		t.Fatal(err)
	}
	if _, err := UpdateCatalogLock(root, managed); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(extended, readFile(t, filepath.Join(root, catalogLockName))) {
		t.Fatal("retired identity disappeared")
	}
	writeFile(t, extra, bytes.ReplaceAll(data, []byte("temperature: 0.1"), []byte("temperature: 0.6")))
	if _, err := UpdateCatalogLock(root, managed); err == nil {
		t.Fatal("retired identity was rebound")
	}
}

func TestCatalogLockRejectsMalformedMissingAndLinkedDocuments(t *testing.T) {
	for _, mode := range []string{"missing", "duplicate", "unknown field", "unknown kind", "bad digest", "oversized", "symlink"} {
		t.Run(mode, func(t *testing.T) {
			root := copyCoreFixture(t)
			managed := filepath.Join(t.TempDir(), "managed")
			snapshot, err := LoadUnionReadOnly(root, managed)
			if err != nil {
				t.Fatal(err)
			}
			path := filepath.Join(root, catalogLockName)
			switch mode {
			case "missing":
			case "duplicate":
				writeFile(t, path, []byte(`{"schema":"contractor.catalog-lock/v1","entries":{},"entries":{}}`))
			case "unknown field":
				writeFile(t, path, []byte(`{"schema":"contractor.catalog-lock/v1","entries":{},"extra":true}`))
			case "unknown kind":
				writeFile(t, path, []byte(`{"schema":"contractor.catalog-lock/v1","entries":{"other/x@1":"sha256:`+strings.Repeat("a", 64)+`"}}`))
			case "bad digest":
				writeFile(t, path, []byte(`{"schema":"contractor.catalog-lock/v1","entries":{"workflows/x@1":"bad"}}`))
			case "oversized":
				writeFile(t, path, bytes.Repeat([]byte(" "), maximumCatalogFileBytes+1))
			case "symlink":
				target := filepath.Join(t.TempDir(), "lock.json")
				writeFile(t, target, []byte(`{}`))
				if err := os.Symlink(target, path); err != nil {
					t.Fatal(err)
				}
			}
			if err := ValidateCatalogLock(root, snapshot, true); err == nil {
				t.Fatal("invalid required lock accepted")
			}
			if mode != "missing" {
				if _, err := LoadUnionReadOnly(root, managed); err == nil {
					t.Fatal("invalid present lock accepted")
				}
			}
		})
	}
}

func TestCatalogLockDoesNotBlockManagedPublication(t *testing.T) {
	root := copyCoreFixture(t)
	managed := filepath.Join(t.TempDir(), "managed")
	if _, err := UpdateCatalogLock(root, managed); err != nil {
		t.Fatal(err)
	}
	manager := newTestManager(t, root, managed, ManagerOptions{})
	lock := readFile(t, filepath.Join(root, catalogLockName))
	if _, err := manager.Publish(t.Context(), PublicationRequest{Kind: ConfigurationModelPolicies, Name: "managed-extra", Version: "1", IdempotencyKey: "publish-managed", ModelPolicy: &ModelPolicyPublication{Model: "extra-model"}}); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadUnionReadOnly(root, managed); err != nil {
		t.Fatal(err)
	}
	if _, err := UpdateCatalogLock(root, managed); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(lock, readFile(t, filepath.Join(root, catalogLockName))) {
		t.Fatal("managed publication changed operator lock")
	}
}
