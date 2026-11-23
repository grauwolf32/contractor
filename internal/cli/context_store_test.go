package cli

import (
	"os"
	"path/filepath"
	"testing"
)

func TestContextStoreRoundTripAndSelection(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nested", "config.json")
	store := NewContextStore(path)
	config, err := store.Put("local", ServerContext{Server: "http://127.0.0.1:8080"}, false)
	if err != nil {
		t.Fatal(err)
	}
	if config.CurrentContext != "local" {
		t.Fatalf("current = %q", config.CurrentContext)
	}
	_, context, err := config.Resolve("")
	if err != nil || context.Server != "http://127.0.0.1:8080" {
		t.Fatalf("context=%+v err=%v", context, err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm() != 0o600 {
		t.Fatalf("mode = %o", info.Mode().Perm())
	}
}

func TestContextStoreRejectsUnknownFields(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(path, []byte(`{"contexts":{},"unexpected":true}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := NewContextStore(path).Load(); err == nil {
		t.Fatal("unknown context field was accepted")
	}
}

func TestContextStoreRejectsSymlink(t *testing.T) {
	directory := t.TempDir()
	target := filepath.Join(directory, "target.json")
	path := filepath.Join(directory, "config.json")
	if err := os.WriteFile(target, []byte(`{"contexts":{}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(target, path); err != nil {
		t.Skipf("symlink unavailable: %v", err)
	}
	if _, err := NewContextStore(path).Load(); err == nil {
		t.Fatal("symlink context was accepted")
	}
}
