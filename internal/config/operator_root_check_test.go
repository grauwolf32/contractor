package config

import (
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

type recordedRootCheck struct {
	roots []string
	fail  error
}

func (r *recordedRootCheck) check(subject string) OperatorRootCheck {
	return OperatorRootCheck{Subject: subject, Check: func(root string) error {
		r.roots = append(r.roots, root)
		return r.fail
	}}
}

func TestOperatorRootChecksRunOnlyOnTheResolvedOperatorRoot(t *testing.T) {
	operator := copyCoreFixture(t)
	managed := filepath.Join(t.TempDir(), "managed")
	if err := os.Mkdir(managed, 0o750); err != nil {
		t.Fatal(err)
	}
	resolved, err := filepath.EvalSymlinks(operator)
	if err != nil {
		t.Fatal(err)
	}
	recorded := &recordedRootCheck{}
	if _, err := LoadUnionReadOnly(operator, managed, MVPDescriptors(), recorded.check("bundle")); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(operator, MVPDescriptors(), recorded.check("bundle")); err != nil {
		t.Fatal(err)
	}
	if want := []string{resolved, resolved}; !slices.Equal(recorded.roots, want) {
		t.Fatalf("checked roots = %v, want %v", recorded.roots, want)
	}
}

func TestOperatorRootCheckFailureRejectsLoadBeforeManifests(t *testing.T) {
	operator := copyCoreFixture(t)
	if err := os.WriteFile(filepath.Join(operator, "workflows", "broken.yaml"), []byte("kind: ["), 0o600); err != nil {
		t.Fatal(err)
	}
	failing := &recordedRootCheck{fail: errors.New("package_invalid")}
	_, err := Load(operator, MVPDescriptors(), failing.check("bundled things"))
	if err == nil || err.Error() != "bundled things: package_invalid" {
		t.Fatalf("load error = %v", err)
	}
	if _, err := Load(operator, MVPDescriptors()); err == nil || strings.Contains(err.Error(), "bundled things") {
		t.Fatalf("catalog error without checks = %v", err)
	}
}

func TestManagerRunsOperatorRootChecksOnStartupAndReload(t *testing.T) {
	operator := copyCoreFixture(t)
	managed := filepath.Join(t.TempDir(), "managed")
	recorded := &recordedRootCheck{}
	manager := newTestManager(t, operator, managed, ManagerOptions{
		OperatorRootChecks: []OperatorRootCheck{recorded.check("bundle")},
	})
	if len(recorded.roots) != 1 {
		t.Fatalf("startup checks = %v", recorded.roots)
	}
	recorded.fail = errors.New("package_invalid")
	_, err := manager.Publish(t.Context(), validPolicyPublication("checked-reload"))
	if err == nil || err.Error() != "reload configuration roots before publication: bundle: package_invalid" {
		t.Fatalf("publication error = %v", err)
	}
	if _, statErr := os.Stat(filepath.Join(managed, "model-policies", "ui-worker@2.yaml")); !errors.Is(statErr, os.ErrNotExist) {
		t.Fatalf("rejected publication wrote a manifest: %v", statErr)
	}
	if _, err := NewManager(ManagerOptions{
		OperatorRoot: operator, ManagedRoot: managed, Descriptors: MVPDescriptors(),
		OperatorRootChecks: []OperatorRootCheck{recorded.check("bundle")},
	}); err == nil || err.Error() != "bundle: package_invalid" {
		t.Fatalf("startup error = %v", err)
	}
}
