package configload

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/configtest"
)

const catalogFixture = "../config/testdata/valid"

func writeFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestLoadValidatesBundledSkillsWithoutReadingManagedRoot(t *testing.T) {
	root := configtest.CopyWithPolicies(t, catalogFixture)
	skill := filepath.Join(root, "skills", "review", "SKILL.md")
	writeFile(t, skill, "---\nname: review\ndescription: Review guidance.\n---\n# Review\n")
	if _, err := Load(root, config.MVPDescriptors()); err != nil {
		t.Fatalf("valid bundled skill rejected: %v", err)
	}

	// Bundles are operator-owned: a skills directory in the managed root is
	// not a bundle and never reaches the validators.
	managed := filepath.Join(t.TempDir(), "managed")
	writeFile(t, filepath.Join(managed, "skills", "review", "SKILL.md"), "not a skill")
	if _, err := LoadUnionReadOnly(root, managed, config.MVPDescriptors()); err != nil {
		t.Fatalf("managed skills directory was validated as a bundle: %v", err)
	}

	secret := "PRIVATE-SKILL-INSTRUCTION"
	writeFile(t, skill, secret)
	for name, load := range map[string]func() error{
		"Load": func() error { _, err := Load(root, config.MVPDescriptors()); return err },
		"LoadUnionReadOnly": func() error {
			_, err := LoadUnionReadOnly(root, managed, config.MVPDescriptors())
			return err
		},
	} {
		err := load()
		if err == nil || !strings.HasPrefix(err.Error(), "bundled skills: ") ||
			strings.Contains(err.Error(), root) || strings.Contains(err.Error(), secret) ||
			!strings.Contains(err.Error(), "skill_manifest_invalid") {
			t.Fatalf("%s: unsafe bundled skill validation error: %v", name, err)
		}
	}
}

func TestLoadValidatesBundledAuditStandards(t *testing.T) {
	root := configtest.CopyWithPolicies(t, catalogFixture)
	writeFile(t, filepath.Join(root, "audit-standards", "broken", "standard.json"), "{}")
	_, err := Load(root, config.MVPDescriptors())
	if err == nil || !strings.HasPrefix(err.Error(), "bundled Audit standards: ") {
		t.Fatalf("invalid bundled Audit standard error = %v", err)
	}
}

func TestManagerRechecksBundlesBeforePublication(t *testing.T) {
	root := configtest.CopyWithPolicies(t, catalogFixture)
	managed := filepath.Join(t.TempDir(), "managed")
	manager, err := NewManager(config.ManagerOptions{
		OperatorRoot: root, ManagedRoot: managed, Descriptors: config.MVPDescriptors(),
	})
	if err != nil {
		t.Fatal(err)
	}
	writeFile(t, filepath.Join(root, "skills", "review", "SKILL.md"), "not a skill")
	_, err = manager.Publish(t.Context(), config.PublicationRequest{
		Kind: config.ConfigurationModelPolicies, Name: "ui-worker", Version: "2",
		IdempotencyKey: "after-bundle-change", ActorID: "user-1",
		ModelPolicy: &config.ModelPolicyPublication{Model: "qwen/new-model"},
	})
	if err == nil || !strings.HasPrefix(err.Error(), "reload configuration roots before publication: bundled skills: ") {
		t.Fatalf("publication after a bundle became invalid: %v", err)
	}
	if _, statErr := os.Stat(filepath.Join(managed, "model-policies", "ui-worker@2.yaml")); !errors.Is(statErr, os.ErrNotExist) {
		t.Fatalf("rejected publication wrote a manifest: %v", statErr)
	}
	if _, err := NewManager(config.ManagerOptions{
		OperatorRoot: root, ManagedRoot: managed, Descriptors: config.MVPDescriptors(),
	}); err == nil || !strings.HasPrefix(err.Error(), "bundled skills: ") {
		t.Fatalf("restart with an invalid bundle: %v", err)
	}
}
