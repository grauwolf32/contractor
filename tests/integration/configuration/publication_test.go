package configuration_test

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"testing"

	"github.com/grauwolf32/contractor/internal/config"
)

// This black-box test pins the durable crash-window contract outside the
// config package: a response is not reported as successful before the snapshot
// swap, while the already-renamed YAML is ordinary startup input.
func TestManagedPublicationRecoversRenameBeforeSnapshotSwap(t *testing.T) {
	operator := copyTree(t, "../../../configs")
	managed := filepath.Join(t.TempDir(), "managed")
	simulatedCrash := errors.New("simulated process crash")
	manager, err := config.NewManager(config.ManagerOptions{
		OperatorRoot: operator, ManagedRoot: managed, Descriptors: config.MVPDescriptors(),
		AfterDurablePublish: func(config.ConfigurationResource) error { return simulatedCrash },
	})
	if err != nil {
		t.Fatal(err)
	}
	request := config.PublicationRequest{
		Kind: config.ConfigurationModelPolicies, Name: "restart-policy", Version: "1",
		IdempotencyKey: "restart-publication",
		ModelPolicy:    &config.ModelPolicyPublication{Model: "restart-model"},
	}
	if _, err := manager.Publish(t.Context(), request); !errors.Is(err, simulatedCrash) {
		t.Fatalf("crash-window publication error = %v", err)
	}
	if _, err := manager.Configuration(config.ConfigurationModelPolicies, "restart-policy@1"); !errors.Is(err, config.ErrConfigurationNotFound) {
		t.Fatalf("pre-crash snapshot changed: %v", err)
	}

	restarted, err := config.NewManager(config.ManagerOptions{
		OperatorRoot: operator, ManagedRoot: managed, Descriptors: config.MVPDescriptors(),
	})
	if err != nil {
		t.Fatal(err)
	}
	resource, err := restarted.Configuration(config.ConfigurationModelPolicies, "restart-policy@1")
	if err != nil || resource.Source != config.ConfigurationSourceManaged {
		t.Fatalf("restart resource = %+v, error %v", resource, err)
	}
	replay, err := restarted.Publish(t.Context(), request)
	if err != nil || !replay.Replayed || replay.Resource.Ref != resource.Ref {
		t.Fatalf("restart replay = %+v, error %v", replay, err)
	}
}

func copyTree(t *testing.T, source string) string {
	t.Helper()
	destination := filepath.Join(t.TempDir(), "operator")
	err := filepath.WalkDir(source, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		relative, err := filepath.Rel(source, path)
		if err != nil {
			return err
		}
		target := filepath.Join(destination, relative)
		if entry.IsDir() {
			return os.MkdirAll(target, 0o755)
		}
		contents, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		return os.WriteFile(target, contents, 0o644)
	})
	if err != nil {
		t.Fatalf("copy configuration tree: %v", err)
	}
	return destination
}
