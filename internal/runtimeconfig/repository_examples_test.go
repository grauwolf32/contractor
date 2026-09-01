package runtimeconfig

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"testing"

	"go.yaml.in/yaml/v4"
)

func TestRepositoryRuntimeConfigExamplesNormalizeWithoutSecrets(t *testing.T) {
	t.Parallel()

	path := filepath.Join("..", "..", "deploy", "runtime-labels", "runtime-configs.example.yaml")
	contents, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	decoder := yaml.NewDecoder(bytes.NewReader(contents))
	versions := make(map[string]Version)
	for {
		var document map[string]any
		if err := decoder.Decode(&document); errors.Is(err, io.EOF) {
			break
		} else if err != nil {
			t.Fatalf("decode %s: %v", path, err)
		}
		if len(document) == 0 {
			continue
		}
		encoded, err := json.Marshal(document)
		if err != nil {
			t.Fatal(err)
		}
		prepared, err := PreparePublication(encoded)
		if err != nil {
			t.Fatalf("prepare RuntimeConfig example: %v", err)
		}
		version, err := prepared.Resolve(context.Background(), nil)
		if err != nil {
			t.Fatalf("resolve RuntimeConfig example: %v", err)
		}
		versions[version.Ref.Name+"@"+version.Ref.Version] = version
	}
	if len(versions) != 3 {
		t.Fatalf("RuntimeConfig examples = %d, want 3", len(versions))
	}
	caido, ok := versions["caido-analysis@1"]
	if !ok {
		t.Fatal("caido-analysis@1 RuntimeConfig example is missing")
	}
	proxy := caido.Spec.Worker.HTTPProxy
	adapter := caido.Spec.Worker.Caido
	if !proxy.Present || proxy.Clear || proxy.Value.Adapter != "http-proxy@1" ||
		len(proxy.Value.Targets) != 1 || proxy.Value.Targets[0] != "tool-http" ||
		proxy.Value.Credential != "" {
		t.Fatalf("caido-analysis@1 HTTP proxy = %+v", proxy)
	}
	if !adapter.Present || adapter.Clear || adapter.Value.Adapter != "caido-graphql@1" ||
		adapter.Value.Credential != "caido-api" || adapter.Value.RequestTimeoutSeconds != 30 {
		t.Fatalf("caido-analysis@1 Caido adapter = %+v", adapter)
	}
	for _, secretCanary := range []string{"replace-me", "change-me", "secret-canary", "password", "bearer-token"} {
		if bytes.Contains(bytes.ToLower(contents), []byte(secretCanary)) {
			t.Errorf("RuntimeConfig examples contain secret canary %q", secretCanary)
		}
	}
}
