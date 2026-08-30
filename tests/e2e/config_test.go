//go:build e2e

package e2e

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// stageE2EConfiguration creates an immutable config root before Server starts.
// The test Gateway endpoint and the explicit development credential are normal
// authored selections; process environment never overrides a pinned Run.
func stageE2EConfiguration(t *testing.T, source, target, gatewayURL string) string {
	t.Helper()
	encodedURL, err := json.Marshal(gatewayURL)
	if err != nil {
		t.Fatal(err)
	}
	gatewayUpdated := false
	workflowUpdates := 0
	err = filepath.WalkDir(source, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		relative, err := filepath.Rel(source, path)
		if err != nil {
			return err
		}
		destination := filepath.Join(target, relative)
		if entry.Type()&os.ModeSymlink != 0 {
			return fmt.Errorf("configuration contains a symbolic link")
		}
		if entry.IsDir() {
			return os.MkdirAll(destination, 0o700)
		}
		if !entry.Type().IsRegular() {
			return fmt.Errorf("configuration contains a non-regular entry")
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		text := string(data)
		if relative == filepath.Join("llm-gateways", "local_litellm.yaml") {
			lines := strings.Split(text, "\n")
			for index, line := range lines {
				if strings.HasPrefix(line, "  url: ") {
					lines[index] = "  url: " + string(encodedURL)
					gatewayUpdated = true
				}
			}
			text = strings.Join(lines, "\n")
		}
		if strings.HasPrefix(relative, "workflows"+string(filepath.Separator)) {
			const gatewaySelection = "      llmGateway: local-litellm@1\n"
			const credentialSelection = gatewaySelection + "      credential: development-worker\n"
			if strings.Contains(text, gatewaySelection) {
				text = strings.ReplaceAll(text, gatewaySelection, credentialSelection)
				workflowUpdates++
			}
		}
		return os.WriteFile(destination, []byte(text), 0o600)
	})
	if err != nil {
		t.Fatalf("stage E2E configuration: %v", err)
	}
	if !gatewayUpdated || workflowUpdates == 0 {
		t.Fatalf("staged config updates = gateway:%t workflows:%d", gatewayUpdated, workflowUpdates)
	}
	return target
}
