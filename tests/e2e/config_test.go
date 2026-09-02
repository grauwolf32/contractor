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
// It changes only the test Gateway endpoint. Role-specific development
// credentials remain authored in the source Workflow manifests so the tests
// exercise the same executionConfig that the demo deployment loads.
func stageE2EConfiguration(t *testing.T, source, target, gatewayURL string) string {
	t.Helper()
	encodedURL, err := json.Marshal(gatewayURL)
	if err != nil {
		t.Fatal(err)
	}
	gatewayUpdated := false
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
		return os.WriteFile(destination, []byte(text), 0o600)
	})
	if err != nil {
		t.Fatalf("stage E2E configuration: %v", err)
	}
	if !gatewayUpdated {
		t.Fatal("staged configuration did not update the Gateway URL")
	}
	return target
}
