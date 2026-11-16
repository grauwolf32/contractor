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

	"github.com/grauwolf32/contractor/internal/configload"
	"go.yaml.in/yaml/v4"
)

// stageE2EConfiguration creates an immutable config root before Server starts.
// It changes only the test Gateway endpoint. Role-specific development
// credentials remain authored in the source Workflow manifests so the tests
// exercise the same executionConfig that the demo deployment loads.
func stageE2EConfiguration(t *testing.T, source, target, gatewayURL string) string {
	t.Helper()
	if _, err := os.Lstat(filepath.Join(source, "catalog.lock.json")); err == nil {
		if _, _, err := configload.LoadUnionReadOnly(source, filepath.Join(t.TempDir(), "managed")); err != nil {
			t.Fatalf("locked source configuration: %v", err)
		}
	} else if !os.IsNotExist(err) {
		t.Fatal(err)
	}
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
		// Source locks were validated above. The private process catalog uses
		// a local Gateway and accepts scenario-specific manifests before startup.
		if relative == "catalog.lock.json" {
			return nil
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

// Ordinary finding provenance is independent of Audit result completion. Keep
// its process fixture in the temporary catalog, with no Audit tool selection.
func installOrdinaryFindingFixture(t *testing.T, root, templateFile, workflowFile, name string) {
	t.Helper()
	read := func(folder, file string) string {
		data, err := os.ReadFile(filepath.Join(root, folder, file+".yaml"))
		if err != nil {
			t.Fatal(err)
		}
		return string(data)
	}
	template := read("agent-templates", templateFile)
	// This fixture tests the generic finding facade independently of tracing.
	template = strings.ReplaceAll(template, "security-findings-trace@1", "security-findings-code@1")
	template = strings.ReplaceAll(template, "    - ref: taint-annotations@1\n      tools: [annotate_trace, annotate_validate, annotate_sink]\n", "")
	template = strings.ReplaceAll(template, "    - ref: workspace-changes@1\n      tools: [changed_paths, diff, rollback_changes]\n", "")
	template = strings.Replace(template, "  name: "+templateFile+"\n", "  name: "+name+"\n", 1)
	template = strings.Replace(template, "    - ref: audit-results@2\n      tools: [read_audit_task, submit_check_result]\n", "", 1)
	if !strings.Contains(template, "ref: text-artifacts@1") {
		template = strings.Replace(template, "  toolsets:\n", "  toolsets:\n    - ref: text-artifacts@1\n      tools: [write_text_artifact]\n", 1)
	}
	workflow := read("workflows", workflowFile)
	workflow = strings.Replace(workflow, "  name: "+strings.ReplaceAll(workflowFile, "_", "-")+"\n", "  name: "+name+"\n", 1)
	workflow = strings.Replace(workflow, "template: "+templateFile+"@1", "template: "+name+"@1", 1)
	var document map[string]any
	if err := yaml.Unmarshal([]byte(workflow), &document); err != nil {
		t.Fatal(err)
	}
	spec := document["spec"].(map[string]any)
	spec["inputs"] = map[string]any{"source": map[string]any{"required": true, "mediaTypes": []string{"application/zip"}}}
	spec["outputs"] = map[string]any{"result": map[string]any{"required": true, "primary": true, "mediaTypes": []string{"text/plain"}}}
	stage := spec["stages"].(map[string]any)["check"].(map[string]any)
	context := stage["context"].(map[string]any)
	context["artifacts"] = map[string]any{"source": map[string]any{"namespace": "inputs", "name": "source", "required": true}}
	if workspace, ok := context["workspace"].(map[string]any); ok {
		delete(workspace, "export")
	}
	stage["workflowOutputs"] = map[string]string{"result": "result"}
	stage["result"] = map[string]any{"artifacts": map[string]any{"result": map[string]any{"required": true, "mediaTypes": []string{"text/plain"}, "from": map[string]string{"namespace": "audit-check", "name": "result"}}}}
	encoded, err := yaml.Marshal(document)
	if err != nil {
		t.Fatal(err)
	}
	workflow = string(encoded)
	replaceInstructions := func(content string) string {
		lines := strings.Split(content, "\n")
		for index, line := range lines {
			if strings.HasPrefix(strings.TrimSpace(line), "ref: instructions/") {
				lines[index] = line[:len(line)-len(strings.TrimLeft(line, " "))] + "ref: instructions/" + name + ".md"
			}
		}
		return strings.Join(lines, "\n")
	}
	template, workflow = replaceInstructions(template), replaceInstructions(workflow)
	if err := os.WriteFile(filepath.Join(root, "instructions", name+".md"), []byte("Inspect the supplied source and record evidence-backed finding proposals. Publish a plain-text result artifact in the assigned namespace. This ordinary Run does not collect Audit assessments.\n"), 0600); err != nil {
		t.Fatal(err)
	}
	for folder, content := range map[string]string{"agent-templates": template, "workflows": workflow} {
		if err := os.WriteFile(filepath.Join(root, folder, name+".yaml"), []byte(content), 0600); err != nil {
			t.Fatal(err)
		}
	}
}
