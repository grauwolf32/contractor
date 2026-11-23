package contextcompaction

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/configload"
)

func TestAuthoredContextVariantsResolveThroughRealCatalog(t *testing.T) {
	_, source, _, _ := runtime.Caller(0)
	root := filepath.Clean(filepath.Join(filepath.Dir(source), "../../.."))
	catalogRoot := t.TempDir()
	if err := os.CopyFS(catalogRoot, os.DirFS(filepath.Join(root, "configs/e2e"))); err != nil {
		t.Fatal(err)
	}
	if err := os.CopyFS(catalogRoot, os.DirFS(filepath.Join(filepath.Dir(source), "catalog"))); err != nil {
		t.Fatal(err)
	}
	snapshot, _, err := configload.LoadUnionReadOnly(catalogRoot, filepath.Join(t.TempDir(), "managed"))
	if err != nil {
		t.Fatal(err)
	}
	for _, variant := range []string{"terminal-off", "terminal-on", "compact-off", "compact-on", "terminal-handoff", "controlled-compact", "masking"} {
		workflow, err := snapshot.Workflow("context-eval-" + variant + "@1")
		if err != nil {
			t.Fatal(err)
		}
		if len(workflow.Stages) != 1 {
			t.Fatal("pilot must preserve one authored Stage")
		}
		for _, stage := range workflow.Stages {
			binding := stage.Agents["builder"]
			if strings.Contains(variant, "compact") {
				if !binding.ContextManagement.IsCompact() || stage.ExecutionConfig.Agents["builder"].SummarizerModelPolicy != nil || stage.ExecutionConfig.Agents["builder"].CompactorModelPolicy == nil {
					t.Fatal("compact selection drift")
				}
			} else if variant != "masking" && (binding.Template.Summarizer == nil || stage.ExecutionConfig.Agents["builder"].SummarizerModelPolicy == nil) {
				t.Fatal("terminal selection drift")
			}
		}
	}
}
