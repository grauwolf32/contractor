package config

import (
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/config/resolved"
)

const auditCompactBinding = `          contextManagement:
            strategy: compact@1
            modelPolicy: test-compactor@1
            instructions: {ref: instructions/worker-compaction.md}
            includeReasoning: false
`

func compactAuditFixture(t *testing.T) string {
	t.Helper()
	root := copyCoreFixture(t, "audit-completion-catalog")
	for _, name := range []string{"model-policies/test-compactor.yaml", "model-policies/test-compact-worker.yaml", "model-policies/test-summarizer.yaml", "instructions/worker-compaction.md"} {
		writeFile(t, filepath.Join(root, name), readFile(t, filepath.Join("testdata/context-compaction", name)))
	}
	path := filepath.Join(root, "workflows/audit_source_check.yaml")
	source := string(readFile(t, path))
	source = strings.Replace(source, "modelPolicy: audit_completion_worker@1", "modelPolicy: test-compact-worker@1", 1)
	source = strings.Replace(source, "          namespace: audit-check\n", "          namespace: audit-check\n"+auditCompactBinding, 1)
	writeFile(t, path, []byte(source))
	path = filepath.Join(root, "agent-templates/audit_source_checker.yaml")
	source = strings.Replace(string(readFile(t, path)), "  runtime: adk@1", "  runtime: adk@1\n  summarizer: {contextWindowRatio: 0.9}", 1)
	writeFile(t, path, []byte(source))
	return root
}

func TestAuditCompactPolicyOverridesSummarizerAcrossCatalogAndSnapshots(t *testing.T) {
	root := compactAuditFixture(t)
	path := filepath.Join(root, "workflows/audit_source_check.yaml")
	source := string(readFile(t, path))
	source = strings.Replace(source, "        failed:\n          fail: {}", `        failed:
          escalate:
            maxAttempts: 1
            executionConfig:
              agents:
                checker: {modelPolicy: test-compact-worker@1}
            then: {fail: {}}`, 1)
	writeFile(t, path, []byte(source))
	snapshot := mustLoad(t, root)
	pin, err := snapshot.AuditProfile("source-checklist@1")
	if err != nil {
		t.Fatal(err)
	}
	workflow := pin.Workflows["check"]
	stage := workflow.Stages["check"]
	if stage.Agents["checker"].Template.Summarizer == nil || !stage.Agents["checker"].ContextManagement.IsCompact() || stage.On.Failed.Escalate == nil {
		t.Fatal("compact policy rewrote its template or lost a reachable variant")
	}
	if target, err := auditprofile.ResolveCompletionTarget(workflow, pin.Program.Workflows["check"]); err != nil || target == nil || target.Agent != "checker" {
		t.Fatalf("completion target=%+v error=%v", target, err)
	}
	data, err := json.Marshal(pin)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := auditprofile.DecodePinnedProgram(data); err != nil {
		t.Fatalf("strict Audit pin read: %v", err)
	}
	data, err = json.Marshal(workflow)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := resolved.DecodeResolvedWorkflowSnapshot(data); err != nil {
		t.Fatalf("strict Workflow read: %v", err)
	}
	stage.Session = "shared"
	workflow.Stages["check"] = stage
	data, err = json.Marshal(workflow)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := resolved.DecodeResolvedWorkflowSnapshot(data); err == nil {
		t.Fatal("snapshot read accepted shared Audit compaction")
	}
}

func TestAuditCompactRejectsTerminalSharedAndEscalationSummarizer(t *testing.T) {
	for _, name := range []string{"shared", "terminal", "failed escalation", "interrupted escalation"} {
		t.Run(name, func(t *testing.T) {
			root := compactAuditFixture(t)
			path := filepath.Join(root, "workflows/audit_source_check.yaml")
			source := string(readFile(t, path))
			switch name {
			case "shared":
				source = strings.Replace(source, "      planner: passthrough@1", "      planner: passthrough@1\n      session: shared", 1)
			case "terminal":
				source = strings.Replace(source, auditCompactBinding, "          contextManagement: {strategy: template@1}\n", 1)
				source = strings.Replace(source, "      modelPolicy: test-compact-worker@1", "      modelPolicy: test-compact-worker@1\n      summarizerModelPolicy: test-summarizer@1", 1)
			default:
				outcome := strings.Fields(name)[0]
				source = strings.Replace(source, "        "+outcome+":\n          fail: {}", "        "+outcome+`:
          escalate:
            maxAttempts: 1
            executionConfig:
              agents:
                checker: {summarizerModelPolicy: test-summarizer@1}
            then: {fail: {}}`, 1)
			}
			writeFile(t, path, []byte(source))
			if _, err := LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "managed")); err == nil {
				t.Fatal("unsupported Audit context selection accepted")
			}
		})
	}
}
