package config

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func auditCompletionConfig(t *testing.T) (string, auditprofile.ResolvedAuditProfile) {
	t.Helper()
	root := copyCoreFixture(t, "audit-completion-catalog")
	path := filepath.Join(root, "audit-profiles/source_checklist.yaml")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	// The fixture profile enables completion. Remove it only in this test copy
	// to prove that the trusted binding contributes to the immutable digest.
	without := strings.Replace(string(raw), "      workerCompletion: {kind: audit-check-results@1, stage: check, agent: checker}\n", "", 1)
	writeFile(t, path, []byte(without))
	before, err := mustLoad(t, root).AuditProfile("source-checklist@1")
	if err != nil {
		t.Fatal(err)
	}
	writeFile(t, path, raw)
	return root, before
}

func TestAuditCompletionOptInDigestSnapshotAndClone(t *testing.T) {
	root, before := auditCompletionConfig(t)
	snapshot := mustLoad(t, root)
	profile, err := snapshot.AuditProfile("source-checklist@1")
	if err != nil {
		t.Fatal(err)
	}
	if before.Workflows["check"].WorkerCompletion != nil || before.Ref.Digest == profile.Ref.Digest {
		t.Fatal("completion choice must alter only explicit opt-in digest")
	}
	raw, err := json.Marshal(profile)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := auditprofile.DecodeResolvedAuditProfileSnapshot(raw)
	if err != nil {
		t.Fatal(err)
	}
	if decoded.Workflows["check"].WorkerCompletion.Kind != contracts.AuditCheckResultsV1 {
		t.Fatal("snapshot lost contract")
	}
	profile.Workflows["check"].WorkerCompletion.Stage = "changed"
	again, err := snapshot.AuditProfile("source-checklist@1")
	if err != nil {
		t.Fatal(err)
	}
	if again.Workflows["check"].WorkerCompletion.Stage != "check" {
		t.Fatal("snapshot shares completion pointer")
	}
	// A reader cannot discard the pinned completion choice under the same digest.
	binding := decoded.Workflows["check"]
	binding.WorkerCompletion = nil
	decoded.Workflows["check"] = binding
	raw, err = json.Marshal(decoded)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := auditprofile.DecodeResolvedAuditProfileSnapshot(raw); err == nil {
		t.Fatal("snapshot accepted a removed completion choice without a matching digest")
	}
}

func TestAuditCompletionValidatesReachableEscalationConfiguration(t *testing.T) {
	root, _ := auditCompletionConfig(t)
	profile, err := mustLoad(t, root).AuditProfile("source-checklist@1")
	if err != nil {
		t.Fatal(err)
	}
	b := profile.Workflows["check"]
	s := b.Workflow.Stages["check"]
	policy := s.ExecutionConfig.Agents["checker"].ModelPolicy.Clone()
	variant := resolved.ResolvedEscalationExecutionConfig{Override: resolved.ResolvedStageExecutionConfigOverride{Agents: map[string]resolved.ResolvedExecutionSelectionOverride{"checker": {ModelPolicy: &policy}}}}
	s.On.Failed = resolved.TransitionAction{Kind: resolved.TransitionEscalate, Escalate: &resolved.EscalateTransition{MaxAttempts: 1, ExecutionConfig: variant, Then: resolved.TransitionAction{Kind: resolved.TransitionFail}}}
	b.Workflow.Stages["check"] = s
	// The escalation's effective configuration is the one a load pins.
	if err := resolved.ResolveEscalationVariants(&b.Workflow); err != nil {
		t.Fatal(err)
	}
	action := b.Workflow.Stages["check"].On.Failed
	if err := auditprofile.ValidateAuditWorkerCompletion(b); err != nil {
		t.Fatal(err)
	}
	delete(action.Escalate.ExecutionConfig.Effective.Agents, "checker")
	if err := auditprofile.ValidateAuditWorkerCompletion(b); err == nil {
		t.Fatal("escalated target lost its effective Worker configuration")
	}
}

func TestAuditCompletionAuthoringIsClosed(t *testing.T) {
	root, _ := auditCompletionConfig(t)
	path := filepath.Join(root, "audit-profiles/source_checklist.yaml")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	writeFile(t, path, []byte(strings.Replace(string(raw), "agent: checker}", "agent: checker, model_override: true}", 1)))
	if _, err := Load(root); err == nil {
		t.Fatal("unknown completion authoring field accepted")
	}
}

func TestAuditCompletionLoadsAuthoredEscalationWithoutChangingOwnership(t *testing.T) {
	root, _ := auditCompletionConfig(t)
	path := filepath.Join(root, "workflows/audit_source_check.yaml")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	replacement := `        failed:
          escalate:
            maxAttempts: 1
            executionConfig:
              agents:
                checker:
                  modelPolicy: test-strong-worker@1
            then:
              fail: {}`
	writeFile(t, path, []byte(strings.Replace(string(raw), "        failed:\n          fail: {}", replacement, 1)))
	profile, err := mustLoad(t, root).AuditProfile("source-checklist@1")
	if err != nil {
		t.Fatal(err)
	}
	if err := auditprofile.ValidateAuditWorkerCompletion(profile.Workflows["check"]); err != nil {
		t.Fatal(err)
	}
}

func TestAuditCompletionRejectsRetiredToolset(t *testing.T) {
	root := copyCoreFixture(t, "audit-completion-catalog")
	path := filepath.Join(root, "agent-templates/audit_source_checker.yaml")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	writeFile(t, path, []byte(strings.Replace(string(raw), "audit-results@2", "audit-results@1", 1)))
	if _, err := Load(root); err == nil || !strings.Contains(err.Error(), `unknown Toolset "audit-results@1"`) {
		t.Fatalf("retired Audit toolset must be rejected during catalog loading: %v", err)
	}
}

// Catalog authoring lives in fixtures now, rather than the removed bundled
// production profiles. Keep their native completion tools and exact binding
// executable in the mandatory release gate.
func TestAuditCompletionFixturesPinNativeWorkerTools(t *testing.T) {
	snapshot := mustLoad(t, copyCoreFixture(t, "audit-completion-catalog"))
	checked := 0
	for _, profile := range snapshot.AuditProfiles() {
		for role, binding := range profile.Workflows {
			if binding.WorkerCompletion == nil {
				continue
			}
			checked++
			if err := auditprofile.ValidateAuditWorkerCompletion(binding); err != nil {
				t.Fatalf("%s/%s: %v", profile.Ref.Name, role, err)
			}
			completion := binding.WorkerCompletion
			agent := binding.Workflow.Stages[completion.Stage].Agents[completion.Agent]
			found := false
			for _, toolset := range agent.Template.Toolsets {
				if toolset.Ref.ToolsetID != "audit-results" {
					continue
				}
				if found || toolset.Ref.Version != "2" || !slices.Equal(toolset.Tools, []string{"read_audit_task", "submit_check_result"}) {
					t.Fatalf("%s/%s has an unexpected completion toolset: %+v", profile.Ref.Name, role, toolset)
				}
				found = true
			}
			if !found {
				t.Fatalf("%s/%s has no native completion tools", profile.Ref.Name, role)
			}
		}
	}
	if checked == 0 {
		t.Fatal("completion fixture selected no opted-in Workers")
	}
}
