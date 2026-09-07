package config

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func auditCompletionConfig(t *testing.T) (string, ResolvedAuditProfile) {
	t.Helper()
	root := copyConfigTree(t)
	path := filepath.Join(root, "agent-templates/audit_source_checker.yaml")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	writeFile(t, path, []byte(strings.Replace(string(raw), "ref: audit-results@1", "ref: audit-results@2", 1)))
	before, err := mustLoad(t, root, MVPDescriptors()).AuditProfile("source-checklist@1")
	if err != nil {
		t.Fatal(err)
	}
	path = filepath.Join(root, "audit-profiles/source-checklist.yaml")
	raw, err = os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	writeFile(t, path, []byte(strings.Replace(string(raw), "      ref: audit-source-check@1", "      ref: audit-source-check@1\n      workerCompletion: {kind: audit-check-results@1, stage: check, agent: checker}", 1)))
	return root, before
}

func TestAuditCompletionOptInDigestSnapshotAndClone(t *testing.T) {
	root, before := auditCompletionConfig(t)
	snapshot := mustLoad(t, root, MVPDescriptors())
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
	decoded, err := DecodeResolvedAuditProfileSnapshot(raw)
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
	if _, err := auditProfileLegacyDigest(Selector{ID: profile.Ref.Name, Version: profile.Ref.Version}, profile); err == nil {
		t.Fatal("legacy digest accepted a new completion choice")
	}
}

func TestAuditCompletionRejectsInvalidTargetsAndTrustedMappings(t *testing.T) {
	root, _ := auditCompletionConfig(t)
	profile, err := mustLoad(t, root, MVPDescriptors()).AuditProfile("source-checklist@1")
	if err != nil {
		t.Fatal(err)
	}
	cases := map[string]func(*ResolvedAuditWorkflowBinding){
		"unknown-kind":     func(b *ResolvedAuditWorkflowBinding) { b.WorkerCompletion.Kind = "unknown@1" },
		"non-check":        func(b *ResolvedAuditWorkflowBinding) { b.Kind = AuditWorkflowDiscovery },
		"missing-stage":    func(b *ResolvedAuditWorkflowBinding) { b.WorkerCompletion.Stage = "absent" },
		"missing-agent":    func(b *ResolvedAuditWorkflowBinding) { b.WorkerCompletion.Agent = "absent" },
		"wrong-output":     func(b *ResolvedAuditWorkflowBinding) { delete(b.Outputs, "result") },
		"missing-task":     func(b *ResolvedAuditWorkflowBinding) { delete(b.Inputs, "task") },
		"missing-manifest": func(b *ResolvedAuditWorkflowBinding) { delete(b.Inputs, "execution_manifest") },
		"router": func(b *ResolvedAuditWorkflowBinding) {
			s := b.Workflow.Stages["check"]
			s.Planner = PlannerRef{PlannerID: "router", Version: "1"}
			b.Workflow.Stages["check"] = s
		},
		"two-workers": func(b *ResolvedAuditWorkflowBinding) {
			s := b.Workflow.Stages["check"]
			s.Agents["other"] = s.Agents["checker"]
		},
		"legacy-tool": func(b *ResolvedAuditWorkflowBinding) {
			s := b.Workflow.Stages["check"]
			a := s.Agents["checker"]
			for i := range a.Template.Toolsets {
				if a.Template.Toolsets[i].Ref.ToolsetID == "audit-results" {
					a.Template.Toolsets[i].Ref.Version = "1"
				}
			}
			s.Agents["checker"] = a
		},
		"summarizer": func(b *ResolvedAuditWorkflowBinding) {
			s := b.Workflow.Stages["check"]
			a := s.Agents["checker"]
			a.Template.Summarizer = &contracts.WorkerSummarizerConfig{}
			s.Agents["checker"] = a
		},
		"foreign-result": func(b *ResolvedAuditWorkflowBinding) {
			s := b.Workflow.Stages["check"]
			r := s.Result.Artifacts["result"]
			r.From.Namespace = "foreign"
			s.Result.Artifacts["result"] = r
		},
		"optional-input": func(b *ResolvedAuditWorkflowBinding) {
			s := b.Workflow.Stages["check"]
			v := s.Context.Artifacts["task"]
			v.Required = false
			s.Context.Artifacts["task"] = v
		},
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			copy := cloneAuditProfile(profile)
			binding := copy.Workflows["check"]
			mutate(&binding)
			if err := ValidateAuditWorkerCompletion(binding); err == nil {
				t.Fatal("invalid completion binding accepted")
			}
			copy.Workflows["check"] = binding
			copy.Ref.Digest, _ = auditProfileDigest(Selector{ID: copy.Ref.Name, Version: copy.Ref.Version}, copy)
			raw, _ := json.Marshal(copy)
			if _, err := DecodeResolvedAuditProfileSnapshot(raw); err == nil {
				t.Fatal("invalid immutable completion closure accepted")
			}
		})
	}
}

func TestAuditCompletionValidatesReachableEscalationConfiguration(t *testing.T) {
	root, _ := auditCompletionConfig(t)
	profile, err := mustLoad(t, root, MVPDescriptors()).AuditProfile("source-checklist@1")
	if err != nil {
		t.Fatal(err)
	}
	b := profile.Workflows["check"]
	s := b.Workflow.Stages["check"]
	policy := cloneModelPolicy(s.ExecutionConfig.Agents["checker"].ModelPolicy)
	variant := ResolvedEscalationExecutionConfig{Override: ResolvedStageExecutionConfigOverride{Agents: map[string]ResolvedExecutionSelectionOverride{"checker": {ModelPolicy: &policy}}}}
	effective := s
	effective.ExecutionConfig = cloneStageExecutionConfig(s.ExecutionConfig)
	if err := applyResolvedStageExecutionConfigOverride(&effective, variant.Override, escalationExecutionConfigOrigin("check", "failed", variant)); err != nil {
		t.Fatal(err)
	}
	variant.Effective = effective.ExecutionConfig
	action := TransitionAction{Kind: TransitionEscalate, Escalate: &EscalateTransition{MaxAttempts: 1, ExecutionConfig: variant, Then: TransitionAction{Kind: TransitionFail}}}
	s.On.Failed = action
	b.Workflow.Stages["check"] = s
	if err := ValidateAuditWorkerCompletion(b); err != nil {
		t.Fatal(err)
	}
	delete(action.Escalate.ExecutionConfig.Effective.Agents, "checker")
	if err := ValidateAuditWorkerCompletion(b); err == nil {
		t.Fatal("escalated target lost its effective Worker configuration")
	}
}

func TestAuditCompletionAuthoringIsClosed(t *testing.T) {
	root, _ := auditCompletionConfig(t)
	path := filepath.Join(root, "audit-profiles/source-checklist.yaml")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	writeFile(t, path, []byte(strings.Replace(string(raw), "agent: checker}", "agent: checker, model_override: true}", 1)))
	if _, err := Load(root, MVPDescriptors()); err == nil {
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
                  modelPolicy: worker@2
            then:
              fail: {}`
	writeFile(t, path, []byte(strings.Replace(string(raw), "        failed:\n          fail: {}", replacement, 1)))
	profile, err := mustLoad(t, root, MVPDescriptors()).AuditProfile("source-checklist@1")
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateAuditWorkerCompletion(profile.Workflows["check"]); err != nil {
		t.Fatal(err)
	}
}
