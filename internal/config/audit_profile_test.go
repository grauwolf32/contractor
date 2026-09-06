package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestAuditProfileLoadsResolvedWorkflowAndReturnsDeepCopies(t *testing.T) {
	t.Parallel()

	root := copyAuditProfileConfigTree(t)
	writeAuditProfile(t, root, "z-security-review", validAuditProfileYAML("z-security-review"))
	writeAuditProfile(t, root, "a-security-review", validAuditProfileYAML("a-security-review"))

	snapshot := mustLoad(t, root, MVPDescriptors())
	if got := snapshot.Counts().AuditProfiles; got != 2 {
		t.Fatalf("AuditProfiles count = %d, want 2", got)
	}
	listed := snapshot.AuditProfiles()
	if len(listed) != 2 || listed[0].Ref.Name != "a-security-review" || listed[1].Ref.Name != "z-security-review" {
		t.Fatalf("AuditProfiles() order = %+v", listed)
	}

	profile, err := snapshot.AuditProfile("a-security-review@1")
	if err != nil {
		t.Fatal(err)
	}
	assertDigest(t, profile.Ref.Digest)
	if profile.Mode != AuditModeCustomChecklist || profile.Inventory.Implementation != "checklist@1" ||
		profile.Inventory.ItemWorkflowRole != "check" ||
		profile.Execution.BatchSize != 1 || profile.Interaction.ActiveChecks != AuditActiveChecksProhibited ||
		profile.Interaction.FindingConfirmation != AuditFindingDisabled {
		t.Fatalf("unexpected resolved AuditProfile: %+v", profile)
	}
	binding := profile.Workflows["check"]
	if binding.Workflow.Ref != (WorkflowRef{Name: "taint-trace-from-workspace", Version: "2"}) ||
		binding.Inputs["source"].Name != "source" ||
		binding.Parameters["target"].Name != "subjectKey" ||
		binding.Outputs["result"] != "taint_report" {
		t.Fatalf("unexpected resolved Audit Workflow binding: %+v", binding)
	}
	if got, want := profile.Inputs["checklist"].MediaTypes, []string{"application/json", "application/yaml"}; !equalStrings(got, want) {
		t.Fatalf("normalized Audit input media types = %v, want %v", got, want)
	}
	if profile.Standards[0].Scheme != "owasp-asvs" || profile.Standards[1].Scheme != "owasp-web-top10" {
		t.Fatalf("normalized standards = %+v", profile.Standards)
	}

	input := profile.Inputs["source"]
	input.MediaTypes[0] = "corrupted/type"
	profile.Inputs["source"] = input
	binding.Inputs["source"] = AuditWorkflowInputMapping{Source: AuditInputFromItemPackage}
	binding.Parameters["target"] = AuditWorkflowParameterMapping{Source: AuditParameterLiteral, Value: "corrupted"}
	binding.Outputs["result"] = "corrupted"
	binding.Workflow.Outputs["taint_report"] = ArtifactSlot{MediaTypes: []string{"corrupted/type"}}
	profile.Workflows["check"] = binding
	profile.Standards[0].Scheme = "corrupted"
	listed[0].Ref.Digest = "corrupted"

	again, err := snapshot.AuditProfile("a-security-review@1")
	if err != nil {
		t.Fatal(err)
	}
	againBinding := again.Workflows["check"]
	if again.Inputs["source"].MediaTypes[0] != "application/zip" ||
		againBinding.Inputs["source"].Source != AuditInputFromAudit ||
		againBinding.Parameters["target"].Source != AuditParameterItemField ||
		againBinding.Outputs["result"] != "taint_report" ||
		againBinding.Workflow.Outputs["taint_report"].MediaTypes[0] != "text/markdown" ||
		again.Standards[0].Scheme != "owasp-asvs" {
		t.Fatalf("caller mutation leaked into AuditProfile Snapshot: %+v", again)
	}
	if _, err := snapshot.AuditProfile("a-security-review"); err == nil {
		t.Fatal("AuditProfile accepted a versionless selector")
	}
	if _, err := snapshot.AuditProfile("missing@1"); err == nil {
		t.Fatal("AuditProfile accepted an unknown selector")
	}
}

func TestAuditProfileManagerReadAccessDoesNotAddManagedPublicationKind(t *testing.T) {
	t.Parallel()

	root := copyAuditProfileConfigTree(t)
	writeAuditProfile(t, root, "api-security-review", validAuditProfileYAML("api-security-review"))
	manager := newTestManager(t, root, filepath.Join(t.TempDir(), "managed"), ManagerOptions{})
	profile, err := manager.AuditProfile("api-security-review@1")
	if err != nil || profile.Ref.Name != "api-security-review" || len(manager.AuditProfiles()) != 1 {
		t.Fatalf("Manager AuditProfile access = (%+v, %v)", profile, err)
	}
	if _, err := ParseConfigurationKind("audit-profiles"); err == nil {
		t.Fatal("AuditProfile unexpectedly became a generic managed configuration kind")
	}
}

func TestRepositoryAuditProfilesPinRunnableOneRoundPrograms(t *testing.T) {
	t.Parallel()

	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	for _, test := range []struct {
		ref            string
		mode           AuditProfileMode
		implementation string
		role           string
		sourceInput    string
	}{
		{"source-checklist@1", AuditModeCustomChecklist, "checklist@1", "check", "checklist"},
		{"openapi-operation-trace@1", AuditModeOperationTracing, "openapi-operations@1", "trace", "openapi"},
	} {
		test := test
		t.Run(test.ref, func(t *testing.T) {
			t.Parallel()
			profile, err := snapshot.AuditProfile(test.ref)
			if err != nil {
				t.Fatal(err)
			}
			binding := profile.Workflows[test.role]
			if profile.Mode != test.mode || profile.Inventory.Implementation != test.implementation ||
				profile.Inventory.SourceInput != test.sourceInput || profile.Inventory.ItemWorkflowRole != test.role ||
				profile.Execution.MaxRounds != 1 || profile.Execution.BatchSize != 1 ||
				profile.Interaction.ActiveChecks != AuditActiveChecksProhibited ||
				profile.Interaction.FindingConfirmation != AuditFindingDisabled ||
				profile.Interaction.NotApplicable != AuditNotApplicableProfileRule ||
				profile.Interaction.ReportAcceptance != AuditReportAutomatic {
				t.Fatalf("repository AuditProfile is not MVP-compatible: %+v", profile)
			}
			if binding.Workflow.Ref != (WorkflowRef{Name: "audit-source-check", Version: "1"}) ||
				binding.Inputs["task"].Source != AuditInputFromItemPackage ||
				binding.Inputs["execution_manifest"].Source != AuditInputFromExecutionManifest ||
				binding.Inputs["source"].Source != AuditInputFromAudit || binding.Outputs["result"] != "result" {
				t.Fatalf("repository Audit Workflow binding is incomplete: %+v", binding)
			}
			stage := binding.Workflow.Stages[binding.Workflow.EntryStage]
			checker := stage.Agents["checker"].Template
			if checker.Ref.TemplateID != "audit_source_checker" ||
				!selectedTool(checker.Toolsets, "audit-results@1", "submit_check_result") {
				t.Fatalf("repository Audit Worker cannot publish result packages: %+v", checker)
			}
		})
	}
}

func selectedTool(toolsets []contracts.ToolsetSelection, ref, tool string) bool {
	for _, selection := range toolsets {
		if selection.Ref.ToolsetID+"@"+selection.Ref.Version == ref {
			for _, selected := range selection.Tools {
				if selected == tool {
					return true
				}
			}
		}
	}
	return false
}

func TestAuditProfileCatalogAcceptsBoundedFuturePoliciesWithoutClaimingCompatibility(t *testing.T) {
	t.Parallel()

	root := copyAuditProfileConfigTree(t)
	manifest := strings.NewReplacer(
		"batchSize: 1", "batchSize: 2",
		"activeChecks: prohibited", "activeChecks: approval-required",
		"findingConfirmation: disabled", "findingConfirmation: human-required",
		"notApplicable: profile-rule", "notApplicable: human-required",
		"reportAcceptance: automatic", "reportAcceptance: human-required",
	).Replace(validAuditProfileYAML("future-review"))
	writeAuditProfile(t, root, "future-review", manifest)

	profile, err := mustLoad(t, root, MVPDescriptors()).AuditProfile("future-review@1")
	if err != nil {
		t.Fatal(err)
	}
	if profile.Execution.BatchSize != 2 ||
		profile.Interaction.ActiveChecks != AuditActiveChecksApprovalRequired ||
		profile.Interaction.FindingConfirmation != AuditFindingHumanRequired ||
		profile.Interaction.NotApplicable != AuditNotApplicableHumanRequired ||
		profile.Interaction.ReportAcceptance != AuditReportHumanRequired {
		t.Fatalf("future bounded policy was not retained: %+v", profile)
	}
}

func TestAuditProfileDigestIsSemanticAndPinsWorkflowClosure(t *testing.T) {
	t.Parallel()

	baselineRoot := copyAuditProfileConfigTree(t)
	writeAuditProfile(t, baselineRoot, "api-security-review", validAuditProfileYAML("api-security-review"))
	presentationRoot := copyAuditProfileConfigTree(t)
	writeAuditProfile(t, presentationRoot, "api-security-review", presentationAuditProfileYAML)
	closureRoot := copyAuditProfileConfigTree(t)
	writeAuditProfile(t, closureRoot, "api-security-review", validAuditProfileYAML("api-security-review"))
	appendFile(t, filepath.Join(closureRoot, "instructions/taint-trace-planner.md"), "\n")
	policyRoot := copyAuditProfileConfigTree(t)
	changed := strings.Replace(validAuditProfileYAML("api-security-review"), "maxRounds: 3", "maxRounds: 4", 1)
	writeAuditProfile(t, policyRoot, "api-security-review", changed)

	baseline, _ := mustLoad(t, baselineRoot, MVPDescriptors()).AuditProfile("api-security-review@1")
	presentation, _ := mustLoad(t, presentationRoot, MVPDescriptors()).AuditProfile("api-security-review@1")
	closure, _ := mustLoad(t, closureRoot, MVPDescriptors()).AuditProfile("api-security-review@1")
	policy, _ := mustLoad(t, policyRoot, MVPDescriptors()).AuditProfile("api-security-review@1")
	if baseline.Ref.Digest != presentation.Ref.Digest {
		t.Fatalf("presentation changed AuditProfile digest: %s != %s", baseline.Ref.Digest, presentation.Ref.Digest)
	}
	if baseline.Ref.Digest == closure.Ref.Digest {
		t.Fatal("resolved child Workflow instruction change did not alter AuditProfile digest")
	}
	if baseline.Ref.Digest == policy.Ref.Digest {
		t.Fatal("Audit execution policy change did not alter AuditProfile digest")
	}
}

func TestAuditProfileRejectsInvalidDocumentsAtomically(t *testing.T) {
	t.Parallel()

	valid := validAuditProfileYAML("api-security-review")
	tests := []struct {
		name        string
		manifest    string
		wantMessage string
	}{
		{
			name:        "unknown field",
			manifest:    strings.Replace(valid, "  mode: custom-checklist", "  mode: custom-checklist\n  mystery: true", 1),
			wantMessage: "field mystery not found",
		},
		{
			name:        "removed maxActiveRuns field",
			manifest:    strings.Replace(valid, "    maxSubmittedRuns: 500", "    maxActiveRuns: 2\n    maxSubmittedRuns: 500", 1),
			wantMessage: "field maxActiveRuns not found",
		},
		{
			name:        "unknown workflow",
			manifest:    strings.Replace(valid, "taint-trace-from-workspace@2", "missing-workflow@1", 1),
			wantMessage: "selects unknown Workflow",
		},
		{
			name:        "required workflow input",
			manifest:    strings.Replace(valid, "      inputs:\n        source: {source: audit-input, name: source}", "      inputs: {}", 1),
			wantMessage: "missing required Workflow input",
		},
		{
			name:        "required workflow parameter",
			manifest:    strings.Replace(valid, "      parameters:\n        target: {source: item-field, name: subjectKey}", "      parameters: {}", 1),
			wantMessage: "missing required Workflow parameter",
		},
		{
			name:        "incompatible media type",
			manifest:    strings.Replace(valid, "source: {required: true, mediaTypes: [application/zip]}", "source: {required: true, mediaTypes: [text/plain]}", 1),
			wantMessage: "media types are incompatible",
		},
		{
			name:        "required workflow input from optional audit input",
			manifest:    strings.Replace(valid, "source: {required: true, mediaTypes: [application/zip]}", "source: {required: false, mediaTypes: [application/zip]}", 1),
			wantMessage: "required Workflow input from an optional Audit input",
		},
		{
			name:        "unsupported inventory",
			manifest:    strings.Replace(valid, "implementation: checklist@1", "implementation: model-memory@1", 1),
			wantMessage: "inventory.implementation is unsupported",
		},
		{
			name:        "inventory incompatible with mode",
			manifest:    strings.Replace(valid, "mode: custom-checklist", "mode: operation-tracing", 1),
			wantMessage: "is incompatible with mode",
		},
		{
			name:        "item role has no item context",
			manifest:    strings.Replace(valid, "{source: item-field, name: subjectKey}", "{source: literal, value: fixed-target}", 1),
			wantMessage: "must receive item-package or item-field context",
		},
		{
			name:        "nonpositive bound",
			manifest:    strings.Replace(valid, "maxRounds: 3", "maxRounds: 0", 1),
			wantMessage: "maxRounds must be between 1",
		},
		{
			name:        "nonpositive batch size",
			manifest:    strings.Replace(valid, "batchSize: 1", "batchSize: 0", 1),
			wantMessage: "batchSize must be between 1",
		},
		{
			name:        "batch exceeds server maximum",
			manifest:    strings.Replace(valid, "batchSize: 1", "batchSize: 101", 1),
			wantMessage: "batchSize must be between 1",
		},
		{
			name:        "incoherent submitted budget",
			manifest:    strings.Replace(valid, "maxSubmittedRuns: 500", "maxSubmittedRuns: 99", 1),
			wantMessage: "maxSubmittedRuns cannot cover maxItemsTotal",
		},
		{
			name:        "invalid interaction policy",
			manifest:    strings.Replace(valid, "activeChecks: prohibited", "activeChecks: model-decides", 1),
			wantMessage: "interaction.activeChecks is invalid",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			root := copyAuditProfileConfigTree(t)
			writeAuditProfile(t, root, "api-security-review", test.manifest)
			snapshot, err := Load(root, MVPDescriptors())
			if err == nil || snapshot != nil || !strings.Contains(err.Error(), "audit-profiles/api-security-review.yaml") ||
				!strings.Contains(err.Error(), test.wantMessage) {
				t.Fatalf("Load invalid AuditProfile = (%+v, %v), want path and %q", snapshot, err, test.wantMessage)
			}
		})
	}
}

func TestAuditProfileRejectsRetainedOutputCycles(t *testing.T) {
	t.Parallel()

	root := copyAuditProfileConfigTree(t)
	writeAuditProfile(t, root, "cycle-review", retainedCycleAuditProfileYAML)
	_, err := Load(root, MVPDescriptors())
	if err == nil || !strings.Contains(err.Error(), "retained-output dependencies contain a cycle") {
		t.Fatalf("retained-output cycle error = %v", err)
	}
}

func TestAuditProfileRejectsManagedRootDocuments(t *testing.T) {
	t.Parallel()

	operator := copyAuditProfileConfigTree(t)
	managed := filepath.Join(t.TempDir(), "managed")
	if _, err := requireStrictRoot(managed, true); err != nil {
		t.Fatal(err)
	}
	writeAuditProfile(t, managed, "managed-review", validAuditProfileYAML("managed-review"))
	_, err := LoadUnion(operator, managed, MVPDescriptors())
	if err == nil || !strings.Contains(err.Error(), "AuditProfile must be operator-authored") {
		t.Fatalf("managed AuditProfile error = %v", err)
	}
}

func writeAuditProfile(t *testing.T, root, name, manifest string) {
	t.Helper()
	writeFile(t, filepath.Join(root, "audit-profiles", name+".yaml"), []byte(manifest))
}

func copyAuditProfileConfigTree(t *testing.T) string {
	t.Helper()
	root := copyConfigTree(t)
	directory := filepath.Join(root, "audit-profiles")
	if err := os.RemoveAll(directory); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	return root
}

func validAuditProfileYAML(name string) string {
	return strings.Replace(auditProfileTemplateYAML, "PROFILE_NAME", name, 1)
}

const auditProfileTemplateYAML = `apiVersion: contractor/v1alpha1
kind: AuditProfile
metadata:
  name: PROFILE_NAME
  version: "1"
spec:
  mode: custom-checklist
  standards:
    - {scheme: owasp-web-top10, version: "2025"}
    - {scheme: owasp-asvs, version: "5.0.0"}
  inputs:
    source: {required: true, mediaTypes: [application/zip]}
    checklist: {required: true, mediaTypes: [application/yaml, application/json]}
  inventory:
    implementation: checklist@1
    sourceInput: checklist
    itemWorkflowRole: check
  workflows:
    check:
      kind: check
      ref: taint-trace-from-workspace@2
      inputs:
        source: {source: audit-input, name: source}
      parameters:
        target: {source: item-field, name: subjectKey}
      outputs:
        result: taint_report
  execution:
    roundMode: fixed-barrier
    maxRounds: 3
    batchSize: 1
    maxItemsPerRound: 100
    maxItemsTotal: 250
    maxSubmittedRuns: 500
    maxItemRunAttempts: 2
    deadlineSeconds: 86400
    maxEvidenceBytes: 67108864
    incompleteRound: assess-with-gaps
  interaction:
    activeChecks: prohibited
    findingConfirmation: disabled
    notApplicable: profile-rule
    reportAcceptance: automatic
`

const presentationAuditProfileYAML = `kind: AuditProfile
apiVersion: contractor/v1alpha1
metadata: {version: "1", name: api-security-review}
spec:
  interaction:
    reportAcceptance: automatic
    notApplicable: profile-rule
    findingConfirmation: disabled
    activeChecks: prohibited
  execution:
    maxEvidenceBytes: 67108864
    deadlineSeconds: 86400
    maxItemRunAttempts: 2
    maxSubmittedRuns: 500
    maxItemsTotal: 250
    maxItemsPerRound: 100
    batchSize: 1
    maxRounds: 3
    incompleteRound: assess-with-gaps
    roundMode: fixed-barrier
  workflows:
    check:
      kind: check
      outputs: {result: taint_report}
      parameters: {target: {name: subjectKey, source: item-field}}
      inputs: {source: {name: source, source: audit-input}}
      ref: taint-trace-from-workspace@2
  inventory: {itemWorkflowRole: check, sourceInput: checklist, implementation: checklist@1}
  inputs:
    checklist: {mediaTypes: [application/json, application/yaml], required: true}
    source: {mediaTypes: [application/zip], required: true}
  standards:
    - {version: "5.0.0", scheme: owasp-asvs}
    - {version: "2025", scheme: owasp-web-top10}
  mode: custom-checklist
`

const retainedCycleAuditProfileYAML = `apiVersion: contractor/v1alpha1
kind: AuditProfile
metadata: {name: cycle-review, version: "1"}
spec:
  mode: risk-assessment
  inputs:
    checklist: {required: true, mediaTypes: [application/json]}
  inventory: {implementation: checklist@1, sourceInput: checklist, itemWorkflowRole: a}
  workflows:
    a:
      kind: check
      ref: security-analysis@2
      inputs:
        context: {source: retained-output, role: b, name: result}
      parameters:
        objective: {source: literal, value: Review the operation}
        target: {source: item-field, name: subjectKey}
        authorization_scope: {source: scope-field, name: authorizationScope}
      outputs: {result: security_report}
    b:
      kind: assessment
      ref: security-analysis@2
      inputs:
        context: {source: retained-output, role: a, name: result}
      parameters:
        objective: {source: literal, value: Verify the evidence}
        target: {source: item-field, name: subjectKey}
        authorization_scope: {source: scope-field, name: authorizationScope}
      outputs: {result: security_report}
  execution:
    roundMode: fixed-barrier
    maxRounds: 2
    batchSize: 1
    maxItemsPerRound: 10
    maxItemsTotal: 20
    maxSubmittedRuns: 40
    maxItemRunAttempts: 2
    deadlineSeconds: 3600
    maxEvidenceBytes: 1048576
    incompleteRound: assess-with-gaps
  interaction:
    activeChecks: approval-required
    findingConfirmation: human-required
    notApplicable: human-required
    reportAcceptance: human-required
`
