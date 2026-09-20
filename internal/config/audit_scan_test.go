package config

import (
	"encoding/json"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestAuditScanUsesContractsInsteadOfCatalogNames(t *testing.T) {
	t.Parallel()
	for _, scanner := range []string{"sqlmap", "nuclei"} {
		t.Run(scanner, func(t *testing.T) {
			t.Parallel()
			root := copyConfigTree(t)
			// Rename every authored resource and the role, Stage and Worker.
			// Also give Audit inputs names different from their Workflow aliases.
			changes := map[string]*strings.Replacer{
				"audit-profiles/openapi_" + scanner + "_scan.yaml": strings.NewReplacer(
					"name: openapi-"+scanner+"-scan,", "name: customer-api-review,",
					"ref: audit-openapi-"+scanner+"-scan@1", "ref: customer-check@1",
					"itemWorkflowRole: scan", "itemWorkflowRole: verify",
					"    scan:\n", "    verify:\n",
					"    openapi: {required:", "    api_document: {required:",
					"source: {source: audit-input, name: openapi}", "source: {source: audit-input, name: api_document}",
					"name: openapi}", "name: api_document}",
					"    settings: {required:", "    scan_options: {required:",
					"settings: {source: audit-input, name: settings}", "settings: {source: audit-input, name: scan_options}",
					"name: settings}", "name: scan_options}",
				),
				"workflows/audit_openapi_" + scanner + "_scan.yaml": strings.NewReplacer(
					"name: audit-openapi-"+scanner+"-scan,", "name: customer-check,",
					"audit-"+scanner+"-scan@1", "customer-scanner@1",
					"entryStage: scan", "entryStage: probe",
					"stage: scan", "stage: probe",
					"    scan:\n", "    probe:\n",
					"worker: scanner", "worker: backend",
					"        scanner: {template:", "        backend: {template:",
				),
				"agent-templates/audit_" + scanner + "_scan.yaml": strings.NewReplacer(
					"name: audit-"+scanner+"-scan,", "name: customer-scanner,",
				),
			}
			for relative, replacements := range changes {
				path := filepath.Join(root, relative)
				writeFile(t, path, []byte(replacements.Replace(string(readFile(t, path)))))
			}
			profile, err := mustLoad(t, root, MVPDescriptors()).AuditProfile("customer-api-review@1")
			if err != nil {
				t.Fatal(err)
			}
			binding := profile.Workflows["verify"]
			stage := binding.Workflow.Stages["probe"]
			if profile.Inventory.Implementation != AuditInventoryOpenAPIScans ||
				binding.Workflow.Ref.Name != "customer-check" ||
				stage.Agents["backend"].Template.Ref.TemplateID != "customer-scanner" ||
				binding.Inputs["openapi"].Name != "api_document" ||
				binding.Inputs["settings"].Name != "scan_options" {
				t.Fatal("resolved scan ignored operator-authored resource names or input mappings")
			}
			data, err := json.Marshal(profile)
			if err != nil {
				t.Fatal(err)
			}
			stored, err := DecodeResolvedAuditProfileSnapshot(data)
			if err != nil || !reflect.DeepEqual(stored, profile) {
				t.Fatalf("renamed scan profile cannot be restored: %v", err)
			}
		})
	}
}

func TestAuditScanValidatesExecutionContract(t *testing.T) {
	t.Parallel()
	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	profile, err := snapshot.AuditProfile("openapi-sqlmap-scan@1")
	if err != nil {
		t.Fatal(err)
	}
	for name, change := range map[string]func(*ResolvedAuditProfile){
		"missing executor declaration": func(p *ResolvedAuditProfile) {
			binding := p.Workflows["scan"]
			binding.Workflow.AuditTask = nil
			p.Workflows["scan"] = binding
		},
		"unsupported task version": func(p *ResolvedAuditProfile) { p.Workflows["scan"].Workflow.AuditTask.Contract = "openapi-scan@99" },
		"different task producer":  func(p *ResolvedAuditProfile) { p.Inventory.Implementation = "openapi-operations@1" },
		"missing executor Stage":   func(p *ResolvedAuditProfile) { p.Workflows["scan"].Workflow.AuditTask.Stage = "missing" },
		"approval omitted":         func(p *ResolvedAuditProfile) { p.Interaction.ActiveChecks = AuditActiveChecksProhibited },
		"multiple assigned items":  func(p *ResolvedAuditProfile) { p.Execution.BatchSize = 2 },
		"settings substituted": func(p *ResolvedAuditProfile) {
			p.Workflows["scan"].Inputs["settings"] = AuditWorkflowInputMapping{Source: AuditInputFromAudit, Name: "openapi"}
		},
		"scanner differs from Worker": func(p *ResolvedAuditProfile) {
			p.Workflows["scan"].Workflow.Stages["scan"].AuditScan.Scanner = "nuclei"
		},
		"unknown outcome redispatch": func(p *ResolvedAuditProfile) {
			p.Workflows["scan"].Workflow.Stages["scan"].AuditScan.UnknownOutcome = "retry"
		},
	} {
		t.Run(name, func(t *testing.T) {
			changed := cloneAuditProfile(profile)
			change(&changed)
			if ValidateAuditTaskProfile(changed) == nil {
				t.Fatal("invalid scan execution contract accepted")
			}
			if err := ValidateAuditTaskProfile(profile); err != nil {
				t.Fatalf("copy changed the original snapshot: %v", err)
			}
		})
	}
}

func TestAuditTaskAllowsSurroundingStagesAndOtherRoles(t *testing.T) {
	t.Parallel()
	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	profile, err := snapshot.AuditProfile("openapi-sqlmap-scan@1")
	if err != nil {
		t.Fatal(err)
	}
	ordinary, err := snapshot.Workflow("artifact-copy@2")
	if err != nil {
		t.Fatal(err)
	}
	binding := profile.Workflows["scan"]
	workflow := &binding.Workflow
	notes := ArtifactSlot{Required: true, MediaTypes: []string{"text/plain"}}
	profile.Inputs["operator_notes"] = AuditProfileInput{Required: true, MediaTypes: notes.MediaTypes}
	workflow.Inputs["notes"] = notes
	binding.Inputs["notes"] = AuditWorkflowInputMapping{Source: AuditInputFromAudit, Name: "operator_notes"}
	for _, name := range []string{"before", "after"} {
		stage := cloneStage(ordinary.Stages[ordinary.EntryStage])
		stage.Context = StageContext{Artifacts: map[string]ContextArtifact{
			"notes": {Namespace: "inputs", Name: "notes", Required: true},
		}}
		stage.WorkflowOutputs = map[string]string{}
		stage.Result.Artifacts["copied"].From.Name = name
		workflow.Stages[name] = stage
	}
	before := workflow.Stages["before"]
	before.On.Succeeded = TransitionAction{Kind: TransitionNext, NextStage: "scan"}
	workflow.Stages["before"] = before
	scan := workflow.Stages["scan"]
	scan.On.Succeeded = TransitionAction{Kind: TransitionNext, NextStage: "after"}
	workflow.Stages["scan"] = scan
	workflow.EntryStage = "before"
	profile.Workflows["scan"] = binding
	// Other check roles declare their own consumer contracts; they do not
	// become the inventory's default route merely by accepting the same type.
	profile.Workflows["second-check"] = binding
	// A separate existing Workflow role is unrelated to task execution.
	profile.Workflows["assessment"] = ResolvedAuditWorkflowBinding{
		Kind: AuditWorkflowAssessment, Workflow: ordinary,
		Inputs: map[string]AuditWorkflowInputMapping{"source": {Source: AuditInputFromAudit, Name: "operator_notes"}}, Parameters: map[string]AuditWorkflowParameterMapping{},
		Outputs: map[string]string{"report": "result"},
	}
	if err := ValidateWorkflowGraph(*workflow); err != nil {
		t.Fatal(err)
	}
	if err := ValidateAuditTaskProfile(profile); err != nil {
		t.Fatal(err)
	}
	profile.Ref.Digest, err = auditProfileDigest(Selector{ID: profile.Ref.Name, Version: profile.Ref.Version}, profile)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(profile)
	if err != nil {
		t.Fatal(err)
	}
	restored, err := DecodeResolvedAuditProfileSnapshot(encoded)
	if err != nil || restored.Workflows["scan"].Workflow.AuditTask.Contract != contracts.AuditTaskOpenAPIScanV1 {
		t.Fatalf("composed Workflow cannot be restored: %v", err)
	}
	// Neither another result producer nor a success branch around the scan
	// may turn the composition into a fabricated successful task result.
	other := cloneStage(workflow.Stages["scan"])
	other.AuditScan, other.ScanPlan = nil, nil
	other.WorkflowOutputs = map[string]string{"result": "report"}
	changed := cloneAuditProfile(profile)
	changed.Workflows["scan"].Workflow.Stages["after"] = other
	if ValidateAuditTaskProfile(changed) == nil {
		t.Fatal("accepted a second canonical result producer")
	}
	changed = cloneAuditProfile(profile)
	bypass := changed.Workflows["scan"].Workflow.Stages["before"]
	bypass.On.Succeeded = TransitionAction{Kind: TransitionSucceed}
	// Keep the executor reachable, so rejection must cover the successful
	// bypass path rather than merely detect an unreachable Stage.
	bypass.On.Failed = TransitionAction{Kind: TransitionNext, NextStage: "scan"}
	changed.Workflows["scan"].Workflow.Stages["before"] = bypass
	if ValidateWorkflowGraph(changed.Workflows["scan"].Workflow) == nil {
		t.Fatal("accepted a successful path without the task result")
	}
}
