package config

import (
	"encoding/json"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/config/auditprofile"
)

func TestAuditScanUsesContractsInsteadOfCatalogNames(t *testing.T) {
	t.Parallel()
	for _, scanner := range []string{"sqlmap", "nuclei"} {
		t.Run(scanner, func(t *testing.T) {
			t.Parallel()
			root := copyCoreFixture(t, "audit-scan-catalog")
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
			profile, err := mustLoad(t, root).AuditProfile("customer-api-review@1")
			if err != nil {
				t.Fatal(err)
			}
			binding := profile.Workflows["verify"]
			stage := binding.Workflow.Stages["probe"]
			if profile.Inventory.Implementation != auditprofile.AuditInventoryOpenAPIScans ||
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
			stored, err := auditprofile.DecodeResolvedAuditProfileSnapshot(data)
			if err != nil || !reflect.DeepEqual(stored, profile) {
				t.Fatalf("renamed scan profile cannot be restored: %v", err)
			}
		})
	}
}

func TestAuditScanValidatesExecutionContract(t *testing.T) {
	t.Parallel()
	snapshot := mustLoad(t, copyCoreFixture(t, "audit-scan-catalog"))
	profile, err := snapshot.AuditProfile("openapi-sqlmap-scan@1")
	if err != nil {
		t.Fatal(err)
	}
	for name, change := range map[string]func(*auditprofile.ResolvedAuditProfile){
		"missing executor declaration": func(p *auditprofile.ResolvedAuditProfile) {
			binding := p.Workflows["scan"]
			binding.Workflow.AuditTask = nil
			p.Workflows["scan"] = binding
		},
		"unsupported task version": func(p *auditprofile.ResolvedAuditProfile) {
			p.Workflows["scan"].Workflow.AuditTask.Contract = "openapi-scan@99"
		},
		"different task producer": func(p *auditprofile.ResolvedAuditProfile) { p.Inventory.Implementation = "openapi-operations@1" },
		"missing executor Stage":  func(p *auditprofile.ResolvedAuditProfile) { p.Workflows["scan"].Workflow.AuditTask.Stage = "missing" },
		"approval omitted": func(p *auditprofile.ResolvedAuditProfile) {
			p.Interaction.ActiveChecks = auditprofile.AuditActiveChecksProhibited
		},
		"multiple assigned items": func(p *auditprofile.ResolvedAuditProfile) { p.Execution.BatchSize = 2 },
		"settings substituted": func(p *auditprofile.ResolvedAuditProfile) {
			p.Workflows["scan"].Inputs["settings"] = auditprofile.AuditWorkflowInputMapping{Source: auditprofile.AuditInputFromAudit, Name: "openapi"}
		},
		"scanner differs from Worker": func(p *auditprofile.ResolvedAuditProfile) {
			p.Workflows["scan"].Workflow.Stages["scan"].AuditScan.Scanner = "nuclei"
		},
		"unknown outcome redispatch": func(p *auditprofile.ResolvedAuditProfile) {
			p.Workflows["scan"].Workflow.Stages["scan"].AuditScan.UnknownOutcome = "retry"
		},
	} {
		t.Run(name, func(t *testing.T) {
			changed := profile.Clone()
			change(&changed)
			if auditprofile.ValidateAuditTaskProfile(changed) == nil {
				t.Fatal("invalid scan execution contract accepted")
			}
			if err := auditprofile.ValidateAuditTaskProfile(profile); err != nil {
				t.Fatalf("copy changed the original snapshot: %v", err)
			}
		})
	}
}
