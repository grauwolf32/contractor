package config

import (
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/config/resolved"
)

func TestScanPreparationIsAnOrdinaryWorkflowCapability(t *testing.T) {
	snapshot := mustLoad(t, copyCoreFixture(t, "audit-scan-catalog"))
	for _, ref := range []string{"openapi-sqlmap-scan@1", "openapi-nuclei-scan@1", "web-scan@1"} {
		t.Run(ref, func(t *testing.T) {
			pin, err := snapshot.AuditProfile(ref)
			if err != nil {
				t.Fatal(err)
			}
			workflow := pin.Workflows["scan"]
			stage := workflow.Stages[workflow.EntryStage]
			if stage.ScanPreparation == nil || stage.ScanPreparation.SettingsArtifact == "" {
				t.Fatal("missing ordinary scan preparation")
			}
			if workflow.AuditTask != nil || pin.Program.Workflows["scan"].Outputs["scan_report"] != "result" {
				t.Fatal("retired task execution survived")
			}
			data, _ := json.Marshal(pin)
			if _, err := auditprofile.DecodePinnedProgram(data); err != nil {
				t.Fatal("pin failed recovery", err)
			}
		})
	}
}
func TestScanPreparationRejectsUnsupportedStageShape(t *testing.T) {
	snapshot := mustLoad(t, copyCoreFixture(t, "audit-scan-catalog"))
	pin, err := snapshot.AuditProfile("openapi-sqlmap-scan@1")
	if err != nil {
		t.Fatal(err)
	}
	for name, change := range map[string]func(*resolved.ResolvedStage){
		"unknown kind":      func(s *resolved.ResolvedStage) { s.ScanPreparation.Kind = "other" },
		"missing operation": func(s *resolved.ResolvedStage) { s.ScanPreparation.OperationArtifact = "" },
		"multiple jobs":     func(s *resolved.ResolvedStage) { s.ScanPlan.MaxJobs = 2 },
	} {
		t.Run(name, func(t *testing.T) {
			workflow := pin.Clone().Workflows["scan"]
			stage := workflow.Stages[workflow.EntryStage]
			change(&stage)
			workflow.Stages[workflow.EntryStage] = stage
			if err := resolved.ValidateScanPlanStage(stage); err == nil {
				t.Fatal("invalid scan Stage accepted")
			}
		})
	}
}
