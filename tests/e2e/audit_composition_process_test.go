//go:build e2e

package e2e

import (
	"testing"
	"time"
)

func TestAuditCompositionAcrossProductionProcesses(t *testing.T) {
	if testing.Short() {
		t.Fatal("Audit composition acceptance cannot run in short mode")
	}
	t.Setenv("CONTRACTOR_SCAN_BROWSER", "1")
	h := startConfiguredScanStackWithBudget(t, 8*time.Minute, "", func(h *scanProcessHarness, root string) {
		stageCommonScanAuditCatalog(t, h, root)
	}, "nuclei")
	t.Run("common_tasks_and_retention", func(t *testing.T) { runCompositionBrowser(t, h) })
}
