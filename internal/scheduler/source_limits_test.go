package scheduler

import (
	"testing"

	"github.com/grauwolf32/contractor/internal/sourcezip"
)

func TestSchedulerMaterializesSourceArchiveBudgets(t *testing.T) {
	harness := newSchedulerHarness(t)
	limits := sourcezip.Limits{MaxExpandedBytes: 128 << 20, MaxFileBytes: 16 << 20, MaxEntries: 10000}
	harness.scheduler.options.RuntimeTransport.SourceArchiveLimits = &limits
	resolved, err := schedulerTestResolvedWorkerConfig(harness.workflow.Stages[harness.workflow.EntryStage].ExecutionConfig.Agents["builder"], false)
	if err != nil {
		t.Fatal(err)
	}
	settings, err := harness.scheduler.materializeRuntimeSettings(t.Context(), resolved)
	if err != nil || settings.SourceArchiveLimits == nil || *settings.SourceArchiveLimits != limits {
		t.Fatalf("source limits = %+v, %v", settings.SourceArchiveLimits, err)
	}
	settings.SourceArchiveLimits.MaxFileBytes = 1
	if limits.MaxFileBytes != 16<<20 {
		t.Fatal("prepared settings alias the process configuration")
	}
}
