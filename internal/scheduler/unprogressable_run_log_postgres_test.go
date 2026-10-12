package scheduler

import (
	"context"
	"log/slog"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// A Run the Scheduler cannot progress keeps only a stable code in durable
// state, so one log entry carries the cause (SCH-R170). An output mapping
// onto a binding of another revision (ART-R90) ends run-1 on the first pass;
// the second pass finds the Run terminal, and the cause is logged once.
func TestPostgresUnprogressableRunLogsItsCauseOnce(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	h, _, service, runArtifacts := newOutputBindingHarness(t, ctx)
	logs := newRecordingLogHandler()
	h.scheduler.options.Logger = slog.New(logs)
	earlier, err := runArtifacts.Write(
		ctx, contracts.ArtifactRef{Namespace: "builder", Name: "earlier"},
		artifacts.Payload{MediaType: "text/plain", Data: []byte("earlier\n")}, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := bindOutput(ctx, service, "run-1", "result", earlier.Ref); err != nil {
		t.Fatal(err)
	}

	for pass := range 2 {
		if _, err := h.scheduler.RunOnce(ctx); err != nil {
			t.Fatalf("pass %d: %v", pass+1, err)
		}
	}

	run, err := h.store.GetRun(ctx, "run-1")
	if err != nil || run.State != runstore.RunFailed || run.StateReason.Code != "scheduler_state_invalid" ||
		run.StateReason.Message != "" {
		t.Fatalf("Run = (%s %+v, %v), want failed with only the code", run.State, run.StateReason, err)
	}
	entries := logs.entries(unprogressableRunLog)
	if len(entries) != 1 {
		t.Fatalf("log entries of the ending = %+v, want exactly one", entries)
	}
	// The cause names the output and the refused binding; the classifying
	// sentinel between them is not part of what this test pins.
	entry := entries[0]
	cause := entry.attrs["cause"]
	if entry.level != slog.LevelError || entry.attrs["run_id"] != "run-1" ||
		entry.attrs["code"] != "scheduler_state_invalid" || entry.attrs["run_state"] != string(runstore.RunFailed) ||
		!strings.HasPrefix(cause, `bind Workflow output "result": `) ||
		!strings.HasSuffix(cause, "artifact binding outputs/result did not satisfy its revision precondition") {
		t.Fatalf("log entry of the ending = %+v", entry)
	}
}
