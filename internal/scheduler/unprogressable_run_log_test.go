package scheduler

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

// unprogressableRunLog is the message of the one log entry that carries the
// cause of a Run the Scheduler could not progress (SCH-R170).
const unprogressableRunLog = "Scheduler ended a WorkflowRun it could not progress"

// The Scheduler logs the cause of every FAIL-SCH-13 ending it commits once,
// with the Run, the code and the Run's resulting state, and the cause follows
// the rule of an execution-report error message: deployment secrets removed,
// at most telemetry.MaxErrorMessageBytes (OBS-R30, LIM-OBS-6).
func TestFailActiveRunLogsItsCauseOnce(t *testing.T) {
	const secret = "scheduler-test-deployment-secret"
	long := errors.New(secret + ": " + strings.Repeat("x", 2*telemetry.MaxErrorMessageBytes))
	for _, test := range []struct {
		name      string
		end       func(context.Context, *schedulerHarness) error
		code      string
		runState  runstore.WorkflowRunState
		wantCause func(string) bool
	}{
		{"unsupported Workflow shape", func(ctx context.Context, harness *schedulerHarness) error {
			harness.store.run.WorkflowSnapshot = json.RawMessage(`{}`)
			_, err := harness.scheduler.RunOnce(ctx)
			return err
		}, "unsupported_workflow_shape", runstore.RunFailed, func(cause string) bool {
			return strings.HasPrefix(cause, ErrUnsupportedWorkflow.Error()+": decode Workflow snapshot: ")
		}},
		{"cancelling Run", func(ctx context.Context, harness *schedulerHarness) error {
			harness.store.run.State = runstore.RunCancelling
			return harness.scheduler.failInvalidRunState(ctx, "run-1", errReviewStateInvalid)
		}, "scheduler_state_invalid", runstore.RunCancelled, func(cause string) bool {
			return cause == errReviewStateInvalid.Error()
		}},
		{"cause with a deployment secret over the bound", func(ctx context.Context, harness *schedulerHarness) error {
			return harness.scheduler.failInvalidRunState(ctx, "run-1", long)
		}, "scheduler_state_invalid", runstore.RunFailed, func(cause string) bool {
			return strings.HasPrefix(cause, "[REDACTED]: xxx") && !strings.Contains(cause, secret) &&
				len(cause) <= telemetry.MaxErrorMessageBytes && strings.HasSuffix(cause, "x…")
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			harness := newSchedulerHarness(t)
			logs := newRecordingLogHandler()
			harness.scheduler.options.Logger = slog.New(logs)
			harness.scheduler.options.TelemetrySecrets = []string{secret}

			if err := test.end(t.Context(), harness); err != nil {
				t.Fatal(err)
			}

			if harness.store.run.State != test.runState {
				t.Fatalf("Run = %s, want %s", harness.store.run.State, test.runState)
			}
			entries := logs.entries(unprogressableRunLog)
			if len(entries) != 1 {
				t.Fatalf("log entries of the ending = %+v, want exactly one", entries)
			}
			entry := entries[0]
			if entry.level != slog.LevelError || entry.attrs["run_id"] != "run-1" ||
				entry.attrs["code"] != test.code || entry.attrs["run_state"] != string(test.runState) ||
				!test.wantCause(entry.attrs["cause"]) {
				t.Fatalf("log entry of the ending = %+v", entry)
			}
		})
	}
}

// A commit that fails ends nothing, so nothing is logged as an ending: the
// cause comes back joined with the commit error for the lane to report.
func TestFailActiveRunLogsNothingWhenItsCommitFails(t *testing.T) {
	harness := newSchedulerHarness(t)
	logs := newRecordingLogHandler()
	harness.scheduler.options.Logger = slog.New(logs)
	commitErr := errors.New("injected commit failure")
	harness.scheduler.persistence = failingRunFailure{AtomicPersistence: harness.persistence, err: commitErr}

	err := harness.scheduler.failInvalidRunState(t.Context(), "run-1", errReviewStateInvalid)

	if !errors.Is(err, errReviewStateInvalid) || !errors.Is(err, commitErr) {
		t.Fatalf("failed ending = %v, want the cause joined with the commit error", err)
	}
	if entries := logs.entries(unprogressableRunLog); len(entries) != 0 {
		t.Fatalf("log entries of an ending that did not commit = %+v", entries)
	}
	if harness.store.run.State != runstore.RunRunning {
		t.Fatalf("Run = %s, want it still running", harness.store.run.State)
	}
}

type failingRunFailure struct {
	AtomicPersistence
	err error
}

func (p failingRunFailure) FailRunWithActiveStages(
	context.Context, string, runstore.WorkflowRunState, runstore.WorkflowRunState,
	runstore.Reason, runstore.StageTermination, string,
) error {
	return p.err
}

// recordingLogHandler is a slog.Handler that keeps every record, so a test
// can check the structured attributes of the entries it expects.
type recordingLogHandler struct {
	mu      *sync.Mutex
	records *[]slog.Record
	attrs   []slog.Attr
}

type logEntry struct {
	level slog.Level
	attrs map[string]string
}

func newRecordingLogHandler() *recordingLogHandler {
	return &recordingLogHandler{mu: &sync.Mutex{}, records: &[]slog.Record{}}
}

func (*recordingLogHandler) Enabled(context.Context, slog.Level) bool { return true }

func (h *recordingLogHandler) Handle(_ context.Context, record slog.Record) error {
	kept := record.Clone()
	kept.AddAttrs(h.attrs...)
	h.mu.Lock()
	defer h.mu.Unlock()
	*h.records = append(*h.records, kept)
	return nil
}

func (h *recordingLogHandler) WithAttrs(attrs []slog.Attr) slog.Handler {
	return &recordingLogHandler{mu: h.mu, records: h.records, attrs: append(slices.Clone(h.attrs), attrs...)}
}

// WithGroup keeps the handler: the Scheduler logs no groups.
func (h *recordingLogHandler) WithGroup(string) slog.Handler { return h }

// entries returns the records logged with message, their attributes by key.
func (h *recordingLogHandler) entries(message string) []logEntry {
	h.mu.Lock()
	defer h.mu.Unlock()
	var result []logEntry
	for _, record := range *h.records {
		if record.Message != message {
			continue
		}
		entry := logEntry{level: record.Level, attrs: map[string]string{}}
		record.Attrs(func(attr slog.Attr) bool {
			entry.attrs[attr.Key] = attr.Value.String()
			return true
		})
		result = append(result, entry)
	}
	return result
}
