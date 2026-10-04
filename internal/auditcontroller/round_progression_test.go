package auditcontroller

import (
	"bytes"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
)

func TestControllerClassifiesNextRoundPreparationFailures(t *testing.T) {
	drifted := &auditservice.RoundPreparationError{
		Diagnostic: "a next-Round proposal descriptor drifted", Cause: errors.New("receipt receipt-a"),
	}
	for _, test := range []struct {
		name        string
		builderErr  error
		wantCode    string
		wantMessage string
	}{
		{name: "transient blob read", builderErr: fmt.Errorf("read proposal: %w", artifacts.ErrBlobIO)},
		{name: "missing blob", builderErr: fmt.Errorf("read proposal: %w", artifacts.ErrBlobMissing)},
		{
			name: "digest mismatch", builderErr: fmt.Errorf("read proposal: %w", artifacts.ErrArtifactIntegrity),
			wantCode: "next_round_invalid",
		},
		{
			name: "inconsistent state", builderErr: fmt.Errorf("prepare: %w", drifted),
			wantCode:    "next_round_contract_invalid",
			wantMessage: "The next Audit Round cannot be prepared because a next-Round proposal descriptor drifted.",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			harness := newControllerHarness(t, 1, 1)
			var logs bytes.Buffer
			harness.controller.options.Logger = slog.New(slog.NewTextHandler(&logs, nil))
			harness.store.mu.Lock()
			harness.store.round.State = auditstore.RoundClosed
			harness.store.items = nil
			harness.store.executions = nil
			harness.store.audit.Limits.MaxRounds = 2
			harness.store.mu.Unlock()
			builder := &fakeRoundBuilder{err: test.builderErr}
			harness.controller.roundBuilder = builder
			for attempt := 1; attempt <= 2; attempt++ {
				worked, err := harness.controller.RunOnce(harness.ctx)
				if test.wantCode == "" {
					if worked || !errors.Is(err, test.builderErr) {
						t.Fatalf("attempt %d = (%t, %v), want a retryable failure", attempt, worked, err)
					}
					continue
				}
				if attempt == 1 && (!worked || err != nil) {
					t.Fatalf("attempt %d = (%t, %v), want the Audit closed", attempt, worked, err)
				}
			}
			audit := harness.store.auditSnapshot()
			if test.wantCode == "" {
				if audit.State != auditstore.AuditActive || audit.StopReason != nil || builder.calls.Load() != 2 {
					t.Fatalf("retryable failure changed Audit or stopped retrying: %+v, calls=%d", audit, builder.calls.Load())
				}
				return
			}
			if audit.State != auditstore.AuditFinalizing || audit.Dispatch != auditstore.DispatchClosed ||
				audit.StopReason == nil || audit.StopReason.Code != test.wantCode ||
				test.wantMessage != "" && audit.StopReason.Message != test.wantMessage || builder.calls.Load() != 1 {
				t.Fatalf("deterministic failure = %+v, calls=%d", audit, builder.calls.Load())
			}
			if test.wantMessage != "" &&
				(!strings.Contains(logs.String(), "inconsistent state") || !strings.Contains(logs.String(), "receipt receipt-a") ||
					!strings.Contains(logs.String(), audit.AuditID)) {
				t.Fatalf("inconsistent next Round was not logged with its diagnostic: %s", logs.String())
			}
		})
	}
}
