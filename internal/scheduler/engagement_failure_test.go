package scheduler

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/engagementcontract"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/engagement"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5/pgconn"
)

func engagementPreparationError(t *testing.T, cause error) error {
	t.Helper()
	resolver := runtimeCredentialResolverFunc(func(context.Context, string, []runtimesettings.RuntimeCredentialKind, func(runtimesettings.RuntimeCredentialKind, []byte) error) error {
		return cause
	})
	projection := &engagementcontract.AllocationProjection{HTTP: &engagementcontract.HTTPAssignment{Identities: []engagementcontract.Identity{
		{ID: "anonymous", Kind: "anonymous"},
		{ID: "account", Kind: "bearer", Credential: &runtimesettings.RuntimeCredentialRef{CredentialID: "account", Kind: runtimesettings.RuntimeCredentialTargetBearer}},
	}}}
	material, err := engagement.MaterializeHTTPIdentities(t.Context(), resolver, projection)
	if err == nil || material != nil {
		t.Fatal("failed identity materialization returned private settings")
	}
	return err
}

func TestEngagementPreparationFailureRetainsSafeRetryClass(t *testing.T) {
	const canary = "private-provider-diagnostic-canary"
	cases := []struct {
		name      string
		cause     error
		retryable bool
	}{
		{"missing", credentials.ErrRuntimeCredentialNotFound, false},
		{"missing-with-deadline", errors.Join(credentials.ErrRuntimeCredentialNotFound, context.DeadlineExceeded), false},
		{"missing-with-cancellation", errors.Join(credentials.ErrRuntimeCredentialNotFound, context.Canceled), false},
		{"invalid", credentials.ErrRuntimeCredentialInvalid, false},
		{"crypto", credentials.ErrCrypto, false},
		{"key-unavailable", credentials.ErrCredentialKeyUnavailable, false},
		{"cancelled", context.Canceled, true},
		{"deadline", context.DeadlineExceeded, true},
		{"storage-conflict", &pgconn.PgError{Code: persistencepostgres.SQLStateSerializationFailure}, true},
		{"connection", pgconn.ErrConnClosed, true},
		{"unknown", errors.New("opaque infrastructure error"), true},
		{"runtime-permanent", &controlplane.RuntimeAPIError{StatusCode: 422, Code: "unsupported_toolset", Retryable: false}, false},
		{"runtime-temporary", &controlplane.RuntimeAPIError{StatusCode: 503, Code: "agent_state_unavailable", Retryable: true}, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := engagementPreparationError(t, fmt.Errorf("%w: %s", tc.cause, canary))
			if !errors.Is(err, tc.cause) {
				t.Fatal("materialization discarded the provider error identity")
			}
			if strings.Contains(err.Error(), canary) {
				t.Fatal("materialization rendered provider diagnostics")
			}
			failure := infrastructureFailure("allocation_preparation_failed", "Worker allocation preparation failed", err)
			if failure.Retryable != tc.retryable || strings.Contains(failure.Message, canary) {
				t.Fatalf("preparation retry class = %v, want %v", failure.Retryable, tc.retryable)
			}
		})
	}
}

func TestSchedulerMissingEngagementCredentialDoesNotConsumeRetries(t *testing.T) {
	for _, cleanup := range []struct {
		name string
		err  error
	}{
		{"none", nil},
		{"runtime-api", &controlplane.RuntimeAPIError{StatusCode: 503, Code: "cleanup_unavailable", Retryable: true}},
		{"deadline", context.DeadlineExceeded},
		{"cancelled", context.Canceled},
	} {
		t.Run("cleanup-"+cleanup.name, func(t *testing.T) {
			h := newSchedulerHarness(t)
			configureRetryWorkflow(t, h, 3)
			cause := engagementPreparationError(t, credentials.ErrRuntimeCredentialNotFound)
			cause = errors.Join(cause, cleanup.err)
			h.workers.prepareError = cause
			worked, err := h.scheduler.RunOnce(t.Context())
			if err != nil || !worked {
				t.Fatalf("RunOnce = %v, %v", worked, err)
			}
			if h.store.run.State != runstore.RunFailed || len(h.store.stages) != 1 || len(h.persistence.decisions) != 1 || h.persistence.decisions[0].Action != runstore.StageTransitionFail {
				t.Fatalf("missing credential scheduled a retry: Run=%s, stages=%d", h.store.run.State, len(h.store.stages))
			}
			termination := h.store.stages[0].Termination
			if termination == nil || termination.Code != "allocation_preparation_failed" || termination.Retryable || termination.Phase != runstore.TerminationPreparing {
				t.Fatalf("permanent preparation termination = %+v", termination)
			}
			if h.workers.prepareCalls != 1 || h.workers.releaseCalls != 1 || h.planners.createCalls != 0 || h.planners.runCalls != 0 {
				t.Fatal("missing credential retried, leaked work or invoked the Planner")
			}
		})
	}
}
