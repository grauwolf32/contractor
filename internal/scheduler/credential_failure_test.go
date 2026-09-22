package scheduler

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"testing"

	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5/pgconn"
)

func TestCredentialFailuresClassifyRetryabilityAndLogSafeCause(t *testing.T) {
	const secret = "credential-secret-in-provider-error"
	for _, test := range []struct {
		name      string
		err       error
		retryable bool
		cause     string
	}{
		{"not found", credentials.ErrNotFound, false, "credential_not_found"},
		{"runtime not found", fmt.Errorf("%w: %s", credentials.ErrRuntimeCredentialNotFound, secret), false, "credential_not_found"},
		{"invalid", fmt.Errorf("%w: %s", credentials.ErrInvalid, secret), false, "credential_invalid"},
		{"crypto", credentials.ErrCrypto, false, "credential_unreadable"},
		{"deadline", context.DeadlineExceeded, true, "context_ended"},
		{"connection", fmt.Errorf("query %s: %w", secret, &pgconn.PgError{Code: "08006"}), true, "storage_unavailable"},
		{"serialization", &pgconn.PgError{Code: "40001"}, true, "storage_unavailable"},
		{"constraint", &pgconn.PgError{Code: "23505"}, false, "configuration_unavailable"},
		{"unclassified", errors.New(secret), false, "configuration_unavailable"},
	} {
		t.Run(test.name, func(t *testing.T) {
			var logs bytes.Buffer
			s := &Scheduler{options: Options{
				Logger: slog.New(slog.NewTextHandler(&logs, nil)),
				Credentials: credentialResolverFunc(func(context.Context, contracts.LLMCredentialRef, contracts.LLMGatewayConfigRef) (contracts.SecretString, error) {
					return contracts.SecretString{}, test.err
				}),
				RuntimeCredentials: runtimeCredentialResolverFunc(func(context.Context, string, []contracts.RuntimeCredentialKind, func(contracts.RuntimeCredentialKind, []byte) error) error {
					return test.err
				}),
			}}
			run := runstore.WorkflowRun{RunID: "run-1"}
			execution := runstore.StageExecution{StageExecutionID: "stage-1"}

			_, plannerErr := s.resolveCredential(t.Context(), workflowconfig.ResolvedConsumerExecutionConfig{
				LLMGateway: &contracts.ResolvedLLMGatewayConfig{},
				Credential: &contracts.LLMCredentialRef{CredentialID: "planner-credential"},
			})
			_, workerErr := s.materializeRuntimeSettings(t.Context(), runtimeconfig.ResolvedRuntimeConfig{
				Caido: &runtimeconfig.CaidoConfig{Credential: "caido-lab"},
			})
			for _, err := range []error{plannerErr, workerErr} {
				if err == nil || strings.Contains(err.Error(), secret) {
					t.Fatalf("credential error = %v", err)
				}
				failure := s.credentialFailure(run, execution, "config_unavailable", "Execution configuration is unavailable", err)
				if failure.Retryable != test.retryable {
					t.Fatalf("Retryable = %v, want %v for %v", failure.Retryable, test.retryable, test.err)
				}
			}
			if strings.Contains(logs.String(), secret) || strings.Count(logs.String(), "cause="+test.cause) != 2 {
				t.Fatalf("credential failure log = %q", logs.String())
			}
		})
	}
}
