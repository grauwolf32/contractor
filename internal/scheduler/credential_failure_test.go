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

type failingCredentialRecords struct{ err error }

func (r failingCredentialRecords) GetCredential(context.Context, string) (credentials.Record, error) {
	return credentials.Record{}, r.err
}

// The production resolver composes development and managed providers; its
// errors must keep storage and context causes so transient failures retry.
func TestCompositeCredentialFailuresKeepRetryableCauses(t *testing.T) {
	const secret = "credential-secret-in-provider-error"
	development, err := credentials.NewStaticProvider(nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name  string
		err   error
		cause string
	}{
		{"serialization", &pgconn.PgError{Code: "40001", Message: secret}, "storage_unavailable"},
		{"deadlock", &pgconn.PgError{Code: "40P01", Message: secret}, "storage_unavailable"},
		{"connection", &pgconn.PgError{Code: "08006", Message: secret}, "storage_unavailable"},
		{"shutdown", context.Canceled, "context_ended"},
	} {
		t.Run(test.name, func(t *testing.T) {
			managed, err := credentials.NewEncryptedProvider(failingCredentialRecords{test.err}, nil)
			if err != nil {
				t.Fatal(err)
			}
			composite, err := credentials.NewCompositeProvider(development, managed)
			if err != nil {
				t.Fatal(err)
			}
			var logs bytes.Buffer
			s := &Scheduler{options: Options{
				Logger: slog.New(slog.NewTextHandler(&logs, nil)), Credentials: composite,
			}}
			_, resolveErr := s.resolveCredential(t.Context(), workflowconfig.ResolvedConsumerExecutionConfig{
				LLMGateway: &contracts.ResolvedLLMGatewayConfig{},
				Credential: &contracts.LLMCredentialRef{CredentialID: "managed-credential"},
			})
			if resolveErr == nil || strings.Contains(resolveErr.Error(), secret) {
				t.Fatalf("credential error = %v", resolveErr)
			}
			failure := s.credentialFailure(
				runstore.WorkflowRun{RunID: "run-1"}, runstore.StageExecution{StageExecutionID: "stage-1"},
				"config_unavailable", "Execution configuration is unavailable", resolveErr,
			)
			if !failure.Retryable || !strings.Contains(logs.String(), "cause="+test.cause) ||
				strings.Contains(logs.String(), secret) {
				t.Fatalf("failure = %+v, log = %q", failure, logs.String())
			}
		})
	}
}
