package runtimeconfig

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/credentialerrors"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

type classificationRuntimeValidator func(context.Context, string, ...string) error

func (f classificationRuntimeValidator) ValidateRuntimeCredential(ctx context.Context, id string, kinds ...string) error {
	return f(ctx, id, kinds...)
}

type classificationLLMLookup func(context.Context, string) (config.CredentialMetadata, error)

func (f classificationLLMLookup) LookupLLMCredential(ctx context.Context, id string) (config.CredentialMetadata, error) {
	return f(ctx, id)
}

func TestRuntimeCredentialLookupErrorsOnlyMarkConfirmedInvalidConfiguration(t *testing.T) {
	spec := Spec{Worker: WorkerPatch{Telemetry: telemetryPatch("https://otel.example/traces", "otel-key")}}
	for _, test := range []struct {
		name    string
		cause   error
		invalid bool
	}{
		{"missing", credentialerrors.RuntimeNotFound, true},
		{"wrong-kind", credentialerrors.RuntimeInvalid, true},
		{"no-row", pgx.ErrNoRows, true},
		{"statement-timeout", &pgconn.PgError{Code: "57014", Message: "private-canary"}, false},
		{"connection-closed", errors.New("private-canary"), false},
	} {
		t.Run(test.name, func(t *testing.T) {
			err := validateSpecRuntimeCredentials(t.Context(), spec, classificationRuntimeValidator(
				func(context.Context, string, ...string) error { return test.cause },
			))
			if !errors.Is(err, test.cause) || errors.Is(err, ErrInvalid) != test.invalid {
				t.Fatalf("validation = %v; cause preserved=%t invalid=%t", err, errors.Is(err, test.cause), errors.Is(err, ErrInvalid))
			}
			if strings.Contains(err.Error(), "private-canary") {
				t.Fatalf("validation exposed lookup details: %v", err)
			}
		})
	}
}

func TestLLMCredentialLookupErrorsOnlyMarkConfirmedInvalidConfiguration(t *testing.T) {
	spec := Spec{Worker: WorkerPatch{LLMGateway: LLMGatewayPatch{
		Present: true, Credential: credentialField("llm-key"),
	}}}
	for _, test := range []struct {
		name    string
		cause   error
		invalid bool
	}{
		{"missing", credentialerrors.LLMNotFound, true},
		{"no-row", pgx.ErrNoRows, true},
		{"recovery-required", errors.New("private-canary"), false},
		{"statement-timeout", &pgconn.PgError{Code: "57014", Message: "private-canary"}, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			err := validateRunLLMCredential(t.Context(), spec, classificationLLMLookup(
				func(context.Context, string) (config.CredentialMetadata, error) {
					return config.CredentialMetadata{}, test.cause
				},
			))
			if !errors.Is(err, test.cause) || errors.Is(err, ErrInvalid) != test.invalid {
				t.Fatalf("validation = %v; cause preserved=%t invalid=%t", err, errors.Is(err, test.cause), errors.Is(err, ErrInvalid))
			}
			if strings.Contains(err.Error(), "private-canary") {
				t.Fatalf("validation exposed lookup details: %v", err)
			}
		})
	}
	for _, test := range []struct {
		name     string
		metadata config.CredentialMetadata
	}{
		{"identity", config.CredentialMetadata{Ref: contracts.LLMCredentialRef{CredentialID: "other"}}},
		{"gateway", config.CredentialMetadata{Ref: contracts.LLMCredentialRef{CredentialID: "llm-key"}}},
	} {
		t.Run(test.name, func(t *testing.T) {
			withGateway := spec
			if test.name == "gateway" {
				withGateway.Worker.LLMGateway.Gateway = gatewayField(testGatewayRef("primary", "1"))
			}
			err := validateRunLLMCredential(t.Context(), withGateway, classificationLLMLookup(
				func(context.Context, string) (config.CredentialMetadata, error) {
					return test.metadata, nil
				},
			))
			if !errors.Is(err, ErrInvalid) {
				t.Fatalf("%s mismatch = %v", test.name, err)
			}
		})
	}
}
