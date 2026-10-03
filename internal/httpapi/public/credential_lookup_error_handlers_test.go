package public

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/credentialerrors"
	"github.com/grauwolf32/contractor/internal/credentials"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5/pgconn"
)

func TestCredentialLookupOutagesRemainRetryableAcrossRuntimeMutations(t *testing.T) {
	transient := persistencepostgres.WrapError("validate RuntimeConfig credential", &pgconn.PgError{
		Code: "57014", Message: "private-database-canary",
	})
	for _, operation := range []string{"publish", "label-create", "label-rebind", "run-create"} {
		t.Run(operation, func(t *testing.T) {
			fixture := newHandlerFixture(t)
			var request *http.Request
			switch operation {
			case "publish":
				fixture.runtimeConfigs.publishErr = transient
				request = runtimeMutationRequest(t, http.MethodPost, "/v1/operations/runtime-configs",
					`{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig","metadata":{"name":"debug","version":"1"},"spec":{}}`,
					"lookup-outage-publish", "", "")
			case "label-create", "label-rebind":
				ref := fixture.runtimeConfigs.bindings[runtimeconfig.DefaultLabel].Ref
				body, err := json.Marshal(runtimeLabelMutationRequest{Config: ref})
				if err != nil {
					t.Fatal(err)
				}
				ifMatch, ifNoneMatch := "", "*"
				if operation == "label-create" {
					fixture.runtimeConfigs.createErr = transient
				} else {
					fixture.runtimeConfigs.rebindErr = transient
					ifMatch, ifNoneMatch = `"1"`, ""
				}
				request = runtimeMutationRequest(t, http.MethodPut, "/v1/operations/runtime-labels/debug",
					string(body), "lookup-outage-"+operation, ifMatch, ifNoneMatch)
			case "run-create":
				fixture.runs.pinRuntimeLabels = func(context.Context, []string) (runtimeconfig.RunSnapshot, error) {
					return runtimeconfig.RunSnapshot{}, transient
				}
				request = authenticatedRequest(http.MethodPost, "/v1/runs", bytes.NewReader([]byte(
					`{"workflow":"artifact-copy@1","runtimeLabels":["debug"],"parameters":{},"artifacts":{"source":{"namespace":"projects","name":"source"}}}`,
				)))
				request.Header.Set(idempotencyKeyHeader, "lookup-outage-run")
			}
			response := httptest.NewRecorder()
			fixture.handler.ServeHTTP(response, request)
			assertErrorCode(t, response, "internal_error")
			var envelope errorResponse
			if err := json.Unmarshal(response.Body.Bytes(), &envelope); err != nil {
				t.Fatal(err)
			}
			if response.Code != http.StatusInternalServerError || !envelope.Retryable ||
				strings.Contains(response.Body.String(), "private-database-canary") {
				t.Fatalf("lookup outage = %d %+v", response.Code, envelope)
			}
		})
	}
}

func TestRunCredentialRecoveryIsRetryableServiceUnavailable(t *testing.T) {
	fixture := newHandlerFixture(t)
	fixture.runs.pinRuntimeLabels = func(context.Context, []string) (runtimeconfig.RunSnapshot, error) {
		return runtimeconfig.RunSnapshot{}, persistencepostgres.WrapError(
			"validate RuntimeConfig LLM credential", credentials.ErrRecoveryRequired,
		)
	}
	request := authenticatedRequest(http.MethodPost, "/v1/runs", bytes.NewReader([]byte(
		`{"workflow":"artifact-copy@1","runtimeLabels":["debug"],"parameters":{},"artifacts":{"source":{"namespace":"projects","name":"source"}}}`,
	)))
	request.Header.Set(idempotencyKeyHeader, "lookup-recovery-run")
	response := httptest.NewRecorder()
	fixture.handler.ServeHTTP(response, request)
	assertErrorCode(t, response, "credential_recovery_required")
	var envelope errorResponse
	if err := json.Unmarshal(response.Body.Bytes(), &envelope); err != nil {
		t.Fatal(err)
	}
	if response.Code != http.StatusServiceUnavailable || !envelope.Retryable {
		t.Fatalf("recovery = %d %+v", response.Code, envelope)
	}
}

func TestMissingRunCredentialRemainsInvalidConfiguration(t *testing.T) {
	fixture := newHandlerFixture(t)
	fixture.runs.pinRuntimeLabels = func(context.Context, []string) (runtimeconfig.RunSnapshot, error) {
		return runtimeconfig.RunSnapshot{}, persistencepostgres.WrapError(
			"invalid RuntimeConfig: Runtime credential is unavailable or incompatible",
			errors.Join(runtimeconfig.ErrInvalid, credentialerrors.RuntimeNotFound),
		)
	}
	request := authenticatedRequest(http.MethodPost, "/v1/runs", bytes.NewReader([]byte(
		`{"workflow":"artifact-copy@1","runtimeLabels":["debug"],"parameters":{},"artifacts":{"source":{"namespace":"projects","name":"source"}}}`,
	)))
	request.Header.Set(idempotencyKeyHeader, "missing-credential-run")
	response := httptest.NewRecorder()
	fixture.handler.ServeHTTP(response, request)
	assertErrorCode(t, response, "runtime_config_invalid")
	var envelope errorResponse
	if err := json.Unmarshal(response.Body.Bytes(), &envelope); err != nil {
		t.Fatal(err)
	}
	if response.Code != http.StatusBadRequest || envelope.Retryable {
		t.Fatalf("missing credential = %d %+v", response.Code, envelope)
	}
}
