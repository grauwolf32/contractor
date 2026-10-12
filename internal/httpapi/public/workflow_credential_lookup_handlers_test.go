package public

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/getkin/kin-openapi/routers/gorillamux"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"github.com/grauwolf32/contractor/internal/credentials"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5/pgconn"
)

// scriptedLLMCredentialLookup answers every LLM credential lookup with one
// scripted failure, or with metadata for the selected credential once the
// failure is cleared.
type scriptedLLMCredentialLookup struct {
	mu       sync.Mutex
	err      error
	metadata map[string]config.CredentialMetadata
}

func (l *scriptedLLMCredentialLookup) LookupLLMCredential(
	_ context.Context, id string,
) (config.CredentialMetadata, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.err != nil {
		return config.CredentialMetadata{}, l.err
	}
	metadata, ok := l.metadata[id]
	if !ok {
		return config.CredentialMetadata{}, credentials.ErrNotFound
	}
	return metadata, nil
}

func (l *scriptedLLMCredentialLookup) fail(err error) {
	l.mu.Lock()
	l.err = err
	l.mu.Unlock()
}

const credentialLookupCanary = "private-credential-lookup-canary"

// workflowCredentialLookupOutcomes are the lookup outcomes every Workflow
// credential revalidation must keep apart: recovery and a definite transaction
// conflict are the documented retryable 503, an unclassified storage failure
// is a retryable 500 and a confirmed missing credential is a permanent 400.
var workflowCredentialLookupOutcomes = []struct {
	name      string
	err       error
	status    int
	code      string
	retryable bool
}{
	{"recovery", persistencepostgres.WrapError("lookup LLM credential", credentials.ErrRecoveryRequired),
		http.StatusServiceUnavailable, "credential_recovery_required", true},
	{"transaction conflict", persistencepostgres.WrapError("lookup LLM credential", &pgconn.PgError{
		Code: persistencepostgres.SQLStateSerializationFailure, Message: credentialLookupCanary,
	}), http.StatusServiceUnavailable, "storage_transaction_conflict", true},
	{"storage outage", persistencepostgres.WrapError("lookup LLM credential", &pgconn.PgError{
		Code: persistencepostgres.SQLStateQueryCanceled, Message: credentialLookupCanary,
	}), http.StatusInternalServerError, "internal_error", true},
	{"missing", credentials.ErrNotFound, http.StatusBadRequest, "invalid_request", false},
}

func assertCredentialLookupResponse(
	t *testing.T, response *httptest.ResponseRecorder, status int, code string, retryable bool,
) {
	t.Helper()
	var envelope errorResponse
	if err := json.Unmarshal(response.Body.Bytes(), &envelope); err != nil {
		t.Fatalf("decode error response %q: %v", response.Body.String(), err)
	}
	if response.Code != status || envelope.Code != code || envelope.Retryable != retryable ||
		strings.Contains(response.Body.String(), credentialLookupCanary) {
		t.Fatalf("lookup outcome = %d %+v, want %d %s retryable=%t", response.Code, envelope, status, code, retryable)
	}
}

func newWorkflowCredentialLookupFixture(t *testing.T) (handlerFixture, *scriptedLLMCredentialLookup) {
	t.Helper()
	lookup := &scriptedLLMCredentialLookup{}
	fixture := newHandlerFixtureWithAuth(
		t, "../../config/testdata/valid", newTestAuthentication(t), mustTestOrigins(t), false, nil,
		func(dependencies *testServices) { dependencies.Credentials = lookup },
	)
	gateway, err := fixture.configs.Snapshot().LLMGateway("local-litellm@1")
	if err != nil {
		t.Fatal(err)
	}
	lookup.metadata = map[string]config.CredentialMetadata{"development-worker": {
		Ref: llmgateway.LLMCredentialRef{CredentialID: "development-worker"}, LLMGateway: gateway.Ref,
	}}
	user, err := fixture.artifacts.User("user-1")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := user.Write(
		t.Context(), contracts.ArtifactRef{Namespace: "projects", Name: "source"},
		artifacts.Payload{MediaType: "text/plain", Data: []byte("source")}, nil,
	); err != nil {
		t.Fatal(err)
	}
	return fixture, lookup
}

const workflowCredentialRunBody = `{"workflow":"artifact-copy@1","parameters":{},` +
	`"artifacts":{"source":{"namespace":"projects","name":"source"}},` +
	`"executionConfig":{"workers":{"credential":"development-worker"}}}`

func TestRunCreateKeepsWorkflowCredentialLookupClass(t *testing.T) {
	router, err := gorillamux.NewRouter(loadPublicOpenAPI(t))
	if err != nil {
		t.Fatal(err)
	}
	for _, outcome := range workflowCredentialLookupOutcomes {
		t.Run(outcome.name, func(t *testing.T) {
			fixture, lookup := newWorkflowCredentialLookupFixture(t)
			lookup.fail(outcome.err)
			request := newPublicContractRequest(http.MethodPost, "/v1/runs", []byte(workflowCredentialRunBody))
			request.Header.Set("Content-Type", "application/json")
			request.Header.Set(idempotencyKeyHeader, "workflow-credential-lookup")
			response := serveAndValidatePublicContract(t, router, fixture.handler, request, true)
			assertCredentialLookupResponse(t, response, outcome.status, outcome.code, outcome.retryable)
			if len(fixture.runs.runs) != 0 {
				t.Fatalf("failed lookup created Runs: %v", fixture.runs.runs)
			}

			// The same request succeeds once the credential store answers.
			lookup.fail(nil)
			retry := newPublicContractRequest(http.MethodPost, "/v1/runs", []byte(workflowCredentialRunBody))
			retry.Header.Set("Content-Type", "application/json")
			retry.Header.Set(idempotencyKeyHeader, "workflow-credential-lookup")
			if response := serveAndValidatePublicContract(t, router, fixture.handler, retry, true); response.Code != http.StatusAccepted {
				t.Fatalf("retry after lookup recovery = %d %s", response.Code, response.Body.String())
			}
		})
	}
}

func TestRunRepeatDraftDoesNotReportLookupOutageAsUnavailableSelection(t *testing.T) {
	router, err := gorillamux.NewRouter(loadPublicOpenAPI(t))
	if err != nil {
		t.Fatal(err)
	}
	for _, outcome := range workflowCredentialLookupOutcomes {
		t.Run(outcome.name, func(t *testing.T) {
			fixture, lookup := newWorkflowCredentialLookupFixture(t)
			request := authenticatedRequest(http.MethodPost, "/v1/runs", bytes.NewReader([]byte(workflowCredentialRunBody)))
			created := httptest.NewRecorder()
			fixture.handler.ServeHTTP(created, request)
			if created.Code != http.StatusAccepted {
				t.Fatalf("create Run = %d %s", created.Code, created.Body.String())
			}
			run := fixture.runs.runs["run_fixed"]
			run.State = runstore.RunSucceeded
			run.FinishedAt = timePointer(run.CreatedAt)
			fixture.runs.runs[run.RunID] = run

			lookup.fail(outcome.err)
			if !outcome.retryable {
				// A confirmed missing credential is a draft the owner must edit.
				response := serveQuery(t, fixture.handler, "/v1/runs/run_fixed/repeat-draft")
				var repeat runRepeatDraftResponse
				decodeQueryResponse(t, response, &repeat)
				if response.Code != http.StatusOK || !hasRepeatNotice(repeat.Notices, "execution_configuration_unavailable") {
					t.Fatalf("missing credential repeat draft = %d %+v", response.Code, repeat.Notices)
				}
				return
			}
			response := serveAndValidatePublicContract(t, router, fixture.handler,
				newPublicContractRequest(http.MethodGet, "/v1/runs/run_fixed/repeat-draft", nil), true)
			assertCredentialLookupResponse(t, response, outcome.status, outcome.code, outcome.retryable)
		})
	}
}
