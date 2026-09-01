package public

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auth"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/credentials"
	publicevents "github.com/grauwolf32/contractor/internal/httpapi/public/events"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

const (
	testBearerToken   = "test-bearer-token"
	testAuthPassword  = "correct horse battery staple"
	testBrowserOrigin = "https://ui.contractor.test"
)

var (
	testAuthenticationOnce sync.Once
	testAuthenticationHash string
	testAuthenticationErr  error
)

type handlerFixture struct {
	handler            http.Handler
	configs            *config.Manager
	repository         *fakeArtifactRepository
	artifacts          *artifacts.Service
	runs               *fakeRunStore
	unit               *fakeUnitOfWork
	notifier           *recordingRunNotifier
	metrics            *fakeMetricsReader
	plans              *fakePlannerPlanReader
	credentials        *fakeManagedCredentials
	runtimeConfigs     *fakeRuntimeConfigManagement
	runtimeCredentials *fakeRuntimeCredentialManagement
	operations         *fakeOperationsReader
}

func newHandlerFixture(t *testing.T) handlerFixture {
	return newHandlerFixtureWithConfig(t, "../../config/testdata/valid")
}

func newHandlerFixtureWithConfig(t *testing.T, configRoot string) handlerFixture {
	return newHandlerFixtureWithAuth(
		t,
		configRoot,
		newTestAuthentication(t),
		mustTestOrigins(t),
		false,
		nil,
	)
}

func newHandlerFixtureWithAuth(
	t *testing.T,
	configRoot string,
	authentication *auth.Service,
	origins auth.OriginPolicy,
	insecureLoopbackCookie bool,
	logger *slog.Logger,
) handlerFixture {
	t.Helper()
	manager, err := config.NewManager(config.ManagerOptions{
		OperatorRoot: configRoot,
		ManagedRoot:  filepath.Join(t.TempDir(), "managed-configs"),
		Descriptors:  config.MVPDescriptors(),
	})
	if err != nil {
		t.Fatalf("load test config: %v", err)
	}
	repository := newFakeArtifactRepository()
	service := artifacts.NewService(repository)
	runs := newFakeRunStore()
	unit := &fakeUnitOfWork{runs: runs, artifacts: service}
	notifier := &recordingRunNotifier{}
	metrics := &fakeMetricsReader{records: map[string]telemetry.StageMetricsRecord{}}
	plans := &fakePlannerPlanReader{plans: map[string]planner.PlannerPlanProjection{}}
	managedCredentials := newFakeManagedCredentials()
	runtimeConfigs := newFakeRuntimeConfigManagement(runtimeconfig.GatewayResolverFunc(
		func(_ context.Context, selector string) (contracts.ResolvedLLMGatewayConfig, error) {
			return manager.LLMGateway(selector)
		},
	))
	runtimeCredentials := newFakeRuntimeCredentialManagement()
	operations := newFakeOperationsReader()
	eventHub, err := publicevents.NewHub(publicevents.Options{
		Context: t.Context(), Authentication: authentication, Origins: origins,
		Runs: runs, Operations: operations,
	})
	if err != nil {
		t.Fatalf("create public event Hub: %v", err)
	}
	t.Cleanup(eventHub.Close)
	handler, err := NewHandler(Dependencies{
		Authentication: authentication, BrowserOrigins: origins,
		InsecureLoopbackCookie: insecureLoopbackCookie,
		Config:                 manager, ConfigurationPublisher: manager,
		Credentials: managedCredentials, ManagedCredentials: managedCredentials,
		RuntimeConfigs: runtimeConfigs, RuntimeCredentials: runtimeCredentials,
		Runs: runs, Artifacts: service, Transactions: unit,
		Operations: operations, OperationsInvalidator: operations, Events: eventHub,
		Metrics:      metrics,
		PlannerPlans: plans,
		BearerToken:  contracts.NewSecretString(testBearerToken),
		NewID:        func(prefix string) (string, error) { return prefix + "fixed", nil },
		NewRequestID: func() (string, error) { return "request-fixed", nil },
		RunNotifier:  notifier,
		Logger:       logger,
		Now: func() time.Time {
			return time.Date(2026, 8, 29, 12, 0, 0, 0, time.UTC)
		},
	})
	if err != nil {
		t.Fatalf("NewHandler: %v", err)
	}
	return handlerFixture{
		handler: handler, configs: manager, repository: repository, artifacts: service,
		runs: runs, unit: unit, notifier: notifier, metrics: metrics, plans: plans,
		credentials:    managedCredentials,
		runtimeConfigs: runtimeConfigs, runtimeCredentials: runtimeCredentials,
		operations: operations,
	}
}

func newTestAuthentication(t *testing.T) *auth.Service {
	t.Helper()
	testAuthenticationOnce.Do(func() {
		testAuthenticationHash, testAuthenticationErr = auth.HashPassword([]byte(testAuthPassword))
	})
	if testAuthenticationErr != nil {
		t.Fatal(testAuthenticationErr)
	}
	bootstrap, err := auth.NewBootstrap("user-1", "admin", testAuthenticationHash)
	if err != nil {
		t.Fatal(err)
	}
	service, err := auth.NewService(bootstrap, auth.Options{})
	if err != nil {
		t.Fatal(err)
	}
	return service
}

func mustTestOrigins(t *testing.T) auth.OriginPolicy {
	t.Helper()
	origins, err := auth.NewOriginPolicy([]string{testBrowserOrigin}, false)
	if err != nil {
		t.Fatal(err)
	}
	return origins
}

func authenticatedRequest(method, target string, body *bytes.Reader) *http.Request {
	request := httptest.NewRequest(method, target, body)
	request.Header.Set("Authorization", "Bearer "+testBearerToken)
	if method == http.MethodPost && target == "/v1/runs" {
		request.Header.Set(idempotencyKeyHeader, "test-create-run")
	}
	return request
}

func TestAuthenticationAndRequestID(t *testing.T) {
	fixture := newHandlerFixture(t)
	for _, authorization := range []string{"", "Bearer wrong"} {
		request := httptest.NewRequest(http.MethodGet, "/v1/runs/missing", nil)
		if authorization != "" {
			request.Header.Set("Authorization", authorization)
		}
		response := httptest.NewRecorder()
		fixture.handler.ServeHTTP(response, request)
		if response.Code != http.StatusUnauthorized || response.Header().Get("X-Request-ID") != "request-fixed" {
			t.Fatalf("auth response = status %d, request ID %q", response.Code, response.Header().Get("X-Request-ID"))
		}
		if response.Header().Get(APIVersionHeader) != APIVersion {
			t.Fatalf("auth response API version = %q", response.Header().Get(APIVersionHeader))
		}
		assertErrorCode(t, response, "unauthorized")
	}
}

func TestManagedCredentialCRUDIsStrictSecretFreeAndIdempotent(t *testing.T) {
	fixture := newHandlerFixture(t)
	gateway, err := fixture.configs.Snapshot().LLMGateway("local-litellm@1")
	if err != nil {
		t.Fatal(err)
	}
	policy, err := fixture.configs.Snapshot().ModelPolicy("worker@1")
	if err != nil {
		t.Fatal(err)
	}
	body, err := json.Marshal(createCredentialRequest{
		CredentialID: "managed-worker", LLMGateway: gateway.Ref, Label: stringPointer("Managed worker"),
		GatewayPolicy: credentials.GatewayPolicy{ModelPolicies: []contracts.ModelPolicyRef{policy.Ref}},
	})
	if err != nil {
		t.Fatal(err)
	}
	create := func(key string, payload []byte) *httptest.ResponseRecorder {
		request := authenticatedRequest(
			http.MethodPost, "/v1/operations/credentials", bytes.NewReader(payload),
		)
		request.Header.Set("Content-Type", "application/json")
		request.Header.Set(idempotencyKeyHeader, key)
		response := httptest.NewRecorder()
		fixture.handler.ServeHTTP(response, request)
		return response
	}
	created := create("create-managed-worker", body)
	if created.Code != http.StatusCreated || strings.Contains(created.Body.String(), "token") ||
		strings.Contains(created.Body.String(), "cipher") || strings.Contains(created.Body.String(), "remoteKey") {
		t.Fatalf("create credential = %d %s", created.Code, created.Body.String())
	}
	if revision := fixture.operations.SnapshotOperations().Cursor.Revision; revision != 1 {
		t.Fatalf("credential creation Operations revision = %d", revision)
	}
	replayed := create("create-managed-worker", body)
	if replayed.Code != http.StatusCreated || replayed.Header().Get("Idempotency-Replayed") != "true" ||
		replayed.Body.String() != created.Body.String() {
		t.Fatalf("create replay = %d headers=%v body=%s", replayed.Code, replayed.Header(), replayed.Body.String())
	}
	if revision := fixture.operations.SnapshotOperations().Cursor.Revision; revision != 1 {
		t.Fatalf("credential replay advanced Operations revision to %d", revision)
	}
	secondBody, err := json.Marshal(createCredentialRequest{
		CredentialID: "managed-worker-2", LLMGateway: gateway.Ref, Label: stringPointer("Managed worker 2"),
		GatewayPolicy: credentials.GatewayPolicy{ModelPolicies: []contracts.ModelPolicyRef{policy.Ref}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if second := create("create-managed-worker-2", secondBody); second.Code != http.StatusCreated {
		t.Fatalf("create second credential = %d %s", second.Code, second.Body.String())
	}
	if revision := fixture.operations.SnapshotOperations().Cursor.Revision; revision != 2 {
		t.Fatalf("second credential Operations revision = %d", revision)
	}
	invalid := create("create-invalid", []byte(`{
		"credentialId":"invalid-token-input",
		"llmGateway":{"gatewayId":"local-litellm","version":"1","digest":"sha256:1111111111111111111111111111111111111111111111111111111111111111"},
		"gatewayPolicy":{"modelPolicies":[]},
		"token":"must-not-be-accepted"
	}`))
	if invalid.Code != http.StatusBadRequest {
		t.Fatalf("secret-bearing create request = %d %s", invalid.Code, invalid.Body.String())
	}

	list := authenticatedRequest(http.MethodGet, "/v1/operations/credentials?limit=1", bytes.NewReader(nil))
	listed := httptest.NewRecorder()
	fixture.handler.ServeHTTP(listed, list)
	var firstPage credentialPageResponse
	if listed.Code != http.StatusOK || json.Unmarshal(listed.Body.Bytes(), &firstPage) != nil ||
		len(firstPage.Items) != 1 || firstPage.Items[0].CredentialID != "managed-worker" ||
		!firstPage.Page.HasMore || firstPage.Page.NextCursor == nil {
		t.Fatalf("list credentials = %d %s", listed.Code, listed.Body.String())
	}
	nextList := authenticatedRequest(
		http.MethodGet, "/v1/operations/credentials?limit=1&cursor="+*firstPage.Page.NextCursor,
		bytes.NewReader(nil),
	)
	nextListed := httptest.NewRecorder()
	fixture.handler.ServeHTTP(nextListed, nextList)
	var secondPage credentialPageResponse
	if nextListed.Code != http.StatusOK || json.Unmarshal(nextListed.Body.Bytes(), &secondPage) != nil ||
		len(secondPage.Items) != 1 || secondPage.Items[0].CredentialID != "managed-worker-2" ||
		secondPage.Page.HasMore || secondPage.Page.NextCursor != nil {
		t.Fatalf("next credential page = %d %s", nextListed.Code, nextListed.Body.String())
	}
	get := authenticatedRequest(http.MethodGet, "/v1/operations/credentials/managed-worker", bytes.NewReader(nil))
	got := httptest.NewRecorder()
	fixture.handler.ServeHTTP(got, get)
	if got.Code != http.StatusOK || got.Body.String() != created.Body.String() {
		t.Fatalf("get credential = %d %s", got.Code, got.Body.String())
	}

	deleteCall := func(key string) *httptest.ResponseRecorder {
		request := authenticatedRequest(
			http.MethodDelete, "/v1/operations/credentials/managed-worker", bytes.NewReader(nil),
		)
		request.Header.Set(idempotencyKeyHeader, key)
		response := httptest.NewRecorder()
		fixture.handler.ServeHTTP(response, request)
		return response
	}
	deleted := deleteCall("delete-managed-worker")
	if deleted.Code != http.StatusNoContent || deleted.Body.Len() != 0 {
		t.Fatalf("delete credential = %d %s", deleted.Code, deleted.Body.String())
	}
	if revision := fixture.operations.SnapshotOperations().Cursor.Revision; revision != 3 {
		t.Fatalf("credential deletion Operations revision = %d", revision)
	}
	if replay := deleteCall("delete-managed-worker"); replay.Code != http.StatusNoContent {
		t.Fatalf("delete replay = %d %s", replay.Code, replay.Body.String())
	}
	if revision := fixture.operations.SnapshotOperations().Cursor.Revision; revision != 3 {
		t.Fatalf("credential deletion replay advanced Operations revision to %d", revision)
	}
	missing := httptest.NewRecorder()
	fixture.handler.ServeHTTP(missing, get)
	if missing.Code != http.StatusNotFound {
		t.Fatalf("get deleted credential = %d %s", missing.Code, missing.Body.String())
	}
}

func TestCredentialInUseAndGatewayFailuresHaveBoundedPublicErrors(t *testing.T) {
	fixture := newHandlerFixture(t)
	fixture.credentials.records["managed-worker"] = credentials.Record{
		CredentialID: "managed-worker",
		LLMGateway: contracts.LLMGatewayConfigRef{
			GatewayID: "local-litellm", Version: "1", Digest: "sha256:" + strings.Repeat("1", 64),
		},
		EffectivePolicy: credentials.EffectiveGatewayPolicy{
			ModelPolicies: []contracts.ModelPolicyRef{{
				PolicyID: "worker", Version: "1", Digest: "sha256:" + strings.Repeat("2", 64),
			}}, Models: []string{"test-model"},
		},
		CreatedAt: time.Date(2026, 8, 30, 12, 0, 0, 0, time.UTC),
	}
	fixture.credentials.deleteErr = &credentials.CredentialInUseError{RunIDs: []string{"run-2", "run-1"}}
	request := authenticatedRequest(
		http.MethodDelete, "/v1/operations/credentials/managed-worker", bytes.NewReader(nil),
	)
	request.Header.Set(idempotencyKeyHeader, "delete-in-use")
	response := httptest.NewRecorder()
	fixture.handler.ServeHTTP(response, request)
	if response.Code != http.StatusConflict || !strings.Contains(response.Body.String(), `"kind":"credential_in_use"`) ||
		!strings.Contains(response.Body.String(), `"runIds":["run-2","run-1"]`) {
		t.Fatalf("credential-in-use response = %d %s", response.Code, response.Body.String())
	}
	fixture.credentials.deleteErr = errors.New("provider exposed sk-secret-body")
	response = httptest.NewRecorder()
	fixture.handler.ServeHTTP(response, request)
	if response.Code != http.StatusInternalServerError || strings.Contains(response.Body.String(), "secret") {
		t.Fatalf("unknown Gateway failure response = %d %s", response.Code, response.Body.String())
	}
	fixture.credentials.deleteErr = credentials.ErrGatewayUnavailable
	response = httptest.NewRecorder()
	fixture.handler.ServeHTTP(response, request)
	if response.Code != http.StatusBadGateway || strings.Contains(response.Body.String(), "secret") {
		t.Fatalf("Gateway unavailable response = %d %s", response.Code, response.Body.String())
	}
}

func TestArtifactCreateUpdateAndExactRead(t *testing.T) {
	fixture := newHandlerFixture(t)
	create := authenticatedRequest(http.MethodPut, "/v1/artifacts/projects/source", bytes.NewReader([]byte("first")))
	create.Header.Set("Content-Type", "text/plain")
	create.Header.Set("If-None-Match", "*")
	created := httptest.NewRecorder()
	fixture.handler.ServeHTTP(created, create)
	if created.Code != http.StatusCreated || created.Header().Get("ETag") == "" {
		t.Fatalf("create = status %d, ETag %q, body %s", created.Code, created.Header().Get("ETag"), created.Body.String())
	}

	update := authenticatedRequest(http.MethodPut, "/v1/artifacts/projects/source", bytes.NewReader([]byte("second")))
	update.Header.Set("Content-Type", "text/plain")
	update.Header.Set("If-Match", created.Header().Get("ETag"))
	updated := httptest.NewRecorder()
	fixture.handler.ServeHTTP(updated, update)
	if updated.Code != http.StatusOK || updated.Header().Get("ETag") == created.Header().Get("ETag") {
		t.Fatalf("update = status %d, ETag %q, body %s", updated.Code, updated.Header().Get("ETag"), updated.Body.String())
	}

	exact := authenticatedRequest(http.MethodGet, "/v1/artifacts/projects/source?revision=revision-1", bytes.NewReader(nil))
	read := httptest.NewRecorder()
	fixture.handler.ServeHTTP(read, exact)
	if read.Code != http.StatusOK || read.Body.String() != "first" || read.Header().Get("ETag") != `"revision-1"` {
		t.Fatalf("exact read = status %d, ETag %q, body %q", read.Code, read.Header().Get("ETag"), read.Body.String())
	}
}

func TestArtifactResponseLossRetryFailsCASWithoutAnotherRevision(t *testing.T) {
	fixture := newHandlerFixture(t)
	create := func() *httptest.ResponseRecorder {
		request := authenticatedRequest(
			http.MethodPut,
			"/v1/artifacts/projects/source",
			bytes.NewReader([]byte("first")),
		)
		request.Header.Set("Content-Type", "text/plain")
		request.Header.Set("If-None-Match", "*")
		response := httptest.NewRecorder()
		fixture.handler.ServeHTTP(response, request)
		return response
	}
	first := create()
	if first.Code != http.StatusCreated {
		t.Fatalf("first create = %d %s", first.Code, first.Body.String())
	}
	retry := create()
	if retry.Code != http.StatusConflict || fixture.repository.next != 1 {
		t.Fatalf("lost-response create retry = status %d revisions %d body %s", retry.Code, fixture.repository.next, retry.Body.String())
	}

	oldETag := first.Header().Get("ETag")
	update := func() *httptest.ResponseRecorder {
		request := authenticatedRequest(
			http.MethodPut,
			"/v1/artifacts/projects/source",
			bytes.NewReader([]byte("second")),
		)
		request.Header.Set("Content-Type", "text/plain")
		request.Header.Set("If-Match", oldETag)
		response := httptest.NewRecorder()
		fixture.handler.ServeHTTP(response, request)
		return response
	}
	accepted := update()
	if accepted.Code != http.StatusOK {
		t.Fatalf("first update = %d %s", accepted.Code, accepted.Body.String())
	}
	lostResponseRetry := update()
	if lostResponseRetry.Code != http.StatusConflict || fixture.repository.next != 2 {
		t.Fatalf("lost-response update retry = status %d revisions %d body %s", lostResponseRetry.Code, fixture.repository.next, lostResponseRetry.Body.String())
	}
	current := authenticatedRequest(
		http.MethodGet,
		"/v1/artifacts/projects/source",
		bytes.NewReader(nil),
	)
	read := httptest.NewRecorder()
	fixture.handler.ServeHTTP(read, current)
	if read.Code != http.StatusOK || read.Body.String() != "second" ||
		read.Header().Get("ETag") != accepted.Header().Get("ETag") {
		t.Fatalf("current artifact after retry = %d etag=%q body=%q", read.Code, read.Header().Get("ETag"), read.Body.String())
	}
}

func TestArtifactUploadIsBoundedBeforeRepository(t *testing.T) {
	fixture := newHandlerFixture(t)
	request := authenticatedRequest(
		http.MethodPut, "/v1/artifacts/projects/large",
		bytes.NewReader(bytes.Repeat([]byte("x"), artifacts.MaxPayloadSize+1)),
	)
	request.Header.Set("Content-Type", "application/octet-stream")
	response := httptest.NewRecorder()
	fixture.handler.ServeHTTP(response, request)
	if response.Code != http.StatusRequestEntityTooLarge || fixture.repository.writes != 0 {
		t.Fatalf("oversized upload = status %d, writes %d", response.Code, fixture.repository.writes)
	}
	assertErrorCode(t, response, "artifact_too_large")
}

func TestCreateRunStrictValidationOccursBeforeTransaction(t *testing.T) {
	fixture := newHandlerFixture(t)
	requests := []string{
		`{"workflow":"artifact-copy@1","labels":null,"parameters":{},"artifacts":{"source":{"namespace":"projects","name":"source"}}}`,
		`{"workflow":"artifact-copy@1","parameters":{"objective":42},"artifacts":{"source":{"namespace":"projects","name":"source"}}}`,
		`{"workflow":"artifact-copy@1","parameters":{},"artifacts":{},"extra":true}`,
		`{"workflow":"artifact-copy@1","parameters":{},"artifacts":{}}`,
		`{"workflow":"artifact-copy@1","parameters":{},"artifacts":{"source":{"namespace":"projects","name":"source"}},"executionConfig":null}`,
		`{"workflow":"artifact-copy@1","parameters":{},"artifacts":{"source":{"namespace":"projects","name":"source"}},"executionConfig":{"workers":{"modelPolicy":null}}}`,
		`{"workflow":"artifact-copy@1","parameters":{},"artifacts":{"source":{"namespace":"projects","name":"source"}},"executionConfig":{"workers":{}}}`,
	}
	for _, body := range requests {
		request := authenticatedRequest(http.MethodPost, "/v1/runs", bytes.NewReader([]byte(body)))
		response := httptest.NewRecorder()
		fixture.handler.ServeHTTP(response, request)
		if response.Code != http.StatusBadRequest {
			t.Fatalf("invalid Run request status = %d, body %s", response.Code, response.Body.String())
		}
		assertErrorCode(t, response, "invalid_request")
	}
	if fixture.unit.calls != 0 || len(fixture.runs.runs) != 0 || fixture.notifier.calls != 0 {
		t.Fatalf("invalid requests used transaction %d times or created %d Runs", fixture.unit.calls, len(fixture.runs.runs))
	}
}

func TestCreateRunForksInputAndReturnsRunning(t *testing.T) {
	fixture := newHandlerFixture(t)
	user, _ := fixture.artifacts.User("user-1")
	written, err := user.Write(
		t.Context(), contracts.ArtifactRef{Namespace: "projects", Name: "source"},
		artifacts.Payload{MediaType: "text/plain", Data: []byte("source")}, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	body := `{"workflow":"artifact-copy@1","parameters":{"objective":"copy"},"artifacts":{"source":{"namespace":"projects","name":"source"}}}`
	request := authenticatedRequest(http.MethodPost, "/v1/runs", bytes.NewReader([]byte(body)))
	response := httptest.NewRecorder()
	fixture.handler.ServeHTTP(response, request)
	if response.Code != http.StatusAccepted {
		t.Fatalf("create Run = status %d, body %s", response.Code, response.Body.String())
	}
	if fixture.notifier.calls != 1 {
		t.Fatalf("Scheduler wake calls = %d", fixture.notifier.calls)
	}
	if fixture.credentials.guarded != 1 {
		t.Fatalf("Run initialization guard calls = %d", fixture.credentials.guarded)
	}
	run := fixture.runs.runs["run_fixed"]
	if run.State != runstore.RunRunning || run.OwnerID != "user-1" || len(run.WorkflowSnapshot) == 0 {
		t.Fatalf("stored Run = %+v", run)
	}
	runArtifacts, _ := fixture.artifacts.Run("run_fixed")
	forked, err := runArtifacts.Read(t.Context(), contracts.ArtifactRef{Namespace: "inputs", Name: "source"})
	if err != nil || string(forked.Payload.Data) != "source" || written.Ref.Revision == nil {
		t.Fatalf("forked input = (%+v, %v)", forked, err)
	}
}

func TestCreateRunResponseLossRetryReturnsExistingRun(t *testing.T) {
	fixture := newHandlerFixture(t)
	user, _ := fixture.artifacts.User("user-1")
	if _, err := user.Write(
		t.Context(), contracts.ArtifactRef{Namespace: "projects", Name: "source"},
		artifacts.Payload{MediaType: "text/plain", Data: []byte("source")}, nil,
	); err != nil {
		t.Fatal(err)
	}
	body := []byte(`{"workflow":"artifact-copy@1","parameters":{"objective":"copy"},"artifacts":{"source":{"namespace":"projects","name":"source"}}}`)

	first := authenticatedRequest(http.MethodPost, "/v1/runs", bytes.NewReader(body))
	firstResponse := httptest.NewRecorder()
	fixture.handler.ServeHTTP(firstResponse, first)
	if firstResponse.Code != http.StatusAccepted {
		t.Fatalf("first create = %d %s", firstResponse.Code, firstResponse.Body.String())
	}

	retry := authenticatedRequest(http.MethodPost, "/v1/runs", bytes.NewReader(body))
	retryResponse := httptest.NewRecorder()
	fixture.handler.ServeHTTP(retryResponse, retry)
	if retryResponse.Code != http.StatusAccepted ||
		retryResponse.Header().Get("Idempotency-Replayed") != "true" ||
		retryResponse.Body.String() != firstResponse.Body.String() {
		t.Fatalf("retry create = %d headers=%v body=%s", retryResponse.Code, retryResponse.Header(), retryResponse.Body.String())
	}
	if len(fixture.runs.runs) != 1 || fixture.notifier.calls != 1 || fixture.repository.writes != 1 {
		t.Fatalf("retry side effects = runs:%d wakes:%d artifact writes:%d",
			len(fixture.runs.runs), fixture.notifier.calls, fixture.repository.writes)
	}

	different := []byte(`{"workflow":"artifact-copy@1","parameters":{"objective":"different"},"artifacts":{"source":{"namespace":"projects","name":"source"}}}`)
	conflict := authenticatedRequest(http.MethodPost, "/v1/runs", bytes.NewReader(different))
	conflictResponse := httptest.NewRecorder()
	fixture.handler.ServeHTTP(conflictResponse, conflict)
	if conflictResponse.Code != http.StatusConflict {
		t.Fatalf("idempotency key reuse = %d %s", conflictResponse.Code, conflictResponse.Body.String())
	}
}

func TestCreateRunPinsCanonicalLabelsAndReplaySkipsCurrentBindings(t *testing.T) {
	fixture := newHandlerFixture(t)
	user, _ := fixture.artifacts.User("user-1")
	if _, err := user.Write(
		t.Context(), contracts.ArtifactRef{Namespace: "projects", Name: "source"},
		artifacts.Payload{MediaType: "text/plain", Data: []byte("source")}, nil,
	); err != nil {
		t.Fatal(err)
	}
	pinCalls := 0
	fixture.runs.pinRuntimeLabels = func(
		_ context.Context, labels []string, _ config.CredentialLookup,
	) (runtimeconfig.RunSnapshot, error) {
		pinCalls++
		if strings.Join(labels, ",") != "debug,trace" {
			t.Fatalf("canonical labels = %v", labels)
		}
		result := runtimeconfig.BuiltInRunSnapshot()
		result.Labels = []runtimeconfig.PinnedLabel{
			{Label: "debug", Explicit: true, BindingRevision: 7, Config: result.Default.Config},
			{Label: "trace", Explicit: true, BindingRevision: 9, Config: result.Default.Config},
		}
		return result, nil
	}
	requestBody := func(labels string) []byte {
		return []byte(`{"workflow":"artifact-copy@1","labels":` + labels +
			`,"parameters":{"objective":"copy"},"artifacts":{"source":{"namespace":"projects","name":"source"}}}`)
	}
	first := authenticatedRequest(http.MethodPost, "/v1/runs", bytes.NewReader(requestBody(`["trace","debug"]`)))
	first.Header.Set(idempotencyKeyHeader, "label-replay")
	firstResponse := httptest.NewRecorder()
	fixture.handler.ServeHTTP(firstResponse, first)
	if firstResponse.Code != http.StatusAccepted || pinCalls != 1 {
		t.Fatalf("first labeled Run = %d calls=%d body=%s", firstResponse.Code, pinCalls, firstResponse.Body.String())
	}
	var created createRunResponse
	if err := json.Unmarshal(firstResponse.Body.Bytes(), &created); err != nil ||
		strings.Join(created.Labels, ",") != "debug,trace" ||
		created.RuntimeConfiguration.Labels[0].BindingRevision != "7" {
		t.Fatalf("labeled Run response = (%+v, %v)", created, err)
	}

	// Simulate removal of both mutable bindings after the original commit.
	fixture.runs.pinRuntimeLabels = func(
		context.Context, []string, config.CredentialLookup,
	) (runtimeconfig.RunSnapshot, error) {
		pinCalls++
		return runtimeconfig.RunSnapshot{}, runtimeconfig.ErrNotFound
	}
	retry := authenticatedRequest(http.MethodPost, "/v1/runs", bytes.NewReader(requestBody(`["debug","trace"]`)))
	retry.Header.Set(idempotencyKeyHeader, "label-replay")
	retryResponse := httptest.NewRecorder()
	fixture.handler.ServeHTTP(retryResponse, retry)
	if retryResponse.Code != http.StatusAccepted ||
		retryResponse.Header().Get("Idempotency-Replayed") != "true" ||
		retryResponse.Body.String() != firstResponse.Body.String() || pinCalls != 1 {
		t.Fatalf("binding-independent replay = %d calls=%d headers=%v body=%s",
			retryResponse.Code, pinCalls, retryResponse.Header(), retryResponse.Body.String())
	}
}

func TestCreateRunRejectsMalformedAndUnknownLabels(t *testing.T) {
	for _, test := range []struct {
		name, labels, code string
	}{
		{name: "duplicate", labels: `["debug","debug"]`, code: "runtime_config_invalid"},
		{name: "reserved", labels: `["default"]`, code: "runtime_config_invalid"},
		{name: "unknown", labels: `["missing"]`, code: "runtime_label_unknown"},
	} {
		t.Run(test.name, func(t *testing.T) {
			fixture := newHandlerFixture(t)
			body := []byte(`{"workflow":"artifact-copy@1","labels":` + test.labels +
				`,"parameters":{},"artifacts":{"source":{"namespace":"projects","name":"source"}}}`)
			request := authenticatedRequest(http.MethodPost, "/v1/runs", bytes.NewReader(body))
			request.Header.Set(idempotencyKeyHeader, "invalid-label-"+test.name)
			response := httptest.NewRecorder()
			fixture.handler.ServeHTTP(response, request)
			if response.Code != http.StatusBadRequest {
				t.Fatalf("label rejection = %d %s", response.Code, response.Body.String())
			}
			assertErrorCode(t, response, test.code)
			if test.name != "unknown" && fixture.unit.calls != 0 {
				t.Fatalf("malformed label entered transaction %d times", fixture.unit.calls)
			}
		})
	}
}

func TestCreateRunReplayDoesNotResolveDeletedManagedCredential(t *testing.T) {
	fixture := newHandlerFixture(t)
	user, _ := fixture.artifacts.User("user-1")
	if _, err := user.Write(
		t.Context(), contracts.ArtifactRef{Namespace: "projects", Name: "source"},
		artifacts.Payload{MediaType: "text/plain", Data: []byte("source")}, nil,
	); err != nil {
		t.Fatal(err)
	}
	resolved, err := fixture.configs.ResolveRunWorkflow(
		t.Context(), "artifact-copy@1", config.ExecutionConfigPatch{}, fixture.credentials,
	)
	if err != nil {
		t.Fatal(err)
	}
	gateway := resolved.Stages[resolved.EntryStage].ExecutionConfig.Agents["builder"].LLMGateway.Ref
	fixture.credentials.records["ephemeral-worker"] = credentials.Record{
		CredentialID: "ephemeral-worker", LLMGateway: gateway,
		CreatedAt: time.Date(2026, 8, 30, 12, 0, 0, 0, time.UTC),
	}
	body := []byte(`{"workflow":"artifact-copy@1","parameters":{"objective":"copy"},"artifacts":{"source":{"namespace":"projects","name":"source"}},"executionConfig":{"workers":{"credential":"ephemeral-worker"}}}`)

	first := authenticatedRequest(http.MethodPost, "/v1/runs", bytes.NewReader(body))
	first.Header.Set(idempotencyKeyHeader, "credential-replay")
	firstResponse := httptest.NewRecorder()
	fixture.handler.ServeHTTP(firstResponse, first)
	if firstResponse.Code != http.StatusAccepted {
		t.Fatalf("first create = %d %s", firstResponse.Code, firstResponse.Body.String())
	}
	fixture.credentials.mu.Lock()
	delete(fixture.credentials.records, "ephemeral-worker")
	fixture.credentials.mu.Unlock()

	retry := authenticatedRequest(http.MethodPost, "/v1/runs", bytes.NewReader(body))
	retry.Header.Set(idempotencyKeyHeader, "credential-replay")
	retryResponse := httptest.NewRecorder()
	fixture.handler.ServeHTTP(retryResponse, retry)
	if retryResponse.Code != http.StatusAccepted ||
		retryResponse.Header().Get("Idempotency-Replayed") != "true" ||
		retryResponse.Body.String() != firstResponse.Body.String() {
		t.Fatalf("credential-independent replay = %d headers=%v body=%s",
			retryResponse.Code, retryResponse.Header(), retryResponse.Body.String())
	}
	if fixture.credentials.guarded != 1 {
		t.Fatalf("replay entered mutable credential resolution %d times", fixture.credentials.guarded)
	}
}

func TestCreateRunPinsExecutionConfigAndDetectsSelectorIdempotencyConflict(t *testing.T) {
	fixture := newHandlerFixtureWithConfig(t, "../../../configs")
	user, _ := fixture.artifacts.User("user-1")
	if _, err := user.Write(
		t.Context(), contracts.ArtifactRef{Namespace: "projects", Name: "source"},
		artifacts.Payload{MediaType: "text/plain", Data: []byte("source")}, nil,
	); err != nil {
		t.Fatal(err)
	}
	requestBody := func(policy string) []byte {
		return []byte(`{"workflow":"artifact-copy@1","parameters":{},"artifacts":{"source":{"namespace":"projects","name":"source"}},"executionConfig":{"workers":{"modelPolicy":"` + policy + `"}}}`)
	}

	first := authenticatedRequest(http.MethodPost, "/v1/runs", bytes.NewReader(requestBody("worker@1")))
	firstResponse := httptest.NewRecorder()
	fixture.handler.ServeHTTP(firstResponse, first)
	if firstResponse.Code != http.StatusAccepted {
		t.Fatalf("first create = %d %s", firstResponse.Code, firstResponse.Body.String())
	}
	stored := fixture.runs.runs["run_fixed"]
	var workflow config.ResolvedWorkflow
	if err := json.Unmarshal(stored.WorkflowSnapshot, &workflow); err != nil {
		t.Fatal(err)
	}
	selection := workflow.Stages["copy"].ExecutionConfig.Agents["builder"]
	if selection.ModelPolicy.Ref.PolicyID != "worker" ||
		selection.Origins.ModelPolicy != "run.executionConfig.workers" {
		t.Fatalf("stored resolved executionConfig = %+v", selection)
	}

	conflict := authenticatedRequest(
		http.MethodPost, "/v1/runs", bytes.NewReader(requestBody("domain_worker@1")),
	)
	conflictResponse := httptest.NewRecorder()
	fixture.handler.ServeHTTP(conflictResponse, conflict)
	if conflictResponse.Code != http.StatusConflict || len(fixture.runs.runs) != 1 || fixture.repository.writes != 1 {
		t.Fatalf(
			"selector conflict = status:%d runs:%d writes:%d body:%s",
			conflictResponse.Code, len(fixture.runs.runs), fixture.repository.writes,
			conflictResponse.Body.String(),
		)
	}
}

func TestCreateRunPinsNamedEscalationVariant(t *testing.T) {
	fixture := newHandlerFixtureWithConfig(t, "../../../configs")
	user, _ := fixture.artifacts.User("user-1")
	if _, err := user.Write(
		t.Context(), contracts.ArtifactRef{Namespace: "projects", Name: "source-archive"},
		artifacts.Payload{MediaType: "application/zip", Data: []byte("source")}, nil,
	); err != nil {
		t.Fatal(err)
	}
	body := []byte(`{"workflow":"openapi-from-source@1","parameters":{},"artifacts":{"source":{"namespace":"projects","name":"source-archive"}}}`)
	request := authenticatedRequest(http.MethodPost, "/v1/runs", bytes.NewReader(body))
	response := httptest.NewRecorder()
	fixture.handler.ServeHTTP(response, request)
	if response.Code != http.StatusAccepted {
		t.Fatalf("create = %d %s", response.Code, response.Body.String())
	}
	stored := fixture.runs.runs["run_fixed"]
	var workflow config.ResolvedWorkflow
	if err := json.Unmarshal(stored.WorkflowSnapshot, &workflow); err != nil {
		t.Fatal(err)
	}
	variant := workflow.Stages["openapi_validate"].On.Failed.Escalate.ExecutionConfig
	if variant.Ref == nil || variant.Ref.ConfigID != "strong-oas-review" ||
		variant.Effective.Agents["validator"].ModelPolicy.Ref.PolicyID != "strong_domain_worker" {
		t.Fatalf("stored escalation variant = %+v", variant)
	}
}

func TestCreateRunRequiresValidIdempotencyKey(t *testing.T) {
	fixture := newHandlerFixture(t)
	body := []byte(`{"workflow":"artifact-copy@1","parameters":{},"artifacts":{"source":{"namespace":"projects","name":"source"}}}`)
	for _, key := range []string{"", "two words", strings.Repeat("x", 129)} {
		request := authenticatedRequest(http.MethodPost, "/v1/runs", bytes.NewReader(body))
		request.Header.Del(idempotencyKeyHeader)
		if key != "" {
			request.Header.Set(idempotencyKeyHeader, key)
		}
		response := httptest.NewRecorder()
		fixture.handler.ServeHTTP(response, request)
		if response.Code != http.StatusBadRequest {
			t.Fatalf("key %q status = %d %s", key, response.Code, response.Body.String())
		}
	}
}

func TestCreateRunDigestNormalizesEquivalentEmptyMappings(t *testing.T) {
	omitted, err := createRunRequestDigest(createRunRequest{Workflow: "empty@1"})
	if err != nil {
		t.Fatal(err)
	}
	explicit, err := createRunRequestDigest(createRunRequest{
		Workflow: "empty@1", Parameters: map[string]string{},
		Artifacts: map[string]contracts.ArtifactRef{},
	})
	if err != nil || explicit != omitted {
		t.Fatalf("semantic request digests = omitted:%q explicit:%q error:%v", omitted, explicit, err)
	}
	var emptyExecutionConfig config.ExecutionConfigPatch
	if err := json.Unmarshal([]byte(`{"stages":{}}`), &emptyExecutionConfig); err != nil {
		t.Fatal(err)
	}
	withEmptyExecutionConfig, err := createRunRequestDigest(createRunRequest{
		Workflow: "empty@1", ExecutionConfig: emptyExecutionConfig,
	})
	if err != nil || withEmptyExecutionConfig != omitted {
		t.Fatalf(
			"empty executionConfig digest = %q, omitted = %q, error = %v",
			withEmptyExecutionConfig, omitted, err,
		)
	}
	ordered, err := createRunRequestDigest(createRunRequest{
		Workflow: "empty@1", Labels: runLabels{"debug", "trace"},
	})
	if err != nil {
		t.Fatal(err)
	}
	reversed, err := createRunRequestDigest(createRunRequest{
		Workflow: "empty@1", Labels: runLabels{"trace", "debug"},
	})
	if err != nil || reversed != ordered || reversed == omitted {
		t.Fatalf("label request digests = ordered:%q reversed:%q omitted:%q error:%v", ordered, reversed, omitted, err)
	}
}

type recordingRunNotifier struct {
	calls         int
	cancellations []string
}

func (n *recordingRunNotifier) Wake() { n.calls++ }
func (n *recordingRunNotifier) Cancel(runID string) {
	n.cancellations = append(n.cancellations, runID)
}

func TestCancelRunIsOwnedStrictDurableAndIdempotent(t *testing.T) {
	fixture := newHandlerFixture(t)
	fixture.runs.runs["run-cancel"] = runstore.WorkflowRun{
		RunID: "run-cancel", OwnerID: "user-1", WorkflowName: "artifact-copy", WorkflowVersion: "1",
		State: runstore.RunRunning,
	}

	request := authenticatedRequest(
		http.MethodPost, "/v1/runs/run-cancel/cancel",
		bytes.NewReader([]byte(`{"reason":"  user changed direction  "}`)),
	)
	response := httptest.NewRecorder()
	fixture.handler.ServeHTTP(response, request)
	if response.Code != http.StatusAccepted {
		t.Fatalf("cancel Run = status %d, body %s", response.Code, response.Body.String())
	}
	var first cancelRunResponse
	if err := json.Unmarshal(response.Body.Bytes(), &first); err != nil {
		t.Fatal(err)
	}
	if first.State != runstore.RunCancelling || first.Cancellation == nil ||
		first.Cancellation.Reason == nil || *first.Cancellation.Reason != "user changed direction" ||
		first.Cancellation.RequestedBy == nil || *first.Cancellation.RequestedBy != "user-1" {
		t.Fatalf("cancel response = %+v", first)
	}

	repeat := authenticatedRequest(
		http.MethodPost, "/v1/runs/run-cancel/cancel",
		bytes.NewReader([]byte(`{"reason":"must not replace winner"}`)),
	)
	repeated := httptest.NewRecorder()
	fixture.handler.ServeHTTP(repeated, repeat)
	stored := fixture.runs.runs["run-cancel"]
	if repeated.Code != http.StatusAccepted || stored.Cancellation == nil ||
		stored.Cancellation.Reason == nil || *stored.Cancellation.Reason != "user changed direction" ||
		len(fixture.notifier.cancellations) != 2 {
		t.Fatalf("repeat = status %d, cancellation %+v, notifications %v", repeated.Code, stored.Cancellation, fixture.notifier.cancellations)
	}

	terminalRun := fixture.runs.runs["run-cancel"]
	terminalRun.State = runstore.RunCancelled
	fixture.runs.runs["run-cancel"] = terminalRun
	terminal := httptest.NewRecorder()
	fixture.handler.ServeHTTP(terminal, authenticatedRequest(
		http.MethodPost, "/v1/runs/run-cancel/cancel", bytes.NewReader([]byte(`{}`)),
	))
	if terminal.Code != http.StatusOK || len(fixture.notifier.cancellations) != 2 {
		t.Fatalf("terminal repeat = status %d, notifications %v", terminal.Code, fixture.notifier.cancellations)
	}
}

func TestCancelRunRejectsInvalidOrForeignRequests(t *testing.T) {
	fixture := newHandlerFixture(t)
	fixture.runs.runs["foreign"] = runstore.WorkflowRun{RunID: "foreign", OwnerID: "user-2", State: runstore.RunRunning}
	fixture.runs.runs["owned"] = runstore.WorkflowRun{RunID: "owned", OwnerID: "user-1", State: runstore.RunSucceeded}

	tests := []struct {
		method string
		target string
		body   string
		status int
	}{
		{http.MethodPost, "/v1/runs/foreign/cancel", `{}`, http.StatusNotFound},
		{http.MethodPost, "/v1/runs/owned/cancel?force=true", `{}`, http.StatusBadRequest},
		{http.MethodPost, "/v1/runs/owned/cancel", `{"reason":" "}`, http.StatusBadRequest},
		{http.MethodPost, "/v1/runs/owned/cancel", `{"unexpected":true}`, http.StatusBadRequest},
		{http.MethodPost, "/v1/runs/owned/cancel", `null`, http.StatusBadRequest},
		{http.MethodGet, "/v1/runs/owned/cancel", `{}`, http.StatusMethodNotAllowed},
	}
	for _, test := range tests {
		response := httptest.NewRecorder()
		fixture.handler.ServeHTTP(response, authenticatedRequest(test.method, test.target, bytes.NewReader([]byte(test.body))))
		if response.Code != test.status {
			t.Errorf("%s %s = %d, want %d: %s", test.method, test.target, response.Code, test.status, response.Body.String())
		}
	}
	if fixture.runs.runs["owned"].Cancellation != nil || len(fixture.notifier.cancellations) != 0 {
		t.Fatal("invalid cancellation changed state or notified Scheduler")
	}
}

func TestRunStatusExposesSafeMetricsAndAttemptDiagnostics(t *testing.T) {
	fixture := newHandlerFixture(t)
	snapshot, err := config.Load("../../config/testdata/valid", config.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	workflow, err := snapshot.Workflow("artifact-copy@1")
	if err != nil {
		t.Fatal(err)
	}
	stageSnapshot, err := json.Marshal(workflow.Stages[workflow.EntryStage])
	if err != nil {
		t.Fatal(err)
	}
	fixture.runs.runs["run-metrics"] = runstore.WorkflowRun{
		RunID: "run-metrics", OwnerID: "user-1", WorkflowName: "artifact-copy",
		WorkflowVersion: "1", State: runstore.RunRunning,
	}
	fixture.runs.executions["run-metrics"] = []runstore.StageExecution{{
		StageExecutionID: "stage-metrics", RunID: "run-metrics", StageName: "copy",
		Attempt: 1, ExecutionConfigVariant: runstore.StageExecutionConfigBase,
		StageSpecSnapshot: stageSnapshot, State: runstore.StageRunning,
	}}
	retryable := true
	normalizedWorker, err := telemetry.NewPolicy("must-not-be-public").NormalizeExecutionReport(
		contracts.ExecutionReport{
			ReportID: "worker-secret", Complete: true,
			Metrics: contracts.ExecutionMetrics{Tools: map[string]contracts.ToolMetrics{}},
			ToolCalls: []contracts.ToolCallRecord{{
				CallID: "call-secret", Tool: "probe",
				Arguments: map[string]any{"token": "must-not-be-public"},
				Outcome:   contracts.ToolCallSucceeded,
			}},
			Errors: []contracts.ExecutionError{
				{
					Code:      "worker_result_schema_json_invalid",
					Message:   "Worker result did not match StageContentResult",
					Retryable: &retryable,
				},
				{
					Code:    "gateway_error",
					Message: "provider must-not-be-public at https://provider.example.test/v1",
				},
			},
		},
		telemetry.MaxReportJSONBytes,
	)
	if err != nil {
		t.Fatal(err)
	}
	fixture.metrics.records["stage-metrics"] = telemetry.StageMetricsRecord{
		StageExecutionID: "stage-metrics",
		Metrics: contracts.StageMetrics{
			Workers: map[string]contracts.ExecutionReport{
				"builder": normalizedWorker,
			},
			Runtime: map[string]contracts.RuntimeReport{"builder": {Complete: true}},
		},
		Summary: telemetry.Summary{
			ReportsComplete: true, ModelCalls: 2, ToolCalls: 1, ErrorCount: 2,
		},
	}

	response := httptest.NewRecorder()
	fixture.handler.ServeHTTP(response, authenticatedRequest(
		http.MethodGet, "/v1/runs/run-metrics", bytes.NewReader(nil),
	))
	if response.Code != http.StatusOK {
		t.Fatalf("status response = %d: %s", response.Code, response.Body.String())
	}
	if strings.Contains(response.Body.String(), "must-not-be-public") ||
		strings.Contains(response.Body.String(), "call-secret") ||
		strings.Contains(response.Body.String(), "worker-secret") ||
		strings.Contains(response.Body.String(), "https://provider.example.test") {
		t.Fatalf("public status leaked detailed telemetry: %s", response.Body.String())
	}
	var result runStatusResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if len(result.Attempts) != 1 || result.Attempts[0].Metrics == nil ||
		result.Attempts[0].Metrics.ModelCalls != 2 || result.Attempts[0].Metrics.ToolCalls != 1 {
		t.Fatalf("public metrics summary = %+v", result.Attempts)
	}
	diagnostics := result.Attempts[0].Diagnostics
	if diagnostics == nil || diagnostics.Truncated || len(diagnostics.Items) != 2 ||
		diagnostics.Items[0].Participant != telemetry.AttemptDiagnosticWorker ||
		diagnostics.Items[0].LogicalAgent != "builder" ||
		diagnostics.Items[0].Code != "worker_result_schema_json_invalid" ||
		diagnostics.Items[0].Retryable == nil || !*diagnostics.Items[0].Retryable ||
		!strings.Contains(diagnostics.Items[1].Message, "[REDACTED_URL]") {
		t.Fatalf("public attempt diagnostics = %+v", diagnostics)
	}
	if result.Attempts[0].ExecutionConfig.Variant != runstore.StageExecutionConfigBase ||
		result.Attempts[0].ExecutionConfig.Agents["builder"].ModelPolicy.PolicyID != "worker" {
		t.Fatalf("public effective executionConfig refs = %+v", result.Attempts[0].ExecutionConfig)
	}
}

func TestRunStatusExposesEscalationLineageAndSafeEffectiveRefs(t *testing.T) {
	fixture := newHandlerFixture(t)
	snapshot, err := config.Load("../../../configs", config.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	workflow, err := snapshot.Workflow("openapi-from-source@1")
	if err != nil {
		t.Fatal(err)
	}
	stage := workflow.Stages["openapi_validate"]
	action := stage.On.Failed
	if action.Escalate == nil || action.Escalate.ExecutionConfig.Ref == nil {
		t.Fatal("repository Workflow has no named failed escalation")
	}
	baseSnapshot, err := json.Marshal(stage)
	if err != nil {
		t.Fatal(err)
	}
	stage.ExecutionConfig = action.Escalate.ExecutionConfig.Effective
	escalatedSnapshot, err := json.Marshal(stage)
	if err != nil {
		t.Fatal(err)
	}
	previous, ordinal := "stage-base", 1
	fixture.runs.runs["run-escalation"] = runstore.WorkflowRun{
		RunID: "run-escalation", OwnerID: "user-1", WorkflowName: "openapi-from-source",
		WorkflowVersion: "1", State: runstore.RunFailed,
	}
	fixture.runs.executions["run-escalation"] = []runstore.StageExecution{
		{
			StageExecutionID: "stage-base", RunID: "run-escalation", StageName: "openapi_validate",
			Attempt: 1, ExecutionConfigVariant: runstore.StageExecutionConfigBase,
			StageSpecSnapshot: baseSnapshot, State: runstore.StageFailed,
		},
		{
			StageExecutionID: "stage-escalated", RunID: "run-escalation", StageName: "openapi_validate",
			Attempt: 2, PreviousExecutionID: &previous,
			ExecutionConfigVariant: runstore.StageExecutionConfigFailedEscalation,
			EscalationOrdinal:      &ordinal,
			StageSpecSnapshot:      escalatedSnapshot, State: runstore.StageFailed,
		},
	}
	fixture.runs.decisions["run-escalation"] = []runstore.StageTransitionDecision{
		{
			SourceExecutionID: "stage-base", RunID: "run-escalation",
			Action: runstore.StageTransitionEscalate, TargetStageName: stringTestPointer("openapi_validate"),
			TargetExecutionID: stringTestPointer("stage-escalated"), EscalationOrdinal: &ordinal,
		},
		{
			SourceExecutionID: "stage-escalated", RunID: "run-escalation",
			Action: runstore.StageTransitionFail, EscalationOrdinal: &ordinal,
			EscalationExhausted: true,
		},
	}

	response := httptest.NewRecorder()
	fixture.handler.ServeHTTP(response, authenticatedRequest(
		http.MethodGet, "/v1/runs/run-escalation", bytes.NewReader(nil),
	))
	if response.Code != http.StatusOK {
		t.Fatalf("status response = %d: %s", response.Code, response.Body.String())
	}
	var result runStatusResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if len(result.Attempts) != 2 || len(result.Transitions) != 2 ||
		result.Attempts[1].PreviousExecutionID == nil || *result.Attempts[1].PreviousExecutionID != "stage-base" ||
		result.Attempts[1].ExecutionConfig.Ref == nil ||
		result.Attempts[1].ExecutionConfig.Ref.ConfigID != "strong-oas-review" ||
		result.Attempts[1].ExecutionConfig.Agents["validator"].ModelPolicy.PolicyID != "strong_domain_worker" ||
		result.Transitions[0].Action != runstore.StageTransitionEscalate ||
		!result.Transitions[1].EscalationExhausted {
		t.Fatalf("public escalation read model = %+v", result)
	}
	if strings.Contains(response.Body.String(), "http://litellm") ||
		strings.Contains(response.Body.String(), "credential") {
		t.Fatalf("public escalation read model leaked connection details: %s", response.Body.String())
	}
}

func stringTestPointer(value string) *string { return &value }

func TestRunOutputDownloadRequiresOwnerAndReturnsExactMetadata(t *testing.T) {
	fixture := newHandlerFixture(t)
	fixture.runs.runs["run-owned"] = runstore.WorkflowRun{
		RunID: "run-owned", OwnerID: "user-1", WorkflowName: "artifact-copy", WorkflowVersion: "1", State: runstore.RunSucceeded,
	}
	runArtifacts, _ := fixture.artifacts.Run("run-owned")
	source, err := runArtifacts.Write(
		t.Context(), contracts.ArtifactRef{Namespace: "builder", Name: "result"},
		artifacts.Payload{MediaType: "text/plain", Data: []byte("finished")}, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	bound, err := fixture.artifacts.BindOutputExact(t.Context(), "run-owned", "result", source.Ref, nil)
	if err != nil {
		t.Fatal(err)
	}

	request := authenticatedRequest(http.MethodGet, "/v1/runs/run-owned/outputs/result", bytes.NewReader(nil))
	response := httptest.NewRecorder()
	fixture.handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK || response.Body.String() != "finished" || response.Header().Get("ETag") != quotedETag(bound.TargetRef.Revision) {
		t.Fatalf("output = status %d, ETag %q, body %q", response.Code, response.Header().Get("ETag"), response.Body.String())
	}

	fixture.runs.runs["run-owned"] = runstore.WorkflowRun{RunID: "run-owned", OwnerID: "another-user"}
	readsBefore := fixture.repository.reads
	denied := httptest.NewRecorder()
	fixture.handler.ServeHTTP(denied, request)
	if denied.Code != http.StatusNotFound || fixture.repository.reads != readsBefore {
		t.Fatalf("foreign output = status %d, reads before/after %d/%d", denied.Code, readsBefore, fixture.repository.reads)
	}
}

func assertErrorCode(t *testing.T, response *httptest.ResponseRecorder, expected string) {
	t.Helper()
	var body errorResponse
	if err := json.Unmarshal(response.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode error response: %v; body %q", err, response.Body.String())
	}
	if body.Code != expected || strings.TrimSpace(body.Message) == "" ||
		body.RequestID != "request-fixed" || response.Header().Get("X-Request-ID") != body.RequestID {
		t.Fatalf("error response = %+v, want code %q", body, expected)
	}
}

type fakeMetricsReader struct {
	records map[string]telemetry.StageMetricsRecord
}

func (f *fakeMetricsReader) GetStageMetrics(
	_ context.Context, stageExecutionID string,
) (telemetry.StageMetricsRecord, error) {
	record, ok := f.records[stageExecutionID]
	if !ok {
		return telemetry.StageMetricsRecord{}, telemetry.ErrNotFound
	}
	return record, nil
}
