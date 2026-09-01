package credentials

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestCredentialLifecycleCreatesAndReplaysWithoutExposingToken(t *testing.T) {
	pool, ctx := lifecycleTestPool(t)
	manager := newFakeGatewayManager()
	fixture := newLifecycleFixture(t, pool, manager, ServiceOptions{})
	guardCalled := false
	if err := fixture.service.WithRunCreation(ctx, func() error { guardCalled = true; return nil }); !errors.Is(err, ErrRecoveryRequired) || guardCalled {
		t.Fatalf("unrecovered Run guard = (called=%v, err=%v)", guardCalled, err)
	}
	if _, err := fixture.service.ListCredentials(ctx, "", 1); !errors.Is(err, ErrRecoveryRequired) {
		t.Fatalf("unrecovered credential list error = %v", err)
	}
	if err := fixture.service.Recover(ctx); err != nil {
		t.Fatal(err)
	}
	request := fixture.createRequest("managed-worker", "create-01")
	created, err := fixture.service.Create(ctx, request)
	if err != nil || created.Replayed {
		t.Fatalf("create result = (%+v, %v)", created, err)
	}
	if manager.createCalls() != 1 || manager.remoteCount() != 1 {
		t.Fatalf("manager after create: calls=%d remote=%d", manager.createCalls(), manager.remoteCount())
	}
	encoded, err := json.Marshal(created.Credential)
	if err != nil || bytes.Contains(encoded, []byte(manager.secretFor(request.CredentialID))) ||
		bytes.Contains(encoded, []byte("ciphertext")) || bytes.Contains(encoded, []byte("remoteKey")) {
		t.Fatalf("unsafe credential response = (%s, %v)", encoded, err)
	}
	replayed, err := fixture.service.Create(ctx, request)
	if err != nil || !replayed.Replayed || replayed.Credential.CredentialID != request.CredentialID ||
		manager.createCalls() != 1 {
		t.Fatalf("replayed create = (%+v, %v), calls=%d", replayed, err, manager.createCalls())
	}
	replayedJSON, replayedJSONErr := json.Marshal(replayed.Credential)
	if replayedJSONErr != nil || !bytes.Equal(encoded, replayedJSON) {
		t.Fatalf("replayed credential changed response: first=%s replay=%s error=%v", encoded, replayedJSON, replayedJSONErr)
	}
	conflicting := request
	conflicting.Label = "different"
	if _, err := fixture.service.Create(ctx, conflicting); !errors.Is(err, ErrConflict) {
		t.Fatalf("idempotency conflict error = %v", err)
	}

	provider, _ := NewEncryptedProvider(NewRepository(pool), fixture.cipher)
	resolved, err := provider.ResolveLLMCredential(
		ctx, contracts.LLMCredentialRef{CredentialID: request.CredentialID}, fixture.gateway.Ref,
	)
	if err != nil || resolved.Reveal() != manager.secretFor(request.CredentialID) {
		t.Fatalf("resolved created token = (%s, %v)", resolved, err)
	}
	var ciphertext []byte
	if err := pool.QueryRow(ctx, `SELECT ciphertext FROM llm_credentials WHERE credential_id = $1`, request.CredentialID).Scan(&ciphertext); err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(ciphertext, []byte(manager.secretFor(request.CredentialID))) {
		t.Fatal("database ciphertext contains plaintext token")
	}
	assertCredentialOperationPhase(t, ctx, pool, OperationCreate, request.IdempotencyKey, OperationCompleted)
}

func TestCredentialLifecycleRecoversCreateCrashWindow(t *testing.T) {
	pool, ctx := lifecycleTestPool(t)
	manager := newFakeGatewayManager()
	var hookCalls atomic.Int32
	crash := errors.New("simulated process loss after remote create")
	fixture := newLifecycleFixture(t, pool, manager, ServiceOptions{
		AfterManagerCreate: func() error {
			if hookCalls.Add(1) == 1 {
				return crash
			}
			return nil
		},
	})
	if err := fixture.service.Recover(ctx); err != nil {
		t.Fatal(err)
	}
	request := fixture.createRequest("managed-recovery", "create-crash")
	if _, err := fixture.service.Create(ctx, request); !errors.Is(err, crash) {
		t.Fatalf("crash-window create error = %v", err)
	}
	if _, err := fixture.service.GetCredential(ctx, request.CredentialID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("partial create became active: %v", err)
	}
	assertCredentialOperationPhase(t, ctx, pool, OperationCreate, request.IdempotencyKey, OperationPrepared)
	if manager.remoteCount() != 1 {
		t.Fatalf("remote keys after crash = %d", manager.remoteCount())
	}
	emptyRegistry, _ := NewManagerRegistry()
	blockedService, err := NewService(ServiceOptions{
		Pool: pool, Gateways: staticGatewayLookup{gateway: fixture.gateway}, Managers: emptyRegistry,
		Runs: runstore.NewPostgresStore(pool), Cipher: fixture.cipher,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := blockedService.Recover(ctx); !errors.Is(err, ErrManagerUnavailable) {
		t.Fatalf("startup recovery without manager error = %v", err)
	}
	called := false
	if err := blockedService.WithRunCreation(ctx, func() error { called = true; return nil }); !errors.Is(err, ErrRecoveryRequired) || called {
		t.Fatalf("failed startup recovery allowed Run creation: called=%v err=%v", called, err)
	}

	restarted := newLifecycleFixtureWithCipher(t, pool, manager, fixture.cipher, ServiceOptions{})
	if err := restarted.service.Recover(ctx); err != nil {
		t.Fatalf("recover create: %v", err)
	}
	if manager.recoverCalls() != 1 || manager.createCalls() != 2 || manager.remoteCount() != 1 {
		t.Fatalf("manager recovery calls: recover=%d create=%d remote=%d",
			manager.recoverCalls(), manager.createCalls(), manager.remoteCount())
	}
	if _, err := restarted.service.GetCredential(ctx, request.CredentialID); err != nil {
		t.Fatalf("recovered credential is not active: %v", err)
	}
	assertCredentialOperationPhase(t, ctx, pool, OperationCreate, request.IdempotencyKey, OperationCompleted)
}

func TestCredentialDeletionRejectsPinnedRunThenDeletesAndReplays(t *testing.T) {
	pool, ctx := lifecycleTestPool(t)
	manager := newFakeGatewayManager()
	fixture := newLifecycleFixture(t, pool, manager, ServiceOptions{})
	if err := fixture.service.Recover(ctx); err != nil {
		t.Fatal(err)
	}
	create := fixture.createRequest("managed-delete", "create-delete")
	if _, err := fixture.service.Create(ctx, create); err != nil {
		t.Fatal(err)
	}
	runs := runstore.NewPostgresStore(pool)
	createPinnedRun(t, ctx, runs, "run-pinned", create.CredentialID)
	createPinnedRun(t, ctx, runs, "run-similar", create.CredentialID+"-extra")
	deletion := DeleteRequest{
		CredentialID: create.CredentialID, IdempotencyKey: "delete-01", ActorID: "user-1",
	}
	if _, err := fixture.service.Delete(ctx, deletion); err == nil {
		t.Fatal("deletion pinned by a Run succeeded")
	} else {
		var inUse *CredentialInUseError
		if !errors.As(err, &inUse) || len(inUse.RunIDs) != 1 || inUse.RunIDs[0] != "run-pinned" {
			t.Fatalf("pinned deletion error = %#v", err)
		}
	}
	if manager.deleteCalls() != 0 || countOperations(t, ctx, pool, OperationDelete) != 0 {
		t.Fatalf("pinned deletion had side effects: manager=%d operations=%d",
			manager.deleteCalls(), countOperations(t, ctx, pool, OperationDelete))
	}
	if _, err := runs.TransitionRun(ctx, "run-pinned", runstore.RunInitializing, runstore.RunRunning, runstore.Reason{Code: "started"}); err != nil {
		t.Fatal(err)
	}
	if _, err := runs.TransitionRun(ctx, "run-pinned", runstore.RunRunning, runstore.RunSucceeded, runstore.Reason{Code: "finished"}); err != nil {
		t.Fatal(err)
	}
	result, err := fixture.service.Delete(ctx, deletion)
	if err != nil || result.Replayed {
		t.Fatalf("delete result = (%+v, %v)", result, err)
	}
	if _, err := fixture.service.GetCredential(ctx, create.CredentialID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("deleted credential remains active: %v", err)
	}
	if _, err := NewRepository(pool).GetTombstone(ctx, create.CredentialID); err != nil {
		t.Fatalf("delete tombstone: %v", err)
	}
	replayed, err := fixture.service.Delete(ctx, deletion)
	if err != nil || !replayed.Replayed || manager.deleteCalls() != 1 {
		t.Fatalf("delete replay = (%+v, %v), calls=%d", replayed, err, manager.deleteCalls())
	}
}

func TestCredentialLifecycleRecoversDeleteCrashAndBlocksRunCreation(t *testing.T) {
	pool, ctx := lifecycleTestPool(t)
	manager := newFakeGatewayManager()
	crash := errors.New("simulated process loss after remote delete")
	var hookCalls atomic.Int32
	fixture := newLifecycleFixture(t, pool, manager, ServiceOptions{
		AfterManagerDelete: func() error {
			if hookCalls.Add(1) == 1 {
				return crash
			}
			return nil
		},
	})
	if err := fixture.service.Recover(ctx); err != nil {
		t.Fatal(err)
	}
	create := fixture.createRequest("managed-delete-recovery", "create-before-delete-crash")
	if _, err := fixture.service.Create(ctx, create); err != nil {
		t.Fatal(err)
	}
	deletion := DeleteRequest{
		CredentialID: create.CredentialID, IdempotencyKey: "delete-crash", ActorID: "user-1",
	}
	if _, err := fixture.service.Delete(ctx, deletion); !errors.Is(err, crash) {
		t.Fatalf("delete crash-window error = %v", err)
	}
	if _, err := fixture.service.GetCredential(ctx, create.CredentialID); err != nil {
		t.Fatalf("database record was removed before atomic delete commit: %v", err)
	}
	called := false
	if err := fixture.service.WithRunCreation(ctx, func() error { called = true; return nil }); !errors.Is(err, ErrRecoveryRequired) || called {
		t.Fatalf("Run creation during ambiguous delete = (called=%v, err=%v)", called, err)
	}
	assertCredentialOperationPhase(t, ctx, pool, OperationDelete, deletion.IdempotencyKey, OperationPrepared)

	restarted := newLifecycleFixtureWithCipher(t, pool, manager, fixture.cipher, ServiceOptions{})
	if err := restarted.service.Recover(ctx); err != nil {
		t.Fatalf("recover delete: %v", err)
	}
	if manager.deleteCalls() != 2 || manager.remoteCount() != 0 {
		t.Fatalf("manager after delete recovery: calls=%d remote=%d", manager.deleteCalls(), manager.remoteCount())
	}
	if err := restarted.service.WithRunCreation(ctx, func() error { called = true; return nil }); err != nil || !called {
		t.Fatalf("Run guard after recovery = (called=%v, err=%v)", called, err)
	}
	assertCredentialOperationPhase(t, ctx, pool, OperationDelete, deletion.IdempotencyKey, OperationCompleted)
}

func TestCredentialDeleteSerializesAgainstRunSnapshotCommit(t *testing.T) {
	pool, ctx := lifecycleTestPool(t)
	manager := newFakeGatewayManager()
	fixture := newLifecycleFixture(t, pool, manager, ServiceOptions{})
	if err := fixture.service.Recover(ctx); err != nil {
		t.Fatal(err)
	}
	create := fixture.createRequest("managed-race", "create-race")
	if _, err := fixture.service.Create(ctx, create); err != nil {
		t.Fatal(err)
	}
	runs := runstore.NewPostgresStore(pool)
	entered := make(chan struct{})
	release := make(chan struct{})
	runResult := make(chan error, 1)
	go func() {
		runResult <- fixture.service.WithRunCreation(ctx, func() error {
			close(entered)
			<-release
			_, err := runs.CreateRun(ctx, pinnedRunParams("run-race", create.CredentialID))
			return err
		})
	}()
	<-entered
	deleteResult := make(chan error, 1)
	go func() {
		_, err := fixture.service.Delete(ctx, DeleteRequest{
			CredentialID: create.CredentialID, IdempotencyKey: "delete-race", ActorID: "user-1",
		})
		deleteResult <- err
	}()
	select {
	case err := <-deleteResult:
		t.Fatalf("delete crossed an in-flight Run commit: %v", err)
	case <-time.After(100 * time.Millisecond):
	}
	close(release)
	if err := <-runResult; err != nil {
		t.Fatalf("commit pinned Run: %v", err)
	}
	var inUse *CredentialInUseError
	if err := <-deleteResult; !errors.As(err, &inUse) || len(inUse.RunIDs) != 1 || inUse.RunIDs[0] != "run-race" {
		t.Fatalf("serialized delete error = %#v", err)
	}
	if manager.deleteCalls() != 0 {
		t.Fatalf("serialized delete reached Gateway %d times", manager.deleteCalls())
	}
}

func TestCredentialRunReferenceQueryRejectsUnsafePublicRunIDs(t *testing.T) {
	pool, ctx := lifecycleTestPool(t)
	runs := runstore.NewPostgresStore(pool)
	createPinnedRun(t, ctx, runs, "unsafe run id", "managed-worker")
	if _, err := runs.ListNonTerminalRunIDsByCredential(ctx, "managed-worker", 128); err == nil {
		t.Fatal("unsafe Run ID was returned as public credential-in-use detail")
	}
	if _, err := runs.ListNonTerminalRunIDsByCredential(ctx, "managed-worker", 129); !errors.Is(err, runstore.ErrInvalid) {
		t.Fatalf("unbounded Run-reference query error = %v", err)
	}
}

func TestCredentialManagerFailuresAreRedactedAndLeavePreparedIntent(t *testing.T) {
	pool, ctx := lifecycleTestPool(t)
	manager := newFakeGatewayManager()
	manager.createErr = errors.New("provider body contained sk-highly-sensitive")
	fixture := newLifecycleFixture(t, pool, manager, ServiceOptions{})
	if err := fixture.service.Recover(ctx); err != nil {
		t.Fatal(err)
	}
	request := fixture.createRequest("managed-failure", "create-failure")
	_, err := fixture.service.Create(ctx, request)
	if !errors.Is(err, ErrGatewayUnavailable) || strings.Contains(err.Error(), "sensitive") {
		t.Fatalf("unsafe manager failure = %v", err)
	}
	assertCredentialOperationPhase(t, ctx, pool, OperationCreate, request.IdempotencyKey, OperationPrepared)
}

func TestCredentialManagerValidationRunsBeforeDurableIntent(t *testing.T) {
	pool, ctx := lifecycleTestPool(t)
	manager := newFakeGatewayManager()
	manager.validateErr = fmt.Errorf("%w: unknown exact ModelPolicy", ErrInvalid)
	fixture := newLifecycleFixture(t, pool, manager, ServiceOptions{})
	if err := fixture.service.Recover(ctx); err != nil {
		t.Fatal(err)
	}
	request := fixture.createRequest("managed-invalid-policy", "invalid-policy")
	if _, err := fixture.service.Create(ctx, request); !errors.Is(err, ErrInvalid) {
		t.Fatalf("manager validation error = %v", err)
	}
	if manager.validationCalls() != 1 || manager.createCalls() != 0 ||
		countOperations(t, ctx, pool, OperationCreate) != 0 {
		t.Fatalf("invalid request side effects: validations=%d creates=%d operations=%d",
			manager.validationCalls(), manager.createCalls(), countOperations(t, ctx, pool, OperationCreate))
	}
	var identities int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM llm_credential_identities`).Scan(&identities); err != nil {
		t.Fatal(err)
	}
	if identities != 0 {
		t.Fatalf("invalid request reserved %d credential identities", identities)
	}
}

type lifecycleFixture struct {
	service *Service
	cipher  *TokenCipher
	gateway contracts.ResolvedLLMGatewayConfig
}

func newLifecycleFixture(
	t *testing.T, pool *pgxpool.Pool, manager *fakeGatewayManager, overrides ServiceOptions,
) lifecycleFixture {
	t.Helper()
	cipher, err := NewTokenCipher(bytes.Repeat([]byte{0x55}, 32))
	if err != nil {
		t.Fatal(err)
	}
	return newLifecycleFixtureWithCipher(t, pool, manager, cipher, overrides)
}

func newLifecycleFixtureWithCipher(
	t *testing.T,
	pool *pgxpool.Pool,
	manager *fakeGatewayManager,
	cipher *TokenCipher,
	overrides ServiceOptions,
) lifecycleFixture {
	t.Helper()
	gateway := contracts.ResolvedLLMGatewayConfig{
		Ref:      testPinnedGatewayRef("1", strings.Repeat("1", 64)),
		Protocol: contracts.OpenAICompatibleProtocol, URL: "http://127.0.0.1:4000/v1",
		CredentialManager: &contracts.LLMGatewayCredentialManager{
			Implementation: contracts.LiteLLMVirtualKeysManager,
			ManagementURL:  "http://127.0.0.1:4000",
		},
	}
	registry, err := NewManagerRegistry(ManagerRegistration{
		Implementation: contracts.LiteLLMVirtualKeysManager, Manager: manager,
	})
	if err != nil {
		t.Fatal(err)
	}
	var ids atomic.Int32
	options := ServiceOptions{
		Pool: pool, Gateways: staticGatewayLookup{gateway: gateway}, Managers: registry,
		Runs: runstore.NewPostgresStore(pool), Cipher: cipher, Barrier: overrides.Barrier,
		Now: time.Now,
		NewID: func(prefix string) (string, error) {
			return fmt.Sprintf("%s%d", prefix, ids.Add(1)), nil
		},
		AfterManagerCreate: overrides.AfterManagerCreate,
		AfterManagerDelete: overrides.AfterManagerDelete,
	}
	service, err := NewService(options)
	if err != nil {
		t.Fatal(err)
	}
	return lifecycleFixture{service: service, cipher: cipher, gateway: gateway}
}

func (f lifecycleFixture) createRequest(credentialID, key string) CreateRequest {
	return CreateRequest{
		CredentialID: credentialID, LLMGateway: f.gateway.Ref, Label: "Managed credential",
		GatewayPolicy: GatewayPolicy{ModelPolicies: []contracts.ModelPolicyRef{{
			PolicyID: "worker", Version: "1", Digest: "sha256:" + strings.Repeat("2", 64),
		}}},
		IdempotencyKey: key, ActorID: "user-1",
	}
}

type staticGatewayLookup struct {
	gateway contracts.ResolvedLLMGatewayConfig
}

func (l staticGatewayLookup) LLMGateway(selector string) (contracts.ResolvedLLMGatewayConfig, error) {
	if selector != l.gateway.Ref.GatewayID+"@"+l.gateway.Ref.Version {
		return contracts.ResolvedLLMGatewayConfig{}, errors.New("not found")
	}
	return l.gateway, nil
}

type fakeGatewayManager struct {
	mu          sync.Mutex
	active      map[string]string
	creates     int
	deletes     int
	recoveries  int
	validations int
	validateErr error
	createErr   error
	deleteErr   error
	recoverErr  error
}

func newFakeGatewayManager() *fakeGatewayManager {
	return &fakeGatewayManager{active: make(map[string]string)}
}

func (m *fakeGatewayManager) ValidateCreate(context.Context, ManagerCreateRequest) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.validations++
	return m.validateErr
}

func (m *fakeGatewayManager) Create(_ context.Context, request ManagerCreateRequest) (GeneratedCredential, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.creates++
	if m.createErr != nil {
		return GeneratedCredential{}, m.createErr
	}
	if _, exists := m.active[request.CredentialID]; exists {
		return GeneratedCredential{}, errors.New("duplicate deterministic alias")
	}
	secret := m.secretFor(request.CredentialID)
	token, _ := NewToken(secret)
	remoteID := "remote-" + request.CredentialID
	m.active[request.CredentialID] = remoteID
	refs := append([]contracts.ModelPolicyRef(nil), request.Policy.ModelPolicies...)
	sort.Slice(refs, func(i, j int) bool { return modelPolicyRefKey(refs[i]) < modelPolicyRefKey(refs[j]) })
	return GeneratedCredential{
		Token: token, RemoteKeyID: remoteID,
		EffectivePolicy: EffectiveGatewayPolicy{
			ModelPolicies: refs, Models: []string{"qwen/test-model"},
			MaxBudget: request.Policy.MaxBudget, BudgetDuration: request.Policy.BudgetDuration,
			TPMLimit: request.Policy.TPMLimit, RPMLimit: request.Policy.RPMLimit,
			MaxParallelRequests: request.Policy.MaxParallelRequests,
		},
	}, nil
}

func (m *fakeGatewayManager) Delete(_ context.Context, request ManagerDeleteRequest) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.deletes++
	if m.deleteErr != nil {
		return m.deleteErr
	}
	if remote, exists := m.active[request.CredentialID]; exists && remote != request.RemoteKeyID {
		return errors.New("remote key mismatch")
	}
	delete(m.active, request.CredentialID)
	return nil
}

func (m *fakeGatewayManager) RecoverCreate(_ context.Context, request ManagerCreateRequest) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.recoveries++
	if m.recoverErr != nil {
		return m.recoverErr
	}
	delete(m.active, request.CredentialID)
	return nil
}

func (m *fakeGatewayManager) secretFor(credentialID string) string {
	return "sk-secret-for-" + credentialID
}

func (m *fakeGatewayManager) createCalls() int { m.mu.Lock(); defer m.mu.Unlock(); return m.creates }
func (m *fakeGatewayManager) deleteCalls() int { m.mu.Lock(); defer m.mu.Unlock(); return m.deletes }
func (m *fakeGatewayManager) recoverCalls() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.recoveries
}
func (m *fakeGatewayManager) validationCalls() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.validations
}
func (m *fakeGatewayManager) remoteCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.active)
}

func lifecycleTestPool(t *testing.T) (*pgxpool.Pool, context.Context) {
	t.Helper()
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	t.Cleanup(cancel)
	return isolatedCredentialPool(t, ctx, databaseURL), ctx
}

func pinnedRunParams(runID, credentialID string) runstore.CreateRunParams {
	return runstore.CreateRunParams{
		RunID: runID, OwnerID: "user-1", WorkflowName: "workflow", WorkflowVersion: "1",
		WorkflowSchemaVersion: contracts.APIVersion,
		WorkflowSnapshot: json.RawMessage(fmt.Sprintf(
			`{"stages":{"build":{"executionConfig":{"agents":{"worker":{"credential":{"credentialId":%q}}}}}}}`,
			credentialID,
		)),
		RuntimeConfig: runtimeconfig.BuiltInRunSnapshot(),
		Parameters:    map[string]string{},
	}
}

func createPinnedRun(
	t *testing.T, ctx context.Context, store *runstore.PostgresStore, runID, credentialID string,
) {
	t.Helper()
	if _, err := store.CreateRun(ctx, pinnedRunParams(runID, credentialID)); err != nil {
		t.Fatal(err)
	}
}

func assertCredentialOperationPhase(
	t *testing.T,
	ctx context.Context,
	pool *pgxpool.Pool,
	kind OperationKind,
	key string,
	want OperationPhase,
) {
	t.Helper()
	operation, err := NewRepository(pool).GetOperationByIdempotency(ctx, kind, key)
	if err != nil || operation.Phase != want {
		t.Fatalf("operation phase = (%q, %v), want %q", operation.Phase, err, want)
	}
}

func countOperations(t *testing.T, ctx context.Context, pool *pgxpool.Pool, kind OperationKind) int {
	t.Helper()
	var count int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM credential_operations WHERE operation_kind = $1`, kind).Scan(&count); err != nil {
		t.Fatal(err)
	}
	return count
}
