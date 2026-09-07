package controlplane

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"testing"
	"time"

	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/credentials"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestPlacementPostgresUsesCandidateAdaptersAndPinsBeforeExposure(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedPlacementPool(t, ctx)
	fixture := newPlacementFixture(t, ctx, pool, []string{"debug"})
	withoutAdapter := fixture.registerCandidate(t, ctx, "runtime-a", "1", nil)
	withAdapter := fixture.registerCandidate(
		t, ctx, "runtime-b", "2", []contracts.RuntimeAdapterRef{contracts.RuntimeAdapterOTLPHTTP},
	)

	reservations, err := fixture.allocator.ReserveAllContext(ctx, fixture.request())
	if err != nil {
		t.Fatal(err)
	}
	if len(reservations) != 1 || reservations[0].Grant.RuntimeAgentID != withAdapter ||
		reservations[0].Grant.RuntimeAgentID == withoutAdapter || reservations[0].ResolvedRuntimeConfig == nil {
		t.Fatalf("candidate-aware reservations = %+v", reservations)
	}
	if got := reservations[0].ResolvedRuntimeConfig.RequiredRuntimeAdapters; len(got) != 1 ||
		got[0] != contracts.RuntimeAdapterOTLPHTTP {
		t.Fatalf("required adapters = %v", got)
	}
	allocations, err := runstore.NewPostgresStore(pool).ListStageAllocations(ctx, fixture.stageExecutionID)
	if err != nil || len(allocations) != 1 || allocations[0].RuntimeConfiguration == nil ||
		allocations[0].RuntimeAgentID != withAdapter {
		t.Fatalf("durable allocations = (%+v, %v)", allocations, err)
	}
	if allocations[0].RuntimeConfiguration.Provenance.AgentLabels == nil ||
		allocations[0].RuntimeConfiguration.Provenance.RunLabels[0].Label != "debug" {
		t.Fatalf("durable provenance = %+v", allocations[0].RuntimeConfiguration)
	}
}

func TestPlacementPerformanceCollectionPolicyDoesNotFilterCandidates(t *testing.T) {
	unsupported := contracts.AgentRegistrationV2{}
	supported := contracts.AgentRegistrationV2{
		SupportedPerformanceMetricsVersions: contracts.PerformanceMetricsVersions{1},
	}
	for _, test := range []struct {
		enabled      bool
		registration contracts.AgentRegistrationV2
		want         contracts.PerformanceCollectionPolicy
	}{
		{false, unsupported, contracts.PerformanceCollectionDisabled},
		{false, supported, contracts.PerformanceCollectionDisabled},
		{true, unsupported, contracts.PerformanceCollectionUnsupported},
		{true, supported, contracts.PerformanceCollectionRequested},
	} {
		allocator := &PlacementAllocator{performanceMetrics: test.enabled}
		if got := allocator.collectionPolicy(test.registration); got != test.want {
			t.Fatalf("collection policy enabled=%v supported=%v = %q, want %q", test.enabled, len(test.registration.SupportedPerformanceMetricsVersions) != 0, got, test.want)
		}
	}
}

func TestPlacementPostgresDiscardsProvisionalBatchOnPrincipalRevisionChange(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedPlacementPool(t, ctx)
	fixture := newPlacementFixture(t, ctx, pool, nil)
	runtimeAgentID := fixture.registerCandidate(t, ctx, "runtime-racy", "3", nil)
	fixture.allocator.credentialGuard = placementGuardFunc(func(ctx context.Context, fn func() error) error {
		_, err := runtimeconfig.NewPrincipalRepository(pool).ReplaceLabels(
			ctx, runtimeAgentID, 1, []string{"changed"}, "operator", time.Now().UTC(),
		)
		if err != nil {
			return err
		}
		return fn()
	})

	_, err := fixture.allocator.ReserveAllContext(ctx, fixture.request())
	if !errors.Is(err, ErrInsufficientCapacity) {
		t.Fatalf("revision-change placement error = %v", err)
	}
	agent, err := fixture.registry.GetAgent("runtime-racy")
	if err != nil || agent.AuthoritativeAllocationID != nil {
		t.Fatalf("provisional slot was not discarded: (%+v, %v)", agent, err)
	}
	allocations, err := runstore.NewPostgresStore(pool).ListStageAllocations(ctx, fixture.stageExecutionID)
	if err != nil || len(allocations) != 0 {
		t.Fatalf("revision-change durable allocations = (%+v, %v)", allocations, err)
	}
}

func TestPlacementPostgresKeepsOldAgentBindingAndNextResolutionUsesRebind(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedPlacementPool(t, ctx)
	fixture := newPlacementFixture(t, ctx, pool, nil)
	publisher, err := runtimeconfig.NewPublisher(runtimeconfig.PublisherOptions{
		Pool: pool, RuntimeCredentials: placementRuntimeCatalog{},
		PlannerTelemetryAdapters: runtimeconfig.PlannerTelemetryAdapterCatalogFunc(func(ref string) bool { return ref == "otlp-http@1" }),
	})
	if err != nil {
		t.Fatal(err)
	}
	publish := func(name, endpoint, key string) runtimeconfig.Ref {
		t.Helper()
		document := []byte(`{
  "apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig",
  "metadata":{"name":"` + name + `","version":"1"},
  "spec":{"worker":{"telemetry":{"adapter":"otlp-http@1","endpoint":"` + endpoint + `"}}}
}`)
		published, publishErr := publisher.Publish(ctx, document, key, "operator")
		if publishErr != nil {
			t.Fatal(publishErr)
		}
		return published.Version.Ref
	}
	oldRef := publish("site-old", "https://old-otel.example/v1/traces", "publish-site-old")
	newRef := publish("site-new", "https://new-otel.example/v1/traces", "publish-site-new")
	binding, err := runtimeconfig.NewRepository(pool).CreateBinding(
		ctx, "site", oldRef, "operator", time.Now().UTC(),
	)
	if err != nil {
		t.Fatal(err)
	}
	runtimeAgentID := fixture.registerCandidateWithLabels(
		t, ctx, "runtime-labeled", "4",
		[]contracts.RuntimeAdapterRef{contracts.RuntimeAdapterOTLPHTTP}, []string{"site"},
	)

	reservations, err := fixture.allocator.ReserveAllContext(ctx, fixture.request())
	if err != nil || len(reservations) != 1 || reservations[0].ResolvedRuntimeConfig == nil ||
		reservations[0].ResolvedRuntimeConfig.WorkerTelemetry == nil ||
		reservations[0].ResolvedRuntimeConfig.WorkerTelemetry.Endpoint != "https://old-otel.example/v1/traces" {
		t.Fatalf("old Agent-label allocation = (%+v, %v)", reservations, err)
	}
	if _, err := runtimeconfig.NewRepository(pool).Rebind(
		ctx, "site", binding.Revision, newRef, "operator", time.Now().UTC().Add(time.Second),
	); err != nil {
		t.Fatal(err)
	}
	if got := reservations[0].ResolvedRuntimeConfig.WorkerTelemetry.Endpoint; got != "https://old-otel.example/v1/traces" {
		t.Fatalf("active allocation changed after rebind: %q", got)
	}
	request := fixture.request()
	next, err := fixture.allocator.resolveCandidate(
		ctx, pool, request, request.Bindings[0], AuthenticatedPrincipal{
			RuntimeAgentID: runtimeAgentID, Labels: []string{"site"}, LabelRevision: 1,
		},
	)
	if err != nil || next.WorkerTelemetry == nil ||
		next.WorkerTelemetry.Endpoint != "https://new-otel.example/v1/traces" {
		t.Fatalf("next Agent-label resolution = (%+v, %v)", next, err)
	}
	allocations, err := runstore.NewPostgresStore(pool).ListStageAllocations(ctx, fixture.stageExecutionID)
	if err != nil || len(allocations) != 1 || allocations[0].RuntimeConfiguration == nil ||
		allocations[0].RuntimeConfiguration.Provenance.AgentLabels[0].Config.Name != oldRef.Name {
		t.Fatalf("durable old Agent-label provenance = (%+v, %v)", allocations, err)
	}
}

type placementFixture struct {
	ctx              context.Context
	pool             *pgxpool.Pool
	registry         *InMemoryRegistry
	allocator        *PlacementAllocator
	runID            string
	stageExecutionID string
	runtimeSnapshot  runtimeconfig.RunSnapshot
	template         contracts.ResolvedAgentTemplate
	selection        workflowconfig.ResolvedConsumerExecutionConfig
}

func newPlacementFixture(
	t *testing.T,
	ctx context.Context,
	pool *pgxpool.Pool,
	runLabels []string,
) *placementFixture {
	t.Helper()
	configuration, err := workflowconfig.Load("../../testdata/configs", workflowconfig.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	workflow, err := configuration.Workflow("artifact-copy@1")
	if err != nil {
		t.Fatal(err)
	}
	stage := workflow.Stages[workflow.EntryStage]
	binding := stage.Agents["builder"]
	selection := stage.ExecutionConfig.Agents["builder"]
	if selection.LLMGateway == nil {
		t.Fatal("test Worker has no Gateway")
	}
	catalog := placementRuntimeCatalog{}
	if len(runLabels) != 0 {
		publisher, err := runtimeconfig.NewPublisher(runtimeconfig.PublisherOptions{
			Pool: pool, RuntimeCredentials: catalog,
			PlannerTelemetryAdapters: runtimeconfig.PlannerTelemetryAdapterCatalogFunc(func(ref string) bool { return ref == "otlp-http@1" }),
		})
		if err != nil {
			t.Fatal(err)
		}
		published, err := publisher.Publish(ctx, []byte(`{
  "apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig",
  "metadata":{"name":"debug-telemetry","version":"1"},
  "spec":{"worker":{"telemetry":{"adapter":"otlp-http@1","endpoint":"https://otel.example/v1/traces"}}}
}`), "publish-debug-telemetry", "operator")
		if err != nil {
			t.Fatal(err)
		}
		if _, err := runtimeconfig.NewRepository(pool).CreateBinding(
			ctx, "debug", published.Version.Ref, "operator", time.Now().UTC(),
		); err != nil {
			t.Fatal(err)
		}
	}
	if selection.Credential == nil {
		t.Fatal("test Worker has no credential")
	}
	llmCredentials, err := credentials.NewStaticProvider([]credentials.StaticEntry{{
		Metadata: workflowconfig.CredentialMetadata{
			Ref:          *selection.Credential,
			LLMGateway:   selection.LLMGateway.Ref,
			Unrestricted: true,
		},
		Token: contracts.NewSecretString("placement-test-token"),
	}})
	if err != nil {
		t.Fatal(err)
	}
	tx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback(ctx) }()
	txLookup, err := runtimeconfig.BindTransactionLLMCredentialLookup(
		tx,
		runtimeconfig.TransactionLLMCredentialLookupFactoryFunc(
			func(pgx.Tx) (workflowconfig.CredentialLookup, error) { return llmCredentials, nil },
		),
	)
	if err != nil {
		t.Fatal(err)
	}
	pinned, err := runtimeconfig.PinRunSnapshot(ctx, tx, runLabels, catalog, txLookup)
	if err != nil {
		t.Fatal(err)
	}
	store := runstore.NewPostgresStore(tx)
	runID := "run-placement"
	if _, err := store.CreateRun(ctx, runstore.CreateRunParams{
		RunID: runID, OwnerID: "user-placement", WorkflowName: workflow.Ref.Name,
		WorkflowVersion: workflow.Ref.Version, WorkflowSchemaVersion: contracts.APIVersion,
		WorkflowSnapshot: json.RawMessage(`{}`), Parameters: map[string]string{}, RuntimeConfig: pinned,
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := store.TransitionRun(
		ctx, runID, runstore.RunInitializing, runstore.RunRunning, runstore.Reason{Code: "ready"},
	); err != nil {
		t.Fatal(err)
	}
	if err := tx.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	store = runstore.NewPostgresStore(pool)
	stageExecutionID := "stage-placement"
	if _, err := store.CreateStageExecution(ctx, runstore.CreateStageExecutionParams{
		StageExecutionID: stageExecutionID, RunID: runID, StageName: workflow.EntryStage, Attempt: 1,
		ExecutionConfigVariant: runstore.StageExecutionConfigBase,
		StageSpecSchemaVersion: contracts.APIVersion, StageSpecSnapshot: json.RawMessage(`{}`),
		StageContextSchemaVersion: contracts.APIVersion,
		StageContext: runstore.StageContextSnapshot{
			Parameters: map[string]string{}, Artifacts: map[string]runstore.PinnedContextArtifact{},
		},
	}); err != nil {
		t.Fatal(err)
	}
	registry, err := NewRegistry(RegistryOptions{})
	if err != nil {
		t.Fatal(err)
	}
	allocator, err := NewPlacementAllocator(PlacementAllocatorOptions{
		Pool: pool, Registry: registry,
		Gateways:       placementGatewayLookup{gateway: *selection.LLMGateway},
		LLMCredentials: llmCredentials, RuntimeCredentials: placementRuntimeCredentialLookup{},
		CredentialGuard: placementGuardFunc(func(_ context.Context, fn func() error) error { return fn() }),
	})
	if err != nil {
		t.Fatal(err)
	}
	return &placementFixture{
		ctx: ctx, pool: pool, registry: registry, allocator: allocator,
		runID: runID, stageExecutionID: stageExecutionID, runtimeSnapshot: pinned,
		template: binding.Template, selection: selection,
	}
}

func (f *placementFixture) registerCandidate(
	t *testing.T,
	ctx context.Context,
	instanceID string,
	idDigit string,
	adapters []contracts.RuntimeAdapterRef,
) string {
	return f.registerCandidateWithLabels(t, ctx, instanceID, idDigit, adapters, nil)
}

func (f *placementFixture) registerCandidateWithLabels(
	t *testing.T,
	ctx context.Context,
	instanceID string,
	idDigit string,
	adapters []contracts.RuntimeAdapterRef,
	labels []string,
) string {
	t.Helper()
	runtimeAgentID := strings.Repeat(idDigit, 64)
	now := time.Now().UTC()
	principal := runtimeconfig.RuntimeAgentPrincipal{
		RuntimeAgentID: runtimeAgentID, Labels: append([]string{}, labels...), LabelRevision: 1,
		CreatedBy: "runtime-registration", CreatedAt: now,
		UpdatedBy: "runtime-registration", UpdatedAt: now,
	}
	if _, err := runtimeconfig.NewPrincipalRepository(f.pool).Insert(ctx, principal); err != nil {
		t.Fatal(err)
	}
	registration := contracts.AgentRegistrationV2{
		APIVersion: contracts.APIVersion, PrivateProtocolVersion: contracts.PrivateProtocolVersionV2,
		InstanceID: instanceID, SoftwareVersion: "1.0.0", StartedAt: now,
		ControlURL: "https://" + instanceID + ".test", A2AURL: "https://" + instanceID + ".test",
		InitialLabels:     append([]string{}, labels...),
		SupportedRuntimes: []string{f.template.Runtime.RuntimeID + "@" + f.template.Runtime.Version},
		SupportedToolsets: []contracts.ToolsetCapability{{
			Ref:   f.template.Toolsets[0].Ref.ToolsetID + "@" + f.template.Toolsets[0].Ref.Version,
			Tools: append([]string{}, f.template.Toolsets[0].Tools...),
		}},
		SupportedSandboxProfiles: []string{
			f.template.SandboxProfile.SandboxProfileID + "@" + f.template.SandboxProfile.Version,
		},
		SupportedRuntimeAdapters: append([]contracts.RuntimeAdapterRef{}, adapters...),
		ObservedState:            contracts.AgentIdle,
	}
	authenticated := AuthenticatedPrincipal{
		RuntimeAgentID: runtimeAgentID, Labels: append([]string{}, labels...), LabelRevision: 1,
	}
	if _, err := f.registry.RegisterAuthenticated(authenticated, registration); err != nil {
		t.Fatal(err)
	}
	if _, err := f.registry.HeartbeatAuthenticated(runtimeAgentID, contracts.AgentHeartbeat{
		APIVersion: contracts.APIVersion, InstanceID: instanceID,
		HeartbeatSeq: 1, ObservedState: contracts.AgentIdle,
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := f.registry.HeartbeatAuthenticated(runtimeAgentID, contracts.AgentHeartbeat{
		APIVersion: contracts.APIVersion, InstanceID: instanceID,
		HeartbeatSeq: 2, EchoedAckSeq: 1, ObservedState: contracts.AgentIdle,
	}); err != nil {
		t.Fatal(err)
	}
	return runtimeAgentID
}

func (f *placementFixture) request() ReservationRequest {
	return ReservationRequest{
		RunID: f.runID, StageExecutionID: f.stageExecutionID, RuntimeConfig: &f.runtimeSnapshot,
		Bindings: []BindingRequirement{{
			LogicalAgentName: "builder", Namespace: "builder", AgentTemplate: f.template,
			WorkerSessionMode: contracts.WorkerSessionIsolated,
			ExecutionConfig: AllocationExecutionConfig{
				ModelPolicy: f.selection.ModelPolicy.Ref, LLMGateway: f.selection.LLMGateway.Ref,
				Credential: f.selection.Credential,
			},
			RuntimeSelection: &f.selection,
		}},
	}
}

type placementRuntimeCatalog struct{}

func (placementRuntimeCatalog) ValidateRuntimeCredential(context.Context, string, ...string) error {
	return nil
}

func (placementRuntimeCatalog) WithCredentialReferences(_ context.Context, fn func() error) error {
	return fn()
}

type placementGatewayLookup struct {
	gateway contracts.ResolvedLLMGatewayConfig
}

func (l placementGatewayLookup) LLMGateway(raw string) (contracts.ResolvedLLMGatewayConfig, error) {
	if raw != l.gateway.Ref.GatewayID+"@"+l.gateway.Ref.Version {
		return contracts.ResolvedLLMGatewayConfig{}, workflowconfig.ErrConfigurationNotFound
	}
	return l.gateway, nil
}

type placementRuntimeCredentialLookup struct{}

func (placementRuntimeCredentialLookup) Get(
	context.Context, string,
) (credentials.RuntimeCredentialMetadata, error) {
	return credentials.RuntimeCredentialMetadata{}, credentials.ErrRuntimeCredentialNotFound
}

type placementGuardFunc func(context.Context, func() error) error

func (f placementGuardFunc) WithAllocationReferences(ctx context.Context, fn func() error) error {
	return f(ctx, fn)
}

func isolatedPlacementPool(t *testing.T, ctx context.Context) *pgxpool.Pool {
	t.Helper()
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	admin, err := pgxpool.New(ctx, databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	if err := admin.Ping(ctx); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	random := make([]byte, 8)
	if _, err := rand.Read(random); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	schema := "placement_test_" + hex.EncodeToString(random)
	identifier := pgx.Identifier{schema}.Sanitize()
	if _, err := admin.Exec(ctx, `CREATE SCHEMA `+identifier); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	config, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		admin.Close()
		t.Fatal(err)
	}
	config.ConnConfig.RuntimeParams["search_path"] = schema
	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		admin.Close()
		t.Fatal(err)
	}
	if _, err := persistencepostgres.ApplyMigrations(ctx, pool); err != nil {
		pool.Close()
		admin.Close()
		t.Fatal(err)
	}
	t.Cleanup(func() {
		pool.Close()
		cleanupContext, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_, _ = admin.Exec(cleanupContext, `DROP SCHEMA `+identifier+` CASCADE`)
		admin.Close()
	})
	return pool
}
