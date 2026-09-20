package credentials

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

func TestMCPRuntimeCredentialPostgresPublicationReplayDeletionAndRebindRace(t *testing.T) {
	dsn := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	pool := isolatedCredentialPool(t, ctx, dsn)
	repository := NewRuntimeCredentialRepository(pool)
	cipher, err := NewTokenCipher(bytes.Repeat([]byte{0x65}, 32))
	if err != nil {
		t.Fatal(err)
	}
	service, err := NewRuntimeCredentialService(RuntimeCredentialServiceOptions{Pool: pool, Cipher: cipher, Usage: repository, Barrier: NewLifecycleBarrier()})
	if err != nil {
		t.Fatal(err)
	}
	publisher, err := runtimeconfig.NewPublisher(runtimeconfig.PublisherOptions{Pool: pool, RuntimeCredentials: service, PlannerTelemetryAdapters: runtimeconfig.PlannerTelemetryAdapterCatalogFunc(func(string) bool { return true })})
	if err != nil {
		t.Fatal(err)
	}
	bindings, err := runtimeconfig.NewBindingService(pool, service)
	if err != nil {
		t.Fatal(err)
	}
	create := func(id string) {
		t.Helper()
		material, err := NewMCPHeadersCredential(map[string]string{"Authorization": "Bearer mcp-database-secret-canary"})
		if err != nil {
			t.Fatal(err)
		}
		defer material.Destroy()
		request := RuntimeCredentialCreateRequest{CredentialID: id, Material: material, IdempotencyKey: "create-" + id, ActorID: "operator"}
		first, err := service.Create(ctx, request)
		if err != nil || first.Credential.Kind != RuntimeCredentialMCPHeaders {
			t.Fatalf("create MCP credential: %v", err)
		}
		replay, err := service.Create(ctx, request)
		if err != nil || !replay.Replayed {
			t.Fatalf("MCP create replay: %v", err)
		}
		public, _ := json.Marshal(first)
		if bytes.Contains(public, []byte("mcp-database-secret-canary")) {
			t.Fatal("secret in metadata")
		}
		if err := service.Use(ctx, id, []string{string(RuntimeCredentialMCPHeaders)}, func(m *RuntimeCredentialMaterial) error {
			return m.WithPlaintext(func(kind RuntimeCredentialKind, raw []byte) error {
				if kind != RuntimeCredentialMCPHeaders || !bytes.Equal(raw, []byte(`{"headers":{"authorization":"Bearer mcp-database-secret-canary"}}`)) {
					return errors.New("private material differs")
				}
				return nil
			})
		}); err != nil {
			t.Fatal(err)
		}
		var ciphertext, mac []byte
		if err := pool.QueryRow(ctx, `SELECT c.ciphertext,p.request_mac FROM runtime_credentials c JOIN runtime_credential_creations p ON p.credential_id=c.credential_id WHERE c.credential_id=$1`, id).Scan(&ciphertext, &mac); err != nil {
			t.Fatal(err)
		}
		if bytes.Contains(ciphertext, []byte("mcp-database-secret-canary")) || bytes.Contains(mac, []byte("mcp-database-secret-canary")) {
			t.Fatal("secret stored in plaintext")
		}
	}
	document := func(name, id string) []byte {
		raw, _ := json.Marshal(map[string]any{"apiVersion": contracts.APIVersion, "kind": "RuntimeConfig", "metadata": map[string]string{"name": name, "version": "1"}, "spec": map[string]any{"worker": map[string]any{"toolsets": map[string]any{"mcp@1": map[string]any{"endpoint": "https://knowledge.example/mcp", "credential": id}}}}})
		return raw
	}
	publish := func(name, id string) runtimeconfig.Ref {
		t.Helper()
		raw := document(name, id)
		p, err := publisher.Publish(ctx, raw, "publish-"+name, "operator")
		if err != nil {
			t.Fatal(err)
		}
		replay, err := publisher.Publish(ctx, raw, "publish-"+name, "operator")
		if err != nil || !replay.Replayed || p.Version.Ref != replay.Version.Ref {
			t.Fatalf("MCP publication replay: %v", err)
		}
		if bytes.Contains(p.Version.CanonicalDocument, []byte("mcp-database-secret-canary")) {
			t.Fatal("secret in config")
		}
		return p.Version.Ref
	}
	create("mcp-active")
	ref := publish("mcp-active", "mcp-active")
	binding, err := bindings.Create(ctx, "mcp-active", ref, "operator", time.Now())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := service.Delete(ctx, "mcp-active", "operator"); !errors.Is(err, ErrRuntimeCredentialInUse) {
		t.Fatalf("live MCP binding did not protect credential: %v", err)
	}
	usage, err := repository.InspectRuntimeCredentialUsage(ctx, "mcp-active", 128)
	if err != nil || len(usage.BindingLabels) != 1 || usage.BindingLabels[0] != "mcp-active" {
		t.Fatalf("MCP binding usage: %v", err)
	}
	// The ordinary transaction path must retain the toolset credential after unbinding.
	if err := service.WithCredentialReferences(ctx, func() error {
		tx, err := pool.Begin(ctx)
		if err != nil {
			return err
		}
		defer func() { _ = tx.Rollback(ctx) }()
		provider, err := NewStaticProvider(nil)
		if err != nil {
			return err
		}
		factory, err := NewTransactionLookupFactory(provider)
		if err != nil {
			return err
		}
		lookup, err := runtimeconfig.BindTransactionLLMCredentialLookup(tx, factory)
		if err != nil {
			return err
		}
		store := runstore.NewRunCreationPostgresStore(tx, lookup)
		pins, err := store.PinRuntimeLabels(ctx, []string{"mcp-active"})
		if err != nil {
			return err
		}
		if len(pins.RuntimeCredentialIDs) != 1 || pins.RuntimeCredentialIDs[0] != "mcp-active" {
			return errors.New("MCP credential was not pinned")
		}
		if _, err := store.CreateRun(ctx, runstore.CreateRunParams{RunID: "mcp-run", OwnerID: "user", WorkflowName: "workflow", WorkflowVersion: "1", WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: json.RawMessage(`{}`), Parameters: map[string]string{}, RuntimeConfig: pins}); err != nil {
			return err
		}
		return tx.Commit(ctx)
	}); err != nil {
		t.Fatal(err)
	}
	if err := bindings.Delete(ctx, binding.Label, binding.Revision); err != nil {
		t.Fatal(err)
	}
	if _, err := service.Delete(ctx, "mcp-active", "operator"); !errors.Is(err, ErrRuntimeCredentialInUse) {
		t.Fatalf("Run hold lost MCP credential: %v", err)
	}
	// Two unreleased allocations retain this exact credential independently of
	// the Run hold; releasing the older allocation cannot free its replacement.
	allocationStore := runstore.NewPostgresStore(pool)
	if _, err := allocationStore.CreateRun(ctx, runstore.CreateRunParams{RunID: "mcp-allocation-run", OwnerID: "user", WorkflowName: "workflow", WorkflowVersion: "1", WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: json.RawMessage(`{}`), Parameters: map[string]string{}, RuntimeConfig: runtimeconfig.BuiltInRunSnapshot()}); err != nil {
		t.Fatal(err)
	}
	if _, err := allocationStore.TransitionRun(ctx, "mcp-allocation-run", runstore.RunInitializing, runstore.RunRunning, runstore.Reason{Code: "ready"}); err != nil {
		t.Fatal(err)
	}
	for _, suffix := range []string{"first", "replacement"} {
		stageID := "mcp-stage-" + suffix
		if _, err := allocationStore.CreateStageExecution(ctx, runstore.CreateStageExecutionParams{StageExecutionID: stageID, RunID: "mcp-allocation-run", StageName: suffix, Attempt: 1, StageSpecSchemaVersion: contracts.APIVersion, StageSpecSnapshot: json.RawMessage(`{}`), StageContextSchemaVersion: contracts.APIVersion, StageContext: runstore.StageContextSnapshot{Parameters: map[string]string{}, Artifacts: map[string]runstore.PinnedContextArtifact{}}}); err != nil {
			t.Fatal(err)
		}
		gateway := contracts.LLMGatewayConfigRef{GatewayID: "local", Version: "1", Digest: "sha256:" + strings.Repeat("b", 64)}
		provenance := contracts.ResolvedRuntimeConfigProvenance{Default: contracts.RuntimeLabelBindingProvenance{Label: "default", BindingRevision: 1, Config: contracts.RuntimeConfigRef{Name: runtimeconfig.BuiltInName, Version: runtimeconfig.BuiltInVersion, Digest: runtimeconfig.BuiltInDigest}}, RunLabels: []contracts.RuntimeLabelBindingProvenance{{Label: "mcp-active", BindingRevision: 1, Config: contracts.RuntimeConfigRef{Name: ref.Name, Version: ref.Version, Digest: ref.Digest}}}, AgentLabels: []contracts.RuntimeLabelBindingProvenance{}, RuntimeAdapters: []contracts.RuntimeAdapterRef{}, LLMGatewayConfig: &gateway, RuntimeCredentialRefs: []contracts.RuntimeCredentialRef{{CredentialID: "mcp-active", Kind: contracts.RuntimeCredentialMCPHeaders}}}
		configuration := &runstore.AllocationRuntimeConfiguration{ModelPolicy: contracts.ModelPolicyRef{PolicyID: "worker", Version: "1", Digest: "sha256:" + strings.Repeat("c", 64)}, Origins: runtimeconfig.ResolvedRuntimeConfigOrigins{Toolsets: map[string]runtimeconfig.RuntimeFieldOrigin{"mcp@1": {Layer: runtimeconfig.LayerRunLabels, Configs: []runtimeconfig.Ref{ref}}}}, Provenance: provenance}
		if err := allocationStore.RecordStageAllocation(ctx, runstore.StageAllocation{AllocationID: "mcp-allocation-" + suffix, StageExecutionID: stageID, LogicalAgentName: "worker", Namespace: "worker", AgentTemplateRef: contracts.AgentTemplateRef{TemplateID: "worker", Version: "1", Digest: "sha256:" + strings.Repeat("d", 64)}, WorkerRuntimeRef: contracts.WorkerRuntimeRef{RuntimeID: "adk", Version: "1"}, RuntimeAgentID: strings.Repeat("e", 64), RuntimeAgentInstanceID: "runtime-mcp-" + suffix, RuntimeAgentLabelRevision: 1, RuntimeConfigurationSchemaVersion: runstore.AllocationRuntimeConfigurationSchemaVersion, RuntimeConfiguration: configuration, PerformanceCollectionPolicy: contracts.PerformanceCollectionDisabled}); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := runstore.NewPostgresStore(pool).TransitionRun(ctx, "mcp-run", runstore.RunInitializing, runstore.RunFailed, runstore.Reason{Code: "test_complete"}); err != nil {
		t.Fatal(err)
	}
	for _, suffix := range []string{"first", "replacement"} {
		if _, err := service.Delete(ctx, "mcp-active", "operator"); !errors.Is(err, ErrRuntimeCredentialInUse) {
			t.Fatalf("unreleased allocation lost MCP hold: %v", err)
		}
		if err := allocationStore.MarkStageAllocationReleased(ctx, "mcp-allocation-"+suffix); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := service.Delete(ctx, "mcp-active", "operator"); err != nil {
		t.Fatal(err)
	}
	if _, err := bindings.Create(ctx, "stale", ref, "operator", time.Now()); !errors.Is(err, runtimeconfig.ErrInvalid) {
		t.Fatalf("deleted MCP credential rebound: %v", err)
	}
	if _, err := publisher.Publish(ctx, document("stale", "mcp-active"), "publish-stale", "operator"); !errors.Is(err, runtimeconfig.ErrInvalid) {
		t.Fatalf("deleted MCP credential republished: %v", err)
	}
	// Wrong-kind credentials must fail publication before any immutable row exists.
	wrong, _ := NewOTLPHeadersCredential(map[string]string{"Authorization": "otel-secret"})
	defer wrong.Destroy()
	if _, err := service.Create(ctx, RuntimeCredentialCreateRequest{CredentialID: "wrong-kind", Material: wrong, IdempotencyKey: "create-wrong-kind", ActorID: "operator"}); err != nil {
		t.Fatal(err)
	}
	if _, err := publisher.Publish(ctx, document("wrong-kind", "wrong-kind"), "publish-wrong-kind", "operator"); !errors.Is(err, runtimeconfig.ErrInvalid) {
		t.Fatalf("MCP accepted OTLP credential: %v", err)
	}
	create("mcp-race")
	raceRef := publish("mcp-race", "mcp-race")
	safe := runtimeconfig.Ref{Name: runtimeconfig.BuiltInName, Version: runtimeconfig.BuiltInVersion, Digest: runtimeconfig.BuiltInDigest}
	initial, err := bindings.Create(ctx, "race", safe, "operator", time.Now())
	if err != nil {
		t.Fatal(err)
	}
	start := make(chan struct{})
	rebound := make(chan error, 1)
	deleted := make(chan error, 1)
	go func() {
		<-start
		_, err := bindings.Rebind(ctx, "race", initial.Revision, raceRef, "operator", time.Now())
		rebound <- err
	}()
	go func() { <-start; _, err := service.Delete(ctx, "mcp-race", "operator"); deleted <- err }()
	close(start)
	rebindErr, deleteErr := <-rebound, <-deleted
	if rebindErr == nil && deleteErr == nil {
		t.Fatal("credential deletion and referencing rebind both committed")
	}
	if rebindErr != nil && !errors.Is(rebindErr, runtimeconfig.ErrInvalid) {
		t.Fatalf("unexpected rebind race error: %v", rebindErr)
	}
	if deleteErr != nil && !errors.Is(deleteErr, ErrRuntimeCredentialInUse) {
		t.Fatalf("unexpected delete race error: %v", deleteErr)
	}
}
