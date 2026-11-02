package controlplane

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/agentprincipal"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Query budgets of one placement attempt (CTL-R162). Every test states its
// candidate and binding cardinalities and the exact reads the attempt makes
// outside and inside the pinning transaction, so a pass that repeats a read
// the attempt already made, or reuses one it had to discard, fails it.

// placementReadCounts counts SQL statements by the catalog they read.
type placementReadCounts struct {
	versions, bindings, principals, other int
}

// placementReadTrace splits the statements of a placement attempt into the
// optimistic reads and the pinning transaction (begin through commit or
// rollback).
type placementReadTrace struct {
	mu            sync.Mutex
	inTransaction bool
	optimistic    placementReadCounts
	transaction   placementReadCounts
	// beforeRefresh runs once, before the first principal read outside a
	// transaction: only the principal refresh of a failed pass makes one.
	beforeRefresh func()
}

func (trace *placementReadTrace) TraceQueryStart(
	ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryStartData,
) context.Context {
	statement := strings.TrimSpace(data.SQL)
	var hook func()
	trace.mu.Lock()
	if strings.HasPrefix(statement, "begin") {
		trace.inTransaction = true
	}
	counts := &trace.optimistic
	if trace.inTransaction {
		counts = &trace.transaction
	}
	switch {
	case strings.Contains(statement, "FROM runtime_config_versions"):
		counts.versions++
	case strings.Contains(statement, "FROM runtime_label_bindings"):
		counts.bindings++
	case strings.Contains(statement, "FROM runtime_agent_principals"):
		counts.principals++
		if !trace.inTransaction {
			hook, trace.beforeRefresh = trace.beforeRefresh, nil
		}
	default:
		counts.other++
	}
	if statement == "commit" || statement == "rollback" {
		trace.inTransaction = false
	}
	trace.mu.Unlock()
	if hook != nil {
		hook()
	}
	return ctx
}

func (*placementReadTrace) TraceQueryEnd(context.Context, *pgx.Conn, pgx.TraceQueryEndData) {}

// take returns and resets the counts of the attempts since the last call.
func (trace *placementReadTrace) take() (placementReadCounts, placementReadCounts) {
	trace.mu.Lock()
	defer trace.mu.Unlock()
	optimistic, transaction := trace.optimistic, trace.transaction
	trace.optimistic, trace.transaction = placementReadCounts{}, placementReadCounts{}
	return optimistic, transaction
}

// countPlacementReads routes the allocator's SQL through a counting pool and
// its credential metadata lookups through counting wrappers. Runtime
// credentials resolve to OTLP headers without SQL.
func countPlacementReads(
	t *testing.T, ctx context.Context, f *placementFixture,
) (*placementReadTrace, *placementCountingLLMLookup, *placementCountingRuntimeLookup) {
	t.Helper()
	trace := &placementReadTrace{}
	configuration := f.pool.Config().Copy()
	configuration.ConnConfig.Tracer = trace
	counted, err := pgxpool.NewWithConfig(ctx, configuration)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(counted.Close)
	f.allocator.pool = counted
	llm := &placementCountingLLMLookup{base: f.allocator.llmCredentials}
	runtime := &placementCountingRuntimeLookup{}
	f.allocator.llmCredentials, f.allocator.runtimeCredentials = llm, runtime
	return trace, llm, runtime
}

const (
	plainPlacementSpec = `{"worker":{"telemetry":null}}`
	otlpPlacementSpec  = `{"worker":{"telemetry":{"adapter":"otlp-http@1","endpoint":"https://otel.example/v1/traces"}}}`
)

func credentialPlacementSpec(credential string) string {
	return `{"worker":{"telemetry":{"adapter":"otlp-http@1","endpoint":"https://otel.example/v1/traces",` +
		`"credential":"` + credential + `"}}}`
}

func endpointPlacementSpec(endpoint string) string {
	return `{"worker":{"telemetry":{"adapter":"otlp-http@1","endpoint":"` + endpoint + `"}}}`
}

func publishPlacementConfig(t *testing.T, ctx context.Context, pool *pgxpool.Pool, name, spec string) runtimeconfig.Ref {
	t.Helper()
	publisher, err := runtimeconfig.NewPublisher(runtimeconfig.PublisherOptions{
		Pool: pool, RuntimeCredentials: placementRuntimeCatalog{},
		PlannerTelemetryAdapters: runtimeconfigtest.PlannerTelemetryAdapterCatalogFunc(func(ref string) bool { return ref == "otlp-http@1" }),
	})
	if err != nil {
		t.Fatal(err)
	}
	published, err := publisher.Publish(ctx, []byte(`{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig",`+
		`"metadata":{"name":"`+name+`","version":"1"},"spec":`+spec+`}`), "publish-"+name, "operator")
	if err != nil {
		t.Fatal(err)
	}
	return published.Version.Ref
}

func bindPlacementLabel(t *testing.T, ctx context.Context, pool *pgxpool.Pool, label string, ref runtimeconfig.Ref) {
	t.Helper()
	if _, err := runtimeconfig.NewRepository(pool).CreateBinding(ctx, label, ref, "operator", time.Now().UTC()); err != nil {
		t.Fatal(err)
	}
}

func rebindPlacementLabel(ctx context.Context, pool *pgxpool.Pool, label string, ref runtimeconfig.Ref) error {
	repository := runtimeconfig.NewRepository(pool)
	current, err := repository.GetBinding(ctx, label)
	if err != nil {
		return err
	}
	_, err = repository.Rebind(ctx, label, current.Revision, ref, "operator", time.Now().UTC())
	return err
}

// occupy gives instanceID a provisional reservation of another Stage, so it
// stays live for a model Stage hold but is no placement candidate.
func (f *placementFixture) occupy(t *testing.T, instanceID, runtimeAgentID string) {
	t.Helper()
	request := f.request()
	request.StageExecutionID = "stage-occupying-" + instanceID
	if _, err := f.registry.ReserveCandidateEdges(request, []CandidateEdge{{
		LogicalAgentName: "builder", RuntimeAgentID: runtimeAgentID, RuntimeAgentInstanceID: instanceID,
		RuntimeAgentLabelRevision: 1, RequiredRuntimeAdapters: []runtimesettings.RuntimeAdapterRef{},
	}}); err != nil {
		t.Fatal(err)
	}
}

func modelStageHeld(registry *InMemoryRegistry, instanceID string) bool {
	now := registry.monotonicNow()
	registry.mu.Lock()
	defer registry.mu.Unlock()
	entry, ok := registry.agents[instanceID]
	return ok && entry.modelStageHeldUntil > now
}

// A model Stage that finds no capacity resolves its idle candidates, refreshes
// their principals and computes its hold over every live Agent. Each pinned
// version, binding and credential is read once per attempt however many
// candidates share it, Agents seen only by the hold are still resolved, and
// with no idle candidate only the hold reads anything.
//
// Cardinalities: one binding; the default and one Run label; idle candidates
// that share the Agent label "shared" but lack the OTLP adapter every
// candidate needs; allocated candidates per Agent label "shared" and "other",
// each label with its own Runtime credential. Per attempt: versions = 2
// pinned + 1 per Agent label = 4, bindings = 2, one principal read per idle
// candidate, one LLM credential and two Runtime credential lookups.
func TestPlacementAttemptReusesResolutionsAcrossPassesAndHold(t *testing.T) {
	for _, size := range []struct{ idle, allocatedPerLabel int }{{0, 1}, {1, 1}, {8, 8}} {
		t.Run(fmt.Sprintf("idle=%d,allocated=%d", size.idle, 2*size.allocatedPerLabel), func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
			defer cancel()
			pool := isolatedPlacementPool(t, ctx)
			f := newPlacementFixture(t, ctx, pool, []string{"debug"})
			// The hold must not lapse while the test runs, however slow the database.
			f.registry.modelStageHold = time.Hour
			bindPlacementLabel(t, ctx, pool, "shared", publishPlacementConfig(t, ctx, pool, "shared-agent", credentialPlacementSpec("shared-telemetry")))
			bindPlacementLabel(t, ctx, pool, "other", publishPlacementConfig(t, ctx, pool, "other-agent", credentialPlacementSpec("other-telemetry")))
			principals := 0
			register := func(instanceID string, adapters []runtimesettings.RuntimeAdapterRef, label string) string {
				principals++
				labels := []string{label}
				return f.registerPrincipalProcess(t, ctx, instanceID, fmt.Sprintf("%064x", principals), labels,
					f.candidateRegistration(adapters, labels))
			}
			var idle, allocated []string
			for index := range size.idle {
				instanceID := fmt.Sprintf("idle-%d", index)
				register(instanceID, nil, "shared")
				idle = append(idle, instanceID)
			}
			for index := range size.allocatedPerLabel {
				for _, label := range []string{"shared", "other"} {
					instanceID := fmt.Sprintf("allocated-%s-%d", label, index)
					f.occupy(t, instanceID, register(instanceID,
						[]runtimesettings.RuntimeAdapterRef{runtimesettings.RuntimeAdapterOTLPHTTP}, label))
					allocated = append(allocated, instanceID)
				}
			}
			trace, llm, runtime := countPlacementReads(t, ctx, f)

			if _, err := f.allocator.ReserveAllContext(ctx, f.request()); !errors.Is(err, ErrInsufficientCapacity) {
				t.Fatalf("model Stage without an idle compatible Agent = %v, want capacity", err)
			}
			optimistic, transaction := trace.take()
			want := placementReadCounts{versions: 4, bindings: 2, principals: size.idle}
			if optimistic != want || transaction != (placementReadCounts{}) || llm.count != 1 || runtime.count != 2 {
				t.Fatalf("attempt reads = optimistic %+v, transaction %+v, LLM %d, Runtime %d; want optimistic %+v, no transaction, LLM 1, Runtime 2",
					optimistic, transaction, llm.count, runtime.count, want)
			}
			for _, instanceID := range allocated {
				if !modelStageHeld(f.registry, instanceID) {
					t.Fatalf("hold did not resolve allocated Agent %s", instanceID)
				}
			}
			for _, instanceID := range idle {
				if modelStageHeld(f.registry, instanceID) {
					t.Fatalf("hold took idle Agent %s without the required adapter", instanceID)
				}
			}
		})
	}
}

// A compatible idle candidate that cannot complete a two-Worker matching is
// resolved once for each binding; the hold reuses those resolutions for it
// and for an allocated Agent with the same Agent label, and holds both.
//
// Cardinalities: two bindings with one ExecutionConfig, the default and one
// Run label, one idle and one allocated candidate on Agent label "shared".
// Per attempt: 3 versions, 1 binding, 1 principal, one LLM and one Runtime
// credential lookup.
func TestPlacementAttemptReusesCompatibleCandidateResolutionInHold(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedPlacementPool(t, ctx)
	f := newPlacementFixture(t, ctx, pool, []string{"debug"})
	f.registry.modelStageHold = time.Hour
	bindPlacementLabel(t, ctx, pool, "shared", publishPlacementConfig(t, ctx, pool, "shared-agent", credentialPlacementSpec("shared-telemetry")))
	otlp := []runtimesettings.RuntimeAdapterRef{runtimesettings.RuntimeAdapterOTLPHTTP}
	f.registerCandidateWithLabels(t, ctx, "runtime-idle", "1", otlp, []string{"shared"})
	f.occupy(t, "runtime-allocated", f.registerCandidateWithLabels(t, ctx, "runtime-allocated", "2", otlp, []string{"shared"}))
	trace, llm, runtime := countPlacementReads(t, ctx, f)
	request := f.request()
	second := request.Bindings[0]
	second.LogicalAgentName, second.Namespace = "reviewer", "reviewer"
	request.Bindings = append(request.Bindings, second)

	if _, err := f.allocator.ReserveAllContext(ctx, request); !errors.Is(err, ErrInsufficientCapacity) {
		t.Fatalf("two Workers with one idle Agent = %v, want capacity", err)
	}
	optimistic, transaction := trace.take()
	want := placementReadCounts{versions: 3, bindings: 1, principals: 1}
	if optimistic != want || transaction != (placementReadCounts{}) || llm.count != 1 || runtime.count != 1 {
		t.Fatalf("attempt reads = optimistic %+v, transaction %+v, LLM %d, Runtime %d; want optimistic %+v, no transaction, LLM 1, Runtime 1",
			optimistic, transaction, llm.count, runtime.count, want)
	}
	for _, instanceID := range []string{"runtime-idle", "runtime-allocated"} {
		if !modelStageHeld(f.registry, instanceID) {
			t.Fatalf("hold of the complete live matching omitted %s", instanceID)
		}
	}
}

// A principal refresh proves that a mutation committed after the first pass,
// so the retry reads every binding and credential again and keeps only the
// pinned default version. Here the operator rebinds "next" to a configuration
// without an adapter requirement and moves the stale principal onto "next"
// between the first pass and the refresh: the retry must see the rebind.
//
// Cardinalities: one binding, the default only, two idle candidates without
// adapters, Agent labels "next" and "prior". First pass: 1 + 2 versions, 2
// bindings, one LLM lookup; refresh: 2 principals; retry: 1 version, 1
// binding, one LLM lookup.
func TestPlacementAttemptRefreshDiscardsObservationsOfChangedPrincipals(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedPlacementPool(t, ctx)
	f := newPlacementFixture(t, ctx, pool, nil)
	bindPlacementLabel(t, ctx, pool, "next", publishPlacementConfig(t, ctx, pool, "next-otlp", otlpPlacementSpec))
	bindPlacementLabel(t, ctx, pool, "prior", publishPlacementConfig(t, ctx, pool, "prior-otlp", otlpPlacementSpec))
	plain := publishPlacementConfig(t, ctx, pool, "next-plain", plainPlacementSpec)
	f.registerCandidateWithLabels(t, ctx, "runtime-next", "1", nil, []string{"next"})
	moved := f.registerCandidateWithLabels(t, ctx, "runtime-prior", "2", nil, []string{"prior"})
	trace, llm, _ := countPlacementReads(t, ctx, f)
	trace.beforeRefresh = func() {
		if err := rebindPlacementLabel(ctx, pool, "next", plain); err != nil {
			t.Error(err)
		}
		if _, err := agentprincipal.NewRepository(pool).ReplaceLabels(
			ctx, moved, 1, []string{"next"}, "operator", time.Now().UTC(),
		); err != nil {
			t.Error(err)
		}
	}

	reservations, err := f.allocator.ReserveAllContext(ctx, f.request())
	if err != nil || len(reservations) != 1 || reservations[0].ResolvedRuntimeConfig == nil {
		t.Fatalf("retry after the principal refresh = (%+v, %v)", reservations, err)
	}
	if provenance := reservations[0].ResolvedRuntimeConfig.Provenance.AgentLabels; len(provenance) != 1 ||
		provenance[0].Label != "next" || provenance[0].Config.Name != plain.Name {
		t.Fatalf("retry reused the binding read before the refresh: %+v", provenance)
	}
	optimistic, transaction := trace.take()
	wantOptimistic := placementReadCounts{versions: 4, bindings: 3, principals: 2}
	if optimistic != wantOptimistic || llm.count != 2 {
		t.Fatalf("optimistic reads = %+v, LLM %d; want %+v, LLM 2", optimistic, llm.count, wantOptimistic)
	}
	// The pinning transaction still re-reads the default and the selected
	// label's binding and version under its locks (CTL-R88).
	if transaction.versions != 2 || transaction.bindings != 2 || transaction.principals != 1 {
		t.Fatalf("pinning transaction reads = %+v, want 2 versions, 2 bindings, 1 principal", transaction)
	}
}

// A drift found by the pinning transaction discards the attempt's
// observations, so the model Stage hold that follows reads the binding again
// and holds nothing on an Agent that the new configuration excludes.
//
// Cardinalities: one binding, the default only, one idle candidate without
// adapters on Agent label "site". First pass: 2 versions, 1 binding, one LLM
// lookup; refresh: 1 principal; hold: 1 version, 1 binding, one LLM lookup.
func TestPlacementAttemptTransactionDriftDiscardsObservationsBeforeHold(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedPlacementPool(t, ctx)
	f := newPlacementFixture(t, ctx, pool, nil)
	f.registry.modelStageHold = time.Hour
	bindPlacementLabel(t, ctx, pool, "site", publishPlacementConfig(t, ctx, pool, "site-plain", plainPlacementSpec))
	otlp := publishPlacementConfig(t, ctx, pool, "site-otlp", otlpPlacementSpec)
	f.registerCandidateWithLabels(t, ctx, "runtime-site", "1", nil, []string{"site"})
	trace, llm, _ := countPlacementReads(t, ctx, f)
	request := f.request()
	rebound := false
	request.Admit = func(ctx context.Context, _ []Reservation) error {
		if rebound {
			return errors.New("admission ran again in one attempt")
		}
		rebound = true
		return rebindPlacementLabel(ctx, pool, "site", otlp)
	}

	if _, err := f.allocator.ReserveAllContext(ctx, request); !errors.Is(err, ErrInsufficientCapacity) {
		t.Fatalf("binding changed before pinning = %v, want capacity", err)
	}
	if modelStageHeld(f.registry, "runtime-site") {
		t.Fatal("hold reused the binding that the pinning transaction found changed")
	}
	optimistic, transaction := trace.take()
	wantOptimistic := placementReadCounts{versions: 3, bindings: 2, principals: 1}
	if optimistic != wantOptimistic || llm.count != 2 {
		t.Fatalf("optimistic reads = %+v, LLM %d; want %+v, LLM 2", optimistic, llm.count, wantOptimistic)
	}
	if transaction.versions != 2 || transaction.bindings != 2 || transaction.principals != 1 {
		t.Fatalf("pinning transaction reads = %+v, want 2 versions, 2 bindings, 1 principal", transaction)
	}
	allocations, err := runstore.NewPostgresStore(pool).ListStageAllocations(ctx, f.stageExecutionID)
	if err != nil || len(allocations) != 0 {
		t.Fatalf("drifted placement left durable allocations: (%+v, %v)", allocations, err)
	}
}

// A successful placement reads every optimistic input once and re-reads the
// selected inputs under the pinning transaction.
//
// Cardinalities: one binding, the default and one Run label, four idle
// candidates sharing Agent label "site". Optimistic: 3 versions, 1 binding,
// one LLM lookup; transaction: 3 versions, a locked and a read binding, one
// locked principal.
func TestPlacementAttemptSuccessfulPlacementReadBudget(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedPlacementPool(t, ctx)
	f := newPlacementFixture(t, ctx, pool, []string{"debug"})
	bindPlacementLabel(t, ctx, pool, "site", publishPlacementConfig(t, ctx, pool, "site-plain", plainPlacementSpec))
	for index := 1; index <= 4; index++ {
		f.registerCandidateWithLabels(t, ctx, fmt.Sprintf("runtime-site-%d", index), fmt.Sprintf("%x", index),
			[]runtimesettings.RuntimeAdapterRef{runtimesettings.RuntimeAdapterOTLPHTTP}, []string{"site"})
	}
	trace, llm, runtime := countPlacementReads(t, ctx, f)

	reservations, err := f.allocator.ReserveAllContext(ctx, f.request())
	if err != nil || len(reservations) != 1 {
		t.Fatalf("placement = (%+v, %v)", reservations, err)
	}
	optimistic, transaction := trace.take()
	wantOptimistic := placementReadCounts{versions: 3, bindings: 1}
	if optimistic != wantOptimistic || llm.count != 1 || runtime.count != 0 {
		t.Fatalf("optimistic reads = %+v, LLM %d, Runtime %d; want %+v, LLM 1, Runtime 0",
			optimistic, llm.count, runtime.count, wantOptimistic)
	}
	if transaction.versions != 3 || transaction.bindings != 2 || transaction.principals != 1 {
		t.Fatalf("pinning transaction reads = %+v, want 3 versions, 2 bindings, 1 principal", transaction)
	}
}

// Reuse never crosses attempts: each Scheduler poll observes a rebind, a
// deleted and restored durable LLM credential and a principal label change
// committed after the previous attempt.
func TestPlacementNextAttemptObservesDurableMutations(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedPlacementPool(t, ctx)
	f := newPlacementFixture(t, ctx, pool, nil)
	f.registry.modelStageHold = time.Hour
	bindPlacementLabel(t, ctx, pool, "site", publishPlacementConfig(t, ctx, pool, "site-old",
		endpointPlacementSpec("https://old-otel.example/v1/traces")))
	newRef := publishPlacementConfig(t, ctx, pool, "site-new", endpointPlacementSpec("https://new-otel.example/v1/traces"))
	runtimeAgentID := f.registerCandidateWithLabels(t, ctx, "runtime-site", "1",
		[]runtimesettings.RuntimeAdapterRef{runtimesettings.RuntimeAdapterOTLPHTTP}, []string{"site"})

	record := storePlacementLLMCredential(t, ctx, f)
	repository := credentials.NewRepository(pool)
	cipher, err := credentials.NewTokenCipher(bytes.Repeat([]byte{0x73}, 32))
	if err != nil {
		t.Fatal(err)
	}
	provider, err := credentials.NewEncryptedProvider(repository, cipher)
	if err != nil {
		t.Fatal(err)
	}
	development, err := credentials.NewStaticProvider(nil)
	if err != nil {
		t.Fatal(err)
	}
	if f.allocator.transactionLLMCredentials, err = credentials.NewTransactionLookupFactory(development); err != nil {
		t.Fatal(err)
	}
	llm := &placementCountingLLMLookup{base: provider}
	f.allocator.llmCredentials = llm

	endpoint := func() string {
		t.Helper()
		var observed string
		denied := errors.New("observe and defer admission")
		request := f.request()
		request.Admit = func(_ context.Context, candidates []Reservation) error {
			if len(candidates) == 1 && candidates[0].ResolvedRuntimeConfig != nil &&
				candidates[0].ResolvedRuntimeConfig.WorkerTelemetry != nil {
				observed = candidates[0].ResolvedRuntimeConfig.WorkerTelemetry.Endpoint
			}
			return denied
		}
		if _, err := f.allocator.ReserveAllContext(ctx, request); !errors.Is(err, denied) {
			t.Fatalf("observing attempt = %v", err)
		}
		return observed
	}
	if got := endpoint(); got != "https://old-otel.example/v1/traces" || llm.count != 1 {
		t.Fatalf("first attempt endpoint %q with %d LLM lookups", got, llm.count)
	}
	if err := rebindPlacementLabel(ctx, pool, "site", newRef); err != nil {
		t.Fatal(err)
	}
	if got := endpoint(); got != "https://new-otel.example/v1/traces" || llm.count != 2 {
		t.Fatalf("attempt after rebind endpoint %q with %d LLM lookups", got, llm.count)
	}

	if err := repository.DeleteCredential(ctx, record.CredentialID); err != nil {
		t.Fatal(err)
	}
	if _, err := f.allocator.ReserveAllContext(ctx, f.request()); !errors.Is(err, ErrInsufficientCapacity) {
		t.Fatalf("attempt after credential deletion = %v, want capacity", err)
	}
	// The first pass observes the deletion; the hold reuses that observation.
	if llm.count != 3 || modelStageHeld(f.registry, "runtime-site") {
		t.Fatalf("deleted credential: %d LLM lookups, held %v", llm.count, modelStageHeld(f.registry, "runtime-site"))
	}

	if err := repository.InsertCredential(ctx, record); err != nil {
		t.Fatal(err)
	}
	if _, err := agentprincipal.NewRepository(pool).ReplaceLabels(
		ctx, runtimeAgentID, 1, []string{}, "operator", time.Now().UTC(),
	); err != nil {
		t.Fatal(err)
	}
	reservations, err := f.allocator.ReserveAllContext(ctx, f.request())
	if err != nil || len(reservations) != 1 || reservations[0].RuntimeAgentLabelRevision != 2 ||
		reservations[0].ResolvedRuntimeConfig == nil ||
		len(reservations[0].ResolvedRuntimeConfig.Provenance.AgentLabels) != 0 {
		t.Fatalf("attempt after credential restore and label change = (%+v, %v)", reservations, err)
	}
	// The stale first pass and the retry after the drift each look the
	// restored credential up once.
	if llm.count != 5 {
		t.Fatalf("restored credential attempt made %d LLM lookups in total, want 5", llm.count)
	}
}

// storePlacementLLMCredential stores the fixture selection's LLM credential
// as a managed record sealed with the 0x73 test key.
func storePlacementLLMCredential(t *testing.T, ctx context.Context, f *placementFixture) credentials.Record {
	t.Helper()
	cipher, err := credentials.NewTokenCipher(bytes.Repeat([]byte{0x73}, 32))
	if err != nil {
		t.Fatal(err)
	}
	token, err := credentials.NewToken("attempt-test-token")
	if err != nil {
		t.Fatal(err)
	}
	selection := f.selection
	envelope, err := cipher.Seal(selection.Credential.CredentialID, selection.LLMGateway.Ref, token)
	if err != nil {
		t.Fatal(err)
	}
	record := credentials.Record{
		CredentialID: selection.Credential.CredentialID,
		LLMGateway:   selection.LLMGateway.Ref,
		RemoteKeyID:  strings.Repeat("b", 64),
		EffectivePolicy: credentials.EffectiveGatewayPolicy{
			ModelPolicies: []contracts.ModelPolicyRef{selection.ModelPolicy.Ref},
			Models:        []string{selection.ModelPolicy.Model},
		},
		Envelope:  envelope,
		CreatedAt: time.Now().UTC(),
	}
	repository := credentials.NewRepository(f.pool)
	if err := repository.InsertOperation(ctx, credentials.Operation{
		OperationID: "create:" + record.CredentialID, IdempotencyKey: "reserve-" + record.CredentialID,
		RequestHash: "sha256:" + strings.Repeat("e", 64), CredentialID: record.CredentialID,
		Kind: credentials.OperationCreate, Phase: credentials.OperationPrepared,
		Request:   json.RawMessage(`{"credentialId":"` + record.CredentialID + `"}`),
		CreatedAt: record.CreatedAt, UpdatedAt: record.CreatedAt,
	}); err != nil {
		t.Fatal(err)
	}
	if err := repository.InsertCredential(ctx, record); err != nil {
		t.Fatal(err)
	}
	return record
}

func TestPlacementImpossibleCardinalityReadsNoCandidateConfiguration(t *testing.T) {
	ctx := t.Context()
	pool := isolatedPlacementPool(t, ctx)
	fixture := newPlacementFixture(t, ctx, pool, []string{"debug"})
	fixture.registerCandidate(t, ctx, "only-slot", "1", []runtimesettings.RuntimeAdapterRef{runtimesettings.RuntimeAdapterOTLPHTTP})
	trace, llm, runtime := countPlacementReads(t, ctx, fixture)
	request := fixture.request()
	second := request.Bindings[0]
	second.LogicalAgentName, second.Namespace = "second", "second"
	request.Bindings = append(request.Bindings, second)
	_, err := fixture.allocator.ReserveAllContext(ctx, request)
	if !errors.Is(err, ErrInsufficientCapacity) {
		t.Fatalf("placement = %v", err)
	}
	optimistic, transaction := trace.take()
	if optimistic.versions != 0 || optimistic.bindings != 0 || llm.count != 0 || runtime.count != 0 || transaction != (placementReadCounts{}) {
		t.Fatalf("impossible cardinality read configuration: optimistic=%+v transaction=%+v LLM=%d Runtime=%d", optimistic, transaction, llm.count, runtime.count)
	}
}
