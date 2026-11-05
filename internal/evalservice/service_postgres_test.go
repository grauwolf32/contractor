package evalservice

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/evalcoordinator"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
	"github.com/grauwolf32/contractor/internal/findingintake"
	pg "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/persistence/postgrestest"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/rundeletion"
	"github.com/grauwolf32/contractor/internal/runservice"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type testCredentials struct {
	gateway llmgateway.LLMGatewayConfigRef
}

func (c testCredentials) LookupLLMCredential(_ context.Context, id string) (resolved.CredentialMetadata, error) {
	if id != "development-worker" {
		return resolved.CredentialMetadata{}, errors.New("unexpected fixture credential")
	}
	return resolved.CredentialMetadata{Ref: llmgateway.LLMCredentialRef{CredentialID: id}, LLMGateway: c.gateway, Unrestricted: true}, nil
}

type testBarrier struct{}

func (testBarrier) WithRunCreation(ctx context.Context, fn func() error) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	return fn()
}
func (testBarrier) ValidateRuntimeCredentialUse(
	context.Context, credentials.RuntimeCredentialUser, string, ...string,
) error {
	return nil
}

type serviceHarness struct {
	pool     *pgxpool.Pool
	service  *Service
	resolver *Resolver
	driver   *Driver
	scope    evalstore.Scope
	run      *lostRunResponse
	audit    *lostAuditResponse
}

var errResponseLost = errors.New("fixture: accepted response lost")

type lostRunResponse struct {
	RunCreator
	mu     sync.Mutex
	lost   map[string]bool
	remove bool
	pool   *pgxpool.Pool
}

func (r *lostRunResponse) CreatePublic(ctx context.Context, p runservice.PublicCreateParams) (runservice.CreateResult, error) {
	result, err := r.RunCreator.CreatePublic(ctx, p)
	if err != nil {
		return result, err
	}
	r.mu.Lock()
	first := !r.lost[p.IdempotencyKey]
	r.lost[p.IdempotencyKey] = true
	r.mu.Unlock()
	if first {
		if r.remove {
			store := runstore.NewPostgresStore(r.pool, artifactpgtest.NewStorage())
			if _, err = store.TransitionRun(ctx, result.Run.RunID, runstore.RunPending, runstore.RunRunning, runstore.Reason{Code: "fixture_admitted"}); err != nil {
				return result, err
			}
			if _, err = store.TransitionRun(ctx, result.Run.RunID, runstore.RunRunning, runstore.RunSucceeded, runstore.Reason{Code: "fixture"}); err != nil {
				return result, err
			}
			if err = rundeletion.New(r.pool, artifactpgtest.NewStorage()).DeleteReleasedTerminalRun(ctx, p.OwnerID, result.Run.RunID); err != nil {
				return result, err
			}
		}
		return runservice.CreateResult{}, errResponseLost
	}
	return result, nil
}

type lostAuditResponse struct {
	AuditExecutor
	mu   sync.Mutex
	lost map[string]bool
}

func (a *lostAuditResponse) lose(key string) bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.lost[key] {
		return false
	}
	a.lost[key] = true
	return true
}
func (a *lostAuditResponse) CreateDraft(ctx context.Context, p auditservice.CreateDraftParams) (auditstore.Audit, bool, error) {
	v, created, err := a.AuditExecutor.CreateDraft(ctx, p)
	if err == nil && a.lose(p.IdempotencyKey) {
		return auditstore.Audit{}, false, errResponseLost
	}
	return v, created, err
}
func (a *lostAuditResponse) Start(ctx context.Context, p auditservice.StartParams) (auditservice.StartedAudit, error) {
	v, err := a.AuditExecutor.Start(ctx, p)
	if err == nil && a.lose(p.IdempotencyKey) {
		return auditservice.StartedAudit{}, errResponseLost
	}
	return v, err
}

// Dispatch must consume the retained preparation binding even if mutable
// defaults can no longer resolve. Dependency validation remains the real
// Snapshot implementation on the ordinary PostgreSQL-backed Run/Audit path.
type retainedExecutionCatalog struct{ *config.Snapshot }

func (retainedExecutionCatalog) ResolveRunWorkflow(context.Context, string, resolved.CredentialLookup) (resolved.ResolvedWorkflow, error) {
	return resolved.ResolvedWorkflow{}, errors.New("mutable defaults must not be consulted after preparation")
}

func (retainedExecutionCatalog) ResolveAuditProfile(context.Context, string) (auditprofile.PinnedProgram, error) {
	return auditprofile.PinnedProgram{}, errors.New("mutable defaults must not be consulted after preparation")
}

func newHarness(t *testing.T) *serviceHarness {
	t.Helper()
	pool := serviceTestPool(t)
	catalog := serviceTestCatalog(t)
	gateway, err := catalog.LLMGateway("test-gateway@1")
	if err != nil {
		t.Fatal(err)
	}
	credentialLookup := testCredentials{gateway: gateway.Ref}
	transactionLLMCredentials := runtimeconfigtest.TransactionLLMCredentialLookupFactoryFunc(func(pgx.Tx) (resolved.CredentialLookup, error) { return credentialLookup, nil })
	audits, err := auditservice.New(auditservice.Options{Storage: artifactpgtest.NewStorage(), Pool: pool, Profiles: retainedExecutionCatalog{catalog}, TransactionLLMCredentials: transactionLLMCredentials, CredentialGuard: testBarrier{}})
	if err != nil {
		t.Fatal(err)
	}
	runs, err := runservice.New(runservice.Options{Runs: runstore.NewPostgresStore(pool, artifactpgtest.NewStorage()), Workflows: retainedExecutionCatalog{catalog}, LLMCredentials: credentialLookup, CredentialGuard: testBarrier{}, RuntimeCredentials: testBarrier{}, Projects: projectstore.NewPostgresStore(pool), SkillInitializationAvailable: true, PublicTransaction: func(ctx context.Context, fn func(runservice.PublicRunWriter, *artifacts.Service, runservice.RuntimeCredentialValidator) error) error {
		return pg.InTx(ctx, pool, pgx.TxOptions{IsoLevel: pgx.RepeatableRead}, func(tx pgx.Tx) error {
			lookup, err := runtimeconfig.BindTransactionLLMCredentialLookup(tx, transactionLLMCredentials)
			if err != nil {
				return err
			}
			return fn(runstore.NewRunCreationPostgresStore(tx, lookup, credentials.NewRuntimeCredentialRepository(tx), artifactpgtest.NewStorage()), artifacts.NewService(artifactpgtest.NewStorage().Repository(tx)), credentials.NewRuntimeCredentialRepository(tx))
		})
	}})
	if err != nil {
		t.Fatal(err)
	}
	run := &lostRunResponse{RunCreator: runs, lost: map[string]bool{}, pool: pool}
	audit := &lostAuditResponse{AuditExecutor: audits, lost: map[string]bool{}}
	resolver := &Resolver{Storage: artifactpgtest.NewStorage(), Pool: pool, Catalog: catalog, Credentials: transactionLLMCredentials, Barrier: testBarrier{}}
	driver := &Driver{Storage: artifactpgtest.NewStorage(), Pool: pool, Runs: run, Audits: audit}
	service, err := New(Options{Storage: artifactpgtest.NewStorage(), Pool: pool, Resolver: resolver, Driver: driver})
	if err != nil {
		t.Fatal(err)
	}
	scope := evalstore.Scope{OwnerID: "eval-owner", ProjectID: "eval-project"}
	_, _, err = projectstore.NewPostgresStore(pool).Create(t.Context(), projectstore.CreateParams{OwnerID: scope.OwnerID, ProjectID: scope.ProjectID, Kind: projectstore.KindEvaluation, Name: "Evals", IdempotencyKey: "project", RequestDigest: evaldomain.Digest([]byte("project"))})
	if err != nil {
		t.Fatal(err)
	}
	return &serviceHarness{pool, service, resolver, driver, scope, run, audit}
}
func frozen(t *testing.T, kind string, v any) evaldomain.Frozen {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	f, err := evaldomain.Freeze(kind, b)
	if err != nil {
		t.Fatalf("%s: %v", kind, err)
	}
	return f
}
func identity(t *testing.T, key string, revision int64, doc evaldomain.Frozen) evaldomain.MutationIdentity {
	t.Helper()
	etag := ""
	if revision > 0 {
		etag = fmt.Sprintf(`"%d"`, revision)
	}
	m, err := evaldomain.IdentifyMutation(key, etag, revision > 0, doc.Kind(), doc.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	return m
}
func (h *serviceHarness) get(t *testing.T, id string) evalstore.Experiment {
	t.Helper()
	e, err := evalstore.NewPostgresStore(h.pool, artifactpgtest.NewStorage()).Get(t.Context(), h.scope.OwnerID, id)
	if err != nil {
		t.Fatal(err)
	}
	return e
}
func (h *serviceHarness) dataset(t *testing.T, kind string) (evaldomain.Draft, evaldomain.DatasetInput) {
	t.Helper()
	draft, data, _ := planInputs(t)
	source, _ := artifacts.NewService(artifactpgtest.NewStorage().Repository(h.pool)).User(h.scope.OwnerID)
	for i := range data.Cases {
		c := &data.Cases[i]
		slot, media, payload := "task", "application/zip", []byte("fixture ZIP is not executed")
		if kind == "audit" {
			slot, media, payload = "checklist", "application/json", []byte(`{"fixture":"Exact selected Eval material"}`)
		}
		write, err := source.Write(t.Context(), contracts.ArtifactRef{Namespace: "fixtures", Name: c.ID}, artifacts.Payload{MediaType: media, Data: payload}, nil)
		if err != nil {
			t.Fatal(err)
		}
		meta, err := source.Metadata(t.Context(), write.Ref)
		if err != nil {
			t.Fatal(err)
		}
		c.Task.Parameters = map[string]string{}
		c.Requires = []string{}
		c.Outputs = map[string]evaldomain.Output{}
		c.Inputs = map[string]evaldomain.Artifact{slot: {Scope: "user", ScopeID: h.scope.OwnerID, Namespace: write.Ref.Namespace, Name: write.Ref.Name, Revision: *write.Ref.Revision, SHA256: evaldomain.Digest(payload), MediaType: media, SizeBytes: meta.Size}}
	}
	for i := range draft.Variants {
		v := &draft.Variants[i]
		v.Kind = kind
		v.Selector = "audit-check@1"
		if kind == "audit" {
			v.Selector = "test-checklist@1"
		}
	}
	doc := frozen(t, "DatasetInput", data)
	if err := h.service.tx(t.Context(), func(s *evalstore.Store) error {
		_, err := s.PutDataset(t.Context(), h.scope, "r1", doc, identity(t, "dataset", 0, doc))
		return err
	}); err != nil {
		t.Fatal(err)
	}
	return draft, data
}
func (h *serviceHarness) create(t *testing.T, kind string) evalstore.Experiment {
	t.Helper()
	draft, _ := h.dataset(t, kind)
	doc := frozen(t, "CreateExperiment", evaldomain.CreateExperiment{Name: "Eight members", ControlMode: "server", Draft: &draft})
	receipt, err := h.service.Create(t.Context(), h.scope, doc, identity(t, "create", 0, doc))
	if err != nil {
		t.Fatal(err)
	}
	ref, err := receipt.Experiment()
	if err != nil {
		t.Fatal(err)
	}
	return h.get(t, ref.ExperimentID)
}
func (h *serviceHarness) command(t *testing.T, e evalstore.Experiment, kind string) evalstore.Receipt {
	t.Helper()
	c := evaldomain.Command{Kind: evaldomain.CommandKind(kind)}
	if kind != "prepare" && kind != "duplicate" {
		p, err := evalstore.NewPostgresStore(h.pool, artifactpgtest.NewStorage()).FrozenPlan(t.Context(), e.OwnerID, e.ID)
		if err != nil {
			t.Fatal(err)
		}
		c.PlanSHA256 = p.SHA256
	}
	doc := frozen(t, "Command", c)
	r, err := h.service.Command(t.Context(), h.scope, e.ID, c, identity(t, fmt.Sprintf("%s-%d", kind, e.Revision), e.Revision, doc))
	if err != nil {
		t.Fatalf("%s: %v", kind, err)
	}
	return r
}
func (h *serviceHarness) coordinator(t *testing.T, holder string) *evalcoordinator.Coordinator {
	t.Helper()
	c, err := evalcoordinator.New(evalstore.NewPostgresStore(h.pool, artifactpgtest.NewStorage()), h.service, evalcoordinator.Options{HolderID: holder})
	if err != nil {
		t.Fatal(err)
	}
	return c
}
func tick(t *testing.T, c *evalcoordinator.Coordinator) {
	t.Helper()
	_, err := c.RunOnce(t.Context())
	if err != nil && !errors.Is(err, errResponseLost) {
		t.Fatal(err)
	}
}
func count(t *testing.T, pool *pgxpool.Pool, table string) int {
	t.Helper()
	var n int
	if err := pool.QueryRow(t.Context(), "SELECT count(*) FROM "+pgx.Identifier{table}.Sanitize()).Scan(&n); err != nil {
		t.Fatal(err)
	}
	return n
}
func (h *serviceHarness) prepared(t *testing.T, kind string) evalstore.Experiment {
	t.Helper()
	e := h.create(t, kind)
	h.command(t, e, "prepare")
	tick(t, h.coordinator(t, "prepare"))
	e = h.get(t, e.ID)
	if e.State != "ready" || e.Expected != 8 {
		t.Fatalf("prepare: state=%s members=%d diagnostic=%s", e.State, e.Expected, e.Diagnostic)
	}
	if count(t, h.pool, "workflow_runs") != 0 || count(t, h.pool, "audits") != 0 || count(t, h.pool, "projects") != 1 {
		t.Fatal("prepare executed or provisioned work")
	}
	return e
}

func (h *serviceHarness) preparedWithCapacity(t *testing.T, kind string, capacity int) evalstore.Experiment {
	t.Helper()
	e := h.create(t, kind)
	var draft evaldomain.Draft
	if err := json.Unmarshal(e.Draft.Bytes(), &draft); err != nil {
		t.Fatal(err)
	}
	draft.Budgets.MaxInFlight = capacity
	update := frozen(t, "DraftUpdate", evaldomain.DraftUpdate{Name: e.Name, Draft: draft})
	if err := h.service.tx(t.Context(), func(st *evalstore.Store) error {
		_, err := st.UpdateDraft(t.Context(), h.scope, e.ID, update, identity(t, "capacity", e.Revision, update))
		return err
	}); err != nil {
		t.Fatal(err)
	}
	h.command(t, h.get(t, e.ID), "prepare")
	tick(t, h.coordinator(t, "prepare"))
	e = h.get(t, e.ID)
	if e.State != evaldomain.StateReady || e.Expected != 8 || e.MaxInFlight != capacity {
		t.Fatalf("batch fixture state=%s expected=%d capacity=%d", e.State, e.Expected, e.MaxInFlight)
	}
	return e
}

func (h *serviceHarness) finishRuns(t *testing.T) {
	t.Helper()
	rows, err := h.pool.Query(t.Context(), `SELECT run_id,state FROM workflow_runs WHERE state IN ('pending','running','cancelling') ORDER BY run_id`)
	if err != nil {
		t.Fatal(err)
	}
	states := map[string]runstore.WorkflowRunState{}
	for rows.Next() {
		var id string
		var state runstore.WorkflowRunState
		if err = rows.Scan(&id, &state); err != nil {
			t.Fatal(err)
		}
		states[id] = state
	}
	rows.Close()
	if rows.Err() != nil {
		t.Fatal(rows.Err())
	}
	for id, state := range states {
		if state == runstore.RunPending {
			if _, err := runstore.NewPostgresStore(h.pool, artifactpgtest.NewStorage()).TransitionRun(t.Context(), id, state, runstore.RunRunning, runstore.Reason{Code: "fixture_admitted"}); err != nil {
				t.Fatal(err)
			}
			state = runstore.RunRunning
		}
		target := runstore.RunSucceeded
		if state == runstore.RunCancelling {
			target = runstore.RunCancelled
		}
		if _, err = runstore.NewPostgresStore(h.pool, artifactpgtest.NewStorage()).TransitionRun(t.Context(), id, state, target, runstore.Reason{Code: "fixture_completed"}); err != nil {
			t.Fatal(err)
		}
	}
}
func (h *serviceHarness) finishAudits(t *testing.T) {
	t.Helper()
	store := auditstoretest.NewStore(h.pool, artifactpgtest.NewStorage()).WithPurgeDependencies(findingintake.DeleteAuditHoldsBatch)
	claims, err := store.Claim(t.Context(), auditstore.ClaimParams{HolderID: "fixture-audit-controller", Lease: time.Minute, Limit: 100})
	if err != nil {
		t.Fatal(err)
	}
	for _, claim := range claims {
		a, err := h.audit.Get(t.Context(), h.scope.OwnerID, claim.AuditID)
		if err != nil {
			t.Fatal(err)
		}
		if a.State == auditstore.AuditActive {
			a, err = store.TransitionClaimed(t.Context(), auditstore.ClaimedTransitionParams{Claim: claim, ExpectedRevision: a.Revision, ExpectedState: a.State, TargetState: auditstore.AuditFinalizing})
			if err == nil {
				_, err = store.TransitionClaimed(t.Context(), auditstore.ClaimedTransitionParams{Claim: claim, ExpectedRevision: a.Revision, ExpectedState: a.State, TargetState: auditstore.AuditFailed})
			}
			if err != nil {
				t.Fatal(err)
			}
		}
		if a.State == auditstore.AuditCancelling {
			_, err = store.TransitionClaimed(t.Context(), auditstore.ClaimedTransitionParams{Claim: claim, ExpectedRevision: a.Revision, ExpectedState: a.State, TargetState: auditstore.AuditCancelled})
			if err != nil {
				t.Fatal(err)
			}
		}
		if err = store.ReleaseClaim(t.Context(), claim); err != nil {
			t.Fatal(err)
		}
	}
}

func TestSharedPostgresNativeEightMembersRecoverLostResponsesAndRestart(t *testing.T) {
	for _, kind := range []string{"workflow", "audit"} {
		t.Run(kind, func(t *testing.T) {
			h := newHarness(t)
			e := h.prepared(t, kind)
			h.command(t, e, "start")
			a, b := h.coordinator(t, "process-a"), h.coordinator(t, "process-b")
			for n := 0; n < 80; n++ {
				errs := make(chan error, 2)
				var wg sync.WaitGroup
				for _, c := range []*evalcoordinator.Coordinator{a, b} {
					wg.Add(1)
					go func(c *evalcoordinator.Coordinator) { defer wg.Done(); _, err := c.RunOnce(t.Context()); errs <- err }(c)
				}
				wg.Wait()
				close(errs)
				for err := range errs {
					if err != nil && !errors.Is(err, errResponseLost) {
						t.Fatal(err)
					}
				}
				if kind == "workflow" {
					h.finishRuns(t)
				} else {
					h.finishAudits(t)
				}
				if n == 3 {
					a = h.coordinator(t, "restarted-a")
				}
				e = h.get(t, e.ID)
				if e.State == "finished" {
					break
				}
			}
			if e.State != "finished" || e.Outstanding != 0 {
				t.Fatalf("did not drain: %+v", e)
			}
			table := "workflow_runs"
			if kind == "audit" {
				table = "audits"
			}
			if n := count(t, h.pool, table); n != 8 {
				t.Fatalf("%s executions=%d, want exactly 8", kind, n)
			}
			if n := count(t, h.pool, "eval_submissions"); n != 8 {
				t.Fatal(n)
			}
			var terminals int
			if err := h.pool.QueryRow(t.Context(), `SELECT count(*) FROM eval_submissions WHERE state='terminal'`).Scan(&terminals); err != nil || terminals != 8 {
				t.Fatal(terminals, err)
			}
			if kind == "audit" {
				var projects int
				if err := h.pool.QueryRow(t.Context(), `SELECT count(DISTINCT a.project_id) FROM audits a JOIN projects p USING(project_id) WHERE p.kind='project'`).Scan(&projects); err != nil || projects != 8 {
					t.Fatalf("isolated Audit projects=%d %v", projects, err)
				}
				if count(t, h.pool, "workflow_runs") != 0 {
					t.Fatal("test unexpectedly executed an Audit worker")
				}
			}
		})
	}
}

func TestSharedPostgresNativeBatchAdmissionFillsCapacityAndCreatesByNextTick(t *testing.T) {
	for _, kind := range []string{"workflow", "audit"} {
		t.Run(kind, func(t *testing.T) {
			h := newHarness(t)
			e := h.preparedWithCapacity(t, kind, 8)
			store := evalstore.NewPostgresStore(h.pool, artifactpgtest.NewStorage())
			ordered, err := store.NextMembers(t.Context(), e.OwnerID, e.ID, 8)
			if err != nil || len(ordered) != 8 {
				t.Fatalf("eligible members=%d: %v", len(ordered), err)
			}
			h.command(t, e, "start")
			coordinator := h.coordinator(t, "batch-controller")
			tick(t, coordinator)
			e = h.get(t, e.ID)
			if e.Outstanding != 8 || count(t, h.pool, "eval_submissions") != 8 {
				t.Fatalf("one tick admitted %d of eight members", e.Outstanding)
			}
			rows, err := h.pool.Query(t.Context(), `SELECT member_id FROM eval_submissions WHERE experiment_id=$1 ORDER BY created_at, member_id`, e.ID)
			if err != nil {
				t.Fatal(err)
			}
			admitted := []string{}
			for rows.Next() {
				var memberID string
				if err = rows.Scan(&memberID); err != nil {
					t.Fatal(err)
				}
				admitted = append(admitted, memberID)
			}
			rows.Close()
			if err = rows.Err(); err != nil {
				t.Fatal(err)
			}
			for i, member := range ordered {
				if admitted[i] != member.MemberID {
					t.Fatalf("admission %d=%s, want ordinal member %s", i, admitted[i], member.MemberID)
				}
			}
			tick(t, coordinator)
			table := "workflow_runs"
			if kind == "audit" {
				table = "audits"
			}
			if got := count(t, h.pool, table); got != 8 {
				t.Fatalf("next tick created %d of eight %s executions", got, kind)
			}
		})
	}
}

func TestSharedPostgresNativeBatchAdmissionTwoCoordinatorsDoNotOverfill(t *testing.T) {
	h := newHarness(t)
	e := h.preparedWithCapacity(t, "workflow", 3)
	h.command(t, e, "start")
	a, b := h.coordinator(t, "batch-a"), h.coordinator(t, "batch-b")
	peak := 0
	for round := 0; round < 40 && e.State != evaldomain.StateFinished; round++ {
		errs := make(chan error, 2)
		var wg sync.WaitGroup
		for _, coordinator := range []*evalcoordinator.Coordinator{a, b} {
			wg.Add(1)
			go func(c *evalcoordinator.Coordinator) {
				defer wg.Done()
				_, err := c.RunOnce(t.Context())
				errs <- err
			}(coordinator)
		}
		wg.Wait()
		close(errs)
		for err := range errs {
			if err != nil && !errors.Is(err, errResponseLost) {
				t.Fatal(err)
			}
		}
		var active int
		if err := h.pool.QueryRow(t.Context(), `
SELECT count(*) FROM eval_submissions WHERE experiment_id = $1 AND state IN ('intent', 'accepted')`, e.ID).Scan(&active); err != nil {
			t.Fatal(err)
		}
		if e = h.get(t, e.ID); active > e.MaxInFlight || e.Outstanding != active {
			t.Fatalf("round %d: racing coordinators hold %d active members (outstanding %d) over capacity %d", round, active, e.Outstanding, e.MaxInFlight)
		}
		peak = max(peak, active)
		h.finishRuns(t)
	}
	if e.State != evaldomain.StateFinished || peak != e.MaxInFlight {
		t.Fatalf("racing coordinators did not fill capacity and drain: state=%s peak=%d", e.State, peak)
	}
	if count(t, h.pool, "eval_submissions") != 8 || count(t, h.pool, "workflow_runs") != 8 {
		t.Fatal("racing coordinators admitted or executed a member twice")
	}
}

func TestSharedPostgresPauseResumeKeepsClockAndDuplicateStartsFresh(t *testing.T) {
	h := newHarness(t)
	e := h.prepared(t, "workflow")
	h.command(t, e, "start")
	c := h.coordinator(t, "controller")
	tick(t, c)
	e = h.get(t, e.ID)
	if e.Outstanding != 1 {
		t.Fatal("member was not admitted")
	}
	start, deadline := *e.StartedAt, *e.DeadlineAt
	h.command(t, e, "pause")
	tick(t, c)
	e = h.get(t, e.ID)
	if e.State != "pausing" {
		t.Fatal("uncertain Run was reported drained")
	}
	tick(t, c)
	h.finishRuns(t)
	tick(t, c)
	e = h.get(t, e.ID)
	if e.State != "paused" || e.Outstanding != 0 {
		t.Fatal(e.State, e.Outstanding)
	}
	for i := 0; i < 3; i++ {
		tick(t, c)
	}
	if count(t, h.pool, "workflow_runs") != 1 {
		t.Fatal("pause dispatched more work")
	}
	h.command(t, e, "resume")
	tick(t, c)
	e = h.get(t, e.ID)
	if !e.StartedAt.Equal(start) || !e.DeadlineAt.Equal(deadline) {
		t.Fatal("resume reset allowance")
	}
	receipt := h.command(t, e, "duplicate")
	ref, err := receipt.Experiment()
	if err != nil {
		t.Fatal(err)
	}
	duplicate := h.get(t, ref.ExperimentID)
	if duplicate.State != "draft" || duplicate.ID == e.ID || duplicate.PortableID == e.PortableID || duplicate.StartedAt != nil || duplicate.Outstanding != 0 || duplicate.Expected != 0 {
		t.Fatal("duplicate reused execution state")
	}
	if _, err = evalstore.NewPostgresStore(h.pool, artifactpgtest.NewStorage()).FrozenPlan(t.Context(), e.OwnerID, duplicate.ID); !notFound(err) {
		t.Fatal("duplicate reused frozen plan", err)
	}
}

func TestPostgresExpiredPausedExperimentSettlesWhenClaimed(t *testing.T) {
	h := newHarness(t)
	ctx := t.Context()
	_, err := h.pool.Exec(ctx, `
INSERT INTO eval_experiments
  (experiment_id,owner_id,project_id,portable_id,control_mode,name,state,max_in_flight,wall_ms,started_at,deadline_at)
VALUES ('expired-pause',$1,$2,'expired-pause','external','expired pause','paused',1,1000,
        clock_timestamp()-interval '7 days',clock_timestamp()-interval '1 day')`,
		h.scope.OwnerID, h.scope.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := h.pool.Exec(ctx, `INSERT INTO eval_controller_claims(experiment_id) VALUES ('expired-pause')`); err != nil {
		t.Fatal(err)
	}
	store := evalstore.NewPostgresStore(h.pool, artifactpgtest.NewStorage())
	claims, err := store.Claim(ctx, "expired-pause-controller", time.Minute, 1)
	if err != nil || len(claims) != 1 || claims[0].ExperimentID != "expired-pause" {
		t.Fatalf("expired paused claim = %+v, %v", claims, err)
	}
	progressed, memberErr, err := h.service.tickExecution(ctx, claims[0])
	if err != nil || memberErr != nil || !progressed {
		t.Fatalf("expired paused tick = %v, %v, %v", progressed, memberErr, err)
	}
	if state := h.get(t, "expired-pause").State; state != evaldomain.StateSettling {
		t.Fatalf("expired pause state = %s, want settling", state)
	}
	if _, memberErr, err := h.service.tickExecution(ctx, claims[0]); err != nil || memberErr != nil {
		t.Fatal(memberErr, err)
	}
	if state := h.get(t, "expired-pause").State; state != evaldomain.StateFinished {
		t.Fatalf("expired pause state = %s, want finished", state)
	}
}

func TestSharedPostgresDeletedRunAfterResponseLossIsNeverRecreated(t *testing.T) {
	h := newHarness(t)
	e := h.prepared(t, "workflow")
	h.run.remove = true
	h.command(t, e, "start")
	c := h.coordinator(t, "controller")
	for n := 0; n < 40; n++ {
		tick(t, c)
		e = h.get(t, e.ID)
		if e.State == "finished" {
			break
		}
	}
	if e.State != "finished" || e.Outstanding != 0 {
		t.Fatalf("deleted Run drain: %s %d", e.State, e.Outstanding)
	}
	if count(t, h.pool, "workflow_runs") != 0 || count(t, h.pool, "eval_execution_tombstones") != 8 {
		t.Fatal("deleted Run was resurrected or lost")
	}
	h.run.mu.Lock()
	n := len(h.run.lost)
	h.run.mu.Unlock()
	if n != 8 {
		t.Fatal(n)
	}
}

func TestSharedPostgresPrepareFailureRetainsSchemaValidDiagnostic(t *testing.T) {
	h := newHarness(t)
	e := h.create(t, "workflow")
	// The pinned source becomes unavailable before preparation, without execution.
	if _, err := h.pool.Exec(t.Context(), `DELETE FROM artifact_bindings WHERE scope_kind='user'`); err != nil {
		t.Fatal(err)
	}
	h.command(t, e, "prepare")
	if _, err := h.coordinator(t, "controller").RunOnce(t.Context()); err == nil {
		t.Fatal("preparation failure did not reach coordinator")
	}
	e = h.get(t, e.ID)
	if e.State != "draft" || evaldomain.Validate("Diagnostic", e.Diagnostic) != nil {
		t.Fatalf("invalid preparation failure: %s %s", e.State, e.Diagnostic)
	}
	if count(t, h.pool, "workflow_runs") != 0 || count(t, h.pool, "audits") != 0 {
		t.Fatal("failed prepare executed work")
	}
}

func TestSharedPostgresExternalDispatchRequiresExplicitSubmissionAndFinalizesMissingMembers(t *testing.T) {
	h := newHarness(t)
	draft, data := h.dataset(t, "workflow")
	pre := map[string]Preflight{}
	for _, v := range draft.Variants {
		p, err := h.resolver.Resolve(t.Context(), h.scope.OwnerID, v, data.Cases)
		if err != nil {
			t.Fatal(err)
		}
		pre[v.ID] = p
	}
	bundle, err := BuildPlan("external-invocation", time.Now(), draft, data, pre)
	if err != nil {
		t.Fatal(err)
	}
	manifest, err := evaldomain.PublicPlanProjection(bundle.Plan)
	if err != nil {
		t.Fatal(err)
	}
	var setup struct {
		Checks []evaldomain.Check `json:"checks"`
	}
	if err = json.Unmarshal(bundle.Setup, &setup); err != nil {
		t.Fatal(err)
	}
	recipes := []evaldomain.MemberRecipe{}
	for _, m := range manifest.Members {
		recipes = append(recipes, evaldomain.MemberRecipe{MemberID: m.MemberID, Case: bundle.Cases[m.MemberID]})
	}
	doc := frozen(t, "CreateExperiment", evaldomain.CreateExperiment{Name: "External", ControlMode: "external", Registration: &evaldomain.ExternalRegistration{SchemaVersion: "contractor.eval-registration/v1", SourcePlanSHA256: bundle.Plan.Digest(), Manifest: manifest, Source: evaldomain.Source{System: "fixture", ID: "invocation"}, Variants: draft.Variants, Recipes: recipes, Checks: setup.Checks, Comparison: draft.Comparison, Budgets: draft.Budgets}})
	mutation := identity(t, "external-create", 0, doc)
	receipt, err := h.service.Create(t.Context(), h.scope, doc, mutation)
	if err != nil {
		t.Fatal(err)
	}
	ref, err := receipt.Experiment()
	if err != nil {
		t.Fatal(err)
	}
	c := h.coordinator(t, "native-controller")
	for n := 0; n < 3; n++ {
		tick(t, c)
	}
	e := h.get(t, ref.ExperimentID)
	if e.State != "ready" || e.StartedAt != nil || count(t, h.pool, "eval_submissions") != 0 {
		t.Fatal("native dispatch took over external mode")
	}
	submission := evaldomain.Submission{PlanSHA256: bundle.Plan.Digest()}
	body := frozen(t, "Submission", submission)
	submitMutation := identity(t, "submit", 0, body)
	if _, err = h.service.Submit(t.Context(), h.scope, e.ID, manifest.Members[0].MemberID, submission, submitMutation); err != nil {
		t.Fatal(err)
	}
	tick(t, c)
	tick(t, c)
	h.finishRuns(t)
	tick(t, c)
	e = h.get(t, e.ID)
	if e.State != "running" || e.Outstanding != 0 || count(t, h.pool, "eval_submissions") != 1 {
		t.Fatal("external native admission or inferred completion")
	}
	started, deadline := *e.StartedAt, *e.DeadlineAt
	h.command(t, e, "finalize")
	tick(t, c)
	e = h.get(t, e.ID)
	if e.State != "finished" || e.Expected != 8 {
		t.Fatal("finalization lost missing members")
	}
	if _, err = h.service.Submit(t.Context(), h.scope, e.ID, manifest.Members[0].MemberID, submission, submitMutation); err != nil {
		t.Fatal("exact replay after finalize", err)
	}
	if _, err = h.service.Submit(t.Context(), h.scope, e.ID, manifest.Members[1].MemberID, submission, identity(t, "late", 0, body)); err == nil {
		t.Fatal("late member passed finalize fence")
	}
	e = h.get(t, e.ID)
	if !e.StartedAt.Equal(started) || !e.DeadlineAt.Equal(deadline) {
		t.Fatal("external replay reset clock")
	}
	// An already accepted registration replays before mutable preflight.
	h.resolver.Catalog = nil
	replay, err := h.service.Create(t.Context(), h.scope, doc, mutation)
	if err != nil || !replay.Replayed || string(replay.Response) != string(receipt.Response) {
		t.Fatal("registration replay resolved mutable dependencies", err)
	}
}

func TestSharedPostgresDeadlineAndObservedTokenLimitsDrainWithoutNewAdmissions(t *testing.T) {
	for _, budget := range []string{"deadline", "tokens"} {
		t.Run(budget, func(t *testing.T) {
			h := newHarness(t)
			e := h.create(t, "workflow")
			if budget == "tokens" {
				var draft evaldomain.Draft
				if err := json.Unmarshal(e.Draft.Bytes(), &draft); err != nil {
					t.Fatal(err)
				}
				limit := int64(100)
				draft.Budgets.MaxObservedTotalTokens = &limit
				doc := frozen(t, "DraftUpdate", evaldomain.DraftUpdate{Name: e.Name, Draft: draft})
				if err := h.service.tx(t.Context(), func(s *evalstore.Store) error {
					_, err := s.UpdateDraft(t.Context(), h.scope, e.ID, doc, identity(t, "budget", e.Revision, doc))
					return err
				}); err != nil {
					t.Fatal(err)
				}
				e = h.get(t, e.ID)
			}
			h.command(t, e, "prepare")
			tick(t, h.coordinator(t, "prepare"))
			e = h.get(t, e.ID)
			h.command(t, e, "start")
			c := h.coordinator(t, "controller")
			tick(t, c)
			tick(t, c)
			tick(t, c)
			e = h.get(t, e.ID)
			original := *e.StartedAt
			if budget == "deadline" {
				h.service.now = func() time.Time { return e.DeadlineAt.Add(time.Second) }
			} else {
				store := evalstore.NewPostgresStore(h.pool, artifactpgtest.NewStorage())
				claims, err := store.Claim(t.Context(), "observe", time.Minute, 1)
				if err != nil || len(claims) != 1 {
					t.Fatal(err)
				}
				members, err := store.ReconciliationCandidates(t.Context(), e.OwnerID, e.ID, membersPerTick)
				if err != nil || len(members) != 1 {
					t.Fatal(members, err)
				}
				for _, observed := range []int64{100, 50, 100} {
					if err = h.service.tx(t.Context(), func(s *evalstore.Store) error {
						return s.RecoverMember(h.scope, e.ID, members[0], claims[0]).ObserveTokens(t.Context(), observed)
					}); err != nil {
						t.Fatal(err)
					}
				}
				if err = store.ReleaseClaim(t.Context(), claims[0]); err != nil {
					t.Fatal(err)
				}
				if h.get(t, e.ID).ObservedTokens != 100 {
					t.Fatal("usage was double counted or reset")
				}
			}
			tick(t, c)
			e = h.get(t, e.ID)
			if e.State != "cancelling" {
				t.Fatal("allowance did not stop admission", e.State)
			}
			tick(t, c)
			h.finishRuns(t)
			tick(t, c)
			e = h.get(t, e.ID)
			if e.State != "cancelled" || e.Outstanding != 0 || count(t, h.pool, "eval_submissions") != 1 || !e.StartedAt.Equal(original) {
				t.Fatal("budget cancellation reset or dispatched", e.State)
			}
		})
	}
}

func TestSharedPostgresRecoveryAfterTransitionBeforeCommandAcknowledgement(t *testing.T) {
	h := newHarness(t)
	e := h.create(t, "workflow")
	h.command(t, e, "prepare")
	store := evalstore.NewPostgresStore(h.pool, artifactpgtest.NewStorage())
	claims, err := store.Claim(t.Context(), "crashed", time.Minute, 1)
	if err != nil || len(claims) != 1 {
		t.Fatal(claims, err)
	}
	e = h.get(t, e.ID)
	if err = h.service.prepare(t.Context(), e, claims[0]); err != nil {
		t.Fatal(err)
	}
	// Simulate process death after its commit and lease expiry without
	// acknowledgement. A held claim's lease ends after it starts, so both move.
	if _, err = h.pool.Exec(t.Context(), `UPDATE eval_controller_claims SET claimed_at=clock_timestamp()-interval '2 seconds', expires_at=clock_timestamp()-interval '1 second' WHERE experiment_id=$1`, e.ID); err != nil {
		t.Fatal(err)
	}
	tick(t, h.coordinator(t, "restarted"))
	pending, err := store.PendingCommands(t.Context(), e.OwnerID, e.ID)
	if err != nil || len(pending) != 0 {
		t.Fatal("ready command acknowledgement stranded", pending, err)
	}
	if _, err = h.service.Tick(t.Context(), claims[0]); !errors.Is(err, evalstore.ErrClaimLost) {
		t.Fatal("stale claim recovered work", err)
	}
}

func TestSharedPostgresCancelAfterAuditCreateBeforeStartDrainsThroughOrdinaryDeletion(t *testing.T) {
	h := newHarness(t)
	e := h.prepared(t, "audit")
	h.command(t, e, "start")
	c := h.coordinator(t, "controller")
	tick(t, c)
	tick(t, c)
	if count(t, h.pool, "audits") != 1 {
		t.Fatal("Audit create did not commit")
	}
	e = h.get(t, e.ID)
	h.command(t, e, "cancel")
	tick(t, c)
	store := auditstoretest.NewStore(h.pool, artifactpgtest.NewStorage()).WithPurgeDependencies(findingintake.DeleteAuditHoldsBatch)
	claims, err := store.Claim(t.Context(), auditstore.ClaimParams{HolderID: "audit-controller", Lease: time.Minute, Limit: 10})
	if err != nil || len(claims) != 1 {
		t.Fatal(claims, err)
	}
	if err = store.PurgeClaimed(t.Context(), claims[0], auditdomain.ArtifactNamespace(claims[0].AuditID)); err != nil {
		t.Fatal(err)
	}
	tick(t, c)
	e = h.get(t, e.ID)
	if e.State != "cancelled" || e.Outstanding != 0 || count(t, h.pool, "audits") != 0 || count(t, h.pool, "eval_execution_tombstones") != 1 {
		t.Fatal("unstarted Audit was recreated or failed to drain", e.State)
	}
	var never bool
	if err = h.pool.QueryRow(t.Context(), `SELECT never_started FROM eval_execution_tombstones`).Scan(&never); err != nil || !never {
		t.Fatal("unstarted Audit counted as execution", err)
	}
}

func TestSharedPostgresPreparationArtifactLossRejectsBeforeExecution(t *testing.T) {
	h := newHarness(t)
	e := h.prepared(t, "workflow")
	if _, err := h.pool.Exec(t.Context(), `DELETE FROM artifact_bindings WHERE scope_kind='user'`); err != nil {
		t.Fatal(err)
	}
	h.command(t, e, "start")
	c := h.coordinator(t, "controller")
	for n := 0; n < 20; n++ {
		tick(t, c)
		e = h.get(t, e.ID)
		if e.State == "finished" {
			break
		}
	}
	if e.State != "finished" || e.Outstanding != 0 || count(t, h.pool, "workflow_runs") != 0 {
		t.Fatal("local input failure became uncertain or executed work", e.State)
	}
}

func TestSharedPostgresProjectDeletionWaitsForUncertainRunAndFencesRemainingMembers(t *testing.T) {
	h := newHarness(t)
	e := h.prepared(t, "workflow")
	h.command(t, e, "start")
	c := h.coordinator(t, "controller")
	tick(t, c)
	tick(t, c)
	project, err := projectstore.NewPostgresStore(h.pool).Get(t.Context(), e.OwnerID, e.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err = projectstore.NewPostgresStore(h.pool).BeginDeletion(t.Context(), projectstore.BeginDeletionParams{OwnerID: e.OwnerID, ProjectID: e.ProjectID, ExpectedRevision: project.Revision}); err != nil {
		t.Fatal(err)
	}
	store := evalstore.NewPostgresStore(h.pool, artifactpgtest.NewStorage())
	blocked, err := store.ProjectBlocked(t.Context(), e.OwnerID, e.ProjectID)
	if err != nil || !blocked {
		t.Fatal("uncertain creation did not block purge", err)
	}
	claims, err := store.Claim(t.Context(), "drain-check", time.Minute, 1)
	if err != nil || len(claims) != 1 || claims[0].ExperimentID != e.ID {
		t.Fatalf("claim the deleting experiment = %+v, %v", claims, err)
	}
	if err = h.service.tx(t.Context(), func(s *evalstore.Store) error {
		_, err := s.Purge(t.Context(), h.scope, e.ID, claims[0])
		return err
	}); !errors.Is(err, evalstore.ErrDrain) {
		t.Fatal("uncertain work purged", err)
	}
	if err = store.ReleaseClaim(t.Context(), claims[0]); err != nil {
		t.Fatal(err)
	}
	tick(t, c)
	h.finishRuns(t)
	tick(t, c)
	if count(t, h.pool, "workflow_runs") != 1 {
		t.Fatal("deletion admitted another member")
	}
	if _, err = store.Get(t.Context(), e.OwnerID, e.ID); !notFound(err) {
		t.Fatal("drained eval was not purged", err)
	}
	if blocked, err = store.ProjectBlocked(t.Context(), e.OwnerID, e.ProjectID); err != nil || blocked {
		t.Fatal("drained experiment blocked Project purge", err)
	}
}

type changedRunPins struct {
	RunCreator
	field string
}

func (s changedRunPins) CreatePublic(ctx context.Context, p runservice.PublicCreateParams) (runservice.CreateResult, error) {
	digest := evaldomain.Digest([]byte("changed configuration"))
	switch s.field {
	case "workflow":
		p.ExpectedWorkflowSHA256 = digest
	case "runtime":
		p.ExpectedRuntimeSHA256 = digest
	case "skills":
		p.ExpectedSkillsSHA256 = digest
	}
	return s.RunCreator.CreatePublic(ctx, p)
}

type changedAuditPins struct {
	AuditExecutor
	field string
}

func (s changedAuditPins) CreateDraft(ctx context.Context, p auditservice.CreateDraftParams) (auditstore.Audit, bool, error) {
	if s.field == "profile" {
		p.ExpectedProfileSHA256 = evaldomain.Digest([]byte("changed profile"))
	}
	return s.AuditExecutor.CreateDraft(ctx, p)
}
func (s changedAuditPins) Start(ctx context.Context, p auditservice.StartParams) (auditservice.StartedAudit, error) {
	digest := evaldomain.Digest([]byte("changed configuration"))
	switch s.field {
	case "runtime":
		p.ExpectedRuntimeSHA256 = digest
	case "skills":
		p.ExpectedSkillsSHA256 = digest
	case "standards":
		p.ExpectedStandardsSHA256 = digest
	}
	return s.AuditExecutor.Start(ctx, p)
}
func TestSharedPostgresPinnedSelectionsRejectBeforeOrdinaryEffects(t *testing.T) {
	for _, kind := range []string{"workflow", "audit"} {
		fields := []string{"workflow", "runtime", "skills"}
		if kind == "audit" {
			fields = []string{"profile", "runtime", "skills", "standards"}
		}
		for _, field := range fields {
			t.Run(kind+"/"+field, func(t *testing.T) {
				h := newHarness(t)
				e := h.prepared(t, kind)
				if kind == "workflow" {
					h.driver.Runs = changedRunPins{h.run, field}
				} else {
					h.driver.Audits = changedAuditPins{h.audit, field}
				}
				h.command(t, e, "start")
				c := h.coordinator(t, "controller")
				for n := 0; n < 3; n++ {
					tick(t, c)
				}
				if count(t, h.pool, "workflow_runs") != 0 || count(t, h.pool, "audit_tasks") != 0 {
					t.Fatal("pin guard admitted ordinary execution effects")
				}
				var rejected int
				if err := h.pool.QueryRow(t.Context(), `SELECT count(*) FROM eval_suboperations WHERE state='rejected' AND kind IN ('run-create','audit-create','audit-start')`).Scan(&rejected); err != nil || rejected == 0 {
					t.Fatal("changed pin not recorded as rejected", err)
				}
				if field == "profile" && count(t, h.pool, "audits") != 0 {
					t.Fatal("changed profile created Audit")
				}
			})
		}
	}
}

func (h *serviceHarness) preparedWithRuntimeLabel(t *testing.T, kind string) evalstore.Experiment {
	t.Helper()
	repository := runtimeconfig.NewRepository(h.pool)
	if _, err := repository.CreateBinding(t.Context(), "gpu",
		runtimeconfigtest.BuiltInRunSnapshot().Default.Config, "operator", time.Now()); err != nil {
		t.Fatal(err)
	}
	draft, _ := h.dataset(t, kind)
	for i := range draft.Variants {
		draft.Variants[i].RuntimeLabels = []string{"gpu"}
	}
	doc := frozen(t, "CreateExperiment", evaldomain.CreateExperiment{
		Name: "Members with removable Runtime label", ControlMode: "server", Draft: &draft,
	})
	receipt, err := h.service.Create(t.Context(), h.scope, doc, identity(t, "create", 0, doc))
	if err != nil {
		t.Fatal(err)
	}
	ref, err := receipt.Experiment()
	if err != nil {
		t.Fatal(err)
	}
	e := h.get(t, ref.ExperimentID)
	h.command(t, e, "prepare")
	tick(t, h.coordinator(t, "prepare"))
	e = h.get(t, e.ID)
	if e.State != evaldomain.StateReady || e.Expected != 8 {
		t.Fatalf("labeled preparation = %s members=%d diagnostic=%s", e.State, e.Expected, e.Diagnostic)
	}
	return e
}

func (h *serviceHarness) purgeDeletingAuditDrafts(t *testing.T) {
	t.Helper()
	store := auditstoretest.NewStore(h.pool, artifactpgtest.NewStorage()).WithPurgeDependencies(findingintake.DeleteAuditHoldsBatch)
	claims, err := store.Claim(t.Context(), auditstore.ClaimParams{
		HolderID: "fixture-draft-purge", Lease: time.Minute, Limit: 100,
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, claim := range claims {
		audit, err := h.audit.Get(t.Context(), h.scope.OwnerID, claim.AuditID)
		if err != nil {
			t.Fatal(err)
		}
		if audit.State == auditstore.AuditDeleting {
			if err := store.PurgeClaimed(t.Context(), claim, auditdomain.ArtifactNamespace(claim.AuditID)); err != nil {
				t.Fatal(err)
			}
		} else if err := store.ReleaseClaim(t.Context(), claim); err != nil {
			t.Fatal(err)
		}
	}
}

func TestSharedPostgresDeletedRuntimeLabelRejectsEvalCreateAndDrains(t *testing.T) {
	for _, kind := range []string{"workflow", "audit"} {
		for _, cancel := range []bool{false, true} {
			name := kind + "/finish"
			if cancel {
				name = kind + "/cancel"
			}
			t.Run(name, func(t *testing.T) {
				h := newHarness(t)
				e := h.preparedWithRuntimeLabel(t, kind)
				h.command(t, e, "start")
				coordinator := h.coordinator(t, "controller")
				tick(t, coordinator) // Admit one member while its label is still bound.
				e = h.get(t, e.ID)
				if e.State != evaldomain.StateRunning || e.Outstanding != 1 {
					t.Fatalf("initial admission = %s outstanding=%d", e.State, e.Outstanding)
				}
				if err := runtimeconfig.NewRepository(h.pool).DeleteBinding(t.Context(), "gpu", 1); err != nil {
					t.Fatal(err)
				}
				opKind := "run-create"
				if kind == "audit" {
					opKind = "audit-start"
				}
				if cancel {
					for n := 0; n < 10; n++ {
						tick(t, coordinator)
						if kind == "audit" {
							h.purgeDeletingAuditDrafts(t)
						}
						var rejected int
						if err := h.pool.QueryRow(t.Context(), `
SELECT count(*) FROM eval_suboperations WHERE kind = $1 AND state = 'rejected'`, opKind).Scan(&rejected); err != nil {
							t.Fatal(err)
						}
						if rejected > 0 {
							break
						}
					}
					e = h.get(t, e.ID)
					h.command(t, e, "cancel")
				}
				want := evaldomain.StateFinished
				if cancel {
					want = evaldomain.StateCancelled
				}
				for n := 0; n < 80; n++ {
					tick(t, coordinator)
					if kind == "audit" {
						h.purgeDeletingAuditDrafts(t)
					}
					e = h.get(t, e.ID)
					if e.State == want {
						break
					}
				}
				if e.State != want || e.Outstanding != 0 {
					t.Fatalf("deleted-label experiment did not drain: state=%s outstanding=%d", e.State, e.Outstanding)
				}
				var rejected int
				if err := h.pool.QueryRow(t.Context(), `
SELECT count(*) FROM eval_suboperations
 WHERE kind = $1 AND state = 'rejected'
   AND convert_from(response, 'UTF8') LIKE '%eval_not_ready%'`, opKind).Scan(&rejected); err != nil || rejected == 0 {
					t.Fatalf("missing definite %s rejection: count=%d err=%v", opKind, rejected, err)
				}
				if !cancel && rejected != 8 {
					t.Fatalf("later members were not all rejected: %d of 8", rejected)
				}
				if kind == "workflow" && count(t, h.pool, "workflow_runs") != 0 {
					t.Fatal("deleted binding still created a Workflow Run")
				}
				if kind == "audit" && count(t, h.pool, "audits") != 0 {
					t.Fatal("rejected Audit draft was not deleted")
				}
				if !cancel && count(t, h.pool, "eval_submissions") != 8 {
					t.Fatal("later members were not admitted and rejected")
				}
			})
		}
	}
}

type failedRunCreate struct {
	RunCreator
	cause error
}

func (f failedRunCreate) CreatePublic(context.Context, runservice.PublicCreateParams) (runservice.CreateResult, error) {
	return runservice.CreateResult{}, f.cause
}

func TestSharedPostgresUnknownEvalCreateErrorKeepsIntent(t *testing.T) {
	for _, test := range []struct {
		name  string
		cause error
	}{
		{name: "deadline", cause: context.DeadlineExceeded},
		{name: "database", cause: errors.New("untyped database failure")},
	} {
		t.Run(test.name, func(t *testing.T) {
			h := newHarness(t)
			e := h.prepared(t, "workflow")
			h.command(t, e, "start")
			coordinator := h.coordinator(t, "controller")
			tick(t, coordinator)
			h.driver.Runs = failedRunCreate{RunCreator: h.run, cause: test.cause}
			_, err := coordinator.RunOnce(t.Context())
			if !errors.Is(err, test.cause) {
				t.Fatalf("unknown creation error = %v", err)
			}
			var intents int
			if err := h.pool.QueryRow(t.Context(), `
SELECT count(*) FROM eval_suboperations WHERE kind = 'run-create' AND state = 'intent'`).Scan(&intents); err != nil || intents != 1 {
				t.Fatalf("uncertain creation intent = %d, %v", intents, err)
			}
		})
	}
}

func serviceTestPool(t *testing.T) *pgxpool.Pool {
	t.Helper()
	ctx := context.Background()
	pool := postgrestest.NewPool(t, ctx, "", postgrestest.Options{
		Configure: func(poolConfig *pgxpool.Config) {
			poolConfig.ConnConfig.RuntimeParams["statement_timeout"] = "10000"
		},
	})
	if _, err := pg.ApplyMigrations(ctx, pool, pg.MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	return pool
}
func serviceTestCatalog(t *testing.T) *config.Snapshot {
	t.Helper()
	root := t.TempDir()
	files := map[string]string{
		"instructions/planner.md": "Inspect the selected exact material.",
		"instructions/worker.md":  "Read the selected material and write an ordinary result.",
		"llm-gateways/test.yaml": `apiVersion: contractor/v1alpha1
kind: LLMGatewayConfig
metadata: {name: test-gateway, version: "1"}
spec:
  protocol: openai-compatible@1
  url: http://127.0.0.1:4000/v1
`,
		"model-policies/worker.yaml": `apiVersion: contractor/v1alpha1
kind: ModelPolicy
metadata: {name: worker, version: "1"}
spec:
  model: worker-model
  maxOutputTokens: 1024
  maxModelCalls: 2
  maxToolCalls: 4
  maxTotalTokens: 4096
  temperature: 0
`,
		"agent-templates/worker.yaml": `apiVersion: contractor/v1alpha1
kind: AgentTemplate
metadata: {name: audit-worker, version: "1"}
spec:
  description: Produces one deterministic test result
  runtime: adk@1
  instructions: {ref: instructions/worker.md}
  toolsets:
    - ref: run-artifacts@1
      tools: [read_artifact, write_artifact]
  sandboxProfile: local-workdir@1
`,
		"workflows/check.yaml": `apiVersion: contractor/v1alpha1
kind: Workflow
metadata: {name: audit-check, version: "1"}
spec:
  parameters: {}
  inputs:
    task: {required: true, mediaTypes: [application/zip, application/json]}
  outputs:
    result: {required: true, mediaTypes: [application/zip]}
  executionConfig:
    workers:
      modelPolicy: worker@1
      llmGateway: test-gateway@1
      credential: development-worker
  entryStage: check
  stages:
    check:
      objective: Inspect one exact material
      instructions: {ref: instructions/planner.md}
      planner: passthrough@1
      agents:
        worker: {template: audit-worker@1}
      context:
        artifacts:
          task: {namespace: inputs, name: task, required: true}
      result:
        artifacts:
          result: {required: true, mediaTypes: [application/zip], from: {namespace: worker, name: result}}
      workflowOutputs: {result: result}
      on:
        succeeded: {succeed: {}}
        failed: {fail: {}}
        interrupted: {fail: {}}
`,
		"audit-profiles/checklist.yaml": `apiVersion: contractor/v1alpha1
kind: AuditProfile
metadata: {name: test-checklist, version: "1"}
spec:
  schema: contractor.audit.program.v1
  inputs:
    checklist: {required: true, media_types: [application/json]}
  workflows:
    check:
      ref: audit-check@1
      inputs:
        task: {source: audit-input, name: checklist}
      parameters: {}
      outputs: {material: result}
  hypotheses: {}
  initial_work:
    - key: inspect-selected-material
      workflow: check
      objective: Inspect the exact selected Eval material
  limits:
    max_tasks: 10
    max_iterations: 1
    max_tasks_per_iteration: 10
    batch_size: 1
    max_submitted_runs: 20
    max_attempts: 2
    deadline_seconds: 3600
    max_evidence_bytes: 1048576
    max_hypotheses: 1
    max_hypothesis_revisions: 1
    max_hypothesis_bytes: 32768
  policy:
    active_operations: prohibited
`,
	}
	for _, variant := range []string{"a", "b"} {
		policy := strings.Replace(files["model-policies/worker.yaml"], "name: worker", "name: variant-worker-"+variant, 1)
		files["model-policies/variant-worker-"+variant+".yaml"] = strings.Replace(policy, "worker-model", "variant-model-"+variant, 1)
		workflow := strings.Replace(files["workflows/check.yaml"], "name: audit-check", "name: variant-check-"+variant, 1)
		files["workflows/variant-check-"+variant+".yaml"] = strings.Replace(workflow, "modelPolicy: worker@1", "modelPolicy: variant-worker-"+variant+"@1", 1)
	}
	for _, name := range []string{"agent-templates/nuclei.yaml", "workflows/nuclei.yaml"} {
		data, err := os.ReadFile(filepath.Join("testdata", "source-scope", name))
		if err != nil {
			t.Fatal(err)
		}
		files[name] = string(data)
	}
	for _, name := range []string{"audit-profiles/web_scan.yaml", "workflows/audit_web_nuclei_scan.yaml", "agent-templates/audit_web_nuclei_scan.yaml"} {
		data, err := os.ReadFile(filepath.Join("..", "config", "testdata", "audit-scan-catalog", name))
		if err != nil {
			t.Fatal(err)
		}
		files[name] = string(data)
	}
	files["instructions/audit-openapi-scan.md"] = "Execute the exact scan assignment."
	for _, directory := range []string{
		"instructions", "llm-gateways", "model-policies",
		"agent-templates", "workflows", "audit-profiles", "skills",
	} {
		if err := os.MkdirAll(filepath.Join(root, directory), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	for name, contents := range files {
		if err := os.WriteFile(filepath.Join(root, name), []byte(contents), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	snapshot, _, err := configload.LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "managed"))
	if err != nil {
		t.Fatalf("load Audit service config: %v", err)
	}
	return snapshot
}

func TestSharedPostgresEvalComparesAuthoredWorkflowPoliciesAndSubmitsFrozenBindings(t *testing.T) {
	h := newHarness(t)
	draft, _ := h.dataset(t, "workflow")
	draft.Variants[0].Selector = "variant-check-a@1"
	draft.Variants[1].Selector = "variant-check-b@1"
	draft.Comparison.AllowedDifferences = append(draft.Comparison.AllowedDifferences, "models", "sampling")
	doc := frozen(t, "CreateExperiment", evaldomain.CreateExperiment{Name: "Authored model comparison", ControlMode: "server", Draft: &draft})
	receipt, err := h.service.Create(t.Context(), h.scope, doc, identity(t, "create", 0, doc))
	if err != nil {
		t.Fatal(err)
	}
	ref, err := receipt.Experiment()
	if err != nil {
		t.Fatal(err)
	}
	e := h.get(t, ref.ExperimentID)
	h.command(t, e, "prepare")
	tick(t, h.coordinator(t, "prepare"))
	e = h.get(t, e.ID)
	if e.State != "ready" {
		t.Fatalf("prepare: %s %s", e.State, e.Diagnostic)
	}
	// The ordinary dispatcher already refuses mutable resolution; make the
	// preflight resolver unavailable too after preparation to expose accidental reuse.
	catalog := h.resolver.Catalog.(*config.Snapshot)
	h.resolver.Catalog = retainedExecutionCatalog{catalog}
	h.command(t, e, "start")
	c := h.coordinator(t, "controller")
	for n := 0; n < 80 && count(t, h.pool, "workflow_runs") < 8; n++ {
		tick(t, c)
		h.finishRuns(t)
	}
	rows, err := h.pool.Query(t.Context(), `SELECT workflow_name,workflow_snapshot FROM workflow_runs ORDER BY run_id`)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	counts := map[string]int{}
	for rows.Next() {
		var name string
		var body []byte
		if err := rows.Scan(&name, &body); err != nil {
			t.Fatal(err)
		}
		var workflow resolved.ResolvedWorkflow
		if err := json.Unmarshal(body, &workflow); err != nil {
			t.Fatal(err)
		}
		variant := strings.TrimPrefix(name, "variant-check-")
		selection := workflow.Stages["check"].ExecutionConfig.Agents["worker"]
		if selection.ModelPolicy.Model != "variant-model-"+variant || selection.Origins.ModelPolicy != "workflow.executionConfig.workers" {
			t.Fatalf("submitted policy %+v", selection)
		}
		counts[variant]++
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	if counts["a"] != 4 || counts["b"] != 4 {
		t.Fatalf("submitted variants: %v", counts)
	}
}
