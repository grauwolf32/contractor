package evalservice

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/evalcoordinator"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
	pg "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runservice"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type testCredentials struct{ gateway contracts.LLMGatewayConfigRef }

func (c testCredentials) LookupLLMCredential(_ context.Context, id string) (config.CredentialMetadata, error) {
	if id != "development-worker" {
		return config.CredentialMetadata{}, errors.New("unexpected fixture credential")
	}
	return config.CredentialMetadata{Ref: contracts.LLMCredentialRef{CredentialID: id}, LLMGateway: c.gateway, Unrestricted: true}, nil
}

type testBarrier struct{}

func (testBarrier) WithRunCreation(ctx context.Context, fn func() error) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	return fn()
}
func (testBarrier) ValidateRuntimeCredential(context.Context, string, ...string) error { return nil }

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
			store := runstore.NewPostgresStore(r.pool)
			if _, err = store.TransitionRun(ctx, result.Run.RunID, runstore.RunRunning, runstore.RunSucceeded, runstore.Reason{Code: "fixture"}); err != nil {
				return result, err
			}
			if err = store.DeleteReleasedTerminalRun(ctx, p.OwnerID, result.Run.RunID); err != nil {
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

func newHarness(t *testing.T) *serviceHarness {
	t.Helper()
	pool := serviceTestPool(t)
	catalog := serviceTestCatalog(t)
	gateway, err := catalog.LLMGateway("test-gateway@1")
	if err != nil {
		t.Fatal(err)
	}
	credentialLookup := testCredentials{gateway: gateway.Ref}
	credentials := runtimeconfig.TransactionLLMCredentialLookupFactoryFunc(func(pgx.Tx) (config.CredentialLookup, error) { return credentialLookup, nil })
	audits, err := auditservice.New(auditservice.Options{Pool: pool, Profiles: catalog, TransactionLLMCredentials: credentials, CredentialGuard: testBarrier{}})
	if err != nil {
		t.Fatal(err)
	}
	runs, err := runservice.New(runservice.Options{Runs: runstore.NewPostgresStore(pool), Workflows: catalog, LLMCredentials: credentialLookup, CredentialGuard: testBarrier{}, RuntimeCredentials: testBarrier{}, Projects: projectstore.NewPostgresStore(pool), SkillInitializationAvailable: true, PublicTransaction: func(ctx context.Context, fn func(runservice.PublicRunWriter, *artifacts.Service) error) error {
		return pg.InTx(ctx, pool, pgx.TxOptions{IsoLevel: pgx.RepeatableRead}, func(tx pgx.Tx) error {
			lookup, err := runtimeconfig.BindTransactionLLMCredentialLookup(tx, credentials)
			if err != nil {
				return err
			}
			return fn(runstore.NewRunCreationPostgresStore(tx, lookup), artifacts.NewService(artifacts.NewPostgresRepository(tx)))
		})
	}})
	if err != nil {
		t.Fatal(err)
	}
	run := &lostRunResponse{RunCreator: runs, lost: map[string]bool{}, pool: pool}
	audit := &lostAuditResponse{AuditExecutor: audits, lost: map[string]bool{}}
	resolver := &Resolver{Pool: pool, Catalog: catalog, Credentials: credentials, Barrier: testBarrier{}}
	driver := &Driver{Pool: pool, Runs: run, Audits: audit}
	service, err := New(Options{Pool: pool, Resolver: resolver, Driver: driver})
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
	e, err := evalstore.NewPostgresStore(h.pool).Get(t.Context(), h.scope.OwnerID, id)
	if err != nil {
		t.Fatal(err)
	}
	return e
}
func (h *serviceHarness) dataset(t *testing.T, kind string) (evaldomain.Draft, evaldomain.DatasetInput) {
	t.Helper()
	draft, data, _ := planInputs(t)
	source, _ := artifacts.NewService(artifacts.NewPostgresRepository(h.pool)).User(h.scope.OwnerID)
	for i := range data.Cases {
		c := &data.Cases[i]
		slot, media, payload := "task", "application/zip", []byte("fixture ZIP is not executed")
		if kind == "audit" {
			slot, media, payload = "checklist", "application/json", []byte(`{"schema":"contractor.audit.checklist.v1","items":[{"key":"item","version":"1","statement":"Check fixture.","applicability":"always","allowed_methods":["static"],"required_evidence":[],"review_policy":"automatic"}]}`)
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
		p, err := evalstore.NewPostgresStore(h.pool).FrozenPlan(t.Context(), e.OwnerID, e.ID)
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
	c, err := evalcoordinator.New(evalstore.NewPostgresStore(h.pool), h.service, evalcoordinator.Options{HolderID: holder})
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

func (h *serviceHarness) finishRuns(t *testing.T) {
	t.Helper()
	rows, err := h.pool.Query(t.Context(), `SELECT run_id,state FROM workflow_runs WHERE state IN ('running','cancelling') ORDER BY run_id`)
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
		target := runstore.RunSucceeded
		if state == runstore.RunCancelling {
			target = runstore.RunCancelled
		}
		if _, err = runstore.NewPostgresStore(h.pool).TransitionRun(t.Context(), id, state, target, runstore.Reason{Code: "fixture_completed"}); err != nil {
			t.Fatal(err)
		}
	}
}
func (h *serviceHarness) finishAudits(t *testing.T) {
	t.Helper()
	store := auditstore.NewPostgresStore(h.pool)
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

func TestPostgresNativeEightMembersRecoverLostResponsesAndRestart(t *testing.T) {
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

func TestPostgresPauseResumeKeepsClockAndDuplicateStartsFresh(t *testing.T) {
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
	if _, err = evalstore.NewPostgresStore(h.pool).FrozenPlan(t.Context(), e.OwnerID, duplicate.ID); !notFound(err) {
		t.Fatal("duplicate reused frozen plan", err)
	}
}

func TestPostgresDeletedRunAfterResponseLossIsNeverRecreated(t *testing.T) {
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

func TestPostgresPrepareFailureRetainsSchemaValidDiagnostic(t *testing.T) {
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

func TestPostgresExternalDispatchRequiresExplicitSubmissionAndFinalizesMissingMembers(t *testing.T) {
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

func TestPostgresDeadlineAndObservedTokenLimitsDrainWithoutNewAdmissions(t *testing.T) {
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
				store := evalstore.NewPostgresStore(h.pool)
				claims, err := store.Claim(t.Context(), "observe", time.Minute, 1)
				if err != nil || len(claims) != 1 {
					t.Fatal(err)
				}
				members, err := store.Outstanding(t.Context(), e.OwnerID, e.ID, 100)
				if err != nil || len(members) != 1 {
					t.Fatal(members, err)
				}
				for _, observed := range []int64{100, 50, 100} {
					if err = h.service.tx(t.Context(), func(s *evalstore.Store) error {
						return s.ObserveTokens(t.Context(), h.scope, e.ID, members[0], claims[0], observed)
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

func TestPostgresRecoveryAfterTransitionBeforeCommandAcknowledgement(t *testing.T) {
	h := newHarness(t)
	e := h.create(t, "workflow")
	h.command(t, e, "prepare")
	store := evalstore.NewPostgresStore(h.pool)
	claims, err := store.Claim(t.Context(), "crashed", time.Minute, 1)
	if err != nil || len(claims) != 1 {
		t.Fatal(claims, err)
	}
	e = h.get(t, e.ID)
	if err = h.service.prepare(t.Context(), e, claims[0]); err != nil {
		t.Fatal(err)
	}
	// Simulate process death after its commit and lease expiry without acknowledgement.
	if _, err = h.pool.Exec(t.Context(), `UPDATE eval_controller_claims SET expires_at=clock_timestamp()-interval '1 second' WHERE experiment_id=$1`, e.ID); err != nil {
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

func TestPostgresCancelAfterAuditCreateBeforeStartDrainsThroughOrdinaryDeletion(t *testing.T) {
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
	store := auditstore.NewPostgresStore(h.pool)
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

func TestPostgresPreparationArtifactLossRejectsBeforeExecution(t *testing.T) {
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

func TestPostgresProjectDeletionWaitsForUncertainRunAndFencesRemainingMembers(t *testing.T) {
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
	store := evalstore.NewPostgresStore(h.pool)
	blocked, err := store.ProjectBlocked(t.Context(), e.OwnerID, e.ProjectID)
	if err != nil || !blocked {
		t.Fatal("uncertain creation did not block purge", err)
	}
	if err = h.service.tx(t.Context(), func(s *evalstore.Store) error { return s.Purge(t.Context(), h.scope, e.ID) }); !errors.Is(err, evalstore.ErrDrain) {
		t.Fatal("uncertain work purged", err)
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
func TestPostgresPinnedSelectionsRejectBeforeOrdinaryEffects(t *testing.T) {
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
				if count(t, h.pool, "workflow_runs") != 0 || count(t, h.pool, "audit_items") != 0 {
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

func serviceTestPool(t *testing.T) *pgxpool.Pool {
	t.Helper()
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx := context.Background()
	admin, err := pgxpool.New(ctx, url)
	if err != nil {
		t.Fatal(err)
	}
	random := make([]byte, 8)
	if _, err = rand.Read(random); err != nil {
		t.Fatal(err)
	}
	schema := "eval_test_" + hex.EncodeToString(random)
	quoted := pgx.Identifier{schema}.Sanitize()
	if _, err = admin.Exec(ctx, `CREATE SCHEMA `+quoted); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	cfg, err := pgxpool.ParseConfig(url)
	if err != nil {
		t.Fatal(err)
	}
	cfg.ConnConfig.RuntimeParams["search_path"] = schema
	cfg.ConnConfig.RuntimeParams["statement_timeout"] = "10000"
	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		pool.Close()
		_, err := admin.Exec(context.Background(), `DROP SCHEMA `+quoted+` CASCADE`)
		admin.Close()
		if err != nil {
			t.Error(err)
		}
	})
	if _, err = pg.ApplyMigrations(ctx, pool); err != nil {
		t.Fatal(err)
	}
	return pool
}
func serviceTestCatalog(t *testing.T) *config.Snapshot {
	t.Helper()
	root := t.TempDir()
	files := map[string]string{
		"instructions/planner.md": "Execute the selected checklist item.",
		"instructions/worker.md":  "Read the task package and write a result package.",
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
  modelPolicy: worker@1
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
    task: {required: true, mediaTypes: [application/zip]}
  outputs:
    result: {required: true, mediaTypes: [application/zip]}
  executionConfig:
    workers:
      llmGateway: test-gateway@1
      credential: development-worker
  entryStage: check
  stages:
    check:
      objective: Evaluate one checklist item
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
  mode: custom-checklist
  standards: []
  inputs:
    checklist: {required: true, mediaTypes: [application/json]}
  inventory:
    implementation: checklist@1
    sourceInput: checklist
    itemWorkflowRole: check
  workflows:
    check:
      kind: check
      ref: audit-check@1
      inputs:
        task: {source: item-package}
      parameters: {}
      outputs: {result: result}
  execution:
    roundMode: fixed-barrier
    maxRounds: 1
    batchSize: 1
    maxItemsPerRound: 10
    maxItemsTotal: 10
    maxSubmittedRuns: 20
    maxItemRunAttempts: 2
    deadlineSeconds: 3600
    maxEvidenceBytes: 1048576
    incompleteRound: assess-with-gaps
  interaction:
    activeChecks: prohibited
    findingConfirmation: disabled
    notApplicable: profile-rule
    reportAcceptance: automatic
`,
	}
	for _, directory := range []string{
		"instructions", "llm-gateways", "model-policies", "execution-configs",
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
	snapshot, err := config.Load(root, config.MVPDescriptors())
	if err != nil {
		t.Fatalf("load Audit service config: %v", err)
	}
	return snapshot
}
