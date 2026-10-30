package app

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/credentials"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runrepeat"
	"github.com/grauwolf32/contractor/internal/runservice"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/grauwolf32/contractor/internal/scancontroller"
	"github.com/grauwolf32/contractor/internal/scanservice"
	"github.com/grauwolf32/contractor/internal/scanstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type sweepWake struct{}

func (sweepWake) Wake()         {}
func (sweepWake) Cancel(string) {}

type recordingSweepNotifier struct {
	sweepWake
	cancelled map[string]bool
}

func (n *recordingSweepNotifier) Cancel(id string) { n.cancelled[id] = true }

func TestPostgresScanSweepsRetainInputsAndUseOrdinaryRunAdmission(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 60*time.Second)
	defer cancel()
	f := newSweepFixture(t, ctx)
	pool, service, controller := f.pool, f.service, f.controller
	sourceService := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	user, _ := sourceService.User("sweep-owner")
	pack, err := user.Write(ctx, contracts.ArtifactRef{Namespace: "rules", Name: "pack"}, artifacts.Payload{MediaType: "application/yaml", Data: []byte("pinned owner pack")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	projects := projectstore.NewPostgresStore(pool)
	ids := []string{}
	for i := range 7 {
		id := fmt.Sprintf("sweep-project-%03d", i)
		ids = append(ids, id)
		if i == 5 {
			continue
		}
		p, _, err := projects.Create(ctx, projectstore.CreateParams{ProjectID: id, OwnerID: "sweep-owner", Kind: projectstore.KindProject, Name: id, IdempotencyKey: id, RequestDigest: contentdigest.Bytes([]byte(id))})
		if err != nil {
			t.Fatal(err)
		}
		if i == 6 {
			if _, _, err := projects.BeginDeletion(ctx, projectstore.BeginDeletionParams{ProjectID: id, OwnerID: "sweep-owner", ExpectedRevision: p.Revision}); err != nil {
				t.Fatal(err)
			}
		}
		if i < 4 {
			_, err = projects.Update(ctx, projectstore.UpdateParams{ProjectID: id, OwnerID: "sweep-owner", ExpectedRevision: p.Revision, Name: id, HTTPTarget: &runtimesettings.HTTPOriginTargetRef{URL: fmt.Sprintf("https://target-%d.invalid/", i)}})
			if err != nil {
				t.Fatal(err)
			}
		}
	}
	request := scanservice.Request{ProjectIDs: ids, Workflow: "nuclei-target@1", Rulepack: &pack.Ref}
	sweep, replayed, err := service.Create(ctx, "sweep-owner", "sweep-key", request)
	if err != nil {
		t.Fatal(err)
	}
	if replayed || sweep.Counts.Pending != 4 || sweep.Counts.Skipped != 3 {
		t.Fatalf("admission=%+v replayed=%v", sweep, replayed)
	}
	if _, err := service.Get(ctx, "another-owner", sweep.SweepID); !errors.Is(err, scanstore.ErrNotFound) {
		t.Fatalf("foreign read=%v", err)
	}
	if _, err := service.Cancel(ctx, "another-owner", sweep.SweepID); !errors.Is(err, scanstore.ErrNotFound) {
		t.Fatalf("foreign cancel=%v", err)
	}
	if err := service.Purge(ctx, "another-owner", sweep.SweepID); !errors.Is(err, scanstore.ErrNotFound) {
		t.Fatalf("foreign purge=%v", err)
	}
	if foreign, err := service.List(ctx, "another-owner", "", 10); err != nil || len(foreign) != 0 {
		t.Fatalf("foreign list=%+v %v", foreign, err)
	}
	if _, _, err := service.Create(ctx, "another-owner", "other-owner-key", request); !errors.Is(err, artifacts.ErrArtifactNotFound) {
		t.Fatalf("cross-owner rulepack=%v", err)
	}
	// Advance both mutable sources after the sweep has captured them.
	if _, err := user.Write(ctx, contracts.ArtifactRef{Namespace: "rules", Name: "pack"}, artifacts.Payload{MediaType: "application/yaml", Data: []byte("new pack")}, pack.Ref.Revision); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `UPDATE projects SET http_target_url='https://changed.invalid/',revision=revision+1,updated_at=clock_timestamp() WHERE owner_id='sweep-owner' AND lifecycle_state='active'`); err != nil {
		t.Fatal(err)
	}
	replay, found, err := service.Create(ctx, "sweep-owner", "sweep-key", request)
	if err != nil || !found || replay.SweepID != sweep.SweepID {
		t.Fatalf("replay=%+v found=%v err=%v", replay, found, err)
	}
	changed := request
	changed.Parameters = map[string]string{"rate_limit": "4"}
	if _, _, err := service.Create(ctx, "sweep-owner", "sweep-key", changed); !errors.Is(err, scanstore.ErrIdempotency) {
		t.Fatalf("changed request=%v", err)
	}
	if err := controller.RunOnce(ctx); err != nil {
		t.Fatal(err)
	}
	members, err := service.Members(ctx, "sweep-owner", sweep.SweepID, -1, 10)
	if err != nil {
		t.Fatal(err)
	}
	admitted := 0
	for _, m := range members {
		if m.State != "admitted" {
			continue
		}
		admitted++
		run, err := runstore.NewPostgresStore(pool).GetRun(ctx, *m.RunID)
		if err != nil {
			t.Fatal(err)
		}
		if run.ProjectHTTPTarget == nil || run.ProjectHTTPTarget.URL != run.Parameters["target"] || run.ProjectHTTPTarget.Credential != nil {
			t.Fatalf("changed Project metadata leaked into pin: %+v", run.ProjectHTTPTarget)
		}
		if run.Parameters["target"] == "https://changed.invalid/" {
			t.Fatal("target was re-read after admission")
		}
		scope, _ := sourceService.Run(*m.RunID)
		input, err := scope.Read(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "rulepack"})
		if err != nil || !bytes.Equal(input.Payload.Data, []byte("pinned owner pack")) {
			t.Fatalf("input=%+v err=%v", input, err)
		}
		repeat, err := runrepeat.Load(ctx, sourceService, *m.RunID)
		if err != nil || !repeat.Inputs["rulepack"].Artifact.SameExact(pack.Ref) {
			t.Fatalf("repeat=%+v err=%v", repeat, err)
		}
	}
	if admitted != 4 {
		t.Fatalf("admitted=%d members=%+v", admitted, members)
	}
	if err := service.Purge(ctx, "sweep-owner", sweep.SweepID); !errors.Is(err, scanstore.ErrConflict) {
		t.Fatalf("active purge=%v", err)
	}
	if _, err := service.Cancel(ctx, "sweep-owner", sweep.SweepID); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `UPDATE scan_sweeps SET next_tick_at=clock_timestamp()`); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `UPDATE scan_sweep_members SET next_attempt_at=next_attempt_at - interval '2 minutes'`); err != nil {
		t.Fatal(err)
	}
	if err := controller.RunOnce(ctx); err != nil {
		t.Fatal(err)
	}
	for _, m := range members {
		if m.RunID == nil {
			continue
		}
		run, err := runstore.NewPostgresStore(pool).GetRun(ctx, *m.RunID)
		if err != nil || run.State != runstore.RunCancelling {
			t.Fatalf("actual cancellation=%+v err=%v", run, err)
		}
		if !f.notifier.cancelled[*m.RunID] {
			t.Fatalf("Scheduler did not receive immediate cancellation for %s", *m.RunID)
		}
		if _, err := runstore.NewPostgresStore(pool).TransitionRun(ctx, *m.RunID, runstore.RunCancelling, runstore.RunCancelled, runstore.Reason{Code: "cancelled"}); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := pool.Exec(ctx, `UPDATE scan_sweeps SET next_tick_at=clock_timestamp()`); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `UPDATE scan_sweep_members SET next_attempt_at=next_attempt_at - interval '2 minutes'`); err != nil {
		t.Fatal(err)
	}
	if err := controller.RunOnce(ctx); err != nil {
		t.Fatal(err)
	}
	settled, err := service.Get(ctx, "sweep-owner", sweep.SweepID)
	if err != nil || settled.State != "cancelled" || settled.Counts.Unsettled != 0 {
		t.Fatalf("settled=%+v err=%v", settled, err)
	}
	if err := service.Purge(ctx, "sweep-owner", sweep.SweepID); err != nil {
		t.Fatal(err)
	}
	if _, _, err := service.Create(ctx, "sweep-owner", "sweep-key", request); !errors.Is(err, scanstore.ErrNotFound) {
		t.Fatalf("purged replay=%v", err)
	}
	var snapshotBindings, runCount int
	if err := pool.QueryRow(ctx, `SELECT (SELECT count(*) FROM artifact_bindings WHERE scope_kind='user' AND namespace='contractor-system'),(SELECT count(*) FROM workflow_runs)`).Scan(&snapshotBindings, &runCount); err != nil || snapshotBindings != 0 || runCount != 4 {
		t.Fatalf("snapshotBindings=%d runs=%d err=%v", snapshotBindings, runCount, err)
	}
	for _, m := range members {
		if m.RunID == nil {
			continue
		}
		scope, _ := sourceService.Run(*m.RunID)
		if _, err := scope.Read(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "rulepack"}); err != nil {
			t.Fatalf("purge removed child content: %v", err)
		}
	}
	if _, err := user.Metadata(ctx, pack.Ref); err != nil {
		t.Fatalf("purge removed owner source: %v", err)
	}
}

type sweepFixture struct {
	notifier      *recordingSweepNotifier
	pool          *pgxpool.Pool
	service       *scanservice.Service
	controller    *scancontroller.Controller
	runs          *runservice.Service
	manager       *config.Manager
	credentialSet runCreationCredentials
}

func newSweepFixture(t *testing.T, ctx context.Context) sweepFixture {
	t.Helper()
	pool := isolatedAppPool(t, ctx)
	root := t.TempDir()
	for _, dir := range []string{"workflows", "agent-templates", "instructions", "model-policies", "llm-gateways", "execution-configs", "audit-profiles"} {
		if err := os.MkdirAll(filepath.Join(root, dir), 0700); err != nil {
			t.Fatal(err)
		}
	}
	for _, relative := range []string{"workflows/nuclei.yaml", "agent-templates/nuclei.yaml", "instructions/scan.md"} {
		data, err := os.ReadFile(filepath.Join("../config/testdata/scan-catalog", relative))
		if err != nil {
			t.Fatal(err)
		}
		text := string(data)
		if strings.HasPrefix(relative, "workflows/") {
			text = strings.Replace(text, "inputs: {}", "inputs:\n    rulepack: {required: false, sourceScope: user, mediaTypes: [application/yaml]}", 1)
			text = strings.Replace(text, "context: {artifacts: {}}", "context: {artifacts: {rulepack: {namespace: inputs, name: rulepack, required: false}}}", 1)
		}
		if strings.HasPrefix(relative, "agent-templates/") {
			text = strings.Replace(text, "url: {source: parameter, name: target}", "url: {source: parameter, name: target}\n      rulepack_ref: {source: artifact, name: rulepack, optional: true}", 1)
		}
		if err := os.WriteFile(filepath.Join(root, relative), []byte(text), 0600); err != nil {
			t.Fatal(err)
		}
	}
	manager, _, err := configload.NewManager(config.ManagerOptions{OperatorRoot: root, ManagedRoot: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	provider, err := credentials.NewStaticProvider(nil)
	if err != nil {
		t.Fatal(err)
	}
	lookup, err := credentials.NewTransactionLookupFactory(provider)
	if err != nil {
		t.Fatal(err)
	}
	guard := recoveryCredentialGuard{}
	credentialSet := runCreationCredentials{provider: provider, guard: guard, runtime: guard, transactionLookup: lookup}
	runs, err := configureRunCreation(pool, manager, credentialSet)
	if err != nil {
		t.Fatal(err)
	}
	notifier := &recordingSweepNotifier{cancelled: map[string]bool{}}
	service, controller, err := configureSweeps(pool, manager, credentialSet, runs, notifier, slog.New(slog.NewTextHandler(io.Discard, nil)))
	if err != nil {
		t.Fatal(err)
	}

	return sweepFixture{notifier: notifier, pool: pool, service: service, controller: controller, runs: runs, manager: manager, credentialSet: credentialSet}
}

func TestPostgresScanSweepsBoundThousandMembersAndRecoverLostAcknowledgements(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 2*time.Minute)
	defer cancel()
	f := newSweepFixture(t, ctx)
	projects := projectstore.NewPostgresStore(f.pool)
	for i := range scanstore.MaxProjects {
		id := fmt.Sprintf("scale-%04d", i)
		p, _, err := projects.Create(ctx, projectstore.CreateParams{ProjectID: id, OwnerID: "scale-owner", Kind: projectstore.KindProject, Name: id, IdempotencyKey: id, RequestDigest: contentdigest.Bytes([]byte(id))})
		if err != nil {
			t.Fatal(err)
		}
		_, err = projects.Update(ctx, projectstore.UpdateParams{ProjectID: id, OwnerID: "scale-owner", ExpectedRevision: p.Revision, Name: id, HTTPTarget: &runtimesettings.HTTPOriginTargetRef{URL: "https://captured.invalid/"}})
		if err != nil {
			t.Fatal(err)
		}
	}
	a := artifacts.NewService(artifacts.NewPostgresRepository(f.pool))
	user, _ := a.User("scale-owner")
	pack, err := user.Write(ctx, contracts.ArtifactRef{Namespace: "rules", Name: "pack"}, artifacts.Payload{MediaType: "application/yaml", Data: []byte("one retained copy")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	request := scanservice.Request{AllProjects: true, Workflow: "nuclei-target@1", Rulepack: &pack.Ref}
	sweep, _, err := f.service.Create(ctx, "scale-owner", "scale-create", request)
	if err != nil {
		t.Fatal(err)
	}
	if sweep.Counts.Total != 1000 || sweep.Counts.Pending != 1000 {
		t.Fatalf("snapshot=%+v", sweep)
	}
	// Deleting the original UserScope binding must leave the retained fork live.
	if err := persistencepostgres.InTx(ctx, f.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if _, err := tx.Exec(ctx, `SELECT set_config('contractor.lifecycle_purge','project',true)`); err != nil {
			return err
		}
		for _, sql := range []string{
			`DELETE FROM artifact_lineage WHERE source_scope_kind='user' AND source_scope_id='scale-owner' AND source_namespace='rules' AND source_name='pack'`,
			`DELETE FROM artifact_bindings WHERE scope_kind='user' AND scope_id='scale-owner' AND namespace='rules' AND name='pack'`,
			`DELETE FROM artifact_binding_revisions WHERE scope_kind='user' AND scope_id='scale-owner' AND namespace='rules' AND name='pack'`,
		} {
			if _, err := tx.Exec(ctx, sql); err != nil {
				return err
			}
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	replay, replayed, err := f.service.Create(ctx, "scale-owner", "scale-create", request)
	if err != nil || !replayed || replay.SweepID != sweep.SweepID {
		t.Fatalf("lost sweep acknowledgement replay=%+v %v %v", replay, replayed, err)
	}
	claim, found, err := scanstore.New(f.pool).Claim(ctx, "competing-controller", time.Minute)
	if err != nil || !found {
		t.Fatalf("claim=%+v %v %v", claim, found, err)
	}
	if err := f.controller.RunOnce(ctx); err != nil {
		t.Fatal(err)
	}
	unclaimed, err := f.service.Get(ctx, "scale-owner", sweep.SweepID)
	if err != nil || unclaimed.Counts.Admitted != 0 {
		t.Fatalf("competing lease admitted: %+v %v", unclaimed, err)
	}
	if err := scanstore.New(f.pool).Release(ctx, claim); err != nil {
		t.Fatal(err)
	}
	base := (postgresPublicUnitOfWork{pool: f.pool, transactionLLMCredentials: f.credentialSet.transactionLookup}).sweepAdmitter(f.runs, sweepWake{})
	calls := 0
	lostAck, err := scancontroller.New(scancontroller.Options{Pool: f.pool, HolderID: "lost-ack-controller", Now: time.Now, Logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
		Admit: func(ctx context.Context, c persistencepostgres.ControllerClaim, s scanstore.Snapshot, m scanstore.Member) error {
			calls++
			if err := base(ctx, c, s, m); err != nil {
				return err
			}
			return context.DeadlineExceeded
		},
		CancelRun: func(context.Context, string, string) (runstore.WorkflowRun, error) {
			return runstore.WorkflowRun{}, errors.New("unexpected cancellation")
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	readySweep(t, ctx, f.pool)
	if err := lostAck.RunOnce(ctx); err != nil {
		t.Fatal(err)
	}
	first, err := f.service.Get(ctx, "scale-owner", sweep.SweepID)
	if err != nil || first.Counts.Admitted != 25 || calls != 25 {
		t.Fatalf("bounded lost ack tick: %+v calls=%d err=%v", first, calls, err)
	}
	// Recompose a new controller as after Server restart. It reads only the
	// immutable snapshot and pending members; admitted Run keys are never sent.
	_, restarted, err := configureSweeps(f.pool, f.manager, f.credentialSet, f.runs, sweepWake{}, slog.New(slog.NewTextHandler(io.Discard, nil)))
	if err != nil {
		t.Fatal(err)
	}
	previous := first.Counts.Admitted
	for range 39 {
		readySweep(t, ctx, f.pool)
		if err := restarted.RunOnce(ctx); err != nil {
			t.Fatal(err)
		}
		current, err := f.service.Get(ctx, "scale-owner", sweep.SweepID)
		if err != nil {
			t.Fatal(err)
		}
		if current.Counts.Admitted-previous > 25 {
			t.Fatalf("tick exceeds member budget: %d -> %d", previous, current.Counts.Admitted)
		}
		previous = current.Counts.Admitted
	}
	if previous != 1000 {
		t.Fatalf("admitted=%d, want 1000", previous)
	}
	var runCount, snapshotCount, wrongTarget int
	if err := f.pool.QueryRow(ctx, `SELECT (SELECT count(*) FROM workflow_runs),
 (SELECT count(*) FROM artifact_bindings WHERE scope_kind='user' AND scope_id='scale-owner' AND namespace='contractor-system'),
 (SELECT count(*) FROM workflow_runs WHERE parameters->>'target'<>'https://captured.invalid/')`).Scan(&runCount, &snapshotCount, &wrongTarget); err != nil || runCount != 1000 || snapshotCount != 1 || wrongTarget != 0 {
		t.Fatalf("runs=%d snapshots=%d wrongTarget=%d error=%v", runCount, snapshotCount, wrongTarget, err)
	}
	// Result pagination must expose every member, not just the first page.
	after := -1
	total := 0
	for {
		page, err := f.service.Members(ctx, "scale-owner", sweep.SweepID, after, 200)
		if err != nil {
			t.Fatal(err)
		}
		if len(page) == 0 {
			break
		}
		for _, m := range page {
			if m.Ordinal <= after || m.RunID == nil {
				t.Fatalf("invalid page member %+v", m)
			}
			after = m.Ordinal
			total++
		}
		if len(page) < 200 {
			break
		}
	}
	if total != 1000 {
		t.Fatalf("paged members=%d", total)
	}
	id := "scale-overflow"
	if _, _, err := projects.Create(ctx, projectstore.CreateParams{ProjectID: id, OwnerID: "scale-owner", Kind: projectstore.KindProject, Name: id, IdempotencyKey: id, RequestDigest: contentdigest.Bytes([]byte(id))}); err != nil {
		t.Fatal(err)
	}
	// Use the already-retained pack in no new request: the failed bound check
	// must still reject before dispatch and roll back any input snapshot fork.
	extra, err := user.Write(ctx, contracts.ArtifactRef{Namespace: "rules", Name: "second"}, artifacts.Payload{MediaType: "application/yaml", Data: []byte("new owner source")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	overflow := request
	overflow.Rulepack = &extra.Ref
	if _, _, err := f.service.Create(ctx, "scale-owner", "overflow-create", overflow); !errors.Is(err, scanstore.ErrInvalid) {
		t.Fatalf("1001 projects=%v", err)
	}
	if err := f.pool.QueryRow(ctx, `SELECT count(*) FROM scan_sweeps`).Scan(&runCount); err != nil || runCount != 1 {
		t.Fatalf("failed create left sweep: %d %v", runCount, err)
	}
}

func readySweep(t *testing.T, ctx context.Context, pool *pgxpool.Pool) {
	t.Helper()
	if _, err := pool.Exec(ctx, `UPDATE scan_sweeps SET next_tick_at=clock_timestamp()`); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `UPDATE scan_sweep_members SET next_attempt_at=next_attempt_at - interval '2 minutes'`); err != nil {
		t.Fatal(err)
	}
}

func TestPostgresScanSweepsCancellationFencesUncommittedAdmission(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	f := newSweepFixture(t, ctx)
	sweep := createSmallSweep(t, ctx, f, "cancel-fence", 1)
	claim, found, err := scanstore.New(f.pool).Claim(ctx, "fenced-admitter", time.Minute)
	if err != nil || !found {
		t.Fatalf("claim=%+v %v %v", claim, found, err)
	}
	snapshot, err := scanstore.New(f.pool).Snapshot(ctx, sweep.SweepID)
	if err != nil {
		t.Fatal(err)
	}
	members, err := f.service.Members(ctx, "cancel-fence", sweep.SweepID, -1, 10)
	if err != nil {
		t.Fatal(err)
	}
	entered, proceed := make(chan struct{}), make(chan struct{})
	lookup := runtimeconfigtest.TransactionLLMCredentialLookupFactoryFunc(func(tx pgx.Tx) (resolved.CredentialLookup, error) {
		close(entered)
		select {
		case <-proceed:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
		return f.credentialSet.transactionLookup.ForTransaction(tx)
	})
	admit := (postgresPublicUnitOfWork{pool: f.pool, transactionLLMCredentials: lookup}).sweepAdmitter(f.runs, sweepWake{})
	outcome := make(chan error, 1)
	go func() { outcome <- admit(ctx, claim, snapshot, members[0]) }()
	select {
	case <-entered:
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	}
	if _, err := f.service.Cancel(ctx, "cancel-fence", sweep.SweepID); err != nil {
		t.Fatal(err)
	}
	close(proceed)
	select {
	case err := <-outcome:
		if !errors.Is(err, scanstore.ErrConflict) {
			t.Fatalf("late admission=%v", err)
		}
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	}
	var count int
	if err := f.pool.QueryRow(ctx, `SELECT count(*) FROM workflow_runs`).Scan(&count); err != nil || count != 0 {
		t.Fatalf("Run admitted after cancel: %d %v", count, err)
	}
	if err := scanstore.New(f.pool).Release(ctx, claim); err != nil {
		t.Fatal(err)
	}
	readySweep(t, ctx, f.pool)
	if err := f.controller.RunOnce(ctx); err != nil {
		t.Fatal(err)
	}
	result, err := f.service.Get(ctx, "cancel-fence", sweep.SweepID)
	if err != nil || result.State != "cancelled" || result.Counts.Cancelled != 1 {
		t.Fatalf("settlement=%+v %v", result, err)
	}
}

func TestPostgresScanSweepsPurgeJoinsControllerAndPreservesChild(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	f := newSweepFixture(t, ctx)
	sweep := createSmallSweep(t, ctx, f, "purge-join", 1)
	terminal, exited := make(chan struct{}), make(chan struct{})
	base := (postgresPublicUnitOfWork{pool: f.pool, transactionLLMCredentials: f.credentialSet.transactionLookup}).sweepAdmitter(f.runs, sweepWake{})
	controller, err := scancontroller.New(scancontroller.Options{Pool: f.pool, HolderID: "join-controller", Now: time.Now, Logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
		Admit: func(ctx context.Context, c persistencepostgres.ControllerClaim, s scanstore.Snapshot, m scanstore.Member) error {
			defer close(exited)
			if err := base(ctx, c, s, m); err != nil {
				return err
			}
			run, found, err := runstore.NewPostgresStore(f.pool).LookupRunIdempotency(ctx, s.OwnerID, m.Key, m.Digest)
			if err != nil || !found {
				return fmt.Errorf("read created Run: %v", err)
			}
			if _, err := runstore.NewPostgresStore(f.pool).RequestRunCancellation(ctx, run.RunID, runstore.WorkflowRunCancellation{Code: runstore.CancellationUserRequested, RequestedAt: time.Now()}); err != nil {
				return err
			}
			if _, err := runstore.NewPostgresStore(f.pool).TransitionRun(ctx, run.RunID, runstore.RunCancelling, runstore.RunCancelled, runstore.Reason{Code: "cancelled"}); err != nil {
				return err
			}
			if err := persistencepostgres.InTx(ctx, f.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
				return scanstore.New(tx).Record(ctx, c, m.Ordinal, "admitted", "", &run.RunID, true, 0)
			}); err != nil {
				return err
			}
			close(terminal)
			<-ctx.Done()
			return ctx.Err()
		}, CancelRun: func(context.Context, string, string) (runstore.WorkflowRun, error) {
			return runstore.WorkflowRun{}, errors.New("unexpected cancellation")
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { done <- controller.RunOnce(ctx) }()
	select {
	case <-terminal:
	case err := <-done:
		t.Fatalf("controller ended before terminal hook: %v", err)
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	}
	if err := controller.Drain(ctx, sweep.SweepID, func() error {
		select {
		case <-exited:
		default:
			t.Error("purge callback started before child operation joined")
		}
		return f.service.Purge(ctx, "purge-join", sweep.SweepID)
	}); err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("drained controller=%v", err)
		}
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	}
	var count int
	if err := f.pool.QueryRow(ctx, `SELECT count(*) FROM workflow_runs`).Scan(&count); err != nil || count != 1 {
		t.Fatalf("purge changed independent child: %d %v", count, err)
	}
}

func createSmallSweep(t *testing.T, ctx context.Context, f sweepFixture, owner string, n int) scanstore.Sweep {
	t.Helper()
	a := artifacts.NewService(artifacts.NewPostgresRepository(f.pool))
	user, _ := a.User(owner)
	pack, err := user.Write(ctx, contracts.ArtifactRef{Namespace: "rules", Name: "pack"}, artifacts.Payload{MediaType: "application/yaml", Data: []byte("retained source")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	projects := projectstore.NewPostgresStore(f.pool)
	for i := range n {
		id := fmt.Sprintf("%s-%03d", owner, i)
		p, _, err := projects.Create(ctx, projectstore.CreateParams{ProjectID: id, OwnerID: owner, Kind: projectstore.KindProject, Name: id, IdempotencyKey: id, RequestDigest: contentdigest.Bytes([]byte(id))})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := projects.Update(ctx, projectstore.UpdateParams{ProjectID: id, OwnerID: owner, ExpectedRevision: p.Revision, Name: id, HTTPTarget: &runtimesettings.HTTPOriginTargetRef{URL: "https://captured.invalid/"}}); err != nil {
			t.Fatal(err)
		}
	}
	sweep, _, err := f.service.Create(ctx, owner, "create", scanservice.Request{AllProjects: true, Workflow: "nuclei-target@1", Rulepack: &pack.Ref})
	if err != nil {
		t.Fatal(err)
	}
	return sweep
}

func TestPostgresScanSweepsAdmissionDeadlineAndBoundedRetention(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	f := newSweepFixture(t, ctx)
	sweep := createSmallSweep(t, ctx, f, "deadline", 50)
	future, err := scancontroller.New(scancontroller.Options{Pool: f.pool, HolderID: "deadline-controller", Now: func() time.Time { return sweep.CreatedAt.Add(scanstore.AdmissionDeadline) }, Logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
		Admit: func(context.Context, persistencepostgres.ControllerClaim, scanstore.Snapshot, scanstore.Member) error {
			t.Error("expired admission invoked")
			return nil
		},
		CancelRun: func(context.Context, string, string) (runstore.WorkflowRun, error) {
			t.Error("unadmitted Run cancelled")
			return runstore.WorkflowRun{}, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	for range 2 {
		readySweep(t, ctx, f.pool)
		if err := future.RunOnce(ctx); err != nil {
			t.Fatal(err)
		}
	}
	settled, err := f.service.Get(ctx, "deadline", sweep.SweepID)
	if err != nil || settled.State != "completed" || settled.Counts.Failed != 50 {
		t.Fatalf("deadline result=%+v %v", settled, err)
	}
	if err := scanstore.New(f.pool).PurgeExpired(ctx, settled.SettledAt.Add(scanstore.Retention-time.Nanosecond)); err != nil {
		t.Fatal(err)
	}
	if _, err := f.service.Get(ctx, "deadline", sweep.SweepID); err != nil {
		t.Fatalf("retention purged before deadline: %v", err)
	}
	for i, want := range []int{25, 0, 0} {
		if err := scanstore.New(f.pool).PurgeExpired(ctx, settled.SettledAt.Add(scanstore.Retention)); err != nil {
			t.Fatal(err)
		}
		var remaining int
		if err := f.pool.QueryRow(ctx, `SELECT count(*) FROM scan_sweep_members`).Scan(&remaining); err != nil || remaining != want {
			t.Fatalf("cleanup tick %d members=%d want=%d err=%v", i, remaining, want, err)
		}
	}
	if _, _, err := f.service.Create(ctx, "deadline", "create", scanservice.Request{AllProjects: true, Workflow: "nuclei-target@1", Rulepack: nil}); !errors.Is(err, scanstore.ErrIdempotency) {
		t.Fatalf("changed purged replay=%v", err)
	}
}
