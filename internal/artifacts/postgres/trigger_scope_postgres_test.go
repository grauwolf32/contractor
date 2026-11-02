package postgres_test

import (
	"context"
	"errors"
	"maps"
	"strings"
	"sync"
	"testing"
	"time"

	. "github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/artifactstest"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/persistence/explaintest"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

const (
	bindingAdmission  = "artifact_bindings_project_admission"
	bindingIdentity   = "artifact_bindings_protect_identity"
	outputsOnInsert   = "eval_outputs_projection"
	outputsOnUpdate   = "eval_outputs_projection_on_update"
	outputsOnDelete   = "eval_outputs_projection_on_delete"
	baselineImmutable = "Audit preparation baseline is immutable"
)

// TestPostgresArtifactBindingWritesFireOnlyTheTriggersTheyConcern counts
// the triggers of artifact_bindings that each artifact write calls
// (PER-R22). Project admission runs for every create and update of a
// Project binding and for no other scope; the identity guard runs for every
// update; the Eval output projection runs for a Run output that is created,
// gets a revision or a frozen flag that differs, or is deleted, and for no
// other binding.
func TestPostgresArtifactBindingWritesFireOnlyTheTriggersTheyConcern(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 60*time.Second)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
	artifactStorage := testBlobStorage(t, ctx, pool)
	project, _ := storageReviewProject(t, ctx, pool, artifactpgtest.NewStorage())
	createArtifactRun(t, ctx, pool, "run-binding-triggers", true)
	tx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback(context.WithoutCancel(ctx)) }()
	probe := explaintest.NewProbe(t, tx)
	service := NewService(artifactStorage.Repository(probe))
	expect := func(step string, want map[string]int64, action func() error) {
		t.Helper()
		probe.Reset()
		if err := action(); err != nil {
			t.Fatalf("%s: %v", step, err)
		}
		if fired := probe.Fired("artifact_bindings"); !maps.Equal(fired, want) {
			t.Errorf("%s fired the artifact_bindings triggers %v, want %v", step, fired, want)
		}
	}
	payload := func(text string) Payload { return Payload{MediaType: "text/plain", Data: []byte(text)} }
	projectStore, _ := service.Project(project.ProjectID)
	target := ArtifactRef{Namespace: "docs", Name: "notes"}
	var written WriteResult
	expect("Project create", map[string]int64{bindingAdmission: 1}, func() error {
		written, err = projectStore.Write(ctx, target, payload("one"), nil)
		return err
	})
	expect("Project update", map[string]int64{bindingAdmission: 1, bindingIdentity: 1}, func() error {
		_, err := projectStore.Write(ctx, target, payload("two"), written.Ref.Revision)
		return err
	})
	const runID = "run-binding-triggers"
	runStore, _ := service.Run(runID)
	intermediate := ArtifactRef{Namespace: "builder", Name: "result"}
	expect("Run create", map[string]int64{}, func() error {
		written, err = runStore.Write(ctx, intermediate, payload("one"), nil)
		return err
	})
	expect("Run update", map[string]int64{bindingIdentity: 1}, func() error {
		written, err = runStore.Write(ctx, intermediate, payload("two"), written.Ref.Revision)
		return err
	})
	expect("output binding", map[string]int64{outputsOnInsert: 1}, func() error {
		_, err := artifactstest.BindOutput(ctx, service, runID, "result", written.Ref)
		return err
	})
	expect("output freeze", map[string]int64{bindingIdentity: 1, outputsOnUpdate: 1}, func() error {
		return artifactstest.FreezeOutputs(ctx, service, runID)
	})
	expect("repeated freeze", map[string]int64{bindingIdentity: 1}, func() error {
		return artifactstest.FreezeOutputs(ctx, service, runID)
	})
	expect("Run purge", map[string]int64{outputsOnDelete: 1}, func() error {
		purger, err := artifactStorage.Purger(probe)
		if err != nil {
			return err
		}
		return purger.PurgeRun(ctx, runID)
	})

	// Each column the identity guard compares, changed alone, is rejected
	// by it, so a column missing from its UPDATE OF list fails here.
	for _, column := range []string{"scope_kind", "scope_id", "namespace", "name", "created_at"} {
		value, err := explaintest.Changed(ctx, tx, "artifact_bindings", column)
		if err != nil {
			t.Fatal(err)
		}
		statement := `UPDATE artifact_bindings SET ` + column + ` = ` + value +
			` WHERE scope_kind = 'project' AND scope_id = $1 AND namespace = 'docs'`
		if _, err := explaintest.Explain(ctx, tx, statement, project.ProjectID); err == nil ||
			!strings.Contains(err.Error(), "Artifact binding identity cannot be changed") {
			t.Errorf("%s = %v, want the binding identity protected", statement, err)
		}
	}
	// A frozen output keeps its revision and its flag.
	revision, err := explaintest.Changed(ctx, tx, "artifact_bindings", "current_revision")
	if err != nil {
		t.Fatal(err)
	}
	for _, assignment := range []string{"current_revision = " + revision, "frozen = false"} {
		statement := `UPDATE artifact_bindings SET ` + assignment + `
WHERE scope_kind = 'run' AND scope_id = 'review-run' AND namespace = 'outputs'`
		if _, err := explaintest.Explain(ctx, tx, statement); err == nil ||
			!strings.Contains(err.Error(), "frozen Artifact binding cannot be changed") {
			t.Errorf("%s = %v, want the frozen output protected", statement, err)
		}
	}
}

// scopeWindowTracer commits a change from another connection right after
// the first statement on the traced pool that creates an artifact scope, so
// that the change lands between a write's scope statement and its binding
// statement.
type scopeWindowTracer struct {
	mu     sync.Mutex
	change func(context.Context) error
	fired  bool
	err    error
}

func (s *scopeWindowTracer) TraceQueryStart(ctx context.Context, _ *pgx.Conn, _ pgx.TraceQueryStartData) context.Context {
	return ctx
}

func (s *scopeWindowTracer) TraceQueryEnd(ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryEndData) {
	if data.Err != nil || !strings.HasPrefix(data.CommandTag.String(), "INSERT") {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.change == nil || s.fired {
		return
	}
	s.fired = true
	s.err = s.change(context.WithoutCancel(ctx))
}

// TestStorageReviewProjectDeletionBetweenScopeAndBinding commits Project
// deletion after a write has made sure of its artifact scope and before it
// stores the binding: the binding's Project admission still rejects the write
// as ErrScopeDeleting and no revision is stored, both for a new binding and
// for a revision of an existing one (DEL-R40). Admission keeps firing on every
// INSERT and UPDATE of a Project binding.
func TestStorageReviewProjectDeletionBetweenScopeAndBinding(t *testing.T) {
	for _, update := range []bool{false, true} {
		name := "create"
		if update {
			name = "update"
		}
		t.Run(name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
			defer cancel()
			pool := isolatedArtifactPool(t, ctx)
			artifactStorage := testBlobStorage(t, ctx, pool)
			project, _ := storageReviewProject(t, ctx, pool, artifactpgtest.NewStorage())
			target := ArtifactRef{Namespace: "docs", Name: "content"}
			var prior *string
			if update {
				store, _ := NewService(artifactStorage.Repository(pool)).Project(project.ProjectID)
				initial, err := store.Write(ctx, target, Payload{MediaType: "text/plain", Data: []byte("initial")}, nil)
				if err != nil {
					t.Fatal(err)
				}
				prior = initial.Ref.Revision
			}
			trace := &scopeWindowTracer{change: func(ctx context.Context) error {
				_, _, err := projectstore.NewPostgresStore(pool).BeginDeletion(ctx, projectstore.BeginDeletionParams{
					OwnerID: project.OwnerID, ProjectID: project.ProjectID, ExpectedRevision: project.Revision,
				})
				return err
			}}
			configuration := pool.Config()
			configuration.ConnConfig.Tracer = trace
			traced, err := pgxpool.NewWithConfig(ctx, configuration)
			if err != nil {
				t.Fatal(err)
			}
			defer traced.Close()
			store, _ := NewService(artifactStorage.Repository(traced)).Project(project.ProjectID)
			_, err = store.Write(ctx, target, Payload{MediaType: "text/plain", Data: []byte("rejected")}, prior)
			if fired, changeErr := trace.fired, trace.err; !fired || changeErr != nil {
				t.Fatalf("deletion between the scope and the binding: fired=%t %v", fired, changeErr)
			}
			if !errors.Is(err, ErrScopeDeleting) {
				t.Fatalf("write after deletion began = %v, want ErrScopeDeleting", err)
			}
			var revisions int
			if err := pool.QueryRow(ctx, `SELECT count(*) FROM artifact_binding_revisions
WHERE scope_kind = 'project' AND scope_id = $1 AND namespace = 'docs'`, project.ProjectID).Scan(&revisions); err != nil {
				t.Fatal(err)
			}
			if want := map[bool]int{false: 0, true: 1}[update]; revisions != want {
				t.Fatalf("the rejected write left %d revisions, want %d", revisions, want)
			}
		})
	}
}

// TestSharedPostgresProjectDeletionKeepsAuditCancellationWrites checks the
// exception of Project admission that its WHEN condition must keep (DEL-R40):
// once deletion asked an Audit of the Project to cancel, the Audit may still
// import into its own namespace of the Project scope, while any other Project
// write is rejected.
func TestSharedPostgresProjectDeletionKeepsAuditCancellationWrites(t *testing.T) {
	f := newPreparationFixture(t, false)
	execution, outputs := f.collecting(t, f.intent("generate-api", 1), "succeeded")
	projects := projectstore.NewPostgresStore(f.pool)
	project, err := projects.Get(f.ctx, f.audit.OwnerID, f.audit.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := projects.BeginDeletion(f.ctx, projectstore.BeginDeletionParams{
		ProjectID: project.ProjectID, OwnerID: project.OwnerID, ExpectedRevision: project.Revision,
	}); err != nil {
		t.Fatal(err)
	}
	target := ArtifactRef{Namespace: auditdomain.ArtifactNamespace(f.audit.AuditID), Name: "cancellation-copy"}
	if _, err := importAuditArtifact(f.ctx, f.artifacts, "run-"+execution.ExecutionID, outputs[0].Source.Ref,
		project.ProjectID, target); !errors.Is(err, ErrScopeDeleting) {
		t.Fatalf("Audit import before its cancellation was requested = %v, want ErrScopeDeleting", err)
	}
	if _, err := f.pool.Exec(f.ctx, `
UPDATE projects
   SET deletion_claim_id = 'claim-cancel-writes', deletion_claimed_at = clock_timestamp(),
       deletion_claim_expires_at = clock_timestamp() + interval '1 minute'
 WHERE project_id = $1`, project.ProjectID); err != nil {
		t.Fatal(err)
	}
	if changed, err := f.store.RequestProjectOwnedDeletion(f.ctx, projectstore.DeletionClaim{
		ProjectID: project.ProjectID, OwnerID: project.OwnerID, ClaimID: "claim-cancel-writes",
		Phase: projectstore.DeletionCancelling,
	}); err != nil || !changed {
		t.Fatalf("Project-owned Audit deletion = (%t, %v)", changed, err)
	}
	if _, err := importAuditArtifact(f.ctx, f.artifacts, "run-"+execution.ExecutionID, outputs[0].Source.Ref,
		project.ProjectID, target); err != nil {
		t.Fatalf("Audit import while the Audit cancels = %v", err)
	}
	store, _ := f.artifacts.Project(project.ProjectID)
	if _, err := store.Write(f.ctx, ArtifactRef{Namespace: "docs", Name: "late"}, Payload{MediaType: "text/plain",
		Data: []byte("late")}, nil); !errors.Is(err, ErrScopeDeleting) {
		t.Fatalf("other Project write during deletion = %v, want ErrScopeDeleting", err)
	}
}

// TestSharedPostgresAuditPreparationGuardsSeeEachGuardedColumn changes each
// column the Audit preparation guard and the preparation execution check
// list in UPDATE OF, alone, and expects the rejection that column is guarded
// by (PER-R22): a column missing from a list fails here. The baseline is
// checked while the Audit prepares, and again after preparation through the
// probe for a prepare execution, which now runs only for a baseline change
// or a not-started phase.
func TestSharedPostgresAuditPreparationGuardsSeeEachGuardedColumn(t *testing.T) {
	f := newPreparationFixture(t, false)
	execution, outputs := f.collecting(t, f.intent("generate-api", 1), "succeeded")
	expectRejected := func(db explaintest.Beginner, table, column, where, message string, args ...any) {
		t.Helper()
		value, err := explaintest.Changed(f.ctx, f.pool, table, column)
		if err != nil {
			t.Fatal(err)
		}
		statement := `UPDATE ` + table + ` SET ` + column + ` = ` + value + ` WHERE ` + where
		if _, err := explaintest.Explain(f.ctx, db, statement, args...); err == nil || !strings.Contains(err.Error(), message) {
			t.Errorf("%s = %v, want %q", statement, err, message)
		}
	}
	for _, column := range []string{
		"execution_id", "audit_id", "round_id", "role", "workflow_role", "role_attempt", "manifest_ref",
		"manifest_digest", "submission_key", "request_digest", "preparation_snapshot",
	} {
		expectRejected(f.pool, "audit_executions", column, "execution_id = $1",
			"Audit preparation execution identity is immutable", execution.ExecutionID)
	}
	// A receipt is validated when it is assigned; this one accepts nothing.
	expectRejected(f.pool, "audit_executions", "collection_receipt_id", "execution_id = $1",
		"Audit preparation must accept all mapped outputs atomically", execution.ExecutionID)
	if _, created, err := f.store.Collect(f.ctx, f.collection(execution, outputs)); err != nil || !created {
		t.Fatalf("collect: %t %v", created, err)
	}
	expectRejected(f.pool, "audit_executions", "preparation_outputs", "execution_id = $1",
		"Audit preparation outputs are immutable", execution.ExecutionID)

	baseline := []string{
		"baseline_snapshot", "profile_snapshot", "input_selection", "profile_name", "profile_version",
		"profile_digest", "owner_id", "project_id",
	}
	notStarted := func(db explaintest.Beginner) {
		t.Helper()
		if _, err := explaintest.Explain(f.ctx, db, `UPDATE audits SET phase = 'not-started' WHERE audit_id = $1`,
			f.audit.AuditID); err == nil || !strings.Contains(err.Error(), "Audit preparation cannot return to not-started") {
			t.Errorf("returning to not-started = %v, want it rejected", err)
		}
	}
	for _, column := range baseline {
		expectRejected(f.pool, "audits", column, "audit_id = $1", baselineImmutable, f.audit.AuditID)
	}
	notStarted(f.pool)
	current := f.current(t)
	if _, err := f.store.CompletePreparation(f.ctx, f.claim, current.Revision); err != nil {
		t.Fatal(err)
	}
	// Past inventory, only the prepare execution shows that preparation
	// began; the Rounds phase is set here without its Round, in a
	// transaction that is rolled back.
	tx, err := f.pool.Begin(f.ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback(context.WithoutCancel(f.ctx)) }()
	if _, err := tx.Exec(f.ctx, `UPDATE audits SET phase = 'rounds' WHERE audit_id = $1`, f.audit.AuditID); err != nil {
		t.Fatal(err)
	}
	for _, column := range baseline {
		expectRejected(tx, "audits", column, "audit_id = $1", baselineImmutable, f.audit.AuditID)
	}
	notStarted(tx)
}
