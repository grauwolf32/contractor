package auditservice

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	managedcredentials "github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
)

// TestAuditStartRetryReusesStagedPackages covers AUD-R225: task packages and
// the worklist are staged frozen before the start transaction, so a start
// whose transaction fails leaves them staged and its retry writes none again.
func TestAuditStartRetryReusesStagedPackages(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedAuditServicePool(t, ctx, databaseURL)
	guard := &failingCredentialGuard{failures: 1}
	service := newStartStagingService(t, pool, guard)
	project, draft := createStartStagingDraft(t, ctx, pool, service, "retry", 3)

	params := StartParams{
		OwnerID: project.OwnerID, AuditID: draft.AuditID, ExpectedRevision: draft.Revision,
		IdempotencyKey: "start-staging-retry", RequestDigest: serviceTestDigest("start-staging-retry"),
	}
	if _, err := service.Start(ctx, params); postgres.SQLState(err) != postgres.SQLStateSerializationFailure {
		t.Fatalf("start with a failed transaction = %v", err)
	}
	staged := auditBindingRevisions(t, ctx, pool, project.ProjectID)
	if len(staged) != 4 {
		t.Fatalf("staged Audit revisions after the failed start = %v, want 3 task packages and the worklist", staged)
	}
	if current, err := service.Get(ctx, project.OwnerID, draft.AuditID); err != nil ||
		current.State != auditstore.AuditDraft || current.Revision != draft.Revision {
		t.Fatalf("draft after the failed start = (%+v, %v)", current, err)
	}

	started, err := service.Start(ctx, params)
	if err != nil || started.Replayed || started.Audit.State != auditstore.AuditActive || len(started.Items) != 3 {
		t.Fatalf("start retry = (%+v, %v)", started, err)
	}
	if after := auditBindingRevisions(t, ctx, pool, project.ProjectID); strings.Join(after, ",") != strings.Join(staged, ",") {
		t.Fatalf("start retry wrote Audit revisions %v, staged %v", after, staged)
	}
	stagedSet := map[string]bool{}
	for _, revision := range staged {
		stagedSet[revision] = true
	}
	for _, item := range started.Items {
		if item.Task.Ref.Revision == nil || !stagedSet[item.Task.Ref.Name+"@"+*item.Task.Ref.Revision] {
			t.Fatalf("started item %s names unstaged task %+v", item.ItemID, item.Task.Ref)
		}
	}
	if started.Round.Manifest.Ref.Revision == nil ||
		!stagedSet[started.Round.Manifest.Ref.Name+"@"+*started.Round.Manifest.Ref.Revision] {
		t.Fatalf("started Round names unstaged worklist %+v", started.Round.Manifest.Ref)
	}
}

// TestAuditStartStagingWaitsForPurge covers the AUD-R225 staging fence: start
// stages under FOR KEY SHARE on the draft row, so an Audit purge that holds
// the row either removes what was staged or, as here, removes the Audit
// first; the start then fails without leaving a binding behind.
func TestAuditStartStagingWaitsForPurge(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedAuditServicePool(t, ctx, databaseURL)
	service := newStartStagingService(t, pool, &failingCredentialGuard{})
	project, draft := createStartStagingDraft(t, ctx, pool, service, "purge", 2)

	purge, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer purge.Rollback(context.Background())
	if _, err := purge.Exec(ctx, `SELECT 1 FROM audits WHERE audit_id = $1 FOR UPDATE`, draft.AuditID); err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() {
		_, err := service.Start(ctx, StartParams{
			OwnerID: project.OwnerID, AuditID: draft.AuditID, ExpectedRevision: draft.Revision,
			IdempotencyKey: "start-staging-purge", RequestDigest: serviceTestDigest("start-staging-purge"),
		})
		done <- err
	}()
	waitForStartStagingLock(t, ctx, pool)
	if _, err := purge.Exec(ctx, `DELETE FROM audits WHERE audit_id = $1`, draft.AuditID); err != nil {
		t.Fatal(err)
	}
	if err := purge.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	if err := <-done; !errors.Is(err, auditstore.ErrNotFound) {
		t.Fatalf("start of a purged draft = %v", err)
	}
	if staged := auditBindingRevisions(t, ctx, pool, project.ProjectID); len(staged) != 0 {
		t.Fatalf("start staged %v for a purged Audit", staged)
	}
}

func newStartStagingService(t *testing.T, pool *pgxpool.Pool, guard CredentialReferenceGuard) *Service {
	t.Helper()
	profiles := loadAuditServiceProfiles(t)
	gateway, err := profiles.LLMGateway("test-gateway@1")
	if err != nil {
		t.Fatal(err)
	}
	development, err := managedcredentials.NewStaticProvider([]managedcredentials.StaticEntry{{
		Metadata: resolved.CredentialMetadata{
			Ref:        llmgateway.LLMCredentialRef{CredentialID: "development-worker"},
			LLMGateway: gateway.Ref, Unrestricted: true,
		},
		Token: contracts.NewSecretString("development-test-token"),
	}})
	if err != nil {
		t.Fatal(err)
	}
	transactionCredentials, err := managedcredentials.NewTransactionLookupFactory(development)
	if err != nil {
		t.Fatal(err)
	}
	service, err := New(Options{
		Pool: pool, Profiles: profiles, TransactionLLMCredentials: transactionCredentials,
		CredentialGuard: guard,
	})
	if err != nil {
		t.Fatal(err)
	}
	return service
}

// createStartStagingDraft creates a checklist draft with items automatic
// checks.
func createStartStagingDraft(
	t *testing.T, ctx context.Context, pool *pgxpool.Pool, service *Service, name string, items int,
) (projectstore.Project, auditstore.Audit) {
	return createStartStagingDraftWithSelection(t, ctx, pool, service, name, items, "")
}

func createStartStagingDraftWithSelection(
	t *testing.T, ctx context.Context, pool *pgxpool.Pool, service *Service, name string, items int,
	expectedProfileSHA256 string,
) (projectstore.Project, auditstore.Audit) {
	t.Helper()
	project, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: "project-staging-" + name, OwnerID: "owner-staging-" + name,
		Kind: projectstore.KindProject, Name: "Start staging " + name,
		IdempotencyKey: "project-staging-" + name, RequestDigest: serviceTestDigest("project-staging-" + name),
	})
	if err != nil {
		t.Fatal(err)
	}
	projectArtifacts, err := artifacts.NewService(artifacts.NewPostgresRepository(pool)).Project(project.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	entries := make([]string, items)
	for index := range entries {
		entries[index] = fmt.Sprintf(`{"key":"check-%d","version":"1","statement":"Verify %d.","applicability":"always","allowed_methods":["static"],"required_evidence":[],"review_policy":"automatic"}`, index, index)
	}
	checklist, err := projectArtifacts.Write(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "checklist-" + name},
		artifacts.Payload{MediaType: "application/json", Data: []byte(
			`{"schema":"contractor.audit.checklist.v1","items":[` + strings.Join(entries, ",") + `]}`,
		)}, nil)
	if err != nil {
		t.Fatal(err)
	}
	draft, _, err := service.CreateDraft(ctx, CreateDraftParams{
		AuditID: "audit-staging-" + name, OwnerID: project.OwnerID, ProjectID: project.ProjectID,
		ExpectedProfileSHA256: expectedProfileSHA256,
		Profile:               ProfileSelector{Name: "test-checklist", Version: "1"},
		Inputs:                map[string]contracts.ArtifactRef{"checklist": checklist.Ref},
		Scope:                 Scope{Objective: "Exercise Audit start staging"},
		IdempotencyKey:        "create-staging-" + name, RequestDigest: serviceTestDigest("create-staging-" + name),
	})
	if err != nil {
		t.Fatal(err)
	}
	return project, draft
}

// auditBindingRevisions lists every Audit-managed revision of projectID as
// name@revision.
func auditBindingRevisions(t *testing.T, ctx context.Context, pool *pgxpool.Pool, projectID string) []string {
	t.Helper()
	rows, err := pool.Query(ctx, `
SELECT name || '@' || revision
  FROM artifact_binding_revisions
 WHERE scope_kind = 'project' AND scope_id = $1 AND starts_with(namespace, 'audit-')
 ORDER BY name, revision`, projectID)
	if err != nil {
		t.Fatal(err)
	}
	result := []string{}
	for rows.Next() {
		var revision string
		if err := rows.Scan(&revision); err != nil {
			t.Fatal(err)
		}
		result = append(result, revision)
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	return result
}

// waitForStartStagingLock waits until a start's staging transaction waits on
// the draft row lock.
func waitForStartStagingLock(t *testing.T, ctx context.Context, pool *pgxpool.Pool) {
	t.Helper()
	deadline := time.NewTimer(5 * time.Second)
	defer deadline.Stop()
	for {
		var waiting int
		if err := pool.QueryRow(ctx, `
SELECT count(*)
  FROM pg_stat_activity
 WHERE datname = current_database() AND wait_event_type = 'Lock'
   AND query LIKE '%FOR KEY SHARE%'`).Scan(&waiting); err != nil {
			t.Fatal(err)
		}
		if waiting == 1 {
			return
		}
		select {
		case <-ctx.Done():
			t.Fatal(ctx.Err())
		case <-deadline.C:
			t.Fatal("start staging did not wait for the Audit row lock")
		case <-time.After(10 * time.Millisecond):
		}
	}
}

// failingCredentialGuard fails its first failures critical sections with a
// definite serialization failure before running them.
type failingCredentialGuard struct {
	mu       sync.Mutex
	failures int
}

func (g *failingCredentialGuard) WithRunCreation(ctx context.Context, fn func() error) error {
	g.mu.Lock()
	fail := g.failures > 0
	if fail {
		g.failures--
	}
	g.mu.Unlock()
	if fail {
		return &pgconn.PgError{Code: postgres.SQLStateSerializationFailure, Message: "test transaction conflict"}
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	return fn()
}

// deploymentBoundCatalog models the Manager boundary using an already valid
// fixture policy, retaining the same model while freezing its default binding.
type deploymentBoundCatalog struct{ ProfileCatalog }

func (c deploymentBoundCatalog) ResolveAuditProfile(ctx context.Context, selector string) (auditprofile.ResolvedAuditProfile, error) {
	profile, err := c.ProfileCatalog.ResolveAuditProfile(ctx, selector)
	if err != nil {
		return profile, err
	}
	for role, binding := range profile.Workflows {
		for name, stage := range binding.Workflow.Stages {
			for _, consumer := range stage.ExecutionConfig.Agents {
				stage.ModelDefaults = map[string]contracts.ResolvedModelPolicy{"worker": consumer.ModelPolicy.Clone()}
				break
			}
			binding.Workflow.Stages[name] = stage
		}
		profile.Workflows[role] = binding
	}
	profile.ClosureDigest, err = auditprofile.ProfileClosureDigest(profile)
	return profile, err
}

func TestAuditDraftCatalogSelectionPrecedesDeploymentBinding(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx := t.Context()
	pool := isolatedAuditServicePool(t, ctx, databaseURL)
	service := newStartStagingService(t, pool, &failingCredentialGuard{})
	catalog, err := service.profiles.AuditProfile("test-checklist@1")
	if err != nil {
		t.Fatal(err)
	}
	selected, err := contentdigest.JSON(catalog)
	if err != nil {
		t.Fatal(err)
	}
	service.profiles = deploymentBoundCatalog{ProfileCatalog: service.profiles}
	project, draft := createStartStagingDraftWithSelection(t, ctx, pool, service, "bound-selection", 1, selected)
	restored, err := auditprofile.DecodeResolvedAuditProfileSnapshot(draft.ProfileSnapshot)
	if err != nil || restored.Ref != catalog.Ref || restored.ClosureDigest == catalog.ClosureDigest {
		t.Fatalf("stored bound selection=%+v: %v", restored.Ref, err)
	}
	started, err := service.Start(ctx, StartParams{OwnerID: project.OwnerID, AuditID: draft.AuditID, ExpectedRevision: draft.Revision, IdempotencyKey: "start-bound-selection", RequestDigest: serviceTestDigest("start-bound-selection")})
	if err != nil || started.Audit.State != auditstore.AuditActive {
		t.Fatalf("bound Audit start=%s: %v", started.Audit.State, err)
	}
}
