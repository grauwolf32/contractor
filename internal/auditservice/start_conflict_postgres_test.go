package auditservice

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

// A deadlock abort is as definite a transaction conflict as a serialization
// failure, so Start re-reads idempotency after either one (FAIL-AUD-4) and
// replays the concurrent winner instead of failing.
func TestAuditStartReplaysTheWinnerAfterADeadlockAbort(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	pool := isolatedAuditServicePool(t, ctx, databaseURL)
	profileSnapshot := loadAuditServiceProfiles(t)
	gateway, err := profileSnapshot.LLMGateway("test-gateway@1")
	if err != nil {
		t.Fatal(err)
	}
	credentials := &switchableCredentialLookup{available: true, gateway: gateway.Ref}
	newService := func(lookup func(pgx.Tx) (config.CredentialLookup, error)) *Service {
		service, err := New(Options{
			Pool: pool, Profiles: &switchableProfileCatalog{snapshot: profileSnapshot, available: true},
			TransactionLLMCredentials: runtimeconfig.TransactionLLMCredentialLookupFactoryFunc(lookup),
			CredentialGuard:           &countingCredentialGuard{}, Now: time.Now,
		})
		if err != nil {
			t.Fatal(err)
		}
		return service
	}
	winner := newService(func(pgx.Tx) (config.CredentialLookup, error) { return credentials, nil })

	project, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: "project-start-conflict", OwnerID: "owner-start-conflict", Kind: projectstore.KindProject,
		Name: "Start conflict project", IdempotencyKey: "create-project", RequestDigest: serviceTestDigest("project"),
	})
	if err != nil {
		t.Fatal(err)
	}
	projectArtifacts, err := artifacts.NewService(artifacts.NewPostgresRepository(pool)).Project(project.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	checklist := writeChecklist(t, ctx, projectArtifacts, "automatic", "automatic")
	draft, _, err := winner.CreateDraft(ctx, CreateDraftParams{
		AuditID: "audit-start-conflict", OwnerID: project.OwnerID, ProjectID: project.ProjectID,
		Profile:       ProfileSelector{Name: "test-checklist", Version: "1"},
		Inputs:        map[string]contracts.ArtifactRef{"checklist": checklist.Ref},
		RuntimeLabels: []string{}, Scope: Scope{Objective: "Review the service"},
		IdempotencyKey: "create-audit", RequestDigest: serviceTestDigest("create-audit"),
	})
	if err != nil {
		t.Fatal(err)
	}
	start := StartParams{
		OwnerID: project.OwnerID, AuditID: draft.AuditID, ExpectedRevision: draft.Revision,
		IdempotencyKey: "start-audit", RequestDigest: serviceTestDigest("start-audit"),
	}

	// The loser passed its replay check and holds its snapshot when the winner
	// commits the same start; the loser's transaction then aborts on a deadlock.
	var won StartedAudit
	var winnerErr error
	loser := newService(func(pgx.Tx) (config.CredentialLookup, error) {
		won, winnerErr = winner.Start(ctx, start)
		return nil, &pgconn.PgError{Code: postgres.SQLStateDeadlockDetected}
	})
	replayed, err := loser.Start(ctx, start)
	if winnerErr != nil || won.Replayed || won.Audit.State != "active" {
		t.Fatalf("winner start = (%+v, %v)", won, winnerErr)
	}
	if err != nil || !replayed.Replayed || replayed.Audit.AuditID != draft.AuditID ||
		replayed.Audit.Revision != won.Audit.Revision {
		t.Fatalf("start after a deadlock abort = (%+v, %v), want the winner's replay", replayed, err)
	}
}
