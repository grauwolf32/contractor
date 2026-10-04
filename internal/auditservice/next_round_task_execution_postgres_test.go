package auditservice

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
)

func TestPostgresNextRoundStopsBeforeWritingUnsupportedScanTasks(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 90*time.Second)
	defer cancel()
	pool := isolatedAuditServicePool(t, ctx, databaseURL)
	root := t.TempDir()
	if err := os.CopyFS(root, os.DirFS(auditServiceCatalogFixture)); err != nil {
		t.Fatal(err)
	}
	profilePath := filepath.Join(root, "audit-profiles", "openapi_sqlmap_scan.yaml")
	profileYAML, err := os.ReadFile(profilePath)
	if err != nil {
		t.Fatal(err)
	}
	modified := strings.Replace(string(profileYAML), "maxRounds: 1", "maxRounds: 2", 1)
	modified = strings.Replace(modified, "findingConfirmation: disabled", "findingConfirmation: human-required", 1)
	if modified == string(profileYAML) {
		t.Fatal("scan profile fixture did not enable later finding-confirmation Rounds")
	}
	if err := os.WriteFile(profilePath, []byte(modified), 0o600); err != nil {
		t.Fatal(err)
	}
	snapshot, err := config.Load(root, config.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	profile, err := snapshot.AuditProfile("openapi-sqlmap-scan@1")
	if err != nil || !ProfileCompatibility(profile).ServerCompatible {
		t.Fatalf("scan profile must be selectable before next-Round task validation: %v", err)
	}
	lookup, err := credentials.NewStaticProvider(nil)
	if err != nil {
		t.Fatal(err)
	}
	service, err := New(Options{
		Pool: pool, Profiles: &switchableProfileCatalog{snapshot: snapshot, available: true},
		CredentialGuard: &countingCredentialGuard{},
		TransactionLLMCredentials: runtimeconfig.TransactionLLMCredentialLookupFactoryFunc(
			func(pgx.Tx) (config.CredentialLookup, error) { return lookup, nil },
		),
	})
	if err != nil {
		t.Fatal(err)
	}
	const ownerID, projectID, auditID = "scan-round-owner", "scan-round-project", "scan-round-audit"
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: projectID, OwnerID: ownerID, Kind: projectstore.KindProject,
		Name: "Scan next Round", IdempotencyKey: "scan-round-project", RequestDigest: serviceTestDigest("scan-round-project"),
	}); err != nil {
		t.Fatal(err)
	}
	projectArtifacts, err := artifacts.NewService(artifacts.NewPostgresRepository(pool)).Project(projectID)
	if err != nil {
		t.Fatal(err)
	}
	inputs := make(map[string]contracts.ArtifactRef)
	for name, file := range map[string]string{
		"openapi": "openapi.json", "settings": "sqlmap-settings.json",
	} {
		payload, err := os.ReadFile(filepath.Join(openAPIScanInputFixture, file))
		if err != nil {
			t.Fatal(err)
		}
		written, err := projectArtifacts.Write(ctx,
			contracts.ArtifactRef{Namespace: "inputs", Name: name},
			artifacts.Payload{MediaType: auditdomain.JSONMediaType, Data: payload}, nil)
		if err != nil {
			t.Fatal(err)
		}
		inputs[name] = written.Ref
	}
	draft, _, err := service.CreateDraft(ctx, CreateDraftParams{
		AuditID: auditID, OwnerID: ownerID, ProjectID: projectID,
		Profile: ProfileSelector{Name: "openapi-sqlmap-scan", Version: "1"}, Inputs: inputs,
		IdempotencyKey: "scan-round-draft", RequestDigest: serviceTestDigest("scan-round-draft"),
	})
	if err != nil {
		t.Fatal(err)
	}
	started, err := service.Start(ctx, StartParams{
		OwnerID: ownerID, AuditID: auditID, ExpectedRevision: draft.Revision,
		IdempotencyKey: "scan-round-start", RequestDigest: serviceTestDigest("scan-round-start"),
	})
	if err != nil {
		t.Fatal(err)
	}
	store := auditstore.NewPostgresStore(pool)
	claims, err := store.Claim(ctx, auditstore.ClaimParams{HolderID: "scan-round-controller", Lease: time.Minute, Limit: 1})
	if err != nil || len(claims) != 1 {
		t.Fatalf("claim = (%+v, %v)", claims, err)
	}
	claim := claims[0]
	round := started.Round
	for _, target := range []auditstore.RoundState{auditstore.RoundExecuting, auditstore.RoundAssessing, auditstore.RoundClosed} {
		round, err = store.TransitionRound(ctx, auditstore.RoundTransitionParams{
			Claim: claim, RoundID: round.RoundID, ExpectedRevision: round.Revision,
			ExpectedState: round.State, TargetState: target,
		})
		if err != nil {
			t.Fatalf("close fixture Round as %s: %v", target, err)
		}
	}
	seedAuditFinding(t, ctx, pool, projectID, ownerID, auditID, "unsupported-scan",
		auditdomain.ProposedCheck{Objective: "Trace access control.", Method: "static-trace"})
	namespace := auditdomain.ArtifactNamespace(auditID)
	var before, after int
	countTasks := `SELECT count(*) FROM artifact_bindings WHERE scope_kind='project' AND scope_id=$1 AND namespace=$2 AND name LIKE 'task-%'`
	if err := pool.QueryRow(ctx, countTasks, projectID, namespace).Scan(&before); err != nil {
		t.Fatal(err)
	}
	live, err := store.Get(ctx, ownerID, auditID)
	if err != nil {
		t.Fatal(err)
	}
	params, reason, err := service.PrepareNextRound(ctx, claim, auditstore.ReconcileSnapshot{Audit: live, Round: &round})
	if err != nil || params.RoundID != "" || reason == nil || reason.Code != "next_round_task_unsupported" {
		t.Fatalf("unsupported next Round = (%+v, %+v, %v)", params, reason, err)
	}
	if err := pool.QueryRow(ctx, countTasks, projectID, namespace).Scan(&after); err != nil || after != before {
		t.Fatalf("next-Round task packages changed from %d to %d: %v", before, after, err)
	}
	rounds, err := store.ListRounds(ctx, auditID)
	if err != nil || len(rounds) != 1 || rounds[0].RoundID != started.Round.RoundID {
		t.Fatalf("unsupported next Round was admitted: %+v, %v", rounds, err)
	}
}
