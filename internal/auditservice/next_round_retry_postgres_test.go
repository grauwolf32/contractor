package auditservice

import (
	"context"
	"errors"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
)

func TestPostgresNextRoundRebuildsWorklistAfterFailedAcceptanceAndInboxGrowth(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 90*time.Second)
	defer cancel()
	pool := isolatedAuditServicePool(t, ctx, databaseURL)
	snapshot := loadAuditServiceProfilesWithProfile(t, func(profile string) string {
		profile = strings.Replace(profile, "maxRounds: 1", "maxRounds: 3", 1)
		return strings.Replace(profile, "findingConfirmation: disabled", "findingConfirmation: human-required", 1)
	})
	profiles := &switchableProfileCatalog{snapshot: snapshot, available: true}
	gateway, err := snapshot.LLMGateway("test-gateway@1")
	if err != nil {
		t.Fatal(err)
	}
	credentials := &switchableCredentialLookup{available: true, gateway: gateway.Ref}
	service, err := New(Options{
		Pool: pool, Profiles: profiles, CredentialGuard: &countingCredentialGuard{},
		TransactionLLMCredentials: runtimeconfig.TransactionLLMCredentialLookupFactoryFunc(
			func(pgx.Tx) (config.CredentialLookup, error) { return credentials, nil },
		),
	})
	if err != nil {
		t.Fatal(err)
	}
	const ownerID, projectID, auditID = "round-retry-owner", "round-retry-project", "round-retry-audit"
	project, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: projectID, OwnerID: ownerID, Kind: projectstore.KindProject,
		Name: "Round retry", IdempotencyKey: "round-retry-project", RequestDigest: serviceTestDigest("round-retry-project"),
	})
	if err != nil {
		t.Fatal(err)
	}
	projectArtifacts, err := artifacts.NewService(artifacts.NewPostgresRepository(pool)).Project(project.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	checklist := writeChecklist(t, ctx, projectArtifacts, "round-retry", "automatic")
	draft, _, err := service.CreateDraft(ctx, CreateDraftParams{
		AuditID: auditID, OwnerID: ownerID, ProjectID: projectID,
		Profile:        ProfileSelector{Name: "test-checklist", Version: "1"},
		Inputs:         map[string]contracts.ArtifactRef{"checklist": checklist.Ref},
		IdempotencyKey: "round-retry-draft", RequestDigest: serviceTestDigest("round-retry-draft"),
	})
	if err != nil {
		t.Fatal(err)
	}
	start := StartParams{
		OwnerID: ownerID, AuditID: auditID, ExpectedRevision: draft.Revision,
		IdempotencyKey: "round-retry-start", RequestDigest: serviceTestDigest("round-retry-start"),
	}
	started, err := service.Start(ctx, start)
	if err != nil {
		t.Fatal(err)
	}
	firstWorklist, err := projectArtifacts.Read(ctx, started.Round.Manifest.Ref)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := auditdomain.ValidatePackage(firstWorklist.Payload.Data); err != nil {
		t.Fatalf("first-Round worklist is invalid: %v", err)
	}
	if replay, err := service.Start(ctx, start); err != nil || !replay.Replayed ||
		!replay.Round.Manifest.Ref.SameExact(started.Round.Manifest.Ref) {
		t.Fatalf("first-Round start replay = (%+v, %v)", replay, err)
	}

	store := auditstore.NewPostgresStore(pool)
	claims, err := store.Claim(ctx, auditstore.ClaimParams{HolderID: "round-retry-controller", Lease: time.Minute, Limit: 1})
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
	seedAuditFinding(t, ctx, pool, projectID, ownerID, auditID, "first",
		auditdomain.ProposedCheck{Objective: "Check the first proposal.", Method: "static-trace"})
	live, err := store.Get(ctx, ownerID, auditID)
	if err != nil {
		t.Fatal(err)
	}
	first, reason, err := service.PrepareNextRound(ctx, claim, auditstore.ReconcileSnapshot{Audit: live, Round: &round})
	if err != nil || reason != nil || len(first.Items) != 1 {
		t.Fatalf("first next-Round preparation = (%+v, %+v, %v)", first, reason, err)
	}
	paused, err := service.Pause(ctx, MutationParams{
		OwnerID: ownerID, AuditID: auditID, ExpectedRevision: live.Revision,
		IdempotencyKey: "round-retry-pause", RequestDigest: serviceTestDigest("round-retry-pause"),
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := store.AcceptNextRound(ctx, first); !errors.Is(err, auditstore.ErrPrecondition) {
		t.Fatalf("stale next-Round acceptance = %v, want precondition", err)
	}
	if _, err := service.Resume(ctx, MutationParams{
		OwnerID: ownerID, AuditID: auditID, ExpectedRevision: paused.Audit.Revision,
		IdempotencyKey: "round-retry-resume", RequestDigest: serviceTestDigest("round-retry-resume"),
	}); err != nil {
		t.Fatal(err)
	}
	seedAuditFinding(t, ctx, pool, projectID, ownerID, auditID, "second",
		auditdomain.ProposedCheck{Objective: "Check the second proposal.", Method: "static-trace"})
	live, err = store.Get(ctx, ownerID, auditID)
	if err != nil {
		t.Fatal(err)
	}
	second, reason, err := service.PrepareNextRound(ctx, claim, auditstore.ReconcileSnapshot{Audit: live, Round: &round})
	if err != nil || reason != nil || len(second.Items) != 2 ||
		second.Manifest.Ref.SameExact(first.Manifest.Ref) || second.Manifest.Digest == first.Manifest.Digest {
		t.Fatalf("expanded next-Round preparation = (%+v, %+v, %v)", second, reason, err)
	}
	again, reason, err := service.PrepareNextRound(ctx, claim, auditstore.ReconcileSnapshot{Audit: live, Round: &round})
	if err != nil || reason != nil || !reflect.DeepEqual(second, again) {
		t.Fatalf("unchanged next-Round replay = (%+v, %+v, %v)", again, reason, err)
	}
	accepted, inserted, err := store.AcceptNextRound(ctx, second)
	if err != nil || !inserted || accepted.ExpectedItemCount != 2 || !accepted.Manifest.Ref.SameExact(second.Manifest.Ref) {
		t.Fatalf("expanded next-Round acceptance = (%+v, %t, %v)", accepted, inserted, err)
	}
	var consumed int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM audit_proposal_items WHERE audit_id=$1`, auditID).Scan(&consumed); err != nil || consumed != 2 {
		t.Fatalf("accepted proposal checks = (%d, %v)", consumed, err)
	}
}
