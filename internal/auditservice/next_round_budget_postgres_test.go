package auditservice

import (
	"context"
	"fmt"
	"os"
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

// TestPostgresNextRoundSpreadsLargeProposalsAndReportsUnschedulable holds
// proposals that together exceed one 8 MiB proposal inventory and one that
// cannot fit any inventory. Later Rounds split the former and skip the
// latter, which the final closure reason names.
func TestPostgresNextRoundSpreadsLargeProposalsAndReportsUnschedulable(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 180*time.Second)
	defer cancel()
	pool := isolatedAuditServicePool(t, ctx, databaseURL)
	snapshot := loadAuditServiceProfilesWithProfile(t, func(profile string) string {
		profile = strings.Replace(profile, "maxRounds: 1", "maxRounds: 4", 1)
		return strings.Replace(profile, "findingConfirmation: disabled", "findingConfirmation: human-required", 1)
	})
	gateway, err := snapshot.LLMGateway("test-gateway@1")
	if err != nil {
		t.Fatal(err)
	}
	credentials := &switchableCredentialLookup{available: true, gateway: gateway.Ref}
	service, err := New(Options{
		Pool: pool, Profiles: &switchableProfileCatalog{snapshot: snapshot, available: true},
		CredentialGuard: &countingCredentialGuard{},
		TransactionLLMCredentials: runtimeconfig.TransactionLLMCredentialLookupFactoryFunc(
			func(pgx.Tx) (config.CredentialLookup, error) { return credentials, nil },
		),
	})
	if err != nil {
		t.Fatal(err)
	}
	const ownerID, projectID, auditID = "round-budget-owner", "round-budget-project", "round-budget-audit"
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: projectID, OwnerID: ownerID, Kind: projectstore.KindProject,
		Name: "Round budget", IdempotencyKey: "round-budget-project", RequestDigest: serviceTestDigest("round-budget-project"),
	}); err != nil {
		t.Fatal(err)
	}
	projectArtifacts, err := artifacts.NewService(artifacts.NewPostgresRepository(pool)).Project(projectID)
	if err != nil {
		t.Fatal(err)
	}
	checklist := writeChecklist(t, ctx, projectArtifacts, "round-budget", "automatic")
	draft, _, err := service.CreateDraft(ctx, CreateDraftParams{
		AuditID: auditID, OwnerID: ownerID, ProjectID: projectID,
		Profile:        ProfileSelector{Name: "test-checklist", Version: "1"},
		Inputs:         map[string]contracts.ArtifactRef{"checklist": checklist.Ref},
		IdempotencyKey: "round-budget-draft", RequestDigest: serviceTestDigest("round-budget-draft"),
	})
	if err != nil {
		t.Fatal(err)
	}
	started, err := service.Start(ctx, StartParams{
		OwnerID: ownerID, AuditID: auditID, ExpectedRevision: draft.Revision,
		IdempotencyKey: "round-budget-start", RequestDigest: serviceTestDigest("round-budget-start"),
	})
	if err != nil {
		t.Fatal(err)
	}
	store := auditstore.NewPostgresStore(pool)
	claims, err := store.Claim(ctx, auditstore.ClaimParams{HolderID: "round-budget-controller", Lease: time.Minute, Limit: 1})
	if err != nil || len(claims) != 1 {
		t.Fatalf("claim = (%+v, %v)", claims, err)
	}
	claim := claims[0]
	round := closeRoundForTest(t, ctx, store, claim, started.Round)

	check := auditdomain.ProposedCheck{Objective: "Confirm the reported condition.", Method: "static-trace"}
	for _, suffix := range []string{"large-a", "large-b", "oversized", "large-c"} {
		document := auditFindingDocument(suffix, check)
		size := 3 << 20
		if suffix == "oversized" {
			// A valid proposal at the document limit leaves no room for its
			// receipt and exact artifact identity in any inventory.
			size = auditdomain.MaximumDocumentBytes - 16
		}
		padFindingPreconditions(t, &document, size)
		seedAuditFindingDocument(t, ctx, pool, projectID, ownerID, auditID, suffix, document)
	}

	var receipts [][]string
	for range 2 {
		live, err := store.Get(ctx, ownerID, auditID)
		if err != nil {
			t.Fatal(err)
		}
		params, reason, err := service.PrepareNextRound(ctx, claim, auditstore.ReconcileSnapshot{Audit: live, Round: &round})
		if err != nil || reason != nil || params.RoundID == "" {
			t.Fatalf("Round %d preparation = (%+v, %+v, %v)", round.Ordinal+1, params.RoundID, reason, err)
		}
		var sources []string
		for _, item := range params.Items {
			sources = append(sources, item.ProposalSources[0].ReceiptID)
		}
		receipts = append(receipts, sources)
		accepted, inserted, err := store.AcceptNextRound(ctx, params)
		if err != nil || !inserted {
			t.Fatalf("accept Round %d = (%t, %v)", params.RoundOrdinal, inserted, err)
		}
		round = closeRoundForTest(t, ctx, store, claim, accepted)
	}
	if strings.Join(receipts[0], ",") != "receipt-large-a,receipt-large-b" ||
		strings.Join(receipts[1], ",") != "receipt-large-c" {
		t.Fatalf("later Round proposal sources = %v", receipts)
	}
	live, err := store.Get(ctx, ownerID, auditID)
	if err != nil {
		t.Fatal(err)
	}
	params, reason, err := service.PrepareNextRound(ctx, claim, auditstore.ReconcileSnapshot{Audit: live, Round: &round})
	if err != nil || params.RoundID != "" || reason == nil || reason.Code != "proposal_inventory_limit_exceeded" ||
		!strings.Contains(reason.Message, "receipt-oversized (proposal_inventory.bytes)") {
		t.Fatalf("closure with only an unschedulable proposal = (%+v, %+v, %v)", params.RoundID, reason, err)
	}
}

func TestPostgresNextRoundPreparesNineNearLimitProposals(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 180*time.Second)
	defer cancel()
	pool := isolatedAuditServicePool(t, ctx, databaseURL)
	snapshot := loadAuditServiceProfilesWithProfile(t, func(profile string) string {
		profile = strings.Replace(profile, "maxRounds: 1", "maxRounds: 4", 1)
		return strings.Replace(profile, "findingConfirmation: disabled", "findingConfirmation: human-required", 1)
	})
	gateway, err := snapshot.LLMGateway("test-gateway@1")
	if err != nil {
		t.Fatal(err)
	}
	credentials := &switchableCredentialLookup{available: true, gateway: gateway.Ref}
	service, err := New(Options{
		Pool: pool, Profiles: &switchableProfileCatalog{snapshot: snapshot, available: true},
		CredentialGuard: &countingCredentialGuard{},
		TransactionLLMCredentials: runtimeconfig.TransactionLLMCredentialLookupFactoryFunc(
			func(pgx.Tx) (config.CredentialLookup, error) { return credentials, nil },
		),
	})
	if err != nil {
		t.Fatal(err)
	}
	const ownerID, projectID, auditID = "round-budget-owner", "round-budget-project", "round-budget-audit"
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: projectID, OwnerID: ownerID, Kind: projectstore.KindProject,
		Name: "Round budget", IdempotencyKey: "round-budget-project", RequestDigest: serviceTestDigest("round-budget-project"),
	}); err != nil {
		t.Fatal(err)
	}
	projectArtifacts, err := artifacts.NewService(artifacts.NewPostgresRepository(pool)).Project(projectID)
	if err != nil {
		t.Fatal(err)
	}
	checklist := writeChecklist(t, ctx, projectArtifacts, "round-budget", "automatic")
	draft, _, err := service.CreateDraft(ctx, CreateDraftParams{
		AuditID: auditID, OwnerID: ownerID, ProjectID: projectID,
		Profile:        ProfileSelector{Name: "test-checklist", Version: "1"},
		Inputs:         map[string]contracts.ArtifactRef{"checklist": checklist.Ref},
		IdempotencyKey: "round-budget-draft", RequestDigest: serviceTestDigest("round-budget-draft"),
	})
	if err != nil {
		t.Fatal(err)
	}
	started, err := service.Start(ctx, StartParams{
		OwnerID: ownerID, AuditID: auditID, ExpectedRevision: draft.Revision,
		IdempotencyKey: "round-budget-start", RequestDigest: serviceTestDigest("round-budget-start"),
	})
	if err != nil {
		t.Fatal(err)
	}
	store := auditstore.NewPostgresStore(pool)
	claims, err := store.Claim(ctx, auditstore.ClaimParams{HolderID: "round-budget-controller", Lease: time.Minute, Limit: 1})
	if err != nil || len(claims) != 1 {
		t.Fatalf("claim = (%+v, %v)", claims, err)
	}
	claim := claims[0]
	round := closeRoundForTest(t, ctx, store, claim, started.Round)

	// Nine exact documents near 8 MiB exceed the 64 MiB read ceiling as a
	// group. Each still fits an otherwise empty proposal inventory.
	check := auditdomain.ProposedCheck{Objective: "Confirm the reported condition.", Method: "static-trace"}
	for n := range 9 {
		suffix := fmt.Sprintf("near-limit-%d", n)
		document := auditFindingDocument(suffix, check)
		padFindingPreconditions(t, &document, auditdomain.MaximumDocumentBytes-8192)
		seedAuditFindingDocument(t, ctx, pool, projectID, ownerID, auditID, suffix, document)
	}
	live, err := store.Get(ctx, ownerID, auditID)
	if err != nil {
		t.Fatal(err)
	}
	params, reason, err := service.PrepareNextRound(ctx, claim, auditstore.ReconcileSnapshot{Audit: live, Round: &round})
	if err != nil || reason != nil || params.RoundID == "" || len(params.Items) != 1 {
		t.Fatalf("next Round from nine near-limit proposals: id=%s items=%d reason=%+v err=%v", params.RoundID, len(params.Items), reason, err)
	}
	if _, inserted, err := store.AcceptNextRound(ctx, params); err != nil || !inserted {
		t.Fatalf("accept prepared next Round: inserted=%t error=%v", inserted, err)
	}
}

func closeRoundForTest(
	t *testing.T, ctx context.Context, store *auditstore.PostgresStore,
	claim auditstore.ControllerClaim, round auditstore.Round,
) auditstore.Round {
	t.Helper()
	for _, target := range []auditstore.RoundState{auditstore.RoundExecuting, auditstore.RoundAssessing, auditstore.RoundClosed} {
		var err error
		round, err = store.TransitionRound(ctx, auditstore.RoundTransitionParams{
			Claim: claim, RoundID: round.RoundID, ExpectedRevision: round.Revision,
			ExpectedState: round.State, TargetState: target,
		})
		if err != nil {
			t.Fatalf("close Round %d as %s: %v", round.Ordinal, target, err)
		}
	}
	return round
}

// padFindingPreconditions fills preconditions until the encoded proposal is
// exactly size bytes. Plain letters encode one byte each; a precondition also
// costs its quotes and, after the first, a separator.
func padFindingPreconditions(t *testing.T, document *auditdomain.FindingProposal, size int) {
	t.Helper()
	const chunk = 60_000
	encoded, err := auditdomain.EncodeFindingProposal(*document)
	if err != nil {
		t.Fatal(err)
	}
	for missing := size - len(encoded); missing > 0; {
		cost := 3
		if len(document.Preconditions) == 0 {
			cost = 2
		}
		length := min(missing-cost, chunk)
		if rest := missing - cost - length; rest > 0 && rest < 4 {
			length -= 4 // Leave room for one more non-empty precondition.
		}
		if length < 1 {
			t.Fatalf("cannot pad the proposal by %d bytes", missing)
		}
		document.Preconditions = append(document.Preconditions, strings.Repeat("p", length))
		missing -= length + cost
	}
	if encoded, err = auditdomain.EncodeFindingProposal(*document); err != nil || len(encoded) != size {
		t.Fatalf("padded proposal = %d bytes, %v; want %d", len(encoded), err, size)
	}
}
