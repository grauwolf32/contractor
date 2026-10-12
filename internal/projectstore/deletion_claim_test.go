package projectstore

import (
	"context"
	"errors"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
)

// TestPostgresProjectDeletionClaimLifecycle checks the deletion claim that the
// Project deletion controller works under: a held claim keeps other
// controllers out, every change checks the claim and its phase and reports a
// lost claim as ErrDeletionClaimLost, a purge phase gets the longer lease, and
// the claim's lock admits the final delete of the Project row.
func TestPostgresProjectDeletionClaimLifecycle(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedProjectPool(t, ctx, databaseURL)
	store := NewPostgresStore(pool)
	lease := func(claimID string) ClaimDeletionParams {
		return ClaimDeletionParams{ClaimID: claimID, Duration: time.Minute, PurgeDuration: time.Hour}
	}
	if _, _, err := store.ClaimDeletion(ctx, ClaimDeletionParams{ClaimID: "claim", Duration: time.Minute}); !errors.Is(err, ErrInvalid) {
		t.Fatalf("claim without a purge lease = %v", err)
	}
	if _, found, err := store.ClaimDeletion(ctx, lease("claim-none")); err != nil || found {
		t.Fatalf("claim without a deleting Project = (%t, %v)", found, err)
	}
	project, _, err := store.Create(ctx, CreateParams{
		ProjectID: "project-claimed", OwnerID: "owner-1", Kind: KindProject, Name: "Claimed",
		IdempotencyKey: "create-claimed", RequestDigest: "sha256:" + strings.Repeat("c", 64),
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := store.BeginDeletion(ctx, BeginDeletionParams{
		ProjectID: project.ProjectID, OwnerID: project.OwnerID, ExpectedRevision: project.Revision,
	}); err != nil {
		t.Fatal(err)
	}

	claim, found, err := store.ClaimDeletion(ctx, lease("claim-a"))
	want := DeletionClaim{ProjectID: project.ProjectID, OwnerID: project.OwnerID, ClaimID: "claim-a", Phase: DeletionCancelling}
	if err != nil || !found || claim != want {
		t.Fatalf("claim = (%+v, %t, %v), want %+v", claim, found, err, want)
	}
	if _, found, err := store.ClaimDeletion(ctx, lease("claim-b")); err != nil || found {
		t.Fatalf("claim of a held Project = (%t, %v)", found, err)
	}
	if err := store.AdvanceDeletion(ctx, claim, DeletionDraining); err != nil {
		t.Fatal(err)
	}
	if err := store.AdvanceDeletion(ctx, claim, DeletionPurgingRuns); !errors.Is(err, ErrDeletionClaimLost) {
		t.Fatalf("advance from a past phase = %v", err)
	}
	claim.Phase = DeletionDraining
	stranger := claim
	stranger.ClaimID = "claim-b"
	for name, change := range map[string]func(DeletionClaim) error{
		"advance": func(claim DeletionClaim) error { return store.AdvanceDeletion(ctx, claim, DeletionPurgingRuns) },
		"defer":   func(claim DeletionClaim) error { return store.DeferDeletionClaim(ctx, claim, time.Second) },
		"release": func(claim DeletionClaim) error { return store.ReleaseDeletionClaim(ctx, claim) },
		"lock": func(claim DeletionClaim) error {
			return pgx.BeginFunc(ctx, pool, func(tx pgx.Tx) error {
				return NewPostgresTxStore(tx).LockDeletionClaim(ctx, claim)
			})
		},
	} {
		if err := change(stranger); !errors.Is(err, ErrDeletionClaimLost) {
			t.Errorf("%s under another claim = %v", name, err)
		}
	}
	if err := store.DeferDeletionClaim(ctx, claim, time.Second); err != nil {
		t.Fatal(err)
	}
	if err := store.AdvanceDeletion(ctx, claim, DeletionPurgingRuns); err != nil {
		t.Fatal(err)
	}
	if err := store.ReleaseDeletionClaim(ctx, claim); err != nil {
		t.Fatal(err)
	}
	if err := store.ReleaseDeletionClaim(ctx, claim); !errors.Is(err, ErrDeletionClaimLost) {
		t.Fatalf("second release = %v", err)
	}

	purge, found, err := store.ClaimDeletion(ctx, lease("claim-purge"))
	if err != nil || !found || purge.Phase != DeletionPurgingRuns {
		t.Fatalf("claim in a purge phase = (%+v, %t, %v)", purge, found, err)
	}
	var leased time.Duration
	if err := pool.QueryRow(ctx, `
SELECT deletion_claim_expires_at - deletion_claimed_at FROM projects WHERE project_id = $1`,
		project.ProjectID).Scan(&leased); err != nil || leased != time.Hour {
		t.Fatalf("purge phase lease = (%v, %v), want the purge lease", leased, err)
	}
	if err := store.AdvanceDeletion(ctx, purge, DeletionPurgingArtifacts); err != nil {
		t.Fatal(err)
	}
	purge.Phase = DeletionPurgingArtifacts
	if err := pgx.BeginFunc(ctx, pool, func(tx pgx.Tx) error {
		projects := NewPostgresTxStore(tx)
		if err := projects.LockDeletionClaim(ctx, purge); err != nil {
			return err
		}
		return projects.DeleteClaimedProject(ctx, purge)
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Get(ctx, project.OwnerID, project.ProjectID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("purged Project lookup = %v", err)
	}
	if err := pgx.BeginFunc(ctx, pool, func(tx pgx.Tx) error {
		return NewPostgresTxStore(tx).DeleteClaimedProject(ctx, purge)
	}); !errors.Is(err, ErrDeletionClaimLost) {
		t.Fatalf("delete of a purged Project = %v", err)
	}
}
