package projectstore

import (
	"encoding/json"
	"errors"
	"os"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/strictjson"
)

func TestSharedProjectActiveChecksPolicy(t *testing.T) {
	var cases []struct {
		Value json.RawMessage
		Valid bool
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "public", "project-active-check-policy.json"), &cases); err != nil {
		t.Fatal(err)
	}
	for _, tc := range cases {
		var policy contracts.ActiveChecksPolicy
		err := strictjson.Decode(tc.Value, &policy)
		if valid := err == nil && policy.Valid(); valid != tc.Valid {
			t.Errorf("policy %s valid=%t, want %t", tc.Value, valid, tc.Valid)
		}
	}
}

func TestPostgresProjectActiveChecksPolicyCASAndDeletion(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx := t.Context()
	pool := isolatedProjectPool(t, ctx, url)
	store := NewPostgresStore(pool)
	p, _, err := store.Create(ctx, CreateParams{ProjectID: "active-policy", OwnerID: "owner", Kind: KindProject, Name: "Policy", IdempotencyKey: "policy", RequestDigest: "sha256:" + strings.Repeat("a", 64)})
	if err != nil || p.ActiveChecks != contracts.ActiveChecksAutomatic {
		t.Fatalf("default=%+v, %v", p, err)
	}
	for _, policy := range []contracts.ActiveChecksPolicy{contracts.ActiveChecksApprovalRequired, contracts.ActiveChecksProhibited, contracts.ActiveChecksAutomatic} {
		updated, err := store.Update(ctx, UpdateParams{ProjectID: p.ProjectID, OwnerID: p.OwnerID, ExpectedRevision: p.Revision, Name: p.Name, ActiveChecks: &policy})
		if err != nil || updated.ActiveChecks != policy || updated.Revision != p.Revision+1 {
			t.Fatalf("update=%+v, %v", updated, err)
		}
		if _, err := store.Update(ctx, UpdateParams{ProjectID: p.ProjectID, OwnerID: p.OwnerID, ExpectedRevision: p.Revision, Name: p.Name, ActiveChecks: &policy}); !errors.Is(err, ErrPrecondition) {
			t.Fatalf("stale update=%v", err)
		}
		p = updated
	}
	invalid := contracts.ActiveChecksPolicy("enabled")
	if _, err := store.Update(ctx, UpdateParams{ProjectID: p.ProjectID, OwnerID: p.OwnerID, ExpectedRevision: p.Revision, Name: p.Name, ActiveChecks: &invalid}); !errors.Is(err, ErrInvalid) {
		t.Fatalf("invalid policy=%v", err)
	}
	p, _, err = store.BeginDeletion(ctx, BeginDeletionParams{ProjectID: p.ProjectID, OwnerID: p.OwnerID, ExpectedRevision: p.Revision})
	if err != nil {
		t.Fatal(err)
	}
	policy := contracts.ActiveChecksProhibited
	if _, err := store.Update(ctx, UpdateParams{ProjectID: p.ProjectID, OwnerID: p.OwnerID, ExpectedRevision: p.Revision, Name: p.Name, ActiveChecks: &policy}); !errors.Is(err, ErrDeleting) {
		t.Fatalf("deleting policy update=%v", err)
	}
	if _, err := pool.Exec(ctx, "UPDATE projects SET active_checks='prohibited', revision=revision+1, updated_at=updated_at+interval '1 second' WHERE project_id=$1", p.ProjectID); err == nil {
		t.Fatal("SQL bypassed deleting Project fence")
	}
}
