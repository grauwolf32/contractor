package runstore

import (
	"errors"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
)

func TestPostgresEngagementCredentialDependenciesAreBoundedAndImmutable(t *testing.T) {
	ctx := t.Context()
	pool := isolatedRunStorePool(t, ctx)
	store := NewPostgresStore(pool, artifactpgtest.NewStorage())
	cases := []struct {
		name  string
		ids   []string
		valid bool
	}{
		{"empty", []string{}, true},
		{"maximum", []string{"a", "b", "c", "d", "e", "f", "g", "h"}, true},
		{"over", []string{"a", "b", "c", "d", "e", "f", "g", "h", "i"}, false},
		{"duplicate", []string{"account", "account"}, false},
		{"unsorted", []string{"z", "a"}, false},
		{"blank", []string{""}, false},
		{"wrong-grammar", []string{"Account"}, false},
		{"oversized", []string{strings.Repeat("a", 129)}, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var valid bool
			if err := pool.QueryRow(ctx, `SELECT contractor_valid_engagement_credential_ids($1::text[])`, tc.ids).Scan(&valid); err != nil || valid != tc.valid {
				t.Fatalf("database validation: %t %v", valid, err)
			}
			params := testRunParams("engagement-deps-" + tc.name)
			params.EngagementCredentialIDs = tc.ids
			run, err := createRun(ctx, store, params)
			if !tc.valid {
				if !errors.Is(err, ErrInvalid) {
					t.Fatalf("invalid dependency set: %v", err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if _, err := pool.Exec(ctx, `UPDATE workflow_runs SET engagement_credential_ids=ARRAY['changed'] WHERE run_id=$1`, run.RunID); err == nil {
				t.Fatal("immutable references changed")
			}
		})
	}
}
