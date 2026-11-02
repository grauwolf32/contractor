package runstore

import (
	"context"
	"encoding/json"
	"errors"
	"slices"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestSharedPostgresPlacementLabels(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	var fixture struct {
		Cases []struct {
			Name   string   `json:"name"`
			Labels []string `json:"labels"`
			Valid  bool     `json:"valid"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "placement-label-cases.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	if len(fixture.Cases) == 0 {
		t.Fatal("missing shared placement cases")
	}
	for _, test := range fixture.Cases {
		var valid bool
		if err := pool.QueryRow(ctx, `SELECT contractor_valid_placement_labels($1::text[])`, test.Labels).Scan(&valid); err != nil || valid != test.Valid {
			t.Errorf("%s: database accepted = %t, want %t: %v", test.Name, valid, test.Valid, err)
		}
	}
	for _, expression := range []string{"NULL::text[]", "ARRAY[NULL]::text[]", "'[0:0]={cloud}'::text[]", "ARRAY[['cloud'],['cloud/aws']]"} {
		var valid bool
		if err := pool.QueryRow(ctx, `SELECT contractor_valid_placement_labels(`+expression+`)`).Scan(&valid); err != nil || valid {
			t.Fatalf("invalid array %s accepted = %t, %v", expression, valid, err)
		}
	}
}

func TestPostgresRunPlacementRequirementsAreImmutableAndReadStrictly(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	store := NewPostgresStore(pool, artifactpgtest.NewStorage())
	params := testRunParams("run-location")
	params.RequiredPlacementLabels = []string{"cloud/aws"}
	params.MetadataLabels = map[string]string{"cloud": "aws"}
	created, err := createRun(ctx, store, params)
	if err != nil || !slices.Equal(created.RequiredPlacementLabels, params.RequiredPlacementLabels) {
		t.Fatalf("create = %+v, %v", created, err)
	}
	params.RequiredPlacementLabels[0] = "local/lab"
	loaded, err := NewPostgresStore(pool, artifactpgtest.NewStorage()).GetRun(ctx, created.RunID)
	if err != nil || !slices.Equal(loaded.RequiredPlacementLabels, []string{"cloud/aws"}) {
		t.Fatalf("fresh store read = %+v, %v", loaded, err)
	}
	if _, err := pool.Exec(ctx, `UPDATE workflow_runs SET required_placement_labels = ARRAY['local/lab'] WHERE run_id = $1`, created.RunID); err == nil {
		t.Fatal("database allowed changing a pinned requirement")
	}
	if _, err := store.TransitionRun(ctx, created.RunID, RunInitializing, RunPending, Reason{Code: "initialized"}); err != nil {
		t.Fatal(err)
	}
	if _, err := store.RequestRunCancellation(ctx, created.RunID, WorkflowRunCancellation{Code: CancellationUserRequested, RequestedAt: time.Now().UTC()}); err != nil {
		t.Fatal(err)
	}
	loaded, err = store.GetRun(ctx, created.RunID)
	if err != nil || !slices.Equal(loaded.RequiredPlacementLabels, []string{"cloud/aws"}) {
		t.Fatalf("cancellation changed placement = %+v, %v", loaded, err)
	}
	bad := testRunParams("invalid-location")
	bad.RequiredPlacementLabels = []string{"cloud", "local"}
	if _, err := createRun(ctx, store, bad); !errors.Is(err, ErrInvalid) {
		t.Fatalf("inconsistent location error = %v", err)
	}
	// A corrupted row must fail closed even after database guards are removed
	// in this isolated schema. A reader must never replace it with unrestricted.
	if _, err := pool.Exec(ctx, `ALTER TABLE workflow_runs DROP CONSTRAINT workflow_runs_placement_labels_valid`); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `ALTER TABLE workflow_runs DISABLE TRIGGER workflow_runs_protect_placement_labels`); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `UPDATE workflow_runs SET required_placement_labels = ARRAY['cloud/*'] WHERE run_id = $1`, created.RunID); err != nil {
		t.Fatal(err)
	}
	if _, err := store.GetRun(ctx, created.RunID); err == nil {
		t.Fatal("reader accepted corrupted placement requirements")
	}
}
