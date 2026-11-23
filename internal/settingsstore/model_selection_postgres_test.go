package settingsstore

import (
	"context"
	"errors"
	"maps"
	"os"
	"reflect"
	"sync"
	"testing"
	"time"

	pg "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestSharedPostgresModelSelectionSeedCASRestartAndTransaction(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool, reopen := isolatedSettingsPool(t, ctx, url)
	store := NewPostgresStore(pool)
	a := ModelSelectionValue{DefaultModel: "fixture-a", Roles: map[string]string{}}
	initial, err := store.InitializeModelSelection(ctx, a)
	if err != nil || initial.Revision != 1 || initial.UpdatedAt.IsZero() {
		t.Fatalf("initialize: %+v %v", initial, err)
	}
	seeded, err := store.InitializeModelSelection(ctx, ModelSelectionValue{DefaultModel: "fixture-b", Roles: map[string]string{"summarizer": "fixture-a"}})
	if err != nil || !reflect.DeepEqual(seeded, initial) {
		t.Fatalf("restart bootstrap overwrote selection: %+v %v", seeded, err)
	}
	noop, err := store.UpdateModelSelection(ctx, a, 1)
	if err != nil || !reflect.DeepEqual(noop, initial) {
		t.Fatalf("no-op changed selection: %+v %v", noop, err)
	}
	for _, value := range []ModelSelectionValue{
		{DefaultModel: "", Roles: map[string]string{}}, {DefaultModel: "fixture-a", Roles: nil},
		{DefaultModel: "fixture-a", Roles: map[string]string{"other": "fixture-b"}},
		{DefaultModel: "fixture-a", Roles: map[string]string{"worker": ""}},
	} {
		if _, err := store.UpdateModelSelection(ctx, value, 1); !errors.Is(err, ErrModelSelectionInvalid) {
			t.Fatalf("invalid value accepted: %v", err)
		}
	}
	for _, roles := range []string{`{"worker":null}`, `{"worker":42}`, `{"unknown":"fixture-a"}`, `[]`} {
		_, err := pool.Exec(ctx, `UPDATE model_selection_settings SET role_models=$1::jsonb,revision=revision+1,updated_at=updated_at+interval '1 microsecond' WHERE singleton=true`, roles)
		if pg.SQLState(err) != pg.SQLStateCheckViolation {
			t.Fatalf("invalid SQL roles %s accepted: %v", roles, err)
		}
	}
	cases := sharedModelProfileIDCases(t)
	for _, group := range []struct {
		values []string
		valid  bool
	}{{cases.Valid, true}, {cases.Invalid, false}} {
		for _, value := range group.values {
			for _, query := range []string{`SELECT contractor_valid_model_selection($1, '{}'::jsonb)`, `SELECT contractor_valid_model_selection('fixture', jsonb_build_object('worker',$1::text))`} {
				var valid bool
				if err := pool.QueryRow(ctx, query, value).Scan(&valid); err != nil || valid != group.valid {
					t.Fatalf("SQL model ID %q valid=%v, want %v: %v", value, valid, group.valid, err)
				}
			}
		}
	}
	start := make(chan struct{})
	var group sync.WaitGroup
	results := make([]ModelSelectionSettings, 2)
	failures := make([]error, 2)
	for i, id := range []string{"fixture-b", "fixture-c"} {
		group.Add(1)
		go func(i int, id string) {
			defer group.Done()
			<-start
			results[i], failures[i] = store.UpdateModelSelection(ctx, ModelSelectionValue{DefaultModel: id, Roles: map[string]string{"summarizer": "fixture-a"}}, 1)
		}(i, id)
	}
	close(start)
	group.Wait()
	winners, conflicts := 0, 0
	var saved ModelSelectionSettings
	for i, err := range failures {
		if err == nil {
			winners++
			saved = results[i]
		} else if errors.Is(err, ErrPrecondition) {
			conflicts++
		} else {
			t.Fatalf("concurrent update: %v", err)
		}
	}
	if winners != 1 || conflicts != 1 || saved.Revision != 2 {
		t.Fatalf("CAS results: %+v %v", results, failures)
	}
	restarted := reopen(t, ctx)
	defer restarted.Close()
	current, err := NewPostgresStore(restarted).GetModelSelection(ctx)
	if err != nil || !reflect.DeepEqual(current, saved) {
		t.Fatalf("restart lost selection: %+v %v", current, err)
	}
	if _, err := store.UpdateModelSelection(ctx, saved.Value, 1); !errors.Is(err, ErrPrecondition) {
		t.Fatalf("stale replay overwrote state: %v", err)
	}
	if _, err := pool.Exec(ctx, `DELETE FROM model_selection_settings`); pg.SQLState(err) != pg.SQLStateCheckViolation {
		t.Fatalf("singleton deletion: %v", err)
	}
	config := pool.Config().Copy()
	config.MaxConns = 1
	single, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		t.Fatal(err)
	}
	defer single.Close()
	if err := pgx.BeginFunc(ctx, single, func(tx pgx.Tx) error {
		bound := WithModelSelectionTransaction(ctx, tx)
		value, err := NewPostgresStore(single).GetModelSelection(bound)
		if err == nil && (!maps.Equal(value.Value.Roles, saved.Value.Roles) || value.Value.DefaultModel != saved.Value.DefaultModel) {
			t.Fatal("transaction model selection differs")
		}
		return err
	}); err != nil {
		t.Fatalf("model selection acquired another connection inside a transaction: %v", err)
	}
}
