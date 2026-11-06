package evalstore

import (
	"bytes"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	pg "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5/pgxpool"
)

type readinessCase struct {
	Name       string                `json:"name"`
	Plan       json.RawMessage       `json:"plan"`
	Comparison evaldomain.Comparison `json:"comparison"`
	Readiness  json.RawMessage       `json:"readiness"`
	ErrorCode  string                `json:"errorCode"`
}

// bypassPlanImmutability runs statement on eval_experiments with
// eval_plans_immutable disabled, which only DDL can do.
func bypassPlanImmutability(t *testing.T, pool *pgxpool.Pool, statement string, args ...any) error {
	t.Helper()
	ctx := t.Context()
	if _, err := pool.Exec(ctx, `ALTER TABLE eval_experiments DISABLE TRIGGER eval_plans_immutable; ALTER TABLE eval_experiments DISABLE TRIGGER eval_experiments_protect`); err != nil {
		t.Fatal(err)
	}
	_, err := pool.Exec(ctx, statement, args...)
	if _, enable := pool.Exec(ctx, `ALTER TABLE eval_experiments ENABLE TRIGGER eval_plans_immutable; ALTER TABLE eval_experiments ENABLE TRIGGER eval_experiments_protect`); enable != nil {
		t.Fatal(enable)
	}
	return err
}

func checkViolation(err error) bool {
	return pg.SQLState(err) == pg.SQLStateCheckViolation
}

// The freeze stores the readiness of each shared fixture plan exactly, and
// FrozenPlanView returns it with the plan metadata in one statement that does
// not select the document. A plan the fixture marks rejected never reaches
// the store: the freeze takes only a document Freeze accepted. An external
// registration stores none.
func TestSharedPostgresFreezeStoresNativeReadinessForTheView(t *testing.T) {
	var fixtures struct {
		Cases []readinessCase `json:"cases"`
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "evals", "readiness-cases.json"), &fixtures); err != nil {
		t.Fatal(err)
	}
	for _, c := range fixtures.Cases {
		t.Run(c.Name, func(t *testing.T) {
			if c.ErrorCode != "" {
				if _, err := evaldomain.Freeze("playground.plan/v1", c.Plan); !code(err, c.ErrorCode) {
					t.Fatalf("Freeze = %v, want %s", err, c.ErrorCode)
				}
				return
			}
			pool := testPool(t)
			ctx := t.Context()
			f := newNativeFreeze(t, pool)
			plan, err := evaldomain.Freeze("playground.plan/v1", c.Plan)
			if err != nil {
				t.Fatal(err)
			}
			var setup map[string]any
			if err = json.Unmarshal(f.setup, &setup); err != nil {
				t.Fatal(err)
			}
			setup["comparison"] = c.Comparison
			f.plan, f.setup = plan, bytesOf(setup)
			mustTx(t, pool, func(s *Store) error { return f.freeze(ctx, s) })
			var want evaldomain.Readiness
			if err = strictjson.Decode(c.Readiness, &want); err != nil {
				t.Fatal(err)
			}
			trace := &queryCounter{}
			config := pool.Config()
			config.ConnConfig.Tracer = trace
			traced, err := pgxpool.NewWithConfig(ctx, config)
			if err != nil {
				t.Fatal(err)
			}
			defer traced.Close()
			view, err := NewPostgresStore(traced, artifactpgtest.NewStorage()).FrozenPlanView(ctx, f.scope.OwnerID, f.e.ID)
			if err != nil {
				t.Fatal(err)
			}
			if queries := trace.take(); queries != 1 {
				t.Fatalf("FrozenPlanView issued %d queries", queries)
			}
			if view.Readiness == nil || !reflect.DeepEqual(*view.Readiness, want) {
				t.Fatalf("stored readiness:\n got %+v\nwant %+v", view.Readiness, want)
			}
			metadata, err := NewPostgresStore(pool, artifactpgtest.NewStorage()).FrozenPlanMetadata(ctx, f.scope.OwnerID, f.e.ID)
			if err != nil || view.PlanMetadata.SHA256 != metadata.SHA256 || view.SHA256 != plan.Digest() || !bytes.Equal(view.Setup, metadata.Setup) {
				t.Fatalf("view metadata %+v differs from %+v: %v", view.PlanMetadata, metadata, err)
			}
			if _, err = NewPostgresStore(pool, artifactpgtest.NewStorage()).FrozenPlanView(ctx, "other-owner", f.e.ID); !code(err, "eval_not_found") {
				t.Fatalf("plan view crossed the owner boundary: %v", err)
			}
		})
	}
	t.Run("external", func(t *testing.T) {
		pool := testPool(t)
		scope := setupProject(t, pool, "owner", "eval")
		e := createExperiment(t, pool, scope, "exp", "trace-1", "external-workflow")
		view, err := NewPostgresStore(pool, artifactpgtest.NewStorage()).FrozenPlanView(t.Context(), e.OwnerID, e.ID)
		if err != nil || view.Readiness != nil || view.SHA256 != planDigest(t, pool, e) {
			t.Fatalf("external plan view %+v: %v", view, err)
		}
	})
}

// Stored readiness is immutable with its plan, present exactly for native
// plans and bounded by the column check. A copy that does not decode strictly
// fails the read as an internal error, not as a client-facing Eval error.
func TestSharedPostgresStoredReadinessIsImmutableAndDecodedStrictly(t *testing.T) {
	pool := testPool(t)
	ctx := t.Context()
	f := newNativeFreeze(t, pool)
	mustTx(t, pool, func(s *Store) error { return f.freeze(ctx, s) })
	external := createExperiment(t, pool, setupProject(t, pool, "owner", "external"), "external", "trace-1", "external-workflow")
	if _, err := pool.Exec(ctx, `UPDATE eval_experiments SET plan_readiness=plan_readiness WHERE experiment_id=$1`, f.e.ID); !checkViolation(err) || !strings.Contains(err.Error(), "immutable") {
		t.Fatalf("stored readiness changed: %v", err)
	}
	for name, update := range map[string]struct {
		id, value string
	}{
		"native without readiness": {f.e.ID, ""},
		"external with readiness":  {external.ID, `{"pins":[],"arms":[]}`},
		"empty readiness":          {f.e.ID, "{"},
	} {
		value := []byte(update.value)
		if update.value == "" {
			value = nil
		}
		if err := bypassPlanImmutability(t, pool, `UPDATE eval_experiments SET plan_readiness=$2 WHERE experiment_id=$1`, update.id, value); !checkViolation(err) {
			t.Fatalf("%s: %v", name, err)
		}
	}
	for _, malformed := range []string{
		`not JSON`,
		`{"pins":[],"arms":[],"extra":1}`,
		`{"pins":[]}`,
		`{"pins":[],"arms":[]} {}`,
		`{"pins":[],"pins":[],"arms":[]}`,
		`{"pins":[{"dimension":"source","requiredEqual":true,"baselineOrigin":"observed","candidateOrigin":"observed"}],"arms":[]}`,
	} {
		if err := bypassPlanImmutability(t, pool, `UPDATE eval_experiments SET plan_readiness=$2 WHERE experiment_id=$1`, f.e.ID, []byte(malformed)); err != nil {
			t.Fatal(err)
		}
		_, err := NewPostgresStore(pool, artifactpgtest.NewStorage()).FrozenPlanView(ctx, f.scope.OwnerID, f.e.ID)
		var domain *evaldomain.Error
		if err == nil || errors.As(err, &domain) {
			t.Fatalf("malformed readiness %s read as %v", malformed, err)
		}
	}
}
