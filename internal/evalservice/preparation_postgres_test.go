package evalservice

import (
	"context"
	"errors"
	"github.com/jackc/pgx/v5/pgconn"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
)

type failingResolver struct {
	BindingResolver
	cause error
}

func (r failingResolver) Resolve(context.Context, string, evaldomain.Variant, []evaldomain.Case) (Preflight, error) {
	return Preflight{}, r.cause
}

func TestPostgresEvalPreparationFailurePreservesCauseAndRetry(t *testing.T) {
	for _, configuration := range []bool{false, true} {
		name := "infrastructure"
		if configuration {
			name = "configuration"
		}
		t.Run(name, func(t *testing.T) {
			h := newHarness(t)
			e := h.create(t, "workflow")
			h.command(t, e, "prepare")
			cause := errors.New("fixture internal database failure must not appear in public diagnostic")
			failure := error(cause)
			if configuration {
				failure = errors.Join(evaldomain.Failure("eval_pin_mismatch"), cause)
			}
			h.service.resolver = failingResolver{h.resolver, failure}
			coordinator := h.coordinator(t, "prepare-failure")
			changed, err := coordinator.RunOnce(t.Context())
			if !changed || !errors.Is(err, cause) {
				t.Fatalf("preparation cause lost: changed=%v err=%v", changed, err)
			}
			e = h.get(t, e.ID)
			if strings.Contains(string(e.Diagnostic), cause.Error()) || evaldomain.Validate("Diagnostic", e.Diagnostic) != nil {
				t.Fatalf("unsafe diagnostic: %s", e.Diagnostic)
			}
			pending, err := evalstore.NewPostgresStore(h.pool).PendingCommands(t.Context(), e.OwnerID, e.ID)
			if err != nil {
				t.Fatal(err)
			}
			if configuration {
				if e.State != evaldomain.StateDraft || len(pending) != 0 || !strings.Contains(string(e.Diagnostic), "eval_pin_mismatch") {
					t.Fatalf("configuration failure: %s %s pending=%d", e.State, e.Diagnostic, len(pending))
				}
			} else {
				if e.State != evaldomain.StatePreparing || len(pending) != 1 || !strings.Contains(string(e.Diagnostic), "eval_preparation_unavailable") {
					t.Fatalf("transient failure: %s %s pending=%d", e.State, e.Diagnostic, len(pending))
				}
				h.service.resolver = h.resolver
				tick(t, coordinator)
				if e = h.get(t, e.ID); e.State != evaldomain.StateReady || len(e.Diagnostic) != 0 {
					t.Fatalf("retry did not prepare: %s", e.State)
				}
				record, err := evalstore.NewPostgresStore(h.pool).CommandRecord(t.Context(), e.OwnerID, e.ID, pending[0].ID)
				if err != nil || record.State != "succeeded" {
					t.Fatalf("retry command: %+v %v", record, err)
				}
			}
			if count(t, h.pool, "workflow_runs") != 0 || count(t, h.pool, "audits") != 0 {
				t.Fatal("preparation dispatched executions")
			}
		})
	}
}

// Exercise the actual SQL/resolver path, rather than only a stubbed prepare error.
func TestPostgresEvalPreparationDatabaseFailureIsRetryable(t *testing.T) {
	h := newHarness(t)
	e := h.create(t, "workflow")
	h.command(t, e, "prepare")
	if _, err := h.pool.Exec(t.Context(), `ALTER TABLE runtime_label_bindings RENAME TO fixture_unavailable_runtime_bindings`); err != nil {
		t.Fatal(err)
	}
	coordinator := h.coordinator(t, "database-recovery")
	_, err := coordinator.RunOnce(t.Context())
	var database *pgconn.PgError
	if !errors.As(err, &database) || database.Code != "42P01" {
		t.Fatalf("lost original database error: %v", err)
	}
	e = h.get(t, e.ID)
	if e.State != evaldomain.StatePreparing || !strings.Contains(string(e.Diagnostic), "eval_preparation_unavailable") {
		t.Fatalf("database failure became configuration rejection: %s %s", e.State, e.Diagnostic)
	}
	if _, err := h.pool.Exec(t.Context(), `ALTER TABLE fixture_unavailable_runtime_bindings RENAME TO runtime_label_bindings`); err != nil {
		t.Fatal(err)
	}
	tick(t, coordinator)
	if e = h.get(t, e.ID); e.State != evaldomain.StateReady || len(e.Diagnostic) != 0 {
		t.Fatalf("database recovery: %s %s", e.State, e.Diagnostic)
	}
}
