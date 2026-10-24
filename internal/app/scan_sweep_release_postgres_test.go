package app

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"strings"
	"testing"
	"time"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/scancontroller"
	"github.com/grauwolf32/contractor/internal/scanstore"
	"github.com/jackc/pgx/v5/pgconn"
)

func sweepClaimHolder(t *testing.T, ctx context.Context, f sweepFixture, sweepID string) (holder *string, epoch int64) {
	t.Helper()
	if err := f.pool.QueryRow(ctx, `SELECT holder_id, epoch FROM scan_sweeps WHERE sweep_id = $1`, sweepID).Scan(&holder, &epoch); err != nil {
		t.Fatal(err)
	}
	return holder, epoch
}

// A tick whose claim release fails reports that failure with the sweep in
// its error instead of dropping it, and the sweep stays claimed until its
// lease ends (PER-R26, FAIL-SCN-15). The isolated test schema refuses one of
// the two release statements with a trigger, a definite store failure; the
// error's text, which the Run loop logs, names the sweep and the failed step
// but never the statement or the database's message.
func TestPostgresScanSweepTickReportsADefiniteReleaseFailure(t *testing.T) {
	for _, step := range []struct{ name, refuse string }{
		{name: "next tick", refuse: `NEW.next_tick_at IS DISTINCT FROM OLD.next_tick_at AND NEW.state = OLD.state`},
		{name: "claim", refuse: `OLD.holder_id IS NOT NULL AND NEW.holder_id IS NULL`},
	} {
		t.Run(step.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
			defer cancel()
			f := newSweepFixture(t, ctx)
			sweep := createSmallSweep(t, ctx, f, "release-failure", 1)
			if _, err := f.pool.Exec(ctx, `
CREATE FUNCTION test_refuse_sweep_release() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'fixture refuses the sweep release';
END
$$;
CREATE TRIGGER test_refuse_sweep_release BEFORE UPDATE ON scan_sweeps FOR EACH ROW
    WHEN (`+step.refuse+`)
    EXECUTE FUNCTION test_refuse_sweep_release();`); err != nil {
				t.Fatal(err)
			}
			err := f.controller.RunOnce(ctx)
			var refused *pgconn.PgError
			if !errors.As(err, &refused) || refused.Message != "fixture refuses the sweep release" ||
				!strings.Contains(err.Error(), sweep.SweepID) {
				t.Fatalf("tick with a refused release = %v, want the release failure of sweep %s", err, sweep.SweepID)
			}
			if strings.Contains(err.Error(), "fixture refuses") || strings.Contains(err.Error(), "UPDATE") {
				t.Fatalf("release failure renders store details: %q", err)
			}
			if holder, _ := sweepClaimHolder(t, ctx, f, sweep.SweepID); holder == nil {
				t.Fatal("the refused release freed the sweep")
			}
			var admitted int
			if err := f.pool.QueryRow(ctx, `SELECT count(*) FROM workflow_runs`).Scan(&admitted); err != nil || admitted != 1 {
				t.Fatalf("tick admitted %d child Runs before its release, want 1 (%v)", admitted, err)
			}
		})
	}
}

// A claim whose lease ended during a tick and that another controller took
// over is not released by the stale holder: the tick reports nothing for it
// and the new holder keeps the sweep (PER-R25).
func TestPostgresScanSweepTickToleratesAClaimTakenOverDuringIt(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	f := newSweepFixture(t, ctx)
	sweep := createSmallSweep(t, ctx, f, "release-lost", 1)
	var rival persistencepostgres.ControllerClaim
	controller, err := scancontroller.New(scancontroller.Options{
		Pool: f.pool, HolderID: "stale-controller", Now: time.Now, Logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
		Admit: func(ctx context.Context, _ persistencepostgres.ControllerClaim, _ scanstore.Snapshot, _ scanstore.Member) error {
			if _, err := f.pool.Exec(ctx, `UPDATE scan_sweeps SET expires_at = clock_timestamp() WHERE sweep_id = $1`, sweep.SweepID); err != nil {
				return err
			}
			claim, found, err := scanstore.New(f.pool).Claim(ctx, "rival-controller", time.Minute)
			if err != nil || !found {
				t.Errorf("rival lease = %+v, %t, %v", claim, found, err)
			}
			rival = claim
			return nil
		},
		CancelRun: func(context.Context, string, string) (runstore.WorkflowRun, error) {
			t.Error("unexpected cancellation")
			return runstore.WorkflowRun{}, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := controller.RunOnce(ctx); err != nil {
		t.Fatalf("tick whose claim was taken over = %v, want no failure", err)
	}
	holder, epoch := sweepClaimHolder(t, ctx, f, sweep.SweepID)
	if holder == nil || *holder != "rival-controller" || epoch != rival.Epoch {
		t.Fatalf("claim after the stale release = %v at epoch %d, want rival-controller at %d", holder, epoch, rival.Epoch)
	}
}
