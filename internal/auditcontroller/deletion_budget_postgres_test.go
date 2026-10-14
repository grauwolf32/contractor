//go:build integration

package auditcontroller

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type deletionBudgetStore struct {
	Store
	lease time.Duration
}

func (s *deletionBudgetStore) RenewClaim(ctx context.Context, claim auditstore.ControllerClaim, lease time.Duration) error {
	s.lease = lease
	return s.Store.RenewClaim(ctx, claim, lease)
}

type deletionBudgetDeleter struct {
	RunDeleter
	pool    *pgxpool.Pool
	checked bool
}

func (d *deletionBudgetDeleter) DeleteReleasedTerminalRun(ctx context.Context, ownerID, runID string) error {
	// Inspect the selected SQL policy, not elapsed wall-clock time. The
	// ordinary reconcile context has no maintenance budget at this boundary.
	if err := persistencepostgres.InTx(ctx, d.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		var cleanup bool
		if err := tx.QueryRow(ctx, `SELECT current_setting('statement_timeout')::interval = interval '60 seconds'`).Scan(&cleanup); err != nil {
			return err
		}
		if !cleanup {
			return errors.New("child Run deletion inherited the ordinary operation budget")
		}
		return nil
	}); err != nil {
		return err
	}
	d.checked = true
	return d.RunDeleter.DeleteReleasedTerminalRun(ctx, ownerID, runID)
}

func seedLargeAuditChildRun(t *testing.T, ctx context.Context, h *postgresControllerHarness, runID string) {
	t.Helper()
	scope, _ := artifacts.RunScope(runID)
	seed, err := h.artifacts.Create(ctx, scope, contracts.ArtifactRef{Namespace: "scratch", Name: "purge-seed"},
		artifacts.Payload{MediaType: "text/plain", Data: []byte("large child Run purge fixture")})
	if err != nil {
		t.Fatal(err)
	}
	var remaining int
	var versionID string
	if err := h.pool.QueryRow(ctx, `SELECT 50000 - revision_count FROM artifact_scopes
WHERE scope_kind = 'run' AND scope_id = $1`, runID).Scan(&remaining); err != nil {
		t.Fatal(err)
	}
	if err := h.pool.QueryRow(ctx, `SELECT version_id FROM artifact_binding_revisions
WHERE scope_kind = 'run' AND scope_id = $1 AND namespace = $2 AND name = $3 AND revision = $4`,
		runID, seed.Ref.Namespace, seed.Ref.Name, seed.Ref.Revision).Scan(&versionID); err != nil {
		t.Fatal(err)
	}
	if _, err := h.pool.Exec(ctx, `INSERT INTO artifact_binding_revisions
(scope_kind,scope_id,namespace,name,revision,version_id)
SELECT 'run',$1,'bulk','item-' || n,'bulk-' || n,$3 FROM generate_series(1,$2::int) AS n`, runID, remaining, versionID); err != nil {
		t.Fatal(err)
	}
	if _, err := h.pool.Exec(ctx, `INSERT INTO artifact_bindings
(scope_kind,scope_id,namespace,name,current_revision)
SELECT 'run',$1,'bulk','item-' || n,'bulk-' || n FROM generate_series(1,$2::int) AS n`, runID, remaining); err != nil {
		t.Fatal(err)
	}
}
