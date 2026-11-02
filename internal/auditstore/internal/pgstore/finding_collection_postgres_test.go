package pgstore

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

// statementCountingDB counts the statements a store sends.
type statementCountingDB struct {
	persistencepostgres.DBTX
	statements atomic.Int64
}

func (c *statementCountingDB) Exec(ctx context.Context, sql string, args ...any) (pgconn.CommandTag, error) {
	c.statements.Add(1)
	return c.DBTX.Exec(ctx, sql, args...)
}

func (c *statementCountingDB) Query(ctx context.Context, sql string, args ...any) (pgx.Rows, error) {
	c.statements.Add(1)
	return c.DBTX.Query(ctx, sql, args...)
}

func (c *statementCountingDB) QueryRow(ctx context.Context, sql string, args ...any) pgx.Row {
	c.statements.Add(1)
	return c.DBTX.QueryRow(ctx, sql, args...)
}

// CollectionReceiptIDs resolves up to the 256 findings a collection request
// may select (LIM-FND-25) in one statement, returns every contributing and
// first receipt sorted, and keeps the exact missing, stale and overflow
// outcomes without dropping a receipt (FND-R159).
func TestPostgresCollectionReceiptIDsReadsSelectedFindingsInOneStatement(t *testing.T) {
	f := newCollectingAuditFixture(t, "receipt-ids", 1<<20)
	const findings = auditdomain.MaximumCollectionEntries
	// Admit one finding more than the bound, then fold the last one into the
	// first as a contributing receipt: 256 findings hold 257 receipts.
	ids := make([]string, findings+1)
	receipts := make([]string, findings+1)
	for index := range ids {
		receipts[index] = fmt.Sprintf("receipt-select-%03d", index)
		proposal := testExact("audit-findings", fmt.Sprintf("candidate-select-%03d", index), "proposal-r1")
		proposal.MediaType, proposal.SizeBytes = "application/json", 128
		ids[index] = insertAuditChildFinding(t, f, receipts[index], proposal)
	}
	// Seed one canonical identity with two independent source memberships.
	tx, err := f.pool.Begin(f.ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback(f.ctx) }()
	for _, statement := range []string{
		"ALTER TABLE project_finding_sources DISABLE TRIGGER project_finding_sources_immutable",
		"UPDATE project_finding_sources SET finding_id=$1 WHERE finding_id=$2",
		"DELETE FROM project_findings WHERE finding_id=$2",
		"ALTER TABLE project_finding_sources ENABLE TRIGGER project_finding_sources_immutable",
	} {
		var err error
		if strings.Contains(statement, "$1") {
			_, err = tx.Exec(f.ctx, statement, ids[0], ids[findings])
		} else if strings.Contains(statement, "$2") {
			_, err = tx.Exec(f.ctx, strings.ReplaceAll(statement, "$2", "$1"), ids[findings])
		} else {
			_, err = tx.Exec(f.ctx, statement)
		}
		if err != nil {
			t.Fatal(err)
		}
	}
	if err := tx.Commit(f.ctx); err != nil {
		t.Fatal(err)
	}
	ids = ids[:findings]
	revisions := func(selected []string) []auditstore.FindingRevision {
		result := make([]auditstore.FindingRevision, len(selected))
		for index, id := range selected {
			result[index] = auditstore.FindingRevision{FindingID: id, Revision: 1}
		}
		return result
	}
	counter := &statementCountingDB{DBTX: f.pool}
	store := NewQueryStore(counter, artifactpgtest.NewStorage())

	// The first finding and 254 others hold exactly the 256 receipts allowed.
	selected := revisions(ids[:findings-1])
	got, err := store.CollectionReceiptIDs(f.ctx, f.audit.OwnerID, f.audit.AuditID, selected)
	want := append(slices.Clone(receipts[:findings-1]), receipts[findings])
	slices.Sort(want)
	if err != nil || !slices.Equal(got, want) {
		t.Fatalf("receipts of %d findings = (%d receipts, %v), want %d", len(selected), len(got), err, len(want))
	}
	if statements := counter.statements.Swap(0); statements != 1 {
		t.Fatalf("resolving %d findings used %d statements, want 1", len(selected), statements)
	}

	stale := revisions(ids)
	stale[findings-1].Revision = 2
	missing := revisions(ids[:2])
	missing[1].FindingID = "finding-missing"
	for _, test := range []struct {
		name       string
		owner      string
		findings   []auditstore.FindingRevision
		want       error
		statements int64
	}{
		{name: "every finding exceeds the receipt bound", findings: revisions(ids), want: auditstore.ErrInvalid, statements: 1},
		{name: "stale finding before the receipt bound", findings: stale, want: auditstore.ErrConflict, statements: 1},
		{name: "missing finding", findings: missing, want: auditstore.ErrNotFound, statements: 1},
		{name: "foreign owner", owner: "another-owner", findings: revisions(ids[:1]), want: auditstore.ErrNotFound, statements: 1},
		{name: "more findings than receipts", findings: revisions(append(slices.Clone(ids), "finding-extra")), want: auditstore.ErrInvalid},
	} {
		t.Run(test.name, func(t *testing.T) {
			owner := f.audit.OwnerID
			if test.owner != "" {
				owner = test.owner
			}
			got, err := store.CollectionReceiptIDs(f.ctx, owner, f.audit.AuditID, test.findings)
			if got != nil || !errors.Is(err, test.want) {
				t.Fatalf("receipts = (%v, %v), want %v", got, err, test.want)
			}
			if statements := counter.statements.Swap(0); statements != test.statements {
				t.Fatalf("rejected selection used %d statements, want %d", statements, test.statements)
			}
		})
	}
	if got, err := store.CollectionReceiptIDs(f.ctx, f.audit.OwnerID, f.audit.AuditID, nil); err != nil || len(got) != 0 {
		t.Fatalf("empty selection = (%v, %v)", got, err)
	}
}
