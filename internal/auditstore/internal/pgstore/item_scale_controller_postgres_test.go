package pgstore

import (
	"context"
	"slices"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/persistence/explaintest"
	"github.com/jackc/pgx/v5"
)

// TestPostgresControllerReadsAtItemScale runs two reads that every
// reconciliation of the 100,000-item Audit makes, in the custom and the
// generic plan.
//
// The expired item-approval search (expiredItemReviewsSQL) probes each
// approval item's own review requests through
// audit_review_requests_item_subject_idx: at most two lookups per request of
// an approval item, where without the index it read every review request of
// the Audit (9,715 in the generic plan) or scanned the table once per item
// (58,290 in the custom plan). Dropping the index inside a rolled-back
// transaction returns the same rows.
//
// The reconcile window (listReconcileItemsSQL) is kept: it must order every
// unsettled item, awaiting review last, before it can take the first
// MaxReconcileRows+1, so it reads the Audit's unsettled items once through
// audit_items_dispatch_idx and none of its settled ones. Each budget adds
// planningEntries.
func TestPostgresControllerReadsAtItemScale(t *testing.T) {
	f := newItemScaleFixture(t)
	if _, err := f.pool.Exec(f.ctx, `
INSERT INTO audit_review_requests (request_id, audit_id, subject_kind, subject_id, kind, subject_revision, subject_digest, requested_actions, state, expires_at, idempotency_key, request_digest)
SELECT 'old-' || item_id, audit_id, 'audit-item-action', item_id, 'active-check-approval', 1, $2, '["approve"]'::jsonb, 'expired', clock_timestamp() - interval '1 day', 'old-' || item_id, $2 FROM audit_items WHERE audit_id = $1 AND state = 'settled' AND ordinal % 10 = 0`, f.auditID, testDigest("9")); err != nil {
		t.Fatal(err)
	}
	// The approval items' live reviews pass their expiry; nothing has stored
	// it yet, as when a reconciliation finds them. VACUUM removes the
	// versions this update superseded.
	if _, err := f.pool.Exec(f.ctx, `
UPDATE audit_review_requests SET expires_at = clock_timestamp() - interval '1 minute'
 WHERE audit_id = $1 AND state = 'pending'`, f.auditID); err != nil {
		t.Fatal(err)
	}
	if _, err := f.pool.Exec(f.ctx, `VACUUM (INDEX_CLEANUP ON, ANALYZE) audit_review_requests`); err != nil {
		t.Fatal(err)
	}
	var reviews, approvalRequests int64
	if err := f.pool.QueryRow(f.ctx, `
SELECT (SELECT count(*) FROM audit_review_requests WHERE audit_id = $1),
       (SELECT count(*) FROM audit_review_requests AS review
          JOIN audit_items AS item ON item.audit_id = review.audit_id AND item.item_id = review.subject_id
         WHERE review.audit_id = $1 AND review.subject_kind = 'audit-item-action'
           AND item.state IN ('ready', 'awaiting_review') AND item.approval_kind <> 'none')`, f.auditID).
		Scan(&reviews, &approvalRequests); err != nil {
		t.Fatal(err)
	}
	if approvalRequests == 0 || reviews < 1_000*approvalRequests {
		t.Fatalf("fixture holds %d review requests, %d of approval items", reviews, approvalRequests)
	}

	t.Run("item review renewal", func(t *testing.T) {
		type candidate struct{ ItemID, Kind, Digest, RequestID, State string }
		expired := func(t *testing.T, q pgx.Tx, limit *int) []candidate {
			t.Helper()
			rows, err := q.Query(f.ctx, expiredItemReviewsSQL, f.auditID, limit)
			if err != nil {
				t.Fatal(err)
			}
			result, err := pgx.CollectRows(rows, pgx.RowToStructByPos[candidate])
			if err != nil {
				t.Fatal(err)
			}
			return result
		}
		read := func(t *testing.T, dropIndex bool, limit *int) []candidate {
			t.Helper()
			tx, err := f.pool.Begin(f.ctx)
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = tx.Rollback(context.WithoutCancel(f.ctx)) }()
			if dropIndex {
				if _, err := tx.Exec(f.ctx, `DROP INDEX audit_review_requests_item_subject_idx`); err != nil {
					t.Fatal(err)
				}
			}
			return expired(t, tx, limit)
		}
		one := 1
		all := read(t, false, nil)
		if int64(len(all)) != approvalRequests {
			t.Fatalf("found %d expired approvals, want one per approval item's request (%d)", len(all), approvalRequests)
		}
		for _, row := range all {
			if row.State != string(auditstore.ReviewPending) {
				t.Fatalf("candidate %+v is not the pending request past its expiry", row)
			}
		}
		if first := read(t, false, &one); !slices.Equal(first, all[:1]) {
			t.Fatalf("first candidate = %+v, want %+v", first, all[:1])
		}
		if without := read(t, true, nil); !slices.Equal(without, all) {
			t.Fatalf("without the index the search finds %+v, want %+v", without, all)
		}
		// The search probes each approval item's requests twice, for the
		// join and for the live-authority lookup.
		budget := 2*approvalRequests + planningEntries
		for _, mode := range explaintest.Modes {
			for _, limit := range []*int{&one, nil} {
				plan := explainPrepared(t, f.ctx, f.pool, mode, expiredItemReviewsSQL, f.auditID, limit)
				if read := plan.Reads.Of("audit_review_requests"); read > budget {
					t.Fatalf("%s: the search read %d review requests, budget %d\n%s", mode, read, budget, plan.Plan)
				}
				t.Logf("%s limit %v: read %d review requests (budget %d)", mode, deref(limit),
					plan.Reads.Of("audit_review_requests"), budget)
			}
		}
	})

	t.Run("reconcile window", func(t *testing.T) {
		reference, err := f.store.ListItems(f.ctx, f.auditID)
		if err != nil {
			t.Fatal(err)
		}
		var active, awaiting []string
		for _, item := range reference {
			switch item.State {
			case auditstore.ItemSettled:
			case auditstore.ItemAwaitingReview:
				awaiting = append(awaiting, item.ItemID)
			default:
				active = append(active, item.ItemID)
			}
		}
		unsettled := int64(len(active) + len(awaiting))
		limit := auditstore.MaxReconcileRows + 1
		want := append(slices.Clone(active), awaiting...)[:limit]
		conn, err := f.pool.Acquire(f.ctx)
		if err != nil {
			t.Fatal(err)
		}
		defer conn.Release()
		for _, mode := range explaintest.Modes {
			if _, err := conn.Exec(f.ctx, "SET plan_cache_mode = "+string(mode)); err != nil {
				t.Fatal(err)
			}
			items, err := NewQueryStore(conn).listReconcileItems(f.ctx, f.auditID, limit)
			if err != nil {
				t.Fatal(err)
			}
			got := make([]string, len(items))
			for index, item := range items {
				got[index] = item.ItemID
			}
			if !slices.Equal(got, want) {
				t.Fatalf("%s: reconcile window holds %d items, want the first %d unsettled, awaiting review last", mode, len(got), limit)
			}
			plan := explainPrepared(t, f.ctx, f.pool, mode, listReconcileItemsSQL, f.auditID, limit)
			if read := plan.Reads.Of("audit_items"); read > unsettled+planningEntries {
				t.Fatalf("%s: the window read %d items, budget %d unsettled\n%s", mode, read, unsettled+planningEntries, plan.Plan)
			}
			t.Logf("%s: read %d items (budget %d)", mode, plan.Reads.Of("audit_items"), unsettled+planningEntries)
		}
	})
}
