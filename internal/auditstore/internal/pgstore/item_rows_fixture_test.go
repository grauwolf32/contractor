package pgstore

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/jackc/pgx/v5/pgxpool"
)

// seedItemRows supplies fixed rows for remaining review readers and purge
// topology tests. It admits no work and does not recreate Round execution.
func seedItemRows(t *testing.T, ctx context.Context, pool *pgxpool.Pool, auditID string, itemIDs ...string) string {
	t.Helper()
	roundID := "review-rows-" + auditID
	ref, _ := json.Marshal(testExact("fixtures", "review-rows", "r1").Ref)
	if _, err := pool.Exec(ctx, `INSERT INTO audit_rounds(round_id,audit_id,ordinal,manifest_ref,manifest_digest,state,expected_item_count)
VALUES($1,$2,1,$3,$4,'accepted',$5)`, roundID, auditID, ref, testDigest("a"), len(itemIDs)); err != nil {
		t.Fatal(err)
	}
	for ordinal, id := range itemIDs {
		origin, _ := json.Marshal(testOrigin(id))
		if _, err := pool.Exec(ctx, `INSERT INTO audit_items(item_id,audit_id,round_id,item_key,ordinal,kind,subject_key,task_ref,task_digest,workflow_role,state,origin,coverage_status,coverage_requested,coverage_completed,coverage_gaps)
VALUES($1,$2,$3,$1,$4,'checklist',$1,$5,$6,'check','ready',$7,'not-tested','[]','[]','[]')`, id, auditID, roundID, ordinal, ref, testDigest("a"), origin); err != nil {
			t.Fatal(err)
		}
	}
	return roundID
}
