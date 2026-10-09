package auditstore

import (
	"strings"
	"testing"
)

func TestPostgresAssessmentRestrictChecksUseIndexes(t *testing.T) {
	f := newCollectingAuditFixture(t, "assessment-indexes", 1<<20)
	collectPurgeFinding(t, f, "receipt-indexes")
	_, err := f.pool.Exec(f.ctx, `
INSERT INTO audit_finding_assessments
 (assessment_id, finding_id, audit_id, receipt_id, semantic_assessment,
  result_ref, result_digest, direct_verification, contract_ref, contract_digest)
SELECT 'bulk-' || n, finding_id, audit_id, receipt_id, semantic_assessment,
       result_ref, result_digest, TRUE, '{}'::jsonb, result_digest
  FROM audit_finding_assessments CROSS JOIN generate_series(1,5000) AS n
 WHERE assessment_id = 'assessment-receipt-indexes'`)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.pool.Exec(f.ctx, `ANALYZE audit_finding_assessments`); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		index, predicate string
		args             []any
	}{
		{"audit_finding_assessments_collection_receipt_idx", "collection_receipt_id=$1", []any{"collection-receipt-indexes"}},
		{"audit_finding_assessments_execution_item_idx", "execution_item_id=$1 AND item_id=$2", []any{f.memberID, f.itemID}},
	} {
		rows, err := f.pool.Query(f.ctx, `EXPLAIN SELECT 1 FROM audit_finding_assessments WHERE `+tc.predicate+` FOR KEY SHARE`, tc.args...)
		if err != nil {
			t.Fatal(err)
		}
		var plan strings.Builder
		for rows.Next() {
			var line string
			if err := rows.Scan(&line); err != nil {
				t.Fatal(err)
			}
			plan.WriteString(line + "\n")
		}
		rows.Close()
		if err := rows.Err(); err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(plan.String(), tc.index) {
			t.Errorf("RESTRICT lookup does not use %s:\n%s", tc.index, plan.String())
		}
	}
}
