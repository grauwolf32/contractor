//go:build integration

package pgstore

import (
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5/pgconn"
)

func TestPostgresPreparedExecutionOwnsPlanShapeAndIdentity(t *testing.T) {
	f, _, _, state := preparedAdmissionFixture(t, 1)
	var data []byte
	if err := f.pool.QueryRow(f.ctx, `SELECT prepared_execution FROM audit_items WHERE item_id='typed-item-000'`).Scan(&data); err != nil {
		t.Fatal(err)
	}
	var execution auditdomain.PreparedExecutionItem
	if err := strictjson.Decode(data, &execution); err != nil {
		t.Fatal(err)
	}
	if _, err := f.pool.Exec(f.ctx, `INSERT INTO audit_rounds(round_id,audit_id,ordinal,manifest_ref,manifest_digest,state,expected_item_count,prepared_generation) SELECT 'plan-shape-round',audit_id,3,manifest_ref,manifest_digest,'proposed',128,'plan-shape' FROM audit_rounds WHERE round_id=$1`, state.RoundID); err != nil {
		t.Fatal(err)
	}
	insert := func(id string, projection any, page, related any) error {
		_, err := f.pool.Exec(f.ctx, `INSERT INTO audit_items(item_id,audit_id,round_id,item_key,ordinal,kind,subject_key,task_ref,task_digest,origin,workflow_role,state,coverage_status,coverage_requested,coverage_completed,coverage_gaps,coverage_rationale,prepared_execution,prepared_page_ordinal,prepared_related_count)
SELECT $1,audit_id,'plan-shape-round',item_key,0,kind,subject_key,task_ref,task_digest,origin,workflow_role,'ready',coverage_status,coverage_requested,coverage_completed,coverage_gaps,coverage_rationale,$2::text::json,$3,$4 FROM audit_items WHERE item_id='typed-item-000'`, id, projection, page, related)
		return err
	}
	valid := string(data)
	cases := []struct {
		name, projection string
		page, related    any
	}{
		{"json-null-projection", "null", 0, 5},
		{"missing-plan", `{}`, 0, 5},
		{"null-plan", `{"accepted_plan":null}`, 0, 5},
		{"array-plan", `{"accepted_plan":[]}`, 0, 5},
		{"scalar-plan", `{"accepted_plan":"plan"}`, 0, 5},
		{"missing-schema", `{"accepted_plan":{"audit_id":"` + f.audit.AuditID + `","item_key":"` + execution.ItemKey + `"}}`, 0, 5},
		{"wrong-schema", `{"accepted_plan":{"schema":"other","audit_id":"` + f.audit.AuditID + `","item_key":"` + execution.ItemKey + `"}}`, 0, 5},
		{"foreign-audit", `{"accepted_plan":{"schema":"contractor.audit.accepted-plan.v1","audit_id":"foreign","item_key":"` + execution.ItemKey + `"}}`, 0, 5},
		{"foreign-item", `{"accepted_plan":{"schema":"contractor.audit.accepted-plan.v1","audit_id":"` + f.audit.AuditID + `","item_key":"foreign"}}`, 0, 5},
		{"missing-page", valid, nil, 5},
		{"missing-related", valid, 0, nil},
		{"page-low", valid, -1, 5},
		{"page-high", valid, 4096, 5},
		{"related-low", valid, 0, -1},
		{"related-high", valid, 0, 513},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			assertPreparedCheckRejection(t, insert("bad-plan", test.projection, test.page, test.related))
		})
	}
	t.Run("sql-null-with-headers", func(t *testing.T) { assertPreparedCheckRejection(t, insert("bad-plan", nil, 0, 5)) })
	// Preserve the exact encoded JSON boundary. Multibyte and HTML characters
	// would fail this check if storage measured a jsonb reserialization instead.
	plan := *execution.AcceptedPlan
	plan.Plan.Objective = "世界 <>&"
	checkBytes, err := auditdomain.EncodeCheckPlan(plan.Plan)
	if err != nil {
		t.Fatal(err)
	}
	plan.PlanDigest = auditdomain.DigestBytes(checkBytes)
	planBytes, err := auditdomain.EncodeAcceptedPlan(plan, execution.Dependencies)
	if err != nil {
		t.Fatal(err)
	}
	plan.Plan.Objective += strings.Repeat("x", auditdomain.MaximumAcceptedPlanBytes-len(planBytes))
	checkBytes, err = auditdomain.EncodeCheckPlan(plan.Plan)
	if err != nil {
		t.Fatal(err)
	}
	plan.PlanDigest = auditdomain.DigestBytes(checkBytes)
	planBytes, err = auditdomain.EncodeAcceptedPlan(plan, execution.Dependencies)
	if err != nil || len(planBytes) != 65536 {
		t.Fatalf("boundary=%d error=%v", len(planBytes), err)
	}
	execution.AcceptedPlan = &plan
	bounded, err := strictjson.Canonical(execution)
	if err != nil {
		t.Fatal(err)
	}
	if err = insert("boundary-plan", string(bounded), 0, 5); err != nil {
		t.Fatalf("exact 65536-byte embedded plan rejected: %v", err)
	}
	var stored []byte
	if err = f.pool.QueryRow(f.ctx, `SELECT (prepared_execution->'accepted_plan')::text FROM audit_items WHERE item_id='boundary-plan'`).Scan(&stored); err != nil || string(stored) != string(planBytes) {
		t.Fatal("embedded JSON bytes normalized", err)
	}
	plan.Plan.Objective += "x"
	checkBytes, err = auditdomain.EncodeCheckPlan(plan.Plan)
	if err != nil {
		t.Fatal(err)
	}
	plan.PlanDigest = auditdomain.DigestBytes(checkBytes)
	tooLarge, err := strictjson.Canonical(execution)
	if err != nil {
		t.Fatal(err)
	}
	t.Run("oversized-plan", func(t *testing.T) { assertPreparedCheckRejection(t, insert("oversized-plan", string(tooLarge), 0, 5)) })
}

func TestPostgresPreparedProjectionRemainsImmutableWithoutPlanColumn(t *testing.T) {
	f, _, _, state := preparedAdmissionFixture(t, 1)
	for index, assignment := range []string{
		`prepared_execution=(jsonb_set(prepared_execution::jsonb,'{accepted_plan,audit_id}','"foreign"'))::json`,
		`prepared_execution='{}'`, `prepared_execution=NULL`, `prepared_page_ordinal=1`, `prepared_related_count=prepared_related_count+1`,
		`item_key='different'`, `workflow_role='different'`, `task_digest='sha256:'||repeat('a',64)`, `approval_kind='active-check-approval'`,
	} {
		t.Run(fmt.Sprintf("mutation-%02d", index), func(t *testing.T) {
			_, err := f.pool.Exec(f.ctx, `UPDATE audit_items SET `+assignment+` WHERE item_id='typed-item-000'`)
			assertPreparedCheckRejection(t, err)
		})
	}
	if _, err := f.pool.Exec(f.ctx, `UPDATE audit_items SET state='settled',final_disposition='excluded' WHERE item_id='typed-item-000'`); err != nil {
		t.Fatalf("ordinary lifecycle update rejected: %v", err)
	}
	var visible int
	if err := f.pool.QueryRow(f.ctx, `SELECT count(*) FROM audit_live_item_reviews WHERE audit_id=$1`, f.audit.AuditID).Scan(&visible); err != nil || visible != 0 {
		t.Fatalf("private review authority=%d error=%v", visible, err)
	}
	for _, typed := range []bool{false, true} {
		t.Run(fmt.Sprint(typed), func(t *testing.T) {
			projection, page, related := "NULL", "NULL", "NULL"
			if typed {
				projection, page, related = `jsonb_set(prepared_execution::jsonb,'{accepted_plan,item_key}','"sealed-extra"')::json`, "prepared_page_ordinal", "prepared_related_count"
			}
			_, err := f.pool.Exec(f.ctx, `INSERT INTO audit_items(item_id,audit_id,round_id,item_key,ordinal,kind,subject_key,task_ref,task_digest,origin,workflow_role,coverage_status,coverage_requested,coverage_completed,coverage_gaps,coverage_rationale,prepared_execution,prepared_page_ordinal,prepared_related_count) SELECT 'sealed-extra',audit_id,$1,'sealed-extra',1,kind,subject_key,task_ref,task_digest,jsonb_set(origin,'{entryKey}','"sealed-extra"'),workflow_role,coverage_status,coverage_requested,coverage_completed,coverage_gaps,coverage_rationale,`+projection+`,`+page+`,`+related+` FROM audit_items WHERE item_id='typed-item-000'`, state.RoundID)
			assertPreparedCheckRejection(t, err)
		})
	}
}

func assertPreparedCheckRejection(t *testing.T, err error) {
	t.Helper()
	var rejected *pgconn.PgError
	if !errors.As(err, &rejected) || rejected.Code != persistencepostgres.SQLStateCheckViolation {
		t.Fatalf("mutation=%v, want PostgreSQL check violation", err)
	}
}
