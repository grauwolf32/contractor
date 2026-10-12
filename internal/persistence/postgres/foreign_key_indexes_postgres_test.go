package postgres

import (
	"context"
	"os"
	"slices"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
)

// probeAllowance explains why a foreign key may lack an index led by all of
// its columns. Exactly one field names the schema object that keeps the probe
// cheap, and the test checks that the object still does so.
type probeAllowance struct {
	// prefixIndex is a usable index of the referencing table led by some of the
	// key's columns; reason states why those columns bound the rows a probe
	// reads.
	prefixIndex string
	// immutableParentTrigger is a BEFORE DELETE OR UPDATE trigger of the
	// referenced table that rejects every deletion and key change, so the
	// probe never runs. Reopen the entry when the referenced rows become
	// deletable.
	immutableParentTrigger string
	reason                 string
}

// foreignKeyProbeAllowances lists every foreign key without an index led by
// all of its columns (V370-023). A new foreign key needs such an index, or an
// entry here naming what bounds its probe.
var foreignKeyProbeAllowances = map[string]probeAllowance{
	"allocation_execution_reports_stage_execution_id_allocation_fkey": {
		prefixIndex: "allocation_execution_reports_stage_idx", reason: "one report per logical agent of a StageExecution",
	},
	"artifact_bindings_scope_kind_scope_id_namespace_name_curre_fkey": {
		prefixIndex: "artifact_bindings_pkey", reason: "the binding identity is unique",
	},
	"audit_execution_items_execution_fkey": {
		prefixIndex: "audit_execution_items_execution_member", reason: "the items of one execution, at most one batch",
	},
	"audit_execution_items_item_fkey": {
		prefixIndex: "audit_execution_items_item_attempt", reason: "the attempts of one item, at most MaxItemRunAttempts",
	},
	"audit_finding_contributions_finding_fkey": {
		prefixIndex: "audit_finding_contributions_pkey", reason: "the contributions of one finding",
	},
	"audit_finding_contributions_receipt_id_fkey": {
		immutableParentTrigger: "finding_proposal_receipts_protect_immutable", reason: "receipts are never deleted or re-keyed",
	},
	"audit_findings_first_receipt_id_fkey": {
		immutableParentTrigger: "finding_proposal_receipts_protect_immutable", reason: "receipts are never deleted or re-keyed",
	},
	"audit_items_last_execution_item_fkey": {
		prefixIndex: "audit_items_pkey", reason: "item_id is unique",
	},
	"audit_items_round_fkey": {
		prefixIndex: "audit_items_round_key", reason: "the items of one Round, which its deletion cascades to",
	},
	"audit_review_decisions_request_fkey": {
		prefixIndex: "audit_review_decisions_request_id_key", reason: "one decision per review request",
	},
	"audits_current_round_fkey": {
		prefixIndex: "audits_pkey", reason: "audit_id is unique",
	},
	"audits_project_owner_fkey": {
		prefixIndex: "audits_project_state_idx", reason: "the Audits of one Project, probed once per Project deletion",
	},
	"eval_selections_experiment_id_member_id_assessment_kind_as_fkey": {
		prefixIndex: "eval_selections_pkey", reason: "one selection per Eval member",
	},
	"eval_selections_experiment_id_member_id_result_kind_result_fkey": {
		prefixIndex: "eval_selections_pkey", reason: "one selection per Eval member",
	},
	"eval_submissions_experiment_id_member_id_execution_kind_fkey": {
		prefixIndex: "eval_submissions_pkey", reason: "one submission per Eval member",
	},
	"projects_http_target_credential_fkey": {
		prefixIndex: "projects_http_target_credential_idx", reason: "the Projects that use one Runtime credential",
	},
	"stage_execution_lineage": {
		prefixIndex: "stage_attempt_identity", reason: "the attempts of one Stage of one Run",
	},
	"stage_execution_planner_session": {
		prefixIndex: "stage_executions_pkey", reason: "stage_execution_id is unique",
	},
	"stage_execution_resume_receipt": {
		prefixIndex: "stage_executions_pkey", reason: "stage_execution_id is unique",
	},
	"stage_transition_source_run": {
		prefixIndex: "stage_transition_decisions_pkey", reason: "source_execution_id is unique",
	},
	"stage_transition_target_run_stage": {
		prefixIndex: "stage_transition_target_once", reason: "target_execution_id is unique",
	},
	"workflow_run_output_publicati_source_scope_kind_run_id_sou_fkey": {
		prefixIndex: "workflow_run_output_publications_pkey", reason: "the output publications of one Run",
	},
	"workflow_run_output_publicati_target_scope_kind_project_id_fkey": {
		prefixIndex: "workflow_run_output_publications_project_idx", reason: "the output publications into one Project",
	},
	"workflow_run_output_publications_run_id_project_id_fkey": {
		prefixIndex: "workflow_run_output_publications_pkey", reason: "the output publications of one Run",
	},
	"workflow_runs_audit_submission_fkey": {
		prefixIndex: "workflow_runs_audit_execution_unique", reason: "audit_execution_id is unique",
	},
}

// foreignKeyProbeIndexesSQL lists every foreign key of the current schema
// with the usable indexes of its referencing table that a probe on the key
// columns can use: the number of leading index keys that are key columns,
// and whether those are all of them. An index is usable when it is valid and
// either complete or restricted to rows whose key column is set, which an
// equality probe implies.
const foreignKeyProbeIndexesSQL = `
WITH fk AS (
    SELECT con.oid, con.conname::text AS name, con.conrelid, con.confrelid, con.conkey
      FROM pg_constraint AS con
     WHERE con.contype = 'f' AND con.connamespace = current_schema()::regnamespace
), usable AS (
    SELECT fk.oid AS fk_oid, index_class.relname::text AS index_name,
           (SELECT coalesce(min(position) - 1, idx.indnkeyatts)
              FROM generate_series(1, idx.indnkeyatts) AS position
             WHERE NOT (idx.indkey[position - 1] = ANY (fk.conkey))) AS prefix
      FROM fk
      JOIN pg_index AS idx ON idx.indrelid = fk.conrelid
      JOIN pg_class AS index_class ON index_class.oid = idx.indexrelid
     WHERE idx.indisvalid AND idx.indisready AND idx.indislive
       AND (idx.indpred IS NULL OR pg_get_expr(idx.indpred, idx.indrelid) IN (
           SELECT '(' || attribute.attname || ' IS NOT NULL)'
             FROM pg_attribute AS attribute
            WHERE attribute.attrelid = fk.conrelid AND attribute.attnum = ANY (fk.conkey)))
)
SELECT fk.name, fk.conrelid::regclass::text, fk.confrelid::regclass::text,
       EXISTS (SELECT 1 FROM usable WHERE usable.fk_oid = fk.oid AND usable.prefix >= cardinality(fk.conkey)),
       ARRAY(SELECT usable.index_name FROM usable WHERE usable.fk_oid = fk.oid AND usable.prefix > 0 ORDER BY 1),
       ARRAY(SELECT trigger.tgname::text FROM pg_trigger AS trigger
              WHERE trigger.tgrelid = fk.confrelid AND NOT trigger.tgisinternal
                -- BEFORE (2), DELETE (8) and UPDATE (16) of pg_trigger.tgtype
                AND trigger.tgtype & 26 = 26
              ORDER BY 1)
  FROM fk
 ORDER BY fk.name`

type foreignKeyProbe struct {
	name, referencing, referenced string
	covered                       bool
	prefixIndexes                 []string
	parentGuards                  []string
}

// TestForeignKeysHaveProbeIndexes checks every foreign key of the migrated
// schema: PostgreSQL probes the referencing table on the key columns for each
// deleted or re-keyed referenced row, so those columns must lead a usable
// index, or the allowlist must name the index or trigger that keeps the probe
// cheap (V370-023).
func TestForeignKeysHaveProbeIndexes(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool, _ := isolatedPools(t, ctx, databaseURL)
	if _, err := ApplyMigrations(ctx, pool); err != nil {
		t.Fatal(err)
	}
	rows, err := pool.Query(ctx, foreignKeyProbeIndexesSQL)
	if err != nil {
		t.Fatal(err)
	}
	probes, err := pgx.CollectRows(rows, func(row pgx.CollectableRow) (foreignKeyProbe, error) {
		var probe foreignKeyProbe
		err := row.Scan(&probe.name, &probe.referencing, &probe.referenced, &probe.covered,
			&probe.prefixIndexes, &probe.parentGuards)
		return probe, err
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(probes) < 50 {
		t.Fatalf("found only %d foreign keys; the catalog query is broken", len(probes))
	}
	allowed := make(map[string]bool, len(foreignKeyProbeAllowances))
	for _, probe := range probes {
		allowance, listed := foreignKeyProbeAllowances[probe.name]
		allowed[probe.name] = listed
		switch {
		case probe.covered && listed:
			t.Errorf("%s: %s now has an index led by all key columns; remove its allowance", probe.name, probe.referencing)
		case probe.covered:
		case !listed:
			t.Errorf("%s: %s has no index led by all key columns (usable prefixes %v); add one or an allowance",
				probe.name, probe.referencing, probe.prefixIndexes)
		case (allowance.prefixIndex == "") == (allowance.immutableParentTrigger == "") || allowance.reason == "":
			t.Errorf("%s: an allowance names exactly one index or trigger and a reason", probe.name)
		case allowance.prefixIndex != "" && !slices.Contains(probe.prefixIndexes, allowance.prefixIndex):
			t.Errorf("%s: %s is not a usable index of %s led by key columns (usable prefixes %v)",
				probe.name, allowance.prefixIndex, probe.referencing, probe.prefixIndexes)
		case allowance.immutableParentTrigger != "" && !slices.Contains(probe.parentGuards, allowance.immutableParentTrigger):
			t.Errorf("%s: %s is not a BEFORE DELETE OR UPDATE trigger of %s (found %v)",
				probe.name, allowance.immutableParentTrigger, probe.referenced, probe.parentGuards)
		}
	}
	for name := range foreignKeyProbeAllowances {
		if _, exists := allowed[name]; !exists {
			t.Errorf("%s: no such foreign key; remove its allowance", name)
		}
	}
}
