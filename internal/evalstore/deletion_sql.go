package evalstore

import (
	"fmt"
	"maps"
	"slices"
	"strings"
)

// SQL statements for deletion.go.

// purgeProjectSQL reports whether a Project still has eval state that blocks
// its purge: an experiment not yet fenced by deletion_requested_at, or an
// experiment suboperation still in 'intent'. A true result makes the caller
// return ErrDrain before deleting rows. Used by Store.PurgeProject.
var purgeProjectSQL = `
SELECT EXISTS(SELECT 1
    FROM eval_experiments
    WHERE owner_id=$1
        AND project_id=$2
        AND deletion_requested_at IS NULL) OR EXISTS(SELECT 1
    FROM eval_suboperations op
    JOIN eval_experiments e USING(experiment_id)
    WHERE e.owner_id=$1
        AND e.project_id=$2
        AND op.state='intent')
`

// purgeBlockedSQL reports whether an experiment still has an unresolved
// suboperation intent or a submission whose execution workspace exists, which
// keeps its purge waiting (ErrDrain). Used by Store.Purge.
const purgeBlockedSQL = `
SELECT EXISTS(SELECT 1
    FROM eval_suboperations
    WHERE experiment_id=$1
        AND state='intent') OR EXISTS(SELECT 1
    FROM eval_submissions s
    JOIN projects p ON p.project_id=s.execution_project_id
    WHERE s.experiment_id=$1)
`

// purgeRowColumns lists every column of each table an Eval purge deletes
// from or counts in its final unit. A row's stored size is the sum of
// pg_column_size over them (purgeRowBytes), which counts a compressed or
// out-of-line value at its stored size without reading it.
// TestSharedPostgresEvalPurgeCoversCascadesAndTriggers keeps the lists equal
// to the schema.
var purgeRowColumns = map[string][]string{
	"eval_collections": {"owner_id", "project_id", "revision"},
	"eval_commands": {"command_id", "experiment_id", "actor_id", "kind", "state", "accepted_revision",
		"diagnostic", "created_at", "finished_at"},
	"eval_controller_claims": {"experiment_id", "epoch", "holder_id", "expires_at", "claimed_at"},
	"eval_dataset_revisions": {"owner_id", "project_id", "dataset_id", "revision", "document", "document_sha256",
		"metadata", "created_at"},
	"eval_evidence_refs": {"experiment_id", "member_id", "record_sha256", "scope_kind", "scope_id", "namespace",
		"name", "revision"},
	"eval_execution_tombstones": {"experiment_id", "member_id", "execution_kind", "execution_id", "terminal_state",
		"never_started", "deleted_at"},
	"eval_experiments": {"experiment_id", "owner_id", "project_id", "portable_id", "control_mode", "name", "state",
		"revision", "draft", "dataset_id", "dataset_revision", "expected_count", "outstanding_count", "max_in_flight",
		"wall_ms", "token_limit", "observed_tokens", "started_at", "deadline_at", "last_producer_activity_at",
		"deletion_requested_at", "diagnostic", "view_generation", "created_at", "updated_at",
		"plan_kind", "plan_document", "plan_document_sha256", "plan_sha256", "plan_setup", "plan_created_at", "plan_readiness"},
	"eval_member_projections": {"experiment_id", "member_id", "revision", "projected_revision", "document",
		"collection_complete", "observed_document", "checked_at"},
	"eval_members": {"experiment_id", "member_id", "pair_id", "ordinal", "suite_id", "case_id", "sample", "variant_id",
		"case_sha256", "binding_sha256", "eligibility", "execution_kind", "recipe", "submission_key",
		"eligibility_reason", "selected_result_kind", "selected_result_sha256", "selected_assessment_kind",
		"selected_assessment_sha256", "selection_revision", "selected_by"},
	"eval_plan_resources":        {"experiment_id", "resource_path", "document_kind", "document", "document_sha256"},
	"eval_progress_observations": {"experiment_id", "sequence", "observed_at", "counts"},
	"eval_projection_queue": {"experiment_id", "revision", "published_revision", "snapshot_id", "generation",
		"summary", "suites", "pins_verified", "content_sha256", "published_at"},
	"eval_records": {"experiment_id", "member_id", "kind", "record_sha256", "document", "actor_id",
		"predecessor_sha256", "created_at"},
	"eval_submissions": {"experiment_id", "member_id", "state", "actor_id", "execution_kind", "execution_id",
		"execution_project_id", "terminal_state", "created_at", "updated_at", "observed_tokens"},
	"eval_suboperations": {"experiment_id", "member_id", "kind", "operation_key", "request", "request_sha256",
		"response", "state"},
	"eval_view_charts": {"experiment_id", "generation", "suite_id", "metric", "document"},
	"eval_view_pairs": {"experiment_id", "generation", "ordinal", "pair_id", "suite_id", "document", "regression",
		"unresolved", "tokens_a", "tokens_b", "duration_a", "duration_b", "complete_a", "complete_b",
		"member_ordinal_a", "member_ordinal_b", "variant_a", "variant_b", "eligibility_a", "eligibility_b",
		"execution_state_a", "execution_state_b", "assessment_a", "assessment_b", "conflicting_a", "conflicting_b"},
}

// purgeRowBytes is the SQL expression of one row's stored size per table.
var purgeRowBytes = func() map[string]string {
	out := make(map[string]string, len(purgeRowColumns))
	for table, columns := range purgeRowColumns {
		terms := make([]string, len(columns))
		for i, column := range columns {
			terms[i] = "COALESCE(pg_column_size(" + column + "), 0)"
		}
		out[table] = strings.Join(terms, " + ")
	}
	return out
}()

// collectionPairBytes is the stored size of the two eval_collections rows, the
// owner's row with an empty project_id and the Project's row, that the
// collection trigger rewrites when an experiment or dataset revision row with
// these owner_id and project_id columns is deleted; revision is a bigint.
const collectionPairBytes = `(2 * pg_column_size(owner_id) + pg_column_size(''::text) + pg_column_size(project_id) + 16)`

// purgeTrigger is a deletion trigger of a purged table that writes rows: the
// rows it writes, and the SQL expression of the bytes the purge charges for
// them, for each row the purge deletes from its table.
type purgeTrigger struct {
	table string
	rows  int64
	bytes string
}

// purgeDeleteTriggers are the deletion triggers that write rows when the
// purge deletes from their tables. The statement trigger of eval_experiments
// and the row trigger of eval_dataset_revisions each rewrite the two
// collection rows of the deleted row's owner and Project; the purge deletes
// one experiment per statement, so the statement trigger also writes two rows
// per experiment. Every other trigger that a purge statement fires only
// checks. TestSharedPostgresEvalPurgeCoversCascadesAndTriggers keeps this map
// equal to the schema, and the purge accounting tests compare the rows it
// declares with the rows PostgreSQL counts.
var purgeDeleteTriggers = map[string]purgeTrigger{
	"eval_experiments_collection_delete": {table: "eval_experiments", rows: 2, bytes: collectionPairBytes},
	"eval_datasets_collection":           {table: "eval_dataset_revisions", rows: 2, bytes: collectionPairBytes},
}

// purgeRowCost is the row mutations that deleting one row of table makes: the
// row itself and the rows its deletion triggers write.
func purgeRowCost(table string) int64 {
	cost := int64(1)
	for _, trigger := range purgeDeleteTriggers {
		if trigger.table == table {
			cost += trigger.rows
		}
	}
	return cost
}

// purgeChargedBytes is the SQL expression of the bytes the purge charges for
// deleting one row of table: the row's stored size and that of the rows its
// deletion triggers write.
func purgeChargedBytes(table string) string {
	terms := []string{purgeRowBytes[table]}
	for _, name := range slices.Sorted(maps.Keys(purgeDeleteTriggers)) {
		if trigger := purgeDeleteTriggers[name]; trigger.table == table {
			terms = append(terms, trigger.bytes)
		}
	}
	return strings.Join(terms, " + ")
}

// experimentPurgePhases are the tables one experiment's purge empties before
// its final unit, leaf first (EVM-R243): the view rows and progress
// observations of its invalidated view; selected record references have
// already been cleared by clearPurgedSelections; the evidence references,
// suboperations and submissions that projection and tombstone triggers find
// members through, before the
// projections and tombstones those triggers write; members and plan
// resources; and the commands, which stay
// readable until then. No table has a child that a later phase or the final
// unit removes (TestSharedPostgresEvalPurgeCoversCascadesAndTriggers).
var experimentPurgePhases = []string{
	"eval_view_pairs", "eval_view_charts", "eval_progress_observations",
	"eval_evidence_refs", "eval_suboperations", "eval_submissions", "eval_records",
	"eval_execution_tombstones", "eval_member_projections", "eval_members", "eval_plan_resources",
	"eval_commands",
}

// finalPurgeCascades are the children of eval_experiments that deleting the
// experiment row, the last unit of its purge, cascades to. Their own children
// are phases, so the cascade removes nothing else.
var finalPurgeCascades = []string{"eval_projection_queue", "eval_controller_claims"}

// projectPurgeTables are the tables a Project pass empties after its
// experiments and before its Eval receipts (pg.DeleteEvalReceiptsWithinBudget):
// its dataset revisions, which no experiment references any more, and then
// its collection row, which their deletion triggers rewrote.
var projectPurgeTables = []string{"eval_dataset_revisions", "eval_collections"}

// lockPurgedQueueSQL locks an experiment's projection queue row for one purge
// pass and reports whether it still names a selected view, with its stored
// size. The projection triggers take this row before they write a member
// projection, so none writes one while the pass runs. Used by
// Store.purgeExperiment.
var lockPurgedQueueSQL = fmt.Sprintf(`
SELECT generation IS NOT NULL, (%s)::bigint
FROM eval_projection_queue
WHERE experiment_id = $1
FOR UPDATE
`, purgeRowBytes["eval_projection_queue"])

// invalidatePurgedViewSQL removes the selected view pointer and its summary
// from an experiment's projection queue row before the purge deletes any row
// of that view, so no reader can resolve a partly deleted view. Used by
// Store.purgeExperiment.
const invalidatePurgedViewSQL = `
UPDATE eval_projection_queue
SET snapshot_id = NULL, generation = NULL, summary = NULL, suites = NULL,
    pins_verified = NULL, content_sha256 = NULL, published_at = NULL
WHERE experiment_id = $1
`

// finalPurgeUnitSQL measures the last unit of an experiment's purge: its row,
// the rows its deletion trigger writes and the rows of finalPurgeCascades its
// deletion cascades to. It returns the unit's row mutations and charged
// bytes. Every other child is already gone, so the cascade removes nothing
// else. Used by Store.purgeExperiment.
var finalPurgeUnitSQL = func() string {
	rows := []string{fmt.Sprint(purgeRowCost("eval_experiments"))}
	bytes := []string{"experiment.charged_bytes"}
	from := fmt.Sprintf(`
FROM (SELECT (%s)::bigint AS charged_bytes
    FROM eval_experiments
    WHERE experiment_id = $1) experiment`, purgeChargedBytes("eval_experiments"))
	for i, table := range finalPurgeCascades {
		alias := fmt.Sprintf("cascade%d", i)
		rows = append(rows, fmt.Sprintf("%d * %s.n", purgeRowCost(table), alias))
		bytes = append(bytes, alias+".charged_bytes")
		from += fmt.Sprintf(`
CROSS JOIN (SELECT count(*) AS n, COALESCE(sum(%s), 0) AS charged_bytes
    FROM %s
    WHERE experiment_id = $1) %s`, purgeChargedBytes(table), table, alias)
	}
	return "\nSELECT " + strings.Join(rows, " + ") + ", (" + strings.Join(bytes, " + ") + ")::bigint" + from + "\n"
}()

// nextPurgedExperimentSQL names the first experiment left in a Project
// purge. Used by Store.PurgeProject.
const nextPurgedExperimentSQL = `
SELECT experiment_id
FROM eval_experiments
WHERE owner_id = $1 AND project_id = $2
ORDER BY experiment_id
LIMIT 1
`

// clearPurgedSelectionsSQL clears the current selection before records are
// deleted. The member itself stays until its records and other children are
// gone. Measure at most the remaining row budget plus one, charge each full
// stored member row, and update only the longest prefix that fits. The purge
// holds the experiment and queue locks; clearing a selection cannot fire
// eval_members_selection_projection (its NEW result must be non-NULL).
// Used by Store.clearPurgedSelections.
var clearPurgedSelectionsSQL = fmt.Sprintf(`
WITH candidate AS MATERIALIZED (
    SELECT member_id, (%s)::bigint AS charged_bytes,
        row_number() OVER (ORDER BY member_id) AS position
    FROM eval_members
    WHERE experiment_id=$1 AND selected_result_sha256 IS NOT NULL
    ORDER BY member_id LIMIT $2::bigint + 1
), measured AS (
    SELECT *, sum(charged_bytes) OVER (ORDER BY position) AS running_bytes
    FROM candidate
), cleared AS (
    UPDATE eval_members m
    SET selected_result_sha256=NULL, selected_assessment_sha256=NULL,
        selection_revision=0, selected_by=NULL
    FROM measured c
    WHERE m.experiment_id=$1 AND m.member_id=c.member_id
        AND c.position <= $2::bigint
        AND (c.running_bytes <= $3::bigint OR ($4::boolean AND c.position=1))
    RETURNING c.charged_bytes
)
SELECT count(*), COALESCE(sum(charged_bytes),0)::bigint, (SELECT count(*) FROM candidate)
FROM cleared`, purgeRowBytes["eval_members"])
