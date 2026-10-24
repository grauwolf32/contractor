package evalstore

import (
	"fmt"
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

// purgeRowColumns lists every column of each table an Eval purge deletes or
// rewrites. A row's stored size is the sum of pg_column_size over them
// (purgeRowBytes), which counts a compressed or out-of-line value at its
// stored size without reading it. TestSharedPostgresEvalPurgeMeasuresEveryColumn
// keeps the lists equal to the schema.
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
		"deletion_requested_at", "diagnostic", "view_generation", "created_at", "updated_at"},
	"eval_frozen_plans": {"experiment_id", "document_kind", "document", "document_sha256", "plan_sha256", "setup",
		"created_at", "readiness"},
	"eval_member_projections": {"experiment_id", "member_id", "revision", "projected_revision", "document",
		"collection_complete", "observed_document", "checked_at"},
	"eval_members": {"experiment_id", "member_id", "pair_id", "ordinal", "suite_id", "case_id", "sample", "variant_id",
		"case_sha256", "binding_sha256", "eligibility", "execution_kind", "recipe", "submission_key",
		"eligibility_reason"},
	"eval_plan_resources":        {"experiment_id", "resource_path", "document_kind", "document", "document_sha256"},
	"eval_progress_observations": {"experiment_id", "sequence", "observed_at", "counts"},
	"eval_projection_queue": {"experiment_id", "revision", "published_revision", "snapshot_id", "generation",
		"summary", "suites", "pins_verified", "content_sha256", "published_at"},
	"eval_records": {"experiment_id", "member_id", "kind", "record_sha256", "document", "actor_id",
		"predecessor_sha256", "created_at"},
	"eval_selections": {"experiment_id", "member_id", "result_kind", "result_sha256", "assessment_kind",
		"assessment_sha256", "revision", "actor_id"},
	"eval_submissions": {"experiment_id", "member_id", "state", "actor_id", "execution_kind", "execution_id",
		"execution_project_id", "terminal_state", "created_at", "updated_at", "observed_tokens"},
	"eval_suboperations": {"experiment_id", "member_id", "kind", "operation_key", "request", "request_sha256",
		"response", "state"},
	"eval_view_charts": {"experiment_id", "generation", "suite_id", "metric", "document"},
	"eval_view_pairs": {"experiment_id", "generation", "ordinal", "pair_id", "suite_id", "document", "regression",
		"unresolved", "tokens_a", "tokens_b", "duration_a", "duration_b", "complete_a", "complete_b"},
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

// experimentPurgePhases are the tables one experiment's purge empties before
// its final unit, leaf first (EVM-R243): the view rows and progress
// observations of its invalidated view; selections before the records they
// name; the evidence references, suboperations and submissions that the
// projection and tombstone triggers find members through, before the
// projections and tombstones those triggers write; members and plan
// resources, the children of the frozen plan; and the commands, which stay
// readable until then.
var experimentPurgePhases = []string{
	"eval_view_pairs", "eval_view_charts", "eval_progress_observations", "eval_selections",
	"eval_evidence_refs", "eval_suboperations", "eval_submissions", "eval_records",
	"eval_execution_tombstones", "eval_member_projections", "eval_members", "eval_plan_resources",
	"eval_commands",
}

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
// the frozen plan, projection queue and controller claim its deletion
// cascades to, and the two collection rows its deletion trigger rewrites. It
// returns the unit's row mutations and stored bytes. Every other child is
// already gone, so the cascade removes nothing else. Used by
// Store.purgeExperiment.
var finalPurgeUnitSQL = fmt.Sprintf(`
SELECT 3 + plan.n + queue.n + claim.n,
    (experiment.stored_bytes + plan.stored_bytes + queue.stored_bytes + claim.stored_bytes)::bigint
FROM (SELECT (%[1]s) + %[2]s AS stored_bytes
    FROM eval_experiments
    WHERE experiment_id = $1) experiment
CROSS JOIN (SELECT count(*) AS n, COALESCE(sum(%[3]s), 0) AS stored_bytes
    FROM eval_frozen_plans
    WHERE experiment_id = $1) plan
CROSS JOIN (SELECT count(*) AS n, COALESCE(sum(%[4]s), 0) AS stored_bytes
    FROM eval_projection_queue
    WHERE experiment_id = $1) queue
CROSS JOIN (SELECT count(*) AS n, COALESCE(sum(%[5]s), 0) AS stored_bytes
    FROM eval_controller_claims
    WHERE experiment_id = $1) claim
`, purgeRowBytes["eval_experiments"], collectionPairBytes, purgeRowBytes["eval_frozen_plans"],
	purgeRowBytes["eval_projection_queue"], purgeRowBytes["eval_controller_claims"])

// nextPurgedExperimentSQL names the first experiment left in a Project
// purge. Used by Store.PurgeProject.
const nextPurgedExperimentSQL = `
SELECT experiment_id
FROM eval_experiments
WHERE owner_id = $1 AND project_id = $2
ORDER BY experiment_id
LIMIT 1
`
