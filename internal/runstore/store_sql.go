package runstore

import (
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

// SQL statements for store.go.

// listNonTerminalRunIDsByCredentialSQL returns up to $2 Run IDs, oldest first,
// that reference LLM credential $1: non-terminal Runs whose Workflow snapshot
// (any credentialId field) or runtime config llmCredentialIds pins it, plus
// Runs holding a not-yet-released Stage allocation configured with it.
// The jsonb_path_query_array test is the expression of
// workflow_runs_workflow_credentials_idx: a superset of the exact
// jsonb_path_exists test, which it narrows to indexed candidates, so both
// arms are index scans over active Runs. The JSON members come from the
// constants beside the encoded types, so the text still matches the indexes.
// Used by PostgresStore.ListNonTerminalRunIDsByCredential.
var listNonTerminalRunIDsByCredentialSQL = `
WITH credential_runs AS (
    SELECT run_id, created_at
    FROM workflow_runs
    WHERE state IN ('initializing', 'pending', 'running', 'waiting', 'cancelling')
      AND (
          (
              jsonb_path_query_array(workflow_snapshot, '$.**.` + llmgateway.CredentialIDMember + `.**', '{}', true) ? $1
              AND jsonb_path_exists(
                  workflow_snapshot,
                  '$.**.` + llmgateway.CredentialIDMember + ` ? (@ == $credential)',
                  jsonb_build_object('credential', to_jsonb($1::text)),
                  true
              )
          )
          OR (runtime_config_snapshot->'` + runtimeconfig.LLMCredentialIDsMember + `') ? $1
      )
    UNION
    SELECT e.run_id, r.created_at
    FROM stage_allocations a
    JOIN stage_executions e ON e.stage_execution_id = a.stage_execution_id
    JOIN workflow_runs r ON r.run_id = e.run_id
    WHERE a.release_completed_at IS NULL
      AND a.runtime_configuration #>> '{` + AllocationProvenanceMember + `,` +
	runtimesettings.LLMCredentialMember + `,` + llmgateway.CredentialIDMember + `}' = $1
)
SELECT run_id
FROM credential_runs
ORDER BY created_at, run_id
LIMIT $2`

// createRunSQL inserts an 'initializing' WorkflowRun and its metadata labels
// (from the $12 JSON object, ordinals in key order) in one statement. It is a
// prefix: the caller appends the column list and "FROM inserted_run". A
// Project-deletion trigger rejects the insert with SQLSTATE 55000.
// Used by PostgresStore.CreateRun.
var createRunSQL = `
WITH inserted_run AS (
INSERT INTO workflow_runs (
    run_id, owner_id, project_id, workflow_name, workflow_version,
    workflow_schema_version, workflow_snapshot, parameters,
    runtime_labels, runtime_config_snapshot, project_http_target_snapshot, metadata_labels,
    state, state_reason_code, state_reason_message
) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8::jsonb, $9, $10::jsonb, $11::jsonb, $12::jsonb, 'initializing', 'created', '')
RETURNING *
)
SELECT `

// createRunIdempotentSQL is createRunSQL plus the owner's idempotency key ($12)
// and request digest ($13), with ON CONFLICT DO NOTHING: any unique conflict
// yields no row, and the caller then resolves the existing Run by key and
// digest. Labels come from $14; the caller appends the column list and
// "FROM inserted_run". Used by PostgresStore.CreateRunIdempotent.
var createRunIdempotentSQL = `
WITH inserted_run AS (
INSERT INTO workflow_runs (
    run_id, owner_id, project_id, workflow_name, workflow_version,
    workflow_schema_version, workflow_snapshot, parameters,
    runtime_labels, runtime_config_snapshot, project_http_target_snapshot,
    request_idempotency_key, request_digest, metadata_labels,
    state, state_reason_code, state_reason_message
) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8::jsonb, $9, $10::jsonb, $11::jsonb, $12, $13, $14::jsonb, 'initializing', 'created', '')
ON CONFLICT DO NOTHING
RETURNING *
)
SELECT `

// listRunsSQL returns one keyset page of an owner's Runs, newest first, with
// optional state, Project, lifecycle (active/terminal) and all-labels-match
// filters; the labels filter is a containment match on the Run's labels.
// Each row carries the metadata labels as a JSON object.
// internal/rundeletion reports which listed Runs are deletable.
// Used by PostgresStore.ListRuns.
var listRunsSQL = `
WITH page AS (
    SELECT run_id, project_id, workflow_name, workflow_version, state, created_at, updated_at, finished_at,
           metadata_labels
    FROM workflow_runs
    WHERE owner_id = $1
      AND ($2::text IS NULL OR state = $2)
      AND ($3::timestamptz IS NULL OR (created_at, run_id) < ($3, $4))
      AND ($8::text IS NULL OR project_id = $8)
      AND (
          $9::text IS NULL
          OR ($9 = 'active' AND state IN ('initializing', 'pending', 'running', 'waiting', 'cancelling'))
          OR ($9 = 'terminal' AND state IN ('succeeded', 'failed', 'cancelled'))
      )
      AND (
          cardinality($6::text[]) = 0
          OR (
              -- Exact duplicates are removed before the query, so a repeated
              -- key means contradictory values: nothing can match.
              (SELECT count(DISTINCT label_key) FROM unnest($6::text[]) AS label_key) = cardinality($6::text[])
              AND metadata_labels @> (
                  SELECT jsonb_object_agg(required.label_key, required.label_value)
                  FROM unnest($6::text[], $7::text[]) AS required(label_key, label_value)
              )
          )
      )
    ORDER BY created_at DESC, run_id DESC
    LIMIT $5
)
SELECT run_id, project_id, workflow_name, workflow_version, state,
       created_at, updated_at, finished_at, metadata_labels
FROM page
ORDER BY created_at DESC, run_id DESC`

// transitionRunSQL moves a Run from expected state $2 to $3 (state CAS) and
// records the reason. started_at is set on first entry to running; finished_at
// is set for terminal states and cleared otherwise, and terminal states also
// clear the scheduler lease. The caller appends the RETURNING column list; no
// row means the expected state did not match.
// Used by PostgresStore.TransitionRun.
var transitionRunSQL = `
UPDATE workflow_runs
SET state = $3,
    state_reason_code = $4,
    state_reason_message = $5,
    updated_at = clock_timestamp(),
    started_at = CASE WHEN $3 = 'running' AND started_at IS NULL THEN clock_timestamp() ELSE started_at END,
    finished_at = CASE WHEN $3 IN ('succeeded', 'failed', 'cancelled') THEN clock_timestamp() ELSE NULL END,
    scheduler_claim_id = CASE WHEN $3 IN ('succeeded', 'failed', 'cancelled') THEN NULL ELSE scheduler_claim_id END,
    scheduler_claimed_at = CASE WHEN $3 IN ('succeeded', 'failed', 'cancelled') THEN NULL ELSE scheduler_claimed_at END,
    scheduler_claim_expires_at = CASE WHEN $3 IN ('succeeded', 'failed', 'cancelled') THEN NULL ELSE scheduler_claim_expires_at END
WHERE run_id = $1 AND state = $2
RETURNING `

// requestRunCancellationSQL moves a non-terminal, not yet cancelling Run to
// 'cancelling' with reason user_cancelled and stores the cancellation payload.
// The caller appends the RETURNING column list; no row means the Run is
// already cancelling or terminal, and the caller re-reads it.
// Used by PostgresStore.RequestRunCancellation.
var requestRunCancellationSQL = `
UPDATE workflow_runs
SET state = 'cancelling',
    state_reason_code = 'user_cancelled',
    state_reason_message = COALESCE($4, ''),
    cancellation_schema_version = $2,
    run_cancellation = $3::jsonb,
    updated_at = clock_timestamp(),
    finished_at = NULL
WHERE run_id = $1 AND state IN ('initializing', 'pending', 'running', 'waiting')
RETURNING `

// claimRunnableRunSQL leases one runnable Run that holds no live Scheduler
// claim, in claim order (SCH-R121, SCH-R123): by scheduler_claim_tier
// (cancelling, then running and waiting, then pending Runs and Runs whose
// Skill initialization is pending), then updated_at, created_at and run_id.
// A Run deferred within the last $3 µs ranks behind all of them unless it is
// cancelling; deferred Runs follow each other in updated_at order.
//
// A pending Run of an owner whose Queue is paused is skipped: admission could
// not start its Stage (SCH-R163). It is still claimed while an active
// StageExecution of it is aborting or holds an allocation whose release has
// not completed, the release work that continues while paused (SCH-R148,
// SCH-R152).
//
// The ready probe reads claim order from workflow_runs_claim_idx and stops at
// the first Run it can lock; the deferred probe, read from
// workflow_runs_deferred_claim_idx, runs only when the ready probe found
// none. Neither sorts the candidates (SCH-R164). FOR UPDATE SKIP LOCKED keeps
// concurrent Schedulers off the same row. Sets claim $1 with a $2 µs lease;
// the caller appends the RETURNING columns.
// Used by PostgresStore.ClaimRunnableRun.
var claimRunnableRunSQL = `
WITH ready AS (
    SELECT run.run_id
    FROM workflow_runs AS run
    WHERE run.scheduler_claim_tier IS NOT NULL
      AND (run.scheduler_claim_id IS NULL OR run.scheduler_claim_expires_at <= clock_timestamp())
      AND NOT (run.scheduler_claim_tier > 0 AND run.scheduler_deferred
               AND run.updated_at > statement_timestamp() - ($3::bigint * interval '1 microsecond'))
      AND NOT (run.state = 'pending'
               AND run.owner_id IN (SELECT owner_id FROM owner_queue_controls WHERE paused)
               AND NOT EXISTS (
                   SELECT 1
                   FROM stage_executions AS stage
                   WHERE stage.run_id = run.run_id
                     AND stage.state IN ('preparing', 'running', 'finalizing', 'aborting')
                     AND (stage.state = 'aborting' OR EXISTS (
                         SELECT 1
                         FROM stage_allocations AS allocation
                         WHERE allocation.stage_execution_id = stage.stage_execution_id
                           AND allocation.release_completed_at IS NULL))))
    ORDER BY run.scheduler_claim_tier, run.updated_at, run.created_at, run.run_id
    LIMIT 1
    FOR UPDATE OF run SKIP LOCKED
), deferred AS (
    SELECT run.run_id
    FROM workflow_runs AS run
    WHERE run.scheduler_deferred AND run.scheduler_claim_tier > 0
      AND run.updated_at > statement_timestamp() - ($3::bigint * interval '1 microsecond')
      AND (run.scheduler_claim_id IS NULL OR run.scheduler_claim_expires_at <= clock_timestamp())
      AND NOT (run.state = 'pending'
               AND run.owner_id IN (SELECT owner_id FROM owner_queue_controls WHERE paused)
               AND NOT EXISTS (
                   SELECT 1
                   FROM stage_executions AS stage
                   WHERE stage.run_id = run.run_id
                     AND stage.state IN ('preparing', 'running', 'finalizing', 'aborting')
                     AND (stage.state = 'aborting' OR EXISTS (
                         SELECT 1
                         FROM stage_allocations AS allocation
                         WHERE allocation.stage_execution_id = stage.stage_execution_id
                           AND allocation.release_completed_at IS NULL))))
    ORDER BY run.updated_at, run.created_at, run.run_id
    LIMIT 1
    FOR UPDATE OF run SKIP LOCKED
), candidate AS (
    SELECT run_id FROM ready
    UNION ALL
    SELECT run_id FROM deferred
    LIMIT 1
)
UPDATE workflow_runs AS run
SET scheduler_claim_id = $1,
    scheduler_claimed_at = clock_timestamp(),
    scheduler_claim_expires_at = clock_timestamp() + ($2::bigint * interval '1 microsecond'),
    updated_at = clock_timestamp()
FROM candidate
WHERE run.run_id = candidate.run_id
RETURNING `
