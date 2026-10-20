package gatewayrecovery

// activeParticipantSQL checks participant $2 of Run $1 by its durable primary
// key. A Planner names its Stage; an allocation names its owning Stage through
// stage_allocations. Only running Stages can start or fail model calls. This
// same predicate is read once on the lock-free path and under the Run row lock
// before a mutating update, which orders it with every Stage-ending commit.
const activeParticipantSQL = `
EXISTS (
 SELECT 1 FROM stage_executions AS stage
 WHERE stage.run_id=$1 AND stage.state='running'
   AND stage.stage_execution_id=CASE
    WHEN left($2::text,8)='planner:' THEN substring($2::text FROM 9)
    WHEN left($2::text,11)='allocation:' THEN (
     SELECT allocation.stage_execution_id FROM stage_allocations AS allocation
     WHERE allocation.allocation_id=substring($2::text FROM 12))
    ELSE NULL
   END)`

// readUnchangedUpdateSQL reads the Run, route and participant from one
// committed snapshot without row locks (SCH-R165). Parameters are Run ID,
// participant ID, route key and model. Used by Service.unchanged.
const readUnchangedUpdateSQL = `
SELECT run.state, route.blocked, route.failure_code, route.probe_id,
       EXISTS (SELECT 1 FROM gateway_recovery_waits WHERE participant_id=$2 AND model=$4),
       ` + activeParticipantSQL + `
FROM workflow_runs AS run, gateway_recovery_routes AS route
WHERE run.run_id=$1 AND route.route_key=$3`

// readActiveParticipantSQL reads the same participant fact after the caller
// locks its Run; it introduces no second lock order. Used by Service.update.
const readActiveParticipantSQL = `SELECT ` + activeParticipantSQL
