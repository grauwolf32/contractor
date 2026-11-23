package runstore

// recordStageTransitionDecisionSQL records a decision only once on its source
// Stage $1 in Run $2. $3..$7 are its action, exact target and escalation facts.
// A missing source or previously recorded decision yields no row; the caller
// distinguishes them without overwriting history. Used by
// PostgresStore.RecordStageTransitionDecision.
const recordStageTransitionDecisionSQL = `
UPDATE stage_executions
SET transition_action = $3,
    transition_target_stage_name = $4,
    transition_target_execution_id = $5,
    transition_escalation_ordinal = $6,
    transition_escalation_exhausted = $7,
    transition_decided_at = clock_timestamp()
WHERE stage_execution_id = $1 AND run_id = $2
    AND transition_action IS NULL
RETURNING ` + stageTransitionDecisionColumns
