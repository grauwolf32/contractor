package scanstore

import persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"

var claims = persistencepostgres.NewControllerClaims("scan_sweeps", "sweep_id")

// claimSQL takes one parent at a time; each tick's member and byte bounds
// therefore apply to the whole controller, not separately to 1000 parents.
var claimSQL = `WITH candidates AS (
 SELECT s.ctid AS claim_tid FROM scan_sweeps s
 WHERE s.state IN ('active','cancelling') AND s.next_tick_at <= clock_timestamp()
 AND ` + claims.AvailableCondition("s") + `
 ORDER BY s.next_tick_at,s.sweep_id LIMIT 1 FOR UPDATE SKIP LOCKED
)` + claims.LeaseSQL()

// sweepColumns deliberately omits the potentially large immutable closure.
// Public polling reads the maintained counters and never scans the members.
const sweepColumns = `sweep_id,workflow,snapshot_digest,state,total,pending,admitted,skipped,failed,cancelled,unsettled,created_at,settled_at`

// membersSQL returns one bounded ordered page and the Run's actual state.
// No Run FK is retained: deleting a Project must not delete admission facts.
const membersSQL = `SELECT m.ordinal,m.project_id,m.target,m.state,m.reason,m.run_id,r.state,m.settled,
 m.child,m.child_key,m.child_digest,m.attempts
FROM scan_sweep_members m LEFT JOIN workflow_runs r ON r.run_id = m.run_id
WHERE m.sweep_id = $1 AND m.ordinal > $2
ORDER BY m.ordinal LIMIT $3`

// workSQL rotates unsettled members by next_attempt_at. Settled members drop
// out of the partial index; neither polling nor a tick scans the whole sweep.
const workSQL = `SELECT m.ordinal,m.project_id,m.target,m.state,m.reason,m.run_id,r.state,m.settled,
 m.child,m.child_key,m.child_digest,m.attempts
FROM scan_sweep_members m LEFT JOIN workflow_runs r ON r.run_id = m.run_id
WHERE m.sweep_id = $1 AND NOT m.settled AND m.next_attempt_at <= clock_timestamp()
ORDER BY m.next_attempt_at,m.ordinal LIMIT $2`

// insertSweepSQL publishes counters and the immutable closure together.
const insertSweepSQL = `INSERT INTO scan_sweeps
(sweep_id,owner_id,workflow,snapshot,snapshot_digest,state,total,pending,skipped,unsettled,settled_at)
VALUES($1,$2,$3,$4,$5,CASE WHEN $7 = 0 THEN 'completed' ELSE 'active' END,$6,$7,$8,$7,
CASE WHEN $7 = 0 THEN clock_timestamp() ELSE NULL END)`

// advanceCountersSQL applies one member's delta under the already locked
// parent. Completion happens only after every child has actually settled.
const advanceCountersSQL = `UPDATE scan_sweeps SET
 pending = pending + $2, admitted = admitted + $3, skipped = skipped + $4,
 failed = failed + $5, cancelled = cancelled + $6, unsettled = unsettled + $7,
 settled_at = CASE WHEN unsettled + $7 = 0 THEN clock_timestamp() ELSE NULL END,
 state = CASE WHEN unsettled + $7 = 0 THEN
   CASE WHEN state = 'cancelling' THEN 'cancelled' ELSE 'completed' END ELSE state END
WHERE sweep_id = $1`
