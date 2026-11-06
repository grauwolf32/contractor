package runstore

import (
	"context"
	"fmt"
)

// The waiting operations move a Run between running and waiting while its
// invocation waits for a blocked model route (RUN-R30). Scheduler projects
// the live model-recovery waits; these operations are the only writers of those
// edges and of the reason they record. Each runs in the caller's
// transaction, which already holds the Run row lock (LockRunState) and checks
// the exact active Stage.

// gatewayWaitMessage is the state reason message of a Run that waits for a
// blocked model route, whether it waits to be admitted or in its invocation.
const gatewayWaitMessage = "Waiting for the model gateway"

// gatewayRecoveredReason is the state reason code of a Run that left
// waiting because none of its invocations waits for a model route any more.
const gatewayRecoveredReason = "gateway_recovered"

// EnterWaiting moves running Run runID to waiting, or gives a waiting Run
// another reason, with reasonCode, the failure code of the blocked route. A
// Run in any other state is left as it is: only an admitted Run has an
// invocation that can wait.
func (s *PostgresTxStore) EnterWaiting(ctx context.Context, runID, reasonCode string) error {
	if err := validateGatewayWait(runID, reasonCode); err != nil {
		return err
	}
	if _, err := s.tx.Exec(ctx, `
UPDATE workflow_runs
SET state = 'waiting', state_reason_code = $2, state_reason_message = $3, updated_at = clock_timestamp()
WHERE run_id = $1 AND state IN ('running', 'waiting')
  AND (state <> 'waiting' OR state_reason_code IS DISTINCT FROM $2)`,
		runID, reasonCode, gatewayWaitMessage); err != nil {
		return fmt.Errorf("move WorkflowRun %q to waiting: %w", runID, err)
	}
	return nil
}

// LeaveWaiting returns waiting Run runID to running once no invocation of it
// waits for a model route. A Run in any other state is left as it is.
func (s *PostgresTxStore) LeaveWaiting(ctx context.Context, runID string) error {
	if err := validateOpaque("runID", runID); err != nil {
		return err
	}
	if _, err := s.tx.Exec(ctx, `
UPDATE workflow_runs
SET state = 'running', state_reason_code = $2, state_reason_message = '', updated_at = clock_timestamp()
WHERE run_id = $1 AND state = 'waiting'`, runID, gatewayRecoveredReason); err != nil {
		return fmt.Errorf("return WorkflowRun %q to running: %w", runID, err)
	}
	return nil
}

// RecordAdmissionWait gives pending Run runID, whose Stage admission waits
// for a blocked model route, reasonCode, the failure code of that route. A
// Run in any other state, or one that already has this reason, is left as it
// is, so a Run deferred again keeps its claim order (updated_at).
func (s *PostgresTxStore) RecordAdmissionWait(ctx context.Context, runID, reasonCode string) error {
	if err := validateGatewayWait(runID, reasonCode); err != nil {
		return err
	}
	if _, err := s.tx.Exec(ctx, `
UPDATE workflow_runs
SET state_reason_code = $2, state_reason_message = $3, updated_at = clock_timestamp()
WHERE run_id = $1 AND state = 'pending' AND state_reason_code IS DISTINCT FROM $2`,
		runID, reasonCode, gatewayWaitMessage); err != nil {
		return fmt.Errorf("record the admission wait of WorkflowRun %q: %w", runID, err)
	}
	return nil
}

func validateGatewayWait(runID, reasonCode string) error {
	if err := validateOpaque("runID", runID); err != nil {
		return err
	}
	return validateReason(Reason{Code: reasonCode})
}
