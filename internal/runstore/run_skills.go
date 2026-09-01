package runstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/jackc/pgx/v5"
)

const SkillInitializationPendingReason = "skill_initialization_pending"

// LockRunSkillInitialization serializes immediate initialization after
// POST /runs with Scheduler recovery of the same durable pending Run.
func (s *PostgresStore) LockRunSkillInitialization(
	ctx context.Context,
	runID string,
) (WorkflowRun, error) {
	if err := validateOpaque("runID", runID); err != nil {
		return WorkflowRun{}, err
	}
	run, err := scanWorkflowRun(s.db.QueryRow(ctx, `
SELECT `+workflowRunColumns+`
FROM workflow_runs
WHERE run_id = $1
	FOR UPDATE`, runID))
	if errors.Is(err, pgx.ErrNoRows) {
		return WorkflowRun{}, ErrNotFound
	}
	if err != nil {
		return WorkflowRun{}, fmt.Errorf("lock WorkflowRun Skill initialization: %w", err)
	}
	return run, nil
}

// SetRunSkillSelections records the complete current-owner selection or
// missing marker set while the Run create transaction is still open.
func (s *PostgresStore) SetRunSkillSelections(
	ctx context.Context,
	runID string,
	skills []contracts.RunSkillSnapshot,
) error {
	if err := validateOpaque("runID", runID); err != nil {
		return err
	}
	if len(skills) == 0 {
		return invalidf("Run Skill selection must not be empty")
	}
	if err := validateRunSkillSnapshot(skills, true); err != nil {
		return err
	}
	encoded, err := json.Marshal(skills)
	if err != nil {
		return fmt.Errorf("encode Run Skill selection: %w", err)
	}
	tag, err := s.db.Exec(ctx, `
UPDATE workflow_runs
SET skill_snapshot = $2::jsonb,
    state_reason_code = $3,
    state_reason_message = '',
    updated_at = clock_timestamp()
WHERE run_id = $1 AND state = 'initializing' AND state_reason_code = 'created'
  AND skill_snapshot = '[]'::jsonb`, runID, encoded, SkillInitializationPendingReason)
	if err != nil {
		return fmt.Errorf("record Run Skill selection: %w", err)
	}
	if tag.RowsAffected() != 1 {
		return &StateConflictError{Resource: "WorkflowRun Skill selection", ID: runID, Expected: "created"}
	}
	return nil
}

// CompleteRunSkillInitialization atomically replaces the selected snapshot
// with exact RunScope forks. The source half must remain byte-for-byte equal.
func (s *PostgresStore) CompleteRunSkillInitialization(
	ctx context.Context,
	runID string,
	skills []contracts.RunSkillSnapshot,
) error {
	if err := validateOpaque("runID", runID); err != nil {
		return err
	}
	if len(skills) == 0 {
		return invalidf("initialized Run Skill snapshot must not be empty")
	}
	if err := validateRunSkillSnapshot(skills, false); err != nil {
		return err
	}
	for _, skill := range skills {
		if !skill.Initialized() {
			return invalidf("Run Skill %q is not initialized", skill.Name)
		}
	}
	current, err := s.GetRun(ctx, runID)
	if err != nil {
		return err
	}
	if current.State != RunInitializing || current.StateReason.Code != SkillInitializationPendingReason || !sameRunSkillSources(current.SkillSnapshot, skills) {
		return &StateConflictError{Resource: "WorkflowRun Skill initialization", ID: runID, Expected: SkillInitializationPendingReason}
	}
	previous, err := json.Marshal(current.SkillSnapshot)
	if err != nil {
		return fmt.Errorf("encode selected Run Skill snapshot: %w", err)
	}
	encoded, err := json.Marshal(skills)
	if err != nil {
		return fmt.Errorf("encode initialized Run Skill snapshot: %w", err)
	}
	tag, err := s.db.Exec(ctx, `
UPDATE workflow_runs
SET skill_snapshot = $2::jsonb, updated_at = clock_timestamp()
WHERE run_id = $1 AND state = 'initializing' AND state_reason_code = $4
  AND skill_snapshot = $3::jsonb`, runID, encoded, previous, SkillInitializationPendingReason)
	if err != nil {
		return fmt.Errorf("complete Run Skill initialization: %w", err)
	}
	if tag.RowsAffected() != 1 {
		return &StateConflictError{Resource: "WorkflowRun Skill initialization", ID: runID, Expected: SkillInitializationPendingReason}
	}
	return nil
}

func validateRunSkillSnapshot(skills []contracts.RunSkillSnapshot, selectionsOnly bool) error {
	if len(skills) > contracts.MaxWorkflowRunSkills {
		return invalidf("Run Skill snapshot exceeds %d entries", contracts.MaxWorkflowRunSkills)
	}
	previous := ""
	var storedBytes int64
	var expandedBytes int64
	for _, skill := range skills {
		if err := skill.Validate(); err != nil {
			return invalidf("Run Skill %q is invalid: %v", skill.Name, err)
		}
		if previous != "" && skill.Name <= previous {
			return invalidf("Run Skill snapshot must be sorted and unique")
		}
		if selectionsOnly && skill.Initialized() {
			return invalidf("selected Run Skill %q is already initialized", skill.Name)
		}
		previous = skill.Name
		storedBytes += skill.SourceSize
		expandedBytes += skill.ExpandedBytes
	}
	if storedBytes > 256<<20 || expandedBytes > 512<<20 {
		return invalidf("Run Skill snapshot exceeds aggregate byte limits")
	}
	return nil
}

func sameRunSkillSources(left, right []contracts.RunSkillSnapshot) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index].Name != right[index].Name || left[index].SourceDigest != right[index].SourceDigest || left[index].SourceSize != right[index].SourceSize {
			return false
		}
		if (left[index].Source == nil) != (right[index].Source == nil) {
			return false
		}
		if left[index].Source != nil && (left[index].Source.Namespace != right[index].Source.Namespace || left[index].Source.Name != right[index].Source.Name || *left[index].Source.Revision != *right[index].Source.Revision) {
			return false
		}
	}
	return true
}
