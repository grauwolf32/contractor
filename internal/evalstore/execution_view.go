package evalstore

import (
	"context"
	"time"

	"github.com/grauwolf32/contractor/internal/evaldomain"
)

// ExecutionObservation contains only membership and ordinary execution facts.
// It deliberately cannot carry private recipes, expected checks or snapshots.
type ExecutionObservation struct {
	MemberID, PairID, SuiteID, CaseID, VariantID, Eligibility, Kind, CaseSHA256, BindingSHA256 string
	Ordinal, Sample                                                                            int
	SubmissionState                                                                            *string
	ExecutionID                                                                                *string
	OrdinaryState                                                                              *string
	StartedAt, FinishedAt                                                                      *time.Time
	Deleted                                                                                    bool
	Reason                                                                                     *string
}

func (s *Store) ExecutionObservations(ctx context.Context, owner, id string) ([]ExecutionObservation, error) {
	return s.executionObservations(ctx, owner, id, "")
}
func (s *Store) ExecutionObservation(ctx context.Context, owner, id, member string) (ExecutionObservation, error) {
	rows, err := s.executionObservations(ctx, owner, id, member)
	if err != nil {
		return ExecutionObservation{}, err
	}
	if len(rows) != 1 {
		return ExecutionObservation{}, evaldomain.Failure("eval_not_found")
	}
	return rows[0], nil
}
func (s *Store) executionObservations(ctx context.Context, owner, id, member string) ([]ExecutionObservation, error) {
	// One expected matrix is bounded by the frozen format. No historical Run scan.
	rows, err := s.db.Query(ctx, `
SELECT m.member_id,m.pair_id,m.ordinal,m.suite_id,m.case_id,m.sample,m.variant_id,m.eligibility,
       m.execution_kind,m.case_sha256,m.binding_sha256, sub.state,sub.execution_id,COALESCE(run.state,
                                                                                       audit.state),
       COALESCE(run.created_at, audit.created_at),COALESCE(run.finished_at,
                                                      audit.finished_at),t.execution_id IS NOT NULL,
                                                                                           m.eligibility_reason
FROM eval_members m
JOIN eval_experiments e USING(experiment_id)
LEFT JOIN eval_submissions sub USING(experiment_id,member_id)
LEFT JOIN workflow_runs run ON m.execution_kind = 'run'
AND sub.execution_id = run.run_id
AND run.owner_id = e.owner_id
LEFT JOIN audits AUDIT ON m.execution_kind = 'audit'
AND sub.execution_id = audit.audit_id
AND audit.owner_id = e.owner_id
LEFT JOIN eval_execution_tombstones t USING(experiment_id,member_id)
WHERE e.owner_id = $1
    AND e.experiment_id = $2
    AND ($3 = ''
         OR m.member_id = $3)
ORDER BY m.ordinal
LIMIT $4
`, owner, id, member, evaldomain.MaxMembers+1)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	result := []ExecutionObservation{}
	for rows.Next() {
		var r ExecutionObservation
		if err = rows.Scan(&r.MemberID, &r.PairID, &r.Ordinal, &r.SuiteID, &r.CaseID, &r.Sample, &r.VariantID, &r.Eligibility, &r.Kind, &r.CaseSHA256, &r.BindingSHA256, &r.SubmissionState, &r.ExecutionID, &r.OrdinaryState, &r.StartedAt, &r.FinishedAt, &r.Deleted, &r.Reason); err != nil {
			return nil, err
		}
		result = append(result, r)
	}
	if err = rows.Err(); err != nil {
		return nil, err
	}
	if len(result) > evaldomain.MaxMembers {
		return nil, evaldomain.Failure("eval_limit_exceeded")
	}
	return result, nil
}
