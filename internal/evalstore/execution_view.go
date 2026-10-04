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
	rows, err := s.db.Query(ctx, executionObservationsSQL, owner, id, member, evaldomain.MaxMembers+1)
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
