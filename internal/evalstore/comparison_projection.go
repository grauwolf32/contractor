package evalstore

import (
	"context"
	"encoding/json"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/jackc/pgx/v5"
)

// selectedSide is what a pair row stores of one side's selected member
// besides the pair document: its frozen ordinal and collection completeness.
type selectedSide struct {
	ordinal  int
	complete bool
}

// viewPairColumns are the columns projectComparison writes to a pair row. The
// member fields of both sides that member pages filter and sort by come from
// the same member views the pair document encodes (EVM-R245).
var viewPairColumns = []string{
	"experiment_id", "generation", "ordinal", "pair_id", "suite_id", "document", "regression", "unresolved",
	"tokens_a", "tokens_b", "duration_a", "duration_b", "complete_a", "complete_b",
	"member_ordinal_a", "member_ordinal_b", "variant_a", "variant_b", "eligibility_a", "eligibility_b",
	"execution_state_a", "execution_state_b", "assessment_a", "assessment_b", "conflicting_a", "conflicting_b",
}

// executionState is a member view's execution state, or nil when the view has
// no execution.
func executionState(view evaldomain.MemberView) *string {
	if view.Execution == nil {
		return nil
	}
	return &view.Execution.State
}

// projectComparison retains safe pair rows, with each side's collection
// completeness and member filter fields, and whole-cohort chart aggregates in
// the same transaction as the view generation. HTTP reads never collect Runs.
func (s *Store) projectComparison(ctx context.Context, id string, generation int64, view evaldomain.ComparisonView, sides map[string]selectedSide, verified bool) error {
	rows := make([][]any, 0, len(view.Pairs))
	suites := map[string][]evaldomain.Pair{"": view.Pairs}
	for ordinal, pair := range view.Pairs {
		suites[pair.SuiteID] = append(suites[pair.SuiteID], pair)
		ta, tb, tokenGap := evaldomain.ComparableMeasure(pair.A, pair.B, "tokens", verified)
		da, db, durationGap := evaldomain.ComparableMeasure(pair.A, pair.B, "duration", verified)
		var tokensA, tokensB, durationA, durationB *float64
		if tokenGap == "" {
			tokensA, tokensB = ta.Value, tb.Value
		}
		if durationGap == "" {
			durationA, durationB = da.Value, db.Value
		}
		a, foundA := sides[pair.A.Member.ID]
		b, foundB := sides[pair.B.Member.ID]
		if !foundA || !foundB {
			return evaldomain.Failure("eval_member_conflict")
		}
		rows = append(rows, []any{
			id, generation, ordinal, pair.ID, pair.SuiteID, bytesOf(pair), pair.Regression, len(pair.Exclusions) > 0,
			tokensA, tokensB, durationA, durationB, a.complete, b.complete,
			a.ordinal, b.ordinal, pair.A.Member.VariantID, pair.B.Member.VariantID,
			pair.A.Member.Eligibility, pair.B.Member.Eligibility, executionState(pair.A), executionState(pair.B),
			pair.A.Assessment, pair.B.Assessment, pair.A.Conflicting, pair.B.Conflicting,
		})
	}
	if _, err := s.tx.CopyFrom(ctx, pgx.Identifier{"eval_view_pairs"}, viewPairColumns, pgx.CopyFromRows(rows)); err != nil {
		return err
	}
	charts := make([][]any, 0, len(suites)*2)
	for suite, pairs := range suites {
		for _, metric := range []string{"tokens", "duration"} {
			cohort := evaldomain.MeasurementCohort(pairs, "", metric, "", verified)
			// Raw deltas are paged from the retained pairs, not duplicated in every aggregate.
			cohort.Deltas = nil
			charts = append(charts, []any{id, generation, suite, metric, bytesOf(cohort)})
		}
	}
	_, err := s.tx.CopyFrom(ctx, pgx.Identifier{"eval_view_charts"}, []string{"experiment_id", "generation", "suite_id", "metric", "document"}, pgx.CopyFromRows(charts))
	return err
}

func (s *Store) ChartCohort(ctx context.Context, owner, id string, generation int64, suite, metric string) (evaldomain.Cohort, error) {
	var out evaldomain.Cohort
	var raw []byte
	err := s.db.QueryRow(ctx, `
SELECT c.document
FROM eval_view_charts c
JOIN eval_experiments e USING(experiment_id)
WHERE e.owner_id = $1
    AND e.experiment_id = $2
    AND c.generation = $3
    AND c.suite_id = $4
    AND c.metric = $5
`, owner, id, generation, suite, metric).Scan(&raw)
	if err != nil {
		return out, normalize(err)
	}
	err = json.Unmarshal(raw, &out)
	return out, err
}
