package evalstore

import (
	"context"
	"encoding/json"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/jackc/pgx/v5"
)

// projectComparison retains safe pair rows and whole-cohort chart aggregates in
// the same transaction as the member snapshot. HTTP reads never collect Runs.
func (s *Store) projectComparison(ctx context.Context, id string, generation int64, view evaldomain.ComparisonView, verified bool) error {
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
		rows = append(rows, []any{id, generation, ordinal, pair.ID, pair.SuiteID, bytesOf(pair), pair.Regression, len(pair.Exclusions) > 0, tokensA, tokensB, durationA, durationB})
	}
	if _, err := s.tx.CopyFrom(ctx, pgx.Identifier{"eval_view_pairs"}, []string{"experiment_id", "generation", "ordinal", "pair_id", "suite_id", "document", "regression", "unresolved", "tokens_a", "tokens_b", "duration_a", "duration_b"}, pgx.CopyFromRows(rows)); err != nil {
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
