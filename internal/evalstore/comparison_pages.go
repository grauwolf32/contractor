package evalstore

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/grauwolf32/contractor/internal/evaldomain"
)

// BinFilter is constructed only after the HTTP adapter verifies its signature
// and owner/experiment/snapshot/filter context.
type BinFilter struct {
	Metric         string  `json:"metric"`
	Scope          string  `json:"scope"`
	Lower          float64 `json:"lower"`
	Upper          float64 `json:"upper"`
	UpperInclusive bool    `json:"upperInclusive"`
}

type SelectedPageParams struct {
	OwnerID, ExperimentID, SuiteID, VariantID, Filter string
	Generation                                        int64
	AfterOrdinal, Limit                               int
	Bin                                               *BinFilter
	Metric                                            string
	Absolute                                          bool
	AfterDifference                                   *float64
}

type SelectedPage[T any] struct {
	Items          []T
	FilteredCount  int
	HasMore        bool
	LastOrdinal    int
	LastDifference *float64
}

func pairMeasureColumns(metric string) (string, string) {
	if metric == "duration" {
		return "p.duration_a", "p.duration_b"
	}
	return "p.tokens_a", "p.tokens_b"
}

func (p SelectedPageParams) binArgs() (bool, float64, float64, bool) {
	if p.Bin == nil {
		return false, 0, 0, false
	}
	return true, p.Bin.Lower, p.Bin.Upper, p.Bin.UpperInclusive
}

func (s *Store) PairPage(ctx context.Context, p SelectedPageParams) (SelectedPage[evaldomain.Pair], error) {
	a, b := pairMeasureColumns(p.Metric)
	if p.Bin != nil {
		a, b = pairMeasureColumns(p.Bin.Metric)
	}
	bin, lo, hi, inclusive := p.binArgs()
	query := fmt.Sprintf(`
WITH filtered AS MATERIALIZED (
    SELECT p.ordinal, p.document, abs(%s - %s) AS difference
    FROM eval_view_pairs p
    JOIN eval_experiments e USING (experiment_id)
    WHERE e.owner_id = $1 AND p.experiment_id = $2 AND p.generation = $3
        AND ($4 = '' OR p.suite_id = $4)
        AND ($5 IN ('', 'all')
            OR ($5 = 'regressions' AND p.regression)
            OR ($5 = 'unresolved' AND p.unresolved))
        AND (NOT $6 OR (
            (%s >= $7 AND (%s < $8 OR ($9 AND %s = $8)))
            OR (%s >= $7 AND (%s < $8 OR ($9 AND %s = $8)))
        ))
        AND ($10 = '' OR (%s IS NOT NULL AND %s IS NOT NULL))
)
`, b, a, a, a, a, b, b, b, a, b)
	args := []any{p.OwnerID, p.ExperimentID, p.Generation, p.SuiteID, p.Filter, bin, lo, hi, inclusive, p.Metric}
	return readSelectedPage[evaldomain.Pair](ctx, s, query, args, p)
}

func (s *Store) SelectedMemberPage(ctx context.Context, p SelectedPageParams) (SelectedPage[evaldomain.MemberView], error) {
	metric := "tokens"
	if p.Bin != nil {
		metric = p.Bin.Metric
	}
	a, b := pairMeasureColumns(metric)
	bin, lo, hi, inclusive := p.binArgs()
	query := fmt.Sprintf(`
WITH members AS (
    SELECT m.ordinal, m.document, m.collection_complete,
        convert_from(m.document, 'UTF8')::jsonb AS value,
        CASE WHEN convert_from(m.document, 'UTF8')::jsonb #>> '{member,variantId}' =
            convert_from(p.document, 'UTF8')::jsonb #>> '{a,member,variantId}'
            THEN %s ELSE %s
        END AS measure
    FROM eval_view_members m
    JOIN eval_experiments e USING (experiment_id)
    JOIN eval_view_pairs p ON p.experiment_id = m.experiment_id
        AND p.generation = m.generation AND p.pair_id = m.pair_id
    WHERE e.owner_id = $1 AND m.experiment_id = $2 AND m.generation = $3
        AND ($4 = '' OR p.suite_id = $4)
        AND (%s IS NOT NULL AND %s IS NOT NULL OR NOT $7)
), filtered AS MATERIALIZED (
    SELECT ordinal, document, NULL::double precision AS difference
    FROM members
    WHERE ($5 = '' OR value #>> '{member,variantId}' = $5)
        AND ($6 IN ('', 'all')
            OR ($6 = 'unresolved' AND (
                value #>> '{execution,state}' NOT IN ('succeeded', 'failed', 'cancelled')
                OR value ->> 'assessment' NOT IN ('pass', 'fail')
                OR NOT collection_complete OR (value ->> 'conflicting')::boolean
            ))
            OR ($6 = 'failed' AND value #>> '{execution,state}' = 'failed')
            OR ($6 = 'unscored' AND value ->> 'assessment' = 'unscored')
            OR ($6 IN ('unsupported', 'blocked') AND value #>> '{member,eligibility}' = $6)
            OR ($6 = 'conflicting' AND (value ->> 'conflicting')::boolean)
        )
        AND (NOT $7 OR (measure >= $8 AND (measure < $9 OR ($10 AND measure = $9))))
)
`, a, b, a, b)
	args := []any{p.OwnerID, p.ExperimentID, p.Generation, p.SuiteID, p.VariantID, p.Filter, bin, lo, hi, inclusive}
	return readSelectedPage[evaldomain.MemberView](ctx, s, query, args, p)
}

func readSelectedPage[T any](ctx context.Context, s *Store, query string, args []any, p SelectedPageParams) (SelectedPage[T], error) {
	out := SelectedPage[T]{Items: []T{}, LastOrdinal: -1}
	if err := s.db.QueryRow(ctx, query+"SELECT count(*) FROM filtered", args...).Scan(&out.FilteredCount); err != nil {
		return out, err
	}
	order, after := "ordinal", "ordinal>$11"
	args = append(args, p.AfterOrdinal)
	if p.Absolute {
		order = "difference DESC, ordinal"
		after = "($11::integer<0 OR difference<$12 OR (difference=$12 AND ordinal>$11))"
		args = append(args, p.AfterDifference)
	}
	args = append(args, p.Limit+1)
	pageSQL := fmt.Sprintf("SELECT ordinal,document,difference FROM filtered WHERE %s ORDER BY %s LIMIT $%d", after, order, len(args))
	rows, err := s.db.Query(ctx, query+pageSQL, args...)
	if err != nil {
		return out, err
	}
	defer rows.Close()
	for rows.Next() {
		var ordinal int
		var raw []byte
		var difference *float64
		if err = rows.Scan(&ordinal, &raw, &difference); err != nil {
			return out, err
		}
		if len(out.Items) == p.Limit {
			out.HasMore = true
			break
		}
		var item T
		if err = json.Unmarshal(raw, &item); err != nil {
			return out, err
		}
		out.Items = append(out.Items, item)
		out.LastOrdinal = ordinal
		out.LastDifference = difference
	}
	return out, rows.Err()
}

func (s *Store) ViewPair(ctx context.Context, owner, id string, generation int64, pairID string) (evaldomain.Pair, error) {
	var out evaldomain.Pair
	var raw []byte
	err := s.db.QueryRow(ctx, `
SELECT p.document
FROM eval_view_pairs p
JOIN eval_experiments e USING(experiment_id)
WHERE e.owner_id = $1
    AND p.experiment_id = $2
    AND p.generation = $3
    AND p.pair_id = $4
`, owner, id, generation, pairID).Scan(&raw)
	if err != nil {
		return out, normalize(err)
	}
	err = json.Unmarshal(raw, &out)
	return out, err
}
