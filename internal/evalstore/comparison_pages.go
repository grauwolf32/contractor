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
	query := fmt.Sprintf(pairPageSQL, b, a, a, a, a, b, b, b, a, b)
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
	query := fmt.Sprintf(selectedMemberPageSQL, a, b, a, b)
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
