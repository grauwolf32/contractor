package evalstore

import (
	"context"
	"encoding/json"
	"errors"
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

// PairPage reads one page of the view generation's pairs with the exact
// filtered count in one statement (EVM-R245).
func (s *Store) PairPage(ctx context.Context, p SelectedPageParams) (SelectedPage[evaldomain.Pair], error) {
	query, args := pairPageStatement(p)
	return readSelectedPage(ctx, s, query, args, p.Limit, func(_ string, _ int, document []byte) (evaldomain.Pair, error) {
		var pair evaldomain.Pair
		err := json.Unmarshal(document, &pair)
		return pair, err
	})
}

// SelectedMemberPage reads one page of the members of the view generation's
// pairs with the exact filtered count in one statement (EVM-R245). A member's
// view is its side of the pair document; a pair that two members of the page
// share is read and decoded once.
func (s *Store) SelectedMemberPage(ctx context.Context, p SelectedPageParams) (SelectedPage[evaldomain.MemberView], error) {
	query, args := memberPageStatement(p)
	pairs := map[int]evaldomain.Pair{}
	return readSelectedPage(ctx, s, query, args, p.Limit, func(arm string, pairOrdinal int, document []byte) (evaldomain.MemberView, error) {
		if document != nil {
			var pair evaldomain.Pair
			if err := json.Unmarshal(document, &pair); err != nil {
				return evaldomain.MemberView{}, err
			}
			pairs[pairOrdinal] = pair
		}
		pair, ok := pairs[pairOrdinal]
		switch {
		case !ok:
			return evaldomain.MemberView{}, errors.New("evalstore member page row has no pair document")
		case arm == "a":
			return pair.A, nil
		case arm == "b":
			return pair.B, nil
		}
		return evaldomain.MemberView{}, errors.New("evalstore member page row has no side")
	})
}

// pairPageStatement is the statement of pair page p and its arguments.
func pairPageStatement(p SelectedPageParams) (string, []any) {
	a, b := pairMeasureColumns(p.Metric)
	if p.Bin != nil {
		a, b = pairMeasureColumns(p.Bin.Metric)
	}
	bin, lo, hi, inclusive := p.binArgs()
	cohort := fmt.Sprintf(pairCohortSQL, b, a, a, a, a, b, b, b, a, b)
	return selectedPageStatement(cohort, []any{p.OwnerID, p.ExperimentID, p.Generation, p.SuiteID, p.Filter, bin, lo, hi, inclusive, p.Metric}, p)
}

// memberPageStatement is the statement of member page p and its arguments.
func memberPageStatement(p SelectedPageParams) (string, []any) {
	metric := "tokens"
	if p.Bin != nil {
		metric = p.Bin.Metric
	}
	a, b := pairMeasureColumns(metric)
	bin, lo, hi, inclusive := p.binArgs()
	cohort := fmt.Sprintf(memberCohortSQL, a, b, a, b)
	return selectedPageStatement(cohort, []any{p.OwnerID, p.ExperimentID, p.Generation, p.SuiteID, p.VariantID, p.Filter, bin, lo, hi, inclusive}, p)
}

// selectedPageStatement completes cohort, a pairCohortSQL or memberCohortSQL
// statement with its args, into the one statement of page p.
func selectedPageStatement(cohort string, args []any, p SelectedPageParams) (string, []any) {
	order, after := "ordinal", "ordinal > $11"
	args = append(args, p.AfterOrdinal)
	if p.Absolute {
		order = "difference DESC, ordinal"
		after = "($11::integer < 0 OR difference < $12 OR (difference = $12 AND ordinal > $11))"
		args = append(args, p.AfterDifference)
	}
	args = append(args, p.Limit)
	return cohort + fmt.Sprintf(selectedPageSQL, order, after, fmt.Sprintf("$%d::integer", len(args))), args
}

// readSelectedPage runs the statement of a page of at most limit items and
// decodes each item from its side ("a" or "b" of a member page, "" of a pair
// page), its pair's ordinal and the pair document, which is nil when an
// earlier item of the page had the same pair.
func readSelectedPage[T any](ctx context.Context, s *Store, query string, args []any, limit int, item func(arm string, pairOrdinal int, document []byte) (T, error)) (SelectedPage[T], error) {
	out := SelectedPage[T]{Items: []T{}, LastOrdinal: -1}
	rows, err := s.db.Query(ctx, query, args...)
	if err != nil {
		return out, err
	}
	defer rows.Close()
	for rows.Next() {
		var ordinal, pairOrdinal *int
		var arm *string
		var difference *float64
		var document []byte
		if err = rows.Scan(&out.FilteredCount, &ordinal, &pairOrdinal, &arm, &difference, &document); err != nil {
			return out, err
		}
		if ordinal == nil || pairOrdinal == nil {
			continue
		}
		if len(out.Items) == limit {
			out.HasMore = true
			break
		}
		side := ""
		if arm != nil {
			side = *arm
		}
		value, err := item(side, *pairOrdinal, document)
		if err != nil {
			return out, err
		}
		out.Items = append(out.Items, value)
		out.LastOrdinal = *ordinal
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
