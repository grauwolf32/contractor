package evalstore

import (
	"strings"
	"testing"
	"time"
)

// TestSummaryPageQueryHasFourShapes checks that the experiment list has one
// statement for each of its four shapes, every Project or one and first page
// or later, each without an optional predicate and with the arguments its
// placeholders number (EVM-R247).
func TestSummaryPageQueryHasFourShapes(t *testing.T) {
	if strings.Count(summaryPageSQL, summaryPagePredicates) != 1 {
		t.Fatalf("summaryPageSQL must contain %q exactly once", summaryPagePredicates)
	}
	cursor := time.Date(2026, 10, 24, 0, 0, 0, 0, time.UTC)
	texts := map[string]bool{}
	for _, project := range []string{"", "project"} {
		for _, after := range []*time.Time{nil, &cursor} {
			p := SummaryPageParams{OwnerID: "owner", ProjectID: project, Limit: 25, AfterCreatedAt: after, AfterID: "exp-1"}
			query, args := summaryPageQuery(p)
			end := strings.Index(query, "), details AS (")
			if end < 0 {
				t.Fatalf("summaryPageSQL lost its page selection:\n%s", query)
			}
			page := query[:end]
			if strings.Contains(page, summaryPagePredicates) || strings.Contains(page, "''") || strings.Contains(page, " OR ") {
				t.Fatalf("%q %v: the page selection keeps an optional predicate:\n%s", project, after != nil, page)
			}
			want := []any{"owner", project, 26}
			if after != nil {
				want = append(want, cursor, "exp-1")
			}
			if len(args) != len(want) {
				t.Fatalf("%q %v: arguments %v, want %v", project, after != nil, args, want)
			}
			for i := range want {
				if args[i] != want[i] {
					t.Fatalf("%q %v: arguments %v, want %v", project, after != nil, args, want)
				}
			}
			for clause, present := range map[string]bool{
				"WHERE owner_id = $1 AND project_id = $2":               project != "",
				"(created_at, experiment_id) < ($4, $5)":                after != nil,
				"ORDER BY created_at DESC, experiment_id DESC LIMIT $3": true,
			} {
				if strings.Contains(page, clause) != present {
					t.Fatalf("%q %v: the page selection contains %q = %v:\n%s", project, after != nil, clause, !present, page)
				}
			}
			texts[query] = true
		}
	}
	if len(texts) != 4 {
		t.Fatalf("%d distinct statements, want 4", len(texts))
	}
}
