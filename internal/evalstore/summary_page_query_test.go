package evalstore

import (
	"fmt"
	"strings"
	"testing"
	"time"
)

// TestSummaryPageQueryNamesOnlySetFilters checks that every filter and cursor
// combination becomes its own statement whose predicates are plain index
// conditions numbered after the owner, Project and limit (V414-026).
func TestSummaryPageQueryNamesOnlySetFilters(t *testing.T) {
	if strings.Count(summaryPageSQL, summaryPageFilters) != 1 {
		t.Fatalf("summaryPageSQL must contain %q exactly once", summaryPageFilters)
	}
	cursor := time.Date(2026, 10, 24, 0, 0, 0, 0, time.UTC)
	texts := map[string]bool{}
	for mask := 0; mask < 32; mask++ {
		p := SummaryPageParams{ListParams: ListParams{OwnerID: "owner", Limit: 25}}
		if mask&1 != 0 {
			p.ProjectID = "project"
		}
		if mask&2 != 0 {
			p.State = "draft"
		}
		if mask&4 != 0 {
			p.DatasetID = "dataset"
		}
		if mask&8 != 0 {
			p.ControlMode = "managed"
		}
		if mask&16 != 0 {
			p.AfterCreatedAt, p.AfterID = &cursor, "exp-1"
		}
		query, args := summaryPageQuery(p)
		if strings.Contains(query, "/* filters */") || strings.Contains(query, "= ''") || strings.Contains(query, "IS NULL OR") {
			t.Fatalf("mask %d: statement keeps an optional predicate:\n%s", mask, query)
		}
		if args[0] != "owner" || args[1] != p.ProjectID || args[2] != 26 {
			t.Fatalf("mask %d: fixed arguments = %v", mask, args[:3])
		}
		want := []string{}
		if p.ProjectID != "" {
			want = append(want, "AND project_id = $2")
		}
		n := 3
		for _, f := range []struct{ column, value string }{{"state", p.State}, {"dataset_id", p.DatasetID}, {"control_mode", p.ControlMode}} {
			if f.value != "" {
				n++
				want = append(want, fmt.Sprintf("AND %s = $%d", f.column, n))
				if args[n-1] != f.value {
					t.Fatalf("mask %d: $%d = %v, want %q", mask, n, args[n-1], f.value)
				}
			}
		}
		if p.AfterCreatedAt != nil {
			want = append(want, fmt.Sprintf("AND (created_at, experiment_id) < ($%d, $%d)", n+1, n+2))
			n += 2
		}
		if len(args) != n {
			t.Fatalf("mask %d: %d arguments, want %d", mask, len(args), n)
		}
		for _, clause := range want {
			if !strings.Contains(query, "WHERE owner_id = $1") || !strings.Contains(query, clause) {
				t.Fatalf("mask %d: statement lacks %q:\n%s", mask, clause, query)
			}
		}
		texts[query] = true
	}
	if len(texts) != 32 {
		t.Fatalf("%d distinct statements, want 32", len(texts))
	}
}
