package evalstore

import (
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/evaldomain"
)

func TestPostgresEvalProgressBucketsRetainRealObservationsAndGaps(t *testing.T) {
	pool := testPool(t)
	scope := setupProject(t, pool, "owner", "eval")
	e := createExperiment(t, pool, scope, "progress", "trace-1", "external-workflow")
	start := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	e.StartedAt = &start
	// Publication history exceeds the response bound and contains a real gap.
	// Values distinguish whole-experiment counts from one suite's counts.
	for i := 0; i < 500; i++ {
		elapsed := time.Duration(i+1) * 20 * time.Second
		if i >= 250 {
			elapsed += 10000 * time.Second
		}
		counts := map[string]any{"a": i, "b": i + 1, "suites": map[string]any{
			"trace": map[string]any{"counts": map[string]any{
				"a": map[string]int{"terminal": i / 2}, "b": map[string]int{"terminal": i / 3},
			}},
		}}
		if _, err := pool.Exec(t.Context(), `INSERT INTO eval_progress_observations(experiment_id, observed_at, counts) VALUES($1,$2,$3)`, e.ID, start.Add(elapsed), bytesOf(counts)); err != nil {
			t.Fatal(err)
		}
	}
	view := View{CreatedAt: start.Add(20000 * time.Second)}
	store := NewPostgresStore(pool)
	for _, suite := range []string{"", "trace"} {
		points, width, err := store.Progress(t.Context(), e, view, suite, "a", "b")
		if err != nil || len(points) == 0 || len(points) > evaldomain.MaxProgressBuckets || width <= 0 {
			t.Fatal("unbounded progress", len(points), width, err)
		}
		if !points[0].GapBefore {
			t.Fatal("missing initial history became zero progress")
		}
		foundGap := false
		for i, point := range points {
			if point.A == nil || point.B == nil || point.ElapsedMS%20000 != 0 {
				t.Fatal("bucket invented an observation", point)
			}
			if i > 0 && point.GapBefore {
				foundGap = true
			}
		}
		if !foundGap {
			t.Fatal("missing publication history was interpolated")
		}
		wantA, wantB := 499, 500
		if suite != "" {
			wantA, wantB = 499/2, 499/3
		}
		last := points[len(points)-1]
		if *last.A != wantA || *last.B != wantB {
			t.Fatal("last real/suite counts lost", last)
		}
	}
	view.CreatedAt = start.Add(100 * time.Second)
	points, _, err := store.Progress(t.Context(), e, view, "", "a", "b")
	if err != nil || len(points) != 5 || *points[4].A != 4 {
		t.Fatal("later observations leaked into old snapshot", points, err)
	}
	points, _, err = store.Progress(t.Context(), e, view, "missing-suite", "a", "b")
	if err != nil || len(points) == 0 || points[0].A != nil || points[0].B != nil || !points[0].GapBefore {
		t.Fatal("missing counts became zero", points, err)
	}
	e.OwnerID = "foreign"
	points, _, err = store.Progress(t.Context(), e, view, "", "a", "b")
	if err != nil || len(points) != 0 {
		t.Fatal("foreign history exposed", err)
	}
}
