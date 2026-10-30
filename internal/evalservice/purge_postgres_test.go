package evalservice

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"testing"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
	"github.com/jackc/pgx/v5"
)

// A deleted experiment larger than one purge pass is purged one bounded pass
// per controller tick, under the tick's claim. Between passes it still reads
// as itself, deleting and without a view, while every read of its members,
// pairs, report, charts, review or executions refuses with
// eval_project_deleting instead of a partial result; once the last pass has
// run the experiment is gone and its deletion request still replays
// (EVM-R243, EVM-R244, EVM-R226).
func TestSharedPostgresEvalPurgeYieldsBetweenTicksAndRefusesPartialReads(t *testing.T) {
	h := newHarness(t)
	ctx := t.Context()
	var logs bytes.Buffer
	logged, err := New(Options{Pool: h.pool, Resolver: h.resolver, Driver: h.driver, Logger: slog.New(slog.NewJSONHandler(&logs, nil))})
	if err != nil {
		t.Fatal(err)
	}
	h.service = logged
	e := h.prepared(t, "workflow")
	h.command(t, e, "start")
	c := h.coordinator(t, "controller")
	store := evalstore.NewPostgresStore(h.pool)
	for n := 0; n < 40; n++ {
		tick(t, c)
		h.finishRuns(t)
		e = h.get(t, e.ID)
		view, err := store.LatestView(ctx, e.OwnerID, e.ID)
		if err != nil {
			t.Fatal(err)
		}
		if e.State == evaldomain.StateFinished && view != nil {
			break
		}
	}
	member := ""
	if err := h.pool.QueryRow(ctx, `SELECT member_id FROM eval_members WHERE experiment_id = $1 ORDER BY ordinal LIMIT 1`, e.ID).Scan(&member); err != nil {
		t.Fatal(err)
	}
	if view, err := store.LatestView(ctx, e.OwnerID, e.ID); err != nil || view == nil || e.State != evaldomain.StateFinished {
		t.Fatalf("experiment did not finish with a view: %+v, %+v, %v", e, view, err)
	}
	// Evidence references standing in for large results: more rows than one
	// pass may delete. The seed runs without the test pool's statement
	// timeout; the purge passes under test keep it.
	if err := pgx.BeginFunc(ctx, h.pool, func(tx pgx.Tx) error {
		if _, err := tx.Exec(ctx, `SET LOCAL statement_timeout = 0`); err != nil {
			return err
		}
		_, err := tx.Exec(ctx, `
INSERT INTO eval_evidence_refs(experiment_id, member_id, record_sha256, scope_kind, scope_id, namespace, name, revision)
SELECT $1, $2, 'sha256:' || repeat('e', 64), 'user', 'eval-owner', 'fixtures', 'evidence-' || n, '1'
FROM generate_series(1, 2500) AS n`, e.ID, member)
		return err
	}); err != nil {
		t.Fatal(err)
	}
	deletion := frozen(t, "Delete", map[string]any{})
	if _, err := h.service.Delete(ctx, h.scope, e.ID, identity(t, "delete", e.Revision, deletion)); err != nil {
		t.Fatal(err)
	}

	evidence := func() int {
		t.Helper()
		var n int
		if err := h.pool.QueryRow(ctx, `SELECT count(*) FROM eval_evidence_refs WHERE experiment_id = $1`, e.ID).Scan(&n); err != nil {
			t.Fatal(err)
		}
		return n
	}
	tick(t, c)
	left := evidence()
	if left == 0 || left >= 2500 {
		t.Fatalf("one tick left %d evidence references, want one bounded pass", left)
	}
	view, err := h.service.Get(ctx, e.OwnerID, e.ID)
	if err != nil {
		t.Fatalf("Get during the purge: %v", err)
	}
	if view.DeletionRequestedAt == nil || view.ViewSnapshot != nil || view.Summary != nil || view.Freshness != "pending" || view.PlanSHA256 == nil {
		t.Fatalf("experiment during the purge = %+v", view)
	}
	list, err := h.service.List(ctx, evalstore.SummaryPageParams{OwnerID: e.OwnerID, ProjectID: e.ProjectID, Limit: 10})
	if err != nil || len(list.Items) != 1 || list.Items[0].ID != e.ID || list.Items[0].Summary != nil || list.Items[0].Freshness != "pending" {
		t.Fatalf("list during the purge = %+v, %v", list, err)
	}
	params := MemberPageParams{OwnerID: e.OwnerID, ExperimentID: e.ID, Limit: 10}
	pairs := PairPageParams{MemberPageParams: params}
	for name, read := range map[string]func() error{
		"members": func() error { _, err := h.service.Members(ctx, params); return err },
		"pairs":   func() error { _, err := h.service.Pairs(ctx, pairs); return err },
		"pair":    func() error { _, err := h.service.Pair(ctx, params, "pair"); return err },
		"report":  func() error { _, err := h.service.Report(ctx, params); return err },
		"chart": func() error {
			chart := PairPageParams{MemberPageParams: MemberPageParams{OwnerID: e.OwnerID, ExperimentID: e.ID, AfterOrdinal: -1}}
			_, err := h.service.Chart(ctx, ChartParams{PairPageParams: chart, Chart: "quality"})
			return err
		},
		"review":     func() error { _, err := h.service.Review(ctx, e.OwnerID, e.ID, member, ""); return err },
		"executions": func() error { _, err := h.service.Executions(ctx, e.OwnerID, e.ID, "", "", 10, nil); return err },
	} {
		var domainError *evaldomain.Error
		if err := read(); !errors.As(err, &domainError) || domainError.Code != "eval_project_deleting" {
			t.Errorf("%s during the purge = %v, want eval_project_deleting", name, err)
		}
	}
	if left := evidence(); left == 0 {
		t.Fatal("reads changed the purge")
	}

	ticks := 1
	for ; ticks < 20; ticks++ {
		if _, err := store.Get(ctx, e.OwnerID, e.ID); notFound(err) {
			break
		} else if err != nil {
			t.Fatal(err)
		}
		tick(t, c)
	}
	if _, err := store.Get(ctx, e.OwnerID, e.ID); !notFound(err) || ticks < 3 {
		t.Fatalf("purge after %d ticks: %v", ticks, err)
	}
	replay, err := h.service.Delete(ctx, h.scope, e.ID, identity(t, "delete", e.Revision, deletion))
	if err != nil || !replay.Replayed {
		t.Fatalf("deletion replay after the purge = %+v, %v", replay, err)
	}
	for _, table := range []string{"eval_members", "eval_records", "eval_evidence_refs", "eval_member_projections", "eval_view_pairs"} {
		if n := count(t, h.pool, table); n != 0 {
			t.Errorf("%s holds %d rows after the purge", table, n)
		}
	}
	if n := count(t, h.pool, "workflow_runs"); n != 8 {
		t.Fatalf("ordinary Runs = %d, want the eight executions kept", n)
	}
	// The purge meter logged the purge's start and its end with every pass.
	var records []string
	for decoder := json.NewDecoder(&logs); decoder.More(); {
		var record struct {
			Msg          string `json:"msg"`
			ExperimentID string `json:"experiment_id"`
			Passes, Rows int64
		}
		if err := decoder.Decode(&record); err != nil {
			t.Fatal(err)
		}
		if record.ExperimentID == e.ID {
			records = append(records, fmt.Sprintf("%s after %d passes", record.Msg, record.Passes))
			if record.Msg == "Eval purge finished" && record.Rows <= 2500 {
				t.Errorf("finished purge counted %d rows, want more than the 2500 evidence references", record.Rows)
			}
		}
	}
	if want := []string{"Eval purge started after 1 passes", fmt.Sprintf("Eval purge finished after %d passes", ticks)}; !slices.Equal(records, want) {
		t.Fatalf("purge records = %q, want %q", records, want)
	}
}
