package evalstore

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/grauwolf32/contractor/internal/evaldomain"
)

func TestPostgresEvalRecordRevisionsSelectionCASAndRetainedReview(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	scope := setupProject(t, pool, "owner", "eval")
	e := createExperiment(t, pool, scope, "experiment", "trace-1", "external-workflow")
	reader := NewPostgresStore(pool)
	ms := members(t, pool, e)
	var result evaldomain.ResultInput
	if err := json.Unmarshal(fixture(t, "result", "ResultInput").Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	member := result.MemberID
	found := false
	for _, m := range ms {
		found = found || m.MemberID == member
	}
	if !found {
		t.Fatal("result fixture not in plan")
	}
	result.PlanSHA256 = planDigest(t, pool, e)
	doc := freeze(t, "ResultInput", result)
	put := func(key, operation string, d evaldomain.Frozen) (Receipt, error) {
		var r Receipt
		err := transaction(pool, func(st *Store) error {
			var err error
			r, err = st.PutRecord(ctx, RecordParams{Scope: scope, ExperimentID: e.ID, MemberID: member, ActorID: scope.OwnerID, Operation: operation, Mutation: mutation(t, key, 0, d), Build: func(Experiment) (evaldomain.Frozen, error) { return d, nil }})
			return err
		})
		return r, err
	}
	r, err := put("result-key", "result", doc)
	if err != nil {
		t.Fatal(err)
	}
	var a evaldomain.AssessmentInput
	if err = json.Unmarshal(fixture(t, "human-assessment", "AssessmentInput").Bytes(), &a); err != nil {
		t.Fatal(err)
	}
	a.ResultSHA256 = doc.Digest()
	aDoc := freeze(t, "AssessmentInput", a)
	if _, err = put("review-key", "assessment", aDoc); err != nil {
		t.Fatal(err)
	}
	aSHA := aDoc.Digest()
	selection := evaldomain.SelectionInput{PlanSHA256: result.PlanSHA256, Selections: []evaldomain.SelectionEntry{{MemberID: member, ResultSHA256: doc.Digest(), AssessmentSHA256: &aSHA}}}
	selectDoc := freeze(t, "SelectionInput", selection)
	e, err = reader.Get(ctx, scope.OwnerID, e.ID)
	if err != nil {
		t.Fatal(err)
	}
	identity := mutation(t, "select-key", e.Revision, selectDoc)
	var selectedReceipt Receipt
	mustTx(t, pool, func(st *Store) error {
		var err error
		selectedReceipt, err = st.SelectRecords(ctx, scope, e.ID, scope.OwnerID, selection, identity)
		return err
	})
	a.PreviousAssessmentSHA256 = &aSHA
	a.Checks[0].Reason = "A new owner review over the same exact result."
	newReview := freeze(t, "AssessmentInput", a)
	if _, err = put("next-review", "assessment", newReview); err != nil {
		t.Fatal(err)
	}
	newSHA := newReview.Digest()
	selection.Selections[0].AssessmentSHA256 = &newSHA
	nextDoc := freeze(t, "SelectionInput", selection)
	err = transaction(pool, func(st *Store) error {
		_, err := st.SelectRecords(ctx, scope, e.ID, scope.OwnerID, selection, mutation(t, "select-race", e.Revision, nextDoc))
		return err
	})
	if !code(err, "eval_revision_mismatch") {
		t.Fatalf("stale selection: %v", err)
	}
	if _, err = reader.Record(ctx, scope.OwnerID, e.ID, member, "assessment", newSHA); err != nil {
		t.Fatal("review lost on selection conflict", err)
	}
	if _, err = reader.Record(ctx, "foreign", e.ID, member, "assessment", newSHA); !code(err, "eval_not_found") {
		t.Fatal("foreign record exposed", err)
	}
	selected, err := reader.Selected(ctx, scope.OwnerID, e.ID, member)
	if err != nil || selected.AssessmentSHA256 == nil || *selected.AssessmentSHA256 != aSHA {
		t.Fatal("conflict changed selection", selected, err)
	}
	selection.Selections[0].AssessmentSHA256 = &aSHA
	mustTx(t, pool, func(st *Store) error {
		replay, err := st.SelectRecords(ctx, scope, e.ID, scope.OwnerID, selection, identity)
		if err == nil && (!replay.Replayed || string(replay.Response) != string(selectedReceipt.Response)) {
			t.Fatal("selection receipt changed")
		}
		return err
	})
	replay, err := put("result-key", "result", doc)
	if err != nil || !replay.Replayed || string(replay.Response) != string(r.Response) {
		t.Fatal("immutable record replay", err)
	}
	var count int
	if err = pool.QueryRow(ctx, `SELECT count(*) FROM eval_selection_history WHERE experiment_id=$1`, e.ID).Scan(&count); err != nil || count != 1 {
		t.Fatal("selection history", count, err)
	}
	if _, err = pool.Exec(ctx, `UPDATE eval_records SET actor_id='someone-else' WHERE experiment_id=$1`, e.ID); err == nil {
		t.Fatal("record attribution mutated")
	}
}

func TestPostgresEvalCompleteViewPublicationAndInterruptedProjection(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	scope := setupProject(t, pool, "owner", "eval")
	e := createExperiment(t, pool, scope, "experiment", "trace-1", "external-workflow")
	claim := oneClaim(t, pool)
	reader := NewPostgresStore(pool)
	ms := members(t, pool, e)
	comparison := evaldomain.Comparison{Baseline: "a", Candidate: "b", Gates: evaldomain.Gates{MinCandidateEndToEndPass: 1}}
	project := func(changed bool) {
		dirty, err := reader.DirtyMembers(ctx, e.OwnerID, e.ID, 100)
		if err != nil {
			t.Fatal(err)
		}
		for _, d := range dirty {
			var m Member
			for _, row := range ms {
				if row.MemberID == d.MemberID {
					m = row
				}
			}
			v := evaldomain.MemberView{Member: evaldomain.MemberIdentity{ID: m.MemberID, SuiteID: m.SuiteID, CaseID: m.CaseID, Sample: m.Sample, VariantID: m.VariantID, CaseSHA256: m.CaseSHA256, BindingSHA256: m.BindingSHA256, Eligibility: m.Eligibility}, Execution: &evaldomain.ExecutionView{State: "not_submitted"}, Assessment: "unscored"}
			if changed {
				reason := "Fixture execution outcome is being reconciled."
				v.Execution.State = "unknown"
				v.Execution.Reason = &reason
			}
			mustTx(t, pool, func(st *Store) error {
				return st.ProjectMember(ctx, scope, e.ID, m.MemberID, claim, d.Revision, v, false, nil)
			})
		}
	}
	// An incomplete candidate may not leak a partial page or denominator.
	err := transaction(pool, func(st *Store) error { _, err := st.PublishView(ctx, scope, e.ID, claim, comparison, true); return err })
	if !code(err, "eval_not_ready") {
		t.Fatal("partial view published", err)
	}
	project(false)
	var first *View
	mustTx(t, pool, func(st *Store) error {
		var err error
		first, err = st.PublishView(ctx, scope, e.ID, claim, comparison, true)
		return err
	})
	if first == nil || first.Freshness != "current" || first.Summary.Counts["a"].Expected != 4 {
		t.Fatal("complete view", first)
	}
	if _, err = pool.Exec(ctx, `SELECT contractor_eval_dirty($1,$2)`, e.ID, ms[0].MemberID); err != nil {
		t.Fatal(err)
	}
	stale, err := reader.LatestView(ctx, e.OwnerID, e.ID)
	if err != nil || stale.Freshness != "stale" || stale.Snapshot != first.Snapshot {
		t.Fatal("previous complete view not retained stale", stale, err)
	}
	project(true)
	interrupted := errors.New("fixture interrupted before publication commit")
	err = transaction(pool, func(st *Store) error {
		_, err := st.PublishView(ctx, scope, e.ID, claim, comparison, true)
		if err != nil {
			return err
		}
		return interrupted
	})
	if !errors.Is(err, interrupted) {
		t.Fatal(err)
	}
	stale, err = reader.LatestView(ctx, e.OwnerID, e.ID)
	if err != nil || stale.Snapshot != first.Snapshot || stale.Freshness != "stale" {
		t.Fatal("uncommitted generation exposed", stale, err)
	}
	mustTx(t, pool, func(st *Store) error { _, err := st.PublishView(ctx, scope, e.ID, claim, comparison, true); return err })
	latest, err := reader.LatestView(ctx, e.OwnerID, e.ID)
	if err != nil || latest.Snapshot == first.Snapshot || latest.Freshness != "current" {
		t.Fatal("replacement view", latest, err)
	}
	oldRows, err := reader.ViewMembers(ctx, e.OwnerID, e.ID, first.Snapshot)
	if err != nil || len(oldRows) != 8 {
		t.Fatal("old report disappeared", len(oldRows), err)
	}
	foreign, err := reader.ViewMembers(ctx, "foreign", e.ID, first.Snapshot)
	if err != nil || len(foreign) != 0 {
		t.Fatal("foreign view exposed", err)
	}
	// Evidence recovery may restore the same selected documents, but its new
	// observation must not revive an old snapshot or its progress timestamp.
	if _, err = pool.Exec(ctx, `SELECT contractor_eval_dirty($1,$2)`, e.ID, ms[0].MemberID); err != nil {
		t.Fatal(err)
	}
	project(false)
	mustTx(t, pool, func(st *Store) error { _, err := st.PublishView(ctx, scope, e.ID, claim, comparison, true); return err })
	recovered, err := reader.LatestView(ctx, e.OwnerID, e.ID)
	if err != nil || recovered.Snapshot == first.Snapshot || recovered.Generation <= latest.Generation || !recovered.CreatedAt.After(first.CreatedAt) {
		t.Fatal("evidence recovery reused old publication", recovered, err)
	}

}
