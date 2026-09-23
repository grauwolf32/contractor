package public

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalservice"
	"github.com/grauwolf32/contractor/internal/evalstore"
)

func TestEvalPostgresExternalResultsRevisionsAndExplicitSelection(t *testing.T) {
	h := newEvalAPIHarness(t)
	e, manifest, checks := h.registerExternal(t, "workflow")
	memberID := manifest.Members[0].MemberID
	base := "/v1/eval-experiments/" + e.ID
	memberBase := base + "/members/" + memberID
	h.request(t, "POST", memberBase+"/submissions", evaldomain.Submission{PlanSHA256: *e.PlanSHA256}, "submit", "", 202)
	h.tick(t)
	h.finish(t, "workflow")
	h.tick(t)
	page := apiDecode[struct {
		Items []evaldomain.MemberView `json:"items"`
	}](t,
		h.request(t, "GET", base+"/members", nil, "", "", 200))
	var member evaldomain.MemberView
	for _, row := range page.Items {
		if row.Member.ID == memberID {
			member = row
		}
	}
	if member.Execution == nil || member.Usage == nil || member.ResultSHA256 != nil {
		t.Fatal("external collection must use explicit result selection", member)
	}
	result := evaldomain.ResultInput{
		SchemaVersion: evaldomain.ResultSchemaVersion, PlanSHA256: *e.PlanSHA256, MemberID: memberID,
		Source:    evaldomain.Source{System: "fixture", ID: "independent-collector"},
		Execution: *member.Execution, Usage: *member.Usage,
		Collection: evaldomain.Collection{Status: "complete", Gaps: []string{}},
		Outputs:    map[string]evaldomain.Artifact{}, Evidence: []evaldomain.Evidence{},
	}
	before := h.get(t, e.ID)
	accepted := h.request(t, "POST", memberBase+"/results", result, "external-result", "", 201)
	resultDigest := apiDecode[struct {
		SHA string `json:"recordSha256"`
	}](t, accepted).SHA
	recorded := h.get(t, e.ID)
	if recorded.LastProducerActivityAt == nil || !recorded.LastProducerActivityAt.After(*before.LastProducerActivityAt) {
		t.Fatal("accepted producer result did not advance its activity")
	}
	replay := h.request(t, "POST", memberBase+"/results", result, "external-result", "", 201)
	replayed := h.get(t, e.ID)
	if !bytes.Equal(accepted.Body.Bytes(), replay.Body.Bytes()) || replayed.Revision != recorded.Revision || !replayed.LastProducerActivityAt.Equal(*recorded.LastProducerActivityAt) {
		t.Fatal("result replay changed receipt or producer activity")
	}
	decision := evaldomain.AssessmentInput{
		SchemaVersion: evaldomain.AssessmentSchemaVersion, ResultSHA256: resultDigest,
		Source: evaldomain.AssessmentSource{Kind: "external", ProducerID: "independent-scorer", RecordSHA256: evaldomain.Digest([]byte("producer decision 1"))},
		Checks: []evaldomain.CheckResult{{ID: checks[0].ID, Evaluator: checks[0].Evaluator,
			ImplementationSHA256: checks[0].ImplementationSHA256, Status: "pass",
			Reason: "Declared external judgment", EvidenceRefs: []string{}}},
	}
	first := apiDecode[struct {
		SHA string `json:"recordSha256"`
	}](t,
		h.request(t, "POST", memberBase+"/assessments", decision, "assessment-1", "", 201)).SHA
	if selected, err := evalstore.NewPostgresStore(h.pool).Selected(t.Context(), "user-1", e.ID, memberID); err != nil || selected != nil {
		t.Fatal("external records auto-selected", selected, err)
	}
	current := h.get(t, e.ID)
	selectInput := evaldomain.SelectionInput{PlanSHA256: *e.PlanSHA256, Selections: []evaldomain.SelectionEntry{{MemberID: memberID, ResultSHA256: resultDigest, AssessmentSHA256: &first}}}
	h.request(t, "POST", base+"/selections", selectInput, "select-1", fmt.Sprintf("\"%d\"", current.Revision), 201)
	h.tick(t)
	report := apiDecode[evalservice.Report](t, h.request(t, "GET", base+"/report", nil, "", "", 200))
	attributed := false
	for _, source := range report.Sources {
		attributed = attributed || source.System == "external" && source.ID == decision.Source.ProducerID
	}
	if !attributed || report.Summary.Counts[member.Member.VariantID].EndToEndPassed != 1 || report.Summary.Conclusion != "inconclusive" {
		t.Fatal("external attribution or full denominators lost", report)
	}
	markdown := h.request(t, "GET", base+"/report?format=markdown", nil, "", "", 200)
	if !strings.Contains(markdown.Body.String(), "external / independent-scorer") {
		t.Fatal("Markdown export lost external provenance")
	}
	decision.PreviousAssessmentSHA256 = &first
	decision.Source.RecordSHA256 = evaldomain.Digest([]byte("producer decision 2"))
	decision.Checks[0].Status = "fail"
	next := apiDecode[struct {
		SHA string `json:"recordSha256"`
	}](t,
		h.request(t, "POST", memberBase+"/assessments", decision, "assessment-2", "", 201)).SHA
	selectInput.Selections[0].AssessmentSHA256 = &next
	h.request(t, "POST", base+"/selections", selectInput, "stale-selection", fmt.Sprintf("\"%d\"", current.Revision), 412)
	stored, err := evalstore.NewPostgresStore(h.pool).Record(t.Context(), "user-1", e.ID, memberID, evaldomain.RecordKindAssessment, next)
	if err != nil || stored.Predecessor == nil || *stored.Predecessor != first || stored.ActorID != "user-1" {
		t.Fatal("conflict lost attributed assessment revision", stored, err)
	}
	selected, err := evalstore.NewPostgresStore(h.pool).Selected(t.Context(), "user-1", e.ID, memberID)
	if err != nil || selected == nil || selected.AssessmentSHA256 == nil || *selected.AssessmentSHA256 != first {
		t.Fatal("new external revision replaced selected authority", selected, err)
	}
	var count int
	if err = h.pool.QueryRow(t.Context(), "SELECT count(*) FROM workflow_runs").Scan(&count); err != nil || count != 1 || h.get(t, e.ID).State != evaldomain.StateRunning {
		t.Fatal("external assessment dispatched or finalized executions", count, err)
	}
}

// Two identical result requests take their REPEATABLE READ snapshots before
// either commits; the second must replay the first's receipt, not conflict.
func TestEvalPostgresConcurrentResultRetryReplaysReceipt(t *testing.T) {
	h := newEvalAPIHarness(t)
	e, manifest, _ := h.registerExternal(t, "workflow")
	memberID := manifest.Members[0].MemberID
	base := "/v1/eval-experiments/" + e.ID
	h.request(t, "POST", base+"/members/"+memberID+"/submissions", evaldomain.Submission{PlanSHA256: *e.PlanSHA256}, "submit", "", 202)
	h.tick(t)
	h.finish(t, "workflow")
	h.tick(t)
	page := apiDecode[struct {
		Items []evaldomain.MemberView `json:"items"`
	}](t, h.request(t, "GET", base+"/members", nil, "", "", 200))
	var member evaldomain.MemberView
	for _, row := range page.Items {
		if row.Member.ID == memberID {
			member = row
		}
	}
	if member.Execution == nil || member.Usage == nil {
		t.Fatal("member execution was not collected", member)
	}
	raw, err := json.Marshal(evaldomain.ResultInput{
		SchemaVersion: evaldomain.ResultSchemaVersion, PlanSHA256: *e.PlanSHA256, MemberID: memberID,
		Source:    evaldomain.Source{System: "fixture", ID: "independent-collector"},
		Execution: *member.Execution, Usage: *member.Usage,
		Collection: evaldomain.Collection{Status: "complete", Gaps: []string{}},
		Outputs:    map[string]evaldomain.Artifact{}, Evidence: []evaldomain.Evidence{},
	})
	if err != nil {
		t.Fatal(err)
	}
	doc, err := evaldomain.Freeze("ResultInput", raw)
	if err != nil {
		t.Fatal(err)
	}
	mutation, err := evaldomain.IdentifyMutation("concurrent-result", "", false, doc.Kind(), doc.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	scope := evalstore.Scope{OwnerID: "user-1", ProjectID: "evaluation"}
	lockKey, err := json.Marshal([]string{scope.OwnerID, scope.ProjectID, e.ID + ":" + memberID, "result", mutation.Key})
	if err != nil {
		t.Fatal(err)
	}
	holder, err := h.pool.Begin(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	defer holder.Rollback(context.Background())
	if _, err = holder.Exec(t.Context(), `SELECT pg_advisory_xact_lock(hashtextextended($1,0))`, string(lockKey)); err != nil {
		t.Fatal(err)
	}
	type outcome struct {
		receipt evalstore.Receipt
		err     error
	}
	outcomes := make(chan outcome, 2)
	for range 2 {
		go func() {
			receipt, err := h.service.PutRecord(t.Context(), scope, e.ID, memberID, doc, mutation)
			outcomes <- outcome{receipt, err}
		}()
	}
	deadline := time.Now().Add(10 * time.Second)
	for waiting := 0; waiting != 2; {
		if time.Now().After(deadline) {
			t.Fatalf("%d requests queued on the idempotency lock, want 2", waiting)
		}
		time.Sleep(10 * time.Millisecond)
		if err = h.pool.QueryRow(t.Context(), `
SELECT count(*) FROM pg_locks
WHERE locktype = 'advisory' AND NOT granted
    AND database = (SELECT oid FROM pg_database WHERE datname = current_database())
    AND ((classid::bigint << 32) | objid::bigint) = hashtextextended($1,0)`, string(lockKey)).Scan(&waiting); err != nil {
			t.Fatal(err)
		}
	}
	if err = holder.Commit(t.Context()); err != nil {
		t.Fatal(err)
	}
	first, second := <-outcomes, <-outcomes
	if first.err != nil || second.err != nil {
		t.Fatalf("concurrent identical results: %v, %v", first.err, second.err)
	}
	if string(first.receipt.Response) != string(second.receipt.Response) || first.receipt.Replayed == second.receipt.Replayed {
		t.Fatalf("receipts = %+v and %+v, want one original and one exact replay", first.receipt, second.receipt)
	}
}
