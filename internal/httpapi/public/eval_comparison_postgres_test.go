package public

import (
	"bytes"
	"fmt"
	"net/url"
	"strconv"
	"strings"
	"testing"

	"github.com/getkin/kin-openapi/openapi3filter"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalservice"
	"github.com/grauwolf32/contractor/internal/evalstore"
	pg "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// Markdown exports use the same scalar string semantics as text/plain.
func init() { openapi3filter.RegisterBodyDecoder("text/markdown", openapi3filter.PlainBodyDecoder) }

func completedEvalFixture(t *testing.T) (*evalAPIHarness, evalservice.ExperimentView, []evaldomain.MemberView) {
	t.Helper()
	h := newEvalAPIHarness(t)
	draft, _ := h.dataset(t, "workflow")
	draft.Budgets.MaxInFlight = 8
	created := apiDecode[struct {
		ID string `json:"experimentId"`
	}](t, h.request(t, "POST", "/v1/projects/evaluation/eval-experiments", evaldomain.CreateExperiment{Name: "Comparison fixture", ControlMode: "server", Draft: &draft}, "create-comparison", "", 201))
	h.command(t, created.ID, "prepare")
	h.tick(t)
	h.command(t, created.ID, "start")
	var e evalservice.ExperimentView
	for step := 0; step < 32; step++ {
		h.tick(t)
		h.finish(t, "workflow")
		// Deterministic parent intervals; no models or fabricated stage usage.
		if _, err := h.pool.Exec(t.Context(), `UPDATE workflow_runs SET finished_at=created_at+interval '1 second' WHERE state='succeeded' AND finished_at IS DISTINCT FROM created_at+interval '1 second'`); err != nil {
			t.Fatal(err)
		}
		h.tick(t)
		e = h.get(t, created.ID)
		if e.State == evaldomain.StateFinished {
			break
		}
	}
	if e.State != evaldomain.StateFinished {
		t.Fatal("fixture failed to drain", e.State)
	}

	page := apiDecode[struct {
		Items []evaldomain.MemberView `json:"items"`
	}](t, h.request(t, "GET", "/v1/eval-experiments/"+e.ID+"/members", nil, "", "", 200))
	if len(page.Items) != 8 {
		t.Fatalf("expected matrix: %d", len(page.Items))
	}
	for _, member := range page.Items {
		if member.Execution.State != "succeeded" || member.ResultSHA256 == nil || member.Assessment != "incomplete" {
			t.Fatalf("collection must not invent quality: %+v", member)
		}
	}
	return h, e, page.Items
}

func ownerEvalReview(t *testing.T, h *evalAPIHarness, path string) evalservice.ReviewContext {
	t.Helper()
	response := serveAndValidatePublicContract(t, h.contract, h.handler, newPublicContractRequest("GET", path, nil), true)
	if response.Code != 200 {
		t.Fatalf("review: %d %s", response.Code, response.Body.String())
	}
	return apiDecode[evalservice.ReviewContext](t, response)
}

func TestEvalPostgresReviewSelectionReplayAndObservedResultAuthority(t *testing.T) {
	h, e, members := completedEvalFixture(t)
	member := members[0]
	base := "/v1/eval-experiments/" + e.ID
	memberPath := base + "/members/" + member.Member.ID
	review := ownerEvalReview(t, h, memberPath+"/review")
	if len(review.Checks) != 1 || review.Checks[0].Revision != review.Policy[0].RubricRevision {
		t.Fatal("owner rubric pin missing")
	}
	if _, err := h.service.Review(t.Context(), "user-2", e.ID, member.Member.ID, ""); !evaldomain.IsCode(err, "eval_not_found") {
		t.Fatal("foreign review", err)
	}
	check := review.Policy[0]
	assessment := evaldomain.AssessmentInput{SchemaVersion: evaldomain.AssessmentSchemaVersion, Source: evaldomain.AssessmentSource{Kind: "human"}, ResultSHA256: review.ResultSHA256, PreviousAssessmentSHA256: member.AssessmentSHA256, Checks: []evaldomain.CheckResult{{ID: check.ID, Evaluator: check.Evaluator, ImplementationSHA256: check.ImplementationSHA256, Status: "pass", Reason: "Owner decision", EvidenceRefs: []string{}}}}
	receipt := h.request(t, "POST", memberPath+"/assessments", assessment, "review-first", "", 201)
	digest := apiDecode[struct {
		SHA string `json:"recordSha256"`
	}](t, receipt).SHA
	selection := evaldomain.SelectionInput{PlanSHA256: *e.PlanSHA256, Selections: []evaldomain.SelectionEntry{{MemberID: member.Member.ID, ResultSHA256: review.ResultSHA256, AssessmentSHA256: &digest}}}
	etag := strconv.Quote(strconv.FormatInt(review.Revision, 10))
	h.request(t, "POST", base+"/selections", selection, "select-missing-cas", "", 428)
	selected := h.request(t, "POST", base+"/selections", selection, "select-first", etag, 201)
	assessment.PreviousAssessmentSHA256 = &digest
	assessment.Checks[0].Status = "fail"
	second := h.request(t, "POST", memberPath+"/assessments", assessment, "review-next", "", 201)
	nextDigest := apiDecode[struct {
		SHA string `json:"recordSha256"`
	}](t, second).SHA
	selection.Selections[0].AssessmentSHA256 = &nextDigest
	h.request(t, "POST", base+"/selections", selection, "select-stale", etag, 412)
	stored, err := evalstore.NewPostgresStore(h.pool).Record(t.Context(), "user-1", e.ID, member.Member.ID, "assessment", nextDigest)
	if err != nil || stored.ActorID != "user-1" {
		t.Fatal("CAS conflict lost attributed review", err)
	}
	selection.Selections[0].AssessmentSHA256 = &digest
	replay := h.request(t, "POST", base+"/selections", selection, "select-first", etag, 201)
	if !bytes.Equal(selected.Body.Bytes(), replay.Body.Bytes()) {
		t.Fatal("selection replay changed")
	}
	h.tick(t)
	page := apiDecode[struct {
		Summary evaldomain.Summary `json:"experimentSummary"`
	}](t, h.request(t, "GET", base+"/members", nil, "", "", 200))
	if page.Summary.Counts[member.Member.VariantID].EndToEndPassed != 1 {
		t.Fatal("selected owner verdict missing")
	}
	result := review.Result
	result.Source = evaldomain.Source{System: "fixture", ID: "external-collector"}
	result.PreviousResultSHA256 = &review.ResultSHA256
	result.Execution.State = "failed"
	h.request(t, "POST", memberPath+"/results", result, "forged-execution", "", 409)
	result.Execution = review.Result.Execution
	one := 1.0
	result.Usage.TotalTokens.Value = &one
	result.Usage.TotalTokens.Completeness = "partial"
	h.request(t, "POST", memberPath+"/results", result, "forged-usage", "", 409)
	result.Usage = review.Result.Usage
	accepted := h.request(t, "POST", memberPath+"/results", result, "external-result", "", 201)
	replayed := h.request(t, "POST", memberPath+"/results", result, "external-result", "", 201)
	if !bytes.Equal(accepted.Body.Bytes(), replayed.Body.Bytes()) {
		t.Fatal("result replay changed")
	}
	assessment.Source.Kind = "native"
	h.request(t, "POST", memberPath+"/assessments", assessment, "forged-native", "", 422)
	checks := evaldomain.CheckRequest{SchemaVersion: "contractor.eval-check-request/v1", Source: evaldomain.AssessmentSource{Kind: "native"}, ResultSHA256: review.ResultSHA256, CheckIDs: []string{check.ID}}
	h.request(t, "POST", memberPath+"/assessments", checks, "rerun-check", "", 201)
	var executions int
	if err = h.pool.QueryRow(t.Context(), "SELECT count(*) FROM workflow_runs").Scan(&executions); err != nil || executions != 8 {
		t.Fatal("reassessment ran executions", executions, err)
	}
}

func TestEvalPostgresChartsPairPagesAndBoundBinTokens(t *testing.T) {
	h, e, members := completedEvalFixture(t)
	base := "/v1/eval-experiments/" + e.ID
	snapshot := url.QueryEscape(*e.ViewSnapshot)
	chart := apiDecode[evalservice.ChartView](t, h.request(t, "GET", base+"/charts/duration?viewSnapshot="+snapshot, nil, "", "", 200))
	if chart.Bins == nil || len(*chart.Bins) != 1 || chart.Coverage.Included != 4 || *chart.Distributions["a"].P50 != 1000 || *chart.Distributions["b"].P90 != 1000 {
		t.Fatalf("exact all-equal cohort: %+v", chart)
	}
	bin := (*chart.Bins)[0]
	query := "?viewSnapshot=" + snapshot + "&binFilter=" + url.QueryEscape(bin.FilterToken) + "&limit=1"
	first := apiDecode[struct {
		Summary evaldomain.Summary `json:"experimentSummary"`
		Count   int                `json:"filteredCount"`
		Items   []evaldomain.Pair  `json:"items"`
		Page    evalPageInfo       `json:"page"`
	}](t, h.request(t, "GET", base+"/pairs"+query, nil, "", "", 200))
	if first.Count != 4 || first.Page.NextCursor == nil || len(first.Items) != 1 {
		t.Fatal("bin pair page", first.Count)
	}
	second := h.request(t, "GET", base+"/pairs"+query+"&cursor="+url.QueryEscape(*first.Page.NextCursor), nil, "", "", 200)
	if strings.Contains(second.Body.String(), first.Items[0].ID) {
		t.Fatal("pair keyset repeated first pair")
	}
	memberPage := apiDecode[struct {
		Count int `json:"filteredCount"`
	}](t, h.request(t, "GET", base+"/members"+query, nil, "", "", 200))
	if memberPage.Count != bin.Counts["a"]+bin.Counts["b"] {
		t.Fatal("bin counts and drill-down differ")
	}
	h.request(t, "GET", base+"/pairs"+query+"&filter=unresolved", nil, "", "", 422)
	h.request(t, "GET", base+"/members"+query+"&variantId=a", nil, "", "", 422)
	h.request(t, "GET", base+"/pairs"+query+"&measurementScope=audit", nil, "", "", 422)
	h.request(t, "GET", base+"/pairs"+query+"&suiteId=other", nil, "", "", 422)
	h.request(t, "GET", "/v1/eval-experiments/another/pairs"+query, nil, "", "", 422)
	h.request(t, "GET", base+"/pairs?binFilter="+url.QueryEscape(bin.FilterToken+"tampered"), nil, "", "", 422)
	detail := apiDecode[evalservice.PairDetail](t, h.request(t, "GET", base+"/pairs/"+first.Items[0].ID+"?viewSnapshot="+snapshot, nil, "", "", 200))
	if len(detail.Records) != 4 {
		t.Fatal("selected record provenance omitted")
	}
	deltas := apiDecode[struct {
		Differences []evaldomain.Delta `json:"differences"`
		Page        evalPageInfo       `json:"page"`
	}](t, h.request(t, "GET", base+"/charts/pair-deltas?metric=duration&sort=absolute&limit=1", nil, "", "", 200))
	if len(deltas.Differences) != 1 || deltas.Differences[0].Difference != 0 || deltas.Page.NextCursor == nil {
		t.Fatal("raw deltas", deltas)
	}
	h.request(t, "GET", base+"/charts/pair-deltas?metric=duration&sort=absolute&limit=1&cursor="+url.QueryEscape(*deltas.Page.NextCursor), nil, "", "", 200)
	for _, path := range []string{"tokens", "quality", "progress"} {
		h.request(t, "GET", base+"/charts/"+path, nil, "", "", 200)
	}
	for _, bad := range []string{"unknown", "tokens?metric=duration", "duration?measurementScope=audit", "quality?measurementScope=workflow", "tokens?limit=1", "pair-deltas?metric=unknown"} {
		h.request(t, "GET", base+"/charts/"+bad, nil, "", "", 422)
	}
	h.request(t, "GET", base+"/report", nil, "", "", 200)
	h.request(t, "GET", base+"/report?format=markdown", nil, "", "", 200)
	h.request(t, "GET", base+"/members/"+members[0].Member.ID+"/executions?limit=1", nil, "", "", 200)
	// Selected authority changes retain the old snapshot as stale until publication.
	review := ownerEvalReview(t, h, base+"/members/"+members[0].Member.ID+"/review")
	entry := evaldomain.SelectionEntry{MemberID: members[0].Member.ID, ResultSHA256: review.ResultSHA256}
	h.request(t, "POST", base+"/selections", evaldomain.SelectionInput{PlanSHA256: *e.PlanSHA256, Selections: []evaldomain.SelectionEntry{entry}}, "clear-assessment", fmt.Sprintf("\"%d\"", review.Revision), 201)
	stale := apiDecode[evalservice.ChartView](t, h.request(t, "GET", base+"/charts/duration?viewSnapshot="+snapshot, nil, "", "", 200))
	if stale.Freshness != "stale" {
		t.Fatal("stale projection mislabeled")
	}
	h.tick(t)
	h.request(t, "GET", base+"/pairs"+query, nil, "", "", 409)
	h.request(t, "GET", base+"/charts/duration?viewSnapshot="+snapshot, nil, "", "", 409)
}

func TestEvalPostgresDeletedEvidenceInvalidatesSelectedAssessment(t *testing.T) {
	h, e, members := completedEvalFixture(t)
	member := members[0]
	base := "/v1/eval-experiments/" + e.ID
	memberPath := base + "/members/" + member.Member.ID
	review := ownerEvalReview(t, h, memberPath+"/review")
	store, err := artifacts.NewService(artifacts.NewPostgresRepository(h.pool)).Run(member.Execution.Ref.ID)
	if err != nil {
		t.Fatal(err)
	}
	written, err := store.Write(t.Context(), contracts.ArtifactRef{Namespace: "evidence", Name: "proof"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("observed proof")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	ref := evaldomain.Artifact{Scope: "run", ScopeID: member.Execution.Ref.ID, Namespace: written.Ref.Namespace, Name: written.Ref.Name, Revision: *written.Ref.Revision, SHA256: evaldomain.Digest([]byte("observed proof")), MediaType: written.MediaType, SizeBytes: written.Size}
	result := review.Result
	result.Source = evaldomain.Source{System: "fixture", ID: "external-collector"}
	result.PreviousResultSHA256 = &review.ResultSHA256
	result.Evidence = append(result.Evidence, evaldomain.Evidence{ID: "proof", Artifact: ref})
	receipt := h.request(t, "POST", memberPath+"/results", result, "result-with-proof", "", 201)
	resultSHA := apiDecode[struct {
		SHA string `json:"recordSha256"`
	}](t, receipt).SHA
	// A same-owner artifact from this member cannot be attached to another member.
	other := members[1]
	otherReview := ownerEvalReview(t, h, base+"/members/"+other.Member.ID+"/review")
	foreignResult := otherReview.Result
	foreignResult.Source = result.Source
	foreignResult.Evidence = result.Evidence
	h.request(t, "POST", base+"/members/"+other.Member.ID+"/results", foreignResult, "wrong-member-proof", "", 404)
	check := review.Policy[0]
	assessment := evaldomain.AssessmentInput{SchemaVersion: evaldomain.AssessmentSchemaVersion, Source: evaldomain.AssessmentSource{Kind: "human"}, ResultSHA256: resultSHA, Checks: []evaldomain.CheckResult{{ID: check.ID, Evaluator: check.Evaluator, ImplementationSHA256: check.ImplementationSHA256, Status: "pass", Reason: "Reviewed exact proof", EvidenceRefs: []string{"proof"}}}}
	assessmentSHA := apiDecode[struct {
		SHA string `json:"recordSha256"`
	}](t, h.request(t, "POST", memberPath+"/assessments", assessment, "proof-review", "", 201)).SHA
	selection := evaldomain.SelectionInput{PlanSHA256: *e.PlanSHA256, Selections: []evaldomain.SelectionEntry{{MemberID: member.Member.ID, ResultSHA256: resultSHA, AssessmentSHA256: &assessmentSHA}}}
	h.request(t, "POST", base+"/selections", selection, "select-proof", fmt.Sprintf("\"%d\"", h.get(t, e.ID).Revision), 201)
	h.tick(t)
	before := h.get(t, e.ID)
	err = pg.InTx(t.Context(), h.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		purger, err := artifacts.NewPostgresPurger(tx)
		if err != nil {
			return err
		}
		return purger.PurgeRun(t.Context(), member.Execution.Ref.ID)
	})
	if err != nil {
		t.Fatal(err)
	}

	stale := h.get(t, e.ID)
	if stale.Freshness != "stale" || *stale.ViewSnapshot != *before.ViewSnapshot {
		t.Fatal("deletion did not mark old view stale")
	}
	h.tick(t)
	page := apiDecode[struct {
		Items []evaldomain.MemberView `json:"items"`
	}](t, h.request(t, "GET", base+"/members", nil, "", "", 200))
	for _, v := range page.Items {
		if v.Member.ID == member.Member.ID && (v.Assessment != "incomplete" || v.AssessmentSHA256 == nil || *v.AssessmentSHA256 != assessmentSHA || v.Usage.WallMS.Completeness != "complete") {
			t.Fatalf("deleted proof was scored or lost prior authority: %+v", v)
		}
	}
	afterReview := ownerEvalReview(t, h, memberPath+"/review")
	if len(afterReview.Gaps) == 0 {
		t.Fatal("deleted proof missing from owner context")
	}
	if _, err = evalstore.NewPostgresStore(h.pool).Record(t.Context(), "user-1", e.ID, member.Member.ID, "assessment", assessmentSHA); err != nil {
		t.Fatal("prior assessment was erased", err)
	}
}
