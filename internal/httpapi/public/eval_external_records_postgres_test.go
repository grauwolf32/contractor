package public

import (
	"bytes"
	"fmt"
	"strings"
	"testing"

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
