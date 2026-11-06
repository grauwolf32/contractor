package public

import (
	"encoding/json"
	"fmt"
	"maps"
	"strconv"
	"testing"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalservice"
	"github.com/grauwolf32/contractor/internal/evalstore"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

// evalNUL is U+0000, which encoding/json writes as the escape \u0000 that every
// Eval document rejects (EVF-R236).
const evalNUL = "\x00"

// evalEscapeText is the six characters of that escape written as text, and
// evalControl a character PostgreSQL keeps in jsonb; both are valid.
const evalEscapeText, evalControl = `\u0000`, "\x01"

// evalSideEffects counts the rows a rejected Eval write must not add: the
// documents of every public Eval write and the mutation receipts.
func (h *evalAPIHarness) evalSideEffects(t *testing.T) map[string]int {
	t.Helper()
	counts := map[string]int{}
	for _, table := range []string{"eval_dataset_revisions", "eval_experiments", "eval_records", "mutation_receipts"} {
		var count int
		if err := h.pool.QueryRow(t.Context(), "SELECT count(*) FROM "+table).Scan(&count); err != nil {
			t.Fatal(err)
		}
		counts[table] = count
	}
	var drafts string
	if err := h.pool.QueryRow(t.Context(), `SELECT coalesce(string_agg(experiment_id||':'||revision||':'||md5(draft), ',' ORDER BY experiment_id), '') FROM eval_experiments`).Scan(&drafts); err != nil {
		t.Fatal(err)
	}
	counts["drafts "+drafts] = 1
	var selected int
	if err := h.pool.QueryRow(t.Context(), `SELECT count(*) FROM eval_members WHERE selected_result_sha256 IS NOT NULL`).Scan(&selected); err != nil {
		t.Fatal(err)
	}
	counts["selections"] = selected

	return counts
}

// rejectNUL sends a body that holds a NUL escape and requires 422
// eval_invalid, not retryable, with no row and no receipt written.
func (h *evalAPIHarness) rejectNUL(t *testing.T, method, path string, body any, key, etag string) {
	t.Helper()
	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	if !strictjson.HasNULEscape(raw) {
		t.Fatalf("%s %s: body %s holds no NUL escape", method, path, raw)
	}
	before := h.evalSideEffects(t)
	answer := apiDecode[errorResponse](t, h.request(t, method, path, body, key, etag, 422))
	if answer.Code != "eval_invalid" || answer.Retryable {
		t.Fatalf("%s %s: %+v, want eval_invalid, retryable false", method, path, answer)
	}
	if after := h.evalSideEffects(t); !maps.Equal(before, after) {
		t.Fatalf("%s %s wrote rows or a receipt: before %v, after %v", method, path, before, after)
	}
}

// copyEvalDocument returns a deep copy of a document through its JSON encoding.
func copyEvalDocument[T any](t *testing.T, value T) T {
	t.Helper()
	raw, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	var out T
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatal(err)
	}
	return out
}

// listedExperiments reads the owner's experiment list without projectId,
// which reads the setup or draft and the view summary of every experiment of
// the owner as jsonb.
func (h *evalAPIHarness) listedExperiments(t *testing.T) []evalstore.PublicSummary {
	t.Helper()
	return apiDecode[struct {
		Items []evalstore.PublicSummary `json:"items"`
	}](t, h.request(t, "GET", "/v1/eval-experiments", nil, "", "", 200)).Items
}

// V414-017: a NUL escape passed every Eval codec, so a stored draft broke the
// owner's whole experiment list with an unclassified 22P05 (a retryable 500)
// and a dataset failed at its jsonb metadata INSERT the same way. Every Eval
// write now rejects the escape in a free string or a member name with 422
// before side effects, so the same key with a corrected body is a new request;
// other escapes and control characters stay valid and readable.
func TestSharedEvalPostgresNULEscapesAreRejectedBeforeSideEffects(t *testing.T) {
	h := newEvalAPIHarness(t)
	draft, data := h.dataset(t, "workflow")
	datasets := "/v1/projects/evaluation/eval-datasets"

	t.Run("dataset", func(t *testing.T) {
		named := copyEvalDocument(t, data)
		named.DatasetID, named.Name = "nul-dataset", "Trace"+evalNUL+"examples"
		h.rejectNUL(t, "POST", datasets, named, "nul-dataset", "")
		keyed := copyEvalDocument(t, data)
		keyed.DatasetID = "nul-dataset"
		keyed.Cases[0].Task.Parameters = map[string]string{"no" + evalNUL + "te": "x"}
		h.rejectNUL(t, "POST", datasets, keyed, "nul-dataset-key", "")
		// No receipt holds the key: the corrected body is a new request.
		corrected := copyEvalDocument(t, data)
		corrected.DatasetID, corrected.Name = "nul-dataset", "Trace "+evalEscapeText+" examples"+evalControl
		h.request(t, "POST", datasets, corrected, "nul-dataset", "", 201)
	})

	experiments := "/v1/projects/evaluation/eval-experiments"
	valid := copyEvalDocument(t, draft)
	valid.Variants[0].Parameters = map[string]string{"note": "x" + evalEscapeText + "y" + evalControl}
	var path string
	t.Run("create", func(t *testing.T) {
		// The reproduction: a NUL in a parameter value of variant a.
		parameter := copyEvalDocument(t, draft)
		parameter.Variants[0].Parameters = map[string]string{"note": "x" + evalNUL + "y"}
		h.rejectNUL(t, "POST", experiments, evaldomain.CreateExperiment{Name: "Trace instructions", ControlMode: "server", Draft: &parameter}, "nul-create", "")
		parameter.Variants[0].Parameters = map[string]string{"no" + evalNUL + "te": "x"}
		h.rejectNUL(t, "POST", experiments, evaldomain.CreateExperiment{Name: "Trace instructions", ControlMode: "server", Draft: &parameter}, "nul-create-key", "")
		h.rejectNUL(t, "POST", experiments, evaldomain.CreateExperiment{Name: "Trace" + evalNUL + "instructions", ControlMode: "server", Draft: &draft}, "nul-create-name", "")
		created := apiDecode[struct {
			ID string `json:"experimentId"`
		}](t, h.request(t, "POST", experiments, evaldomain.CreateExperiment{Name: "Trace " + evalEscapeText + evalControl, ControlMode: "server", Draft: &valid}, "nul-create", "", 201))
		path = "/v1/eval-experiments/" + created.ID
	})
	t.Run("draft update", func(t *testing.T) {
		if path == "" {
			t.Fatal("no experiment was created")
		}
		h.rejectNUL(t, "PATCH", path, evaldomain.DraftUpdate{Name: "Edited" + evalNUL, Draft: valid}, "nul-edit", `"1"`)
		parameter := copyEvalDocument(t, valid)
		parameter.Variants[0].Parameters = map[string]string{"note": "x" + evalNUL + "y"}
		h.rejectNUL(t, "PATCH", path, evaldomain.DraftUpdate{Name: "Edited", Draft: parameter}, "nul-edit", `"1"`)
		h.request(t, "PATCH", path, evaldomain.DraftUpdate{Name: "Edited " + evalEscapeText, Draft: valid}, "nul-edit", `"1"`, 200)
	})
	// Without the rule a stored NUL draft fails this read of every experiment
	// of the owner with 22P05.
	t.Run("owner list", func(t *testing.T) {
		listed := h.listedExperiments(t)
		if len(listed) != 1 || "/v1/eval-experiments/"+listed[0].ID != path || listed[0].Name != "Edited "+evalEscapeText || listed[0].CaseCount == nil || *listed[0].CaseCount != 2 {
			t.Fatalf("owner experiment list %+v", listed)
		}
	})
}

// Results and assessments of an external producer reject a NUL escape the
// same way, and the comparison report and the owner list, which cast the
// selected records, the pair documents and the summaries to jsonb, stay
// readable for records that hold other escapes and evalControl characters.
func TestSharedEvalPostgresNULEscapesAreRejectedInExternalRecords(t *testing.T) {
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
	result := func(gap string) evaldomain.ResultInput {
		return evaldomain.ResultInput{
			SchemaVersion: evaldomain.ResultSchemaVersion, PlanSHA256: *e.PlanSHA256, MemberID: memberID,
			Source:    evaldomain.Source{System: "fixture", ID: "independent-collector"},
			Execution: *member.Execution, Usage: *member.Usage,
			Collection: evaldomain.Collection{Status: "partial", Gaps: []string{gap}},
			Outputs:    map[string]evaldomain.Artifact{}, Evidence: []evaldomain.Evidence{},
		}
	}
	h.rejectNUL(t, "POST", memberBase+"/results", result("collector"+evalNUL+"gap"), "result", "")
	resultDigest := apiDecode[struct {
		SHA string `json:"recordSha256"`
	}](t, h.request(t, "POST", memberBase+"/results", result("collector "+evalEscapeText+" gap"+evalControl), "result", "", 201)).SHA

	assessment := func(reason string) evaldomain.AssessmentInput {
		return evaldomain.AssessmentInput{
			SchemaVersion: evaldomain.AssessmentSchemaVersion, ResultSHA256: resultDigest,
			Source: evaldomain.AssessmentSource{Kind: "external", ProducerID: "independent-scorer", RecordSHA256: evaldomain.Digest([]byte("producer decision"))},
			Checks: []evaldomain.CheckResult{{ID: checks[0].ID, Evaluator: checks[0].Evaluator,
				ImplementationSHA256: checks[0].ImplementationSHA256, Status: "pass", Reason: reason, EvidenceRefs: []string{}}},
		}
	}
	h.rejectNUL(t, "POST", memberBase+"/assessments", assessment("Declared"+evalNUL+"judgment"), "assessment", "")
	assessmentDigest := apiDecode[struct {
		SHA string `json:"recordSha256"`
	}](t, h.request(t, "POST", memberBase+"/assessments", assessment("Declared "+evalEscapeText+" judgment"+evalControl), "assessment", "", 201)).SHA

	current := h.get(t, e.ID)
	selection := evaldomain.SelectionInput{PlanSHA256: *e.PlanSHA256, Selections: []evaldomain.SelectionEntry{{MemberID: memberID, ResultSHA256: resultDigest, AssessmentSHA256: &assessmentDigest}}}
	h.request(t, "POST", base+"/selections", selection, "select", fmt.Sprintf("%q", strconv.FormatInt(current.Revision, 10)), 201)
	h.tick(t)
	report := apiDecode[evalservice.Report](t, h.request(t, "GET", base+"/report", nil, "", "", 200))
	attributed := false
	for _, source := range report.Sources {
		attributed = attributed || source.System == "external" && source.ID == "independent-scorer"
	}
	if !attributed {
		t.Fatalf("report sources %+v lost the selected assessment", report.Sources)
	}
	if listed := h.listedExperiments(t); len(listed) != 1 || listed[0].ID != e.ID || listed[0].Summary == nil {
		t.Fatalf("owner experiment list %+v", listed)
	}
}
