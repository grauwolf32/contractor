package public

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os/exec"
	"path/filepath"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
)

func TestSharedCompactionEvalHarnessSelectsExactRecordsOverHTTPAndReplaysLostReply(t *testing.T) {
	h := newEvalAPIHarness(t)
	e, manifest, checks := h.registerExternal(t, "workflow")
	memberID := manifest.Members[0].MemberID
	base := "/v1/eval-experiments/" + e.ID
	memberPath := base + "/members/" + memberID
	h.request(t, "POST", memberPath+"/submissions", evaldomain.Submission{PlanSHA256: *e.PlanSHA256}, "submit", "", 202)
	h.tick(t)
	h.finish(t, "workflow")
	h.tick(t)
	page := apiDecode[struct {
		Items []evaldomain.MemberView `json:"items"`
	}](t, h.request(t, "GET", base+"/members", nil, "", "", 200))
	var member evaldomain.MemberView
	for _, item := range page.Items {
		if item.Member.ID == memberID {
			member = item
		}
	}
	if member.Execution == nil || member.Execution.State != "succeeded" || member.Usage == nil {
		t.Fatal("external execution did not settle")
	}
	result := evaldomain.ResultInput{
		SchemaVersion: evaldomain.ResultSchemaVersion, PlanSHA256: *e.PlanSHA256, MemberID: memberID,
		Source: evaldomain.Source{System: "fixture", ID: "context-compaction"}, Execution: *member.Execution,
		Collection: evaldomain.Collection{Status: "complete", Gaps: []string{}},
		Outputs:    map[string]evaldomain.Artifact{}, Evidence: []evaldomain.Evidence{}, Usage: *member.Usage,
	}
	resultSHA := apiDecode[struct {
		SHA string `json:"recordSha256"`
	}](t, h.request(t, "POST", memberPath+"/results", result, "result", "", 201)).SHA
	assessment := evaldomain.AssessmentInput{
		SchemaVersion: evaldomain.AssessmentSchemaVersion, ResultSHA256: resultSHA,
		Source: evaldomain.AssessmentSource{Kind: "external", ProducerID: "context-compaction", RecordSHA256: evaldomain.Digest([]byte("independent verdict"))},
		Checks: []evaldomain.CheckResult{{ID: checks[0].ID, Evaluator: checks[0].Evaluator, ImplementationSHA256: checks[0].ImplementationSHA256, Status: "pass", Reason: "Independent fixture", EvidenceRefs: []string{}}},
	}
	assessmentSHA := apiDecode[struct {
		SHA string `json:"recordSha256"`
	}](t, h.request(t, "POST", memberPath+"/assessments", assessment, "assessment", "", 201)).SHA
	store := evalstore.NewPostgresStore(h.pool, artifactpgtest.NewStorage())
	if selected, err := store.Selected(t.Context(), "user-1", e.ID, memberID); err != nil || selected != nil {
		t.Fatalf("external records were implicitly selected: %+v %v", selected, err)
	}
	before := h.get(t, e.ID)
	var mu sync.Mutex
	var requests []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" || r.URL.Path != base+"/selections" {
			h.handler.ServeHTTP(w, r)
			return
		}
		mu.Lock()
		requests = append(requests, r.Header.Get("Idempotency-Key")+" "+r.Header.Get("If-Match"))
		first := len(requests) == 1
		mu.Unlock()
		if first {
			recorder := httptest.NewRecorder()
			h.handler.ServeHTTP(recorder, r)
			if recorder.Code != http.StatusCreated {
				t.Errorf("selection failed before lost acknowledgement: %d", recorder.Code)
			}
			connection, _, err := w.(http.Hijacker).Hijack()
			if err == nil {
				_ = connection.Close()
			}
			return
		}
		h.handler.ServeHTTP(w, r)
	}))
	defer server.Close()
	helper, err := filepath.Abs("../../../tests/eval/context_compaction")
	if err != nil {
		t.Fatal(err)
	}
	script := `import json,sys
sys.path.insert(0,sys.argv[1])
import harness as h
p=json.load(sys.stdin)
c=h.Client(p["api"],p["token"])
h.select_records(c,p["member"],p["result"],p["assessment"])
h.select_records(c,p["member"],p["result"],p["assessment"],selected={"resultSha256":p["result"],"assessmentSha256":p["assessment"]})
`
	input, err := json.Marshal(map[string]any{"api": server.URL, "token": testBearerToken, "member": map[string]string{"memberId": memberID, "experimentId": e.ID, "planSha256": *e.PlanSHA256}, "result": resultSHA, "assessment": assessmentSHA})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	command := exec.CommandContext(ctx, "python3", "-I", "-S", "-c", script, helper)
	command.Stdin = bytes.NewReader(input)
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("compaction Eval client: %v\n%s", err, output)
	}
	mu.Lock()
	if len(requests) != 2 || requests[0] != requests[1] {
		t.Errorf("ambiguous selection did not replay exact key/CAS: %v", requests)
	}
	mu.Unlock()
	selected, err := store.Selected(t.Context(), "user-1", e.ID, memberID)
	if err != nil || selected == nil || selected.Revision != 1 || selected.ResultSHA256 != resultSHA || selected.AssessmentSHA256 == nil || *selected.AssessmentSHA256 != assessmentSHA {
		t.Fatalf("exact selection: %+v %v", selected, err)
	}
	stale := evaldomain.SelectionInput{PlanSHA256: *e.PlanSHA256, Selections: []evaldomain.SelectionEntry{{MemberID: memberID, ResultSHA256: resultSHA}}}
	h.request(t, "POST", base+"/selections", stale, "stale-selection", strconv.Quote(strconv.FormatInt(before.Revision, 10)), 412)
	h.tick(t)
	current := h.get(t, e.ID)
	var summary evaldomain.Summary
	if err := json.Unmarshal(current.Summary, &summary); err != nil {
		t.Fatal(err)
	}
	if current.Freshness != "current" || summary.Counts[member.Member.VariantID].EndToEndPassed != 1 {
		t.Fatal("explicit selection did not reach the current comparison", fmt.Sprint(summary.Counts))
	}
}
