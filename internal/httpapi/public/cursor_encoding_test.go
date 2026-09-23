package public

import (
	"crypto/sha256"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"testing"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalservice"
	"github.com/grauwolf32/contractor/internal/evalstore"
)

// Cursors and filter tokens are held by clients across Server upgrades, so
// their byte encodings are pinned here: a refactor that changes any of them
// breaks every outstanding cursor.

const (
	pinnedPageCursor     = "eyJ2IjoxLCJrIjoiUERqQ1h6cHBCLTBMbV9PUVpZTW5ISjFUQ0kxV0d6bUJqNkJ5NHRHb3h2SSIsInAiOlsiMjAyNi0wMS0wMlQwMzowNDowNS4wMDAwMDAwMDZaIiwicnVuLTEiXX2mf6iQHrKR4mRxXtEErdSbq-Qv0YJ6BGDrLMlncgU7-g"
	pinnedEvalCursor     = "eyJjb250ZXh0Ijoic2hhMjU2OmQyNjg4ZmY5ZTY0NjcxNDljZWIwOTBiNmU4NTkzYjY2Y2U5ZDAwN2M0MmZiMTM3OTA0ZGRkMWYxMzc1YTMyNTUiLCJwb3NpdGlvbiI6WyI0IiwibWVtYmVyLTkiXSwic25hcHNob3QiOiIxNyJ9CqldmmCpMPsPK4R9DkumRJw8EdFwCYtHlv9WL2dSkss"
	pinnedEvalBinToken   = "eyJjb250ZXh0Ijoic2hhMjU2OmYzNzljMTEzNzcyMzQzNmVhYmUyMzg0NzY4YmVlZGM3OWYzNjA0MWNhM2FlYTRlODcwNWIwNDMwODA5NDc0MTkiLCJzbmFwc2hvdCI6IjE3Iiwic3VpdGUiOiJzdWl0ZS1hIiwiYmluIjp7Im1ldHJpYyI6InNjb3JlIiwic2NvcGUiOiJ0YXNrIiwibG93ZXIiOjAuMjUsInVwcGVyIjowLjUsInVwcGVySW5jbHVzaXZlIjp0cnVlfX1zuOE-SWbOF2ELjnFFN-jiX06LKhbeqggCsXAZFV_5Wg"
	pinnedCursorBearer   = "pinned-cursor-bearer-token"
	pinnedEvalCursorPath = "/v1/evals/experiments/exp-1/members"
)

func pinnedCursorHandler() *handler {
	return &handler{tokenDigest: sha256.Sum256([]byte(pinnedCursorBearer))}
}

func pinnedEvalRequest(rawQuery string) (*http.Request, url.Values) {
	r := httptest.NewRequest(http.MethodGet, pinnedEvalCursorPath+"?"+rawQuery, nil)
	r.SetPathValue("id", "exp-1")
	return r, r.URL.Query()
}

func TestPageCursorEncodingIsPinned(t *testing.T) {
	h := pinnedCursorHandler()
	encoded, err := h.encodePageCursor("runs:state=all", "2026-01-02T03:04:05.000000006Z", "run-1")
	if err != nil {
		t.Fatalf("encodePageCursor: %v", err)
	}
	if encoded != pinnedPageCursor {
		t.Fatalf("encodePageCursor = %q, want pinned %q", encoded, pinnedPageCursor)
	}
	values, err := h.decodePageCursor(pinnedPageCursor, "runs:state=all", 2)
	if err != nil || !reflect.DeepEqual(values, []string{"2026-01-02T03:04:05.000000006Z", "run-1"}) {
		t.Fatalf("decodePageCursor(pinned) = %v, %v", values, err)
	}
	if _, err := h.decodePageCursor(pinnedPageCursor, "runs:state=active", 2); err == nil {
		t.Fatal("decodePageCursor accepted a cursor of another kind")
	}
	other := &handler{tokenDigest: sha256.Sum256([]byte("another-bearer"))}
	if _, err := other.decodePageCursor(pinnedPageCursor, "runs:state=all", 2); err == nil {
		t.Fatal("decodePageCursor accepted a cursor signed with another key")
	}
}

func TestEvalCursorEncodingIsPinned(t *testing.T) {
	h := pinnedCursorHandler()
	r, q := pinnedEvalRequest("limit=2&filter=failed")
	page, err := h.evalPage(r, q, true, "17", "4", "member-9")
	if err != nil || !page.HasMore || page.NextCursor == nil {
		t.Fatalf("evalPage = %+v, %v", page, err)
	}
	if *page.NextCursor != pinnedEvalCursor {
		t.Fatalf("evalPage cursor = %q, want pinned %q", *page.NextCursor, pinnedEvalCursor)
	}
	r, q = pinnedEvalRequest("limit=2&filter=failed&cursor=" + pinnedEvalCursor)
	cursor, err := h.readEvalCursor(r, q, 2)
	if err != nil || cursor.Snapshot != "17" || !reflect.DeepEqual(cursor.Position, []string{"4", "member-9"}) {
		t.Fatalf("readEvalCursor(pinned) = %+v, %v", cursor, err)
	}
	r, q = pinnedEvalRequest("limit=3&filter=failed&cursor=" + pinnedEvalCursor)
	if _, err := h.readEvalCursor(r, q, 2); err == nil {
		t.Fatal("readEvalCursor accepted a cursor from another query")
	}
}

func TestEvalBinTokenEncodingIsPinned(t *testing.T) {
	h := pinnedCursorHandler()
	r, _ := pinnedEvalRequest("")
	suite, scope := "suite-a", "task"
	chart := evalservice.ChartView{
		ViewMetadata: evalservice.ViewMetadata{Snapshot: "17"},
		Chart:        "score", SuiteID: &suite, MeasurementScope: &scope,
	}
	token, err := h.evalBinToken(r, chart, evaldomain.Bin{Lower: 0.25, Upper: 0.5, UpperInclusive: true})
	if err != nil {
		t.Fatalf("evalBinToken: %v", err)
	}
	if token != pinnedEvalBinToken {
		t.Fatalf("evalBinToken = %q, want pinned %q", token, pinnedEvalBinToken)
	}
	params := evalservice.MemberPageParams{SuiteID: suite}
	if err := h.readEvalBin(r, url.Values{"binFilter": {pinnedEvalBinToken}}, &params); err != nil {
		t.Fatalf("readEvalBin(pinned): %v", err)
	}
	want := evalstore.BinFilter{Metric: "score", Scope: "task", Lower: 0.25, Upper: 0.5, UpperInclusive: true}
	if params.Snapshot != "17" || params.Bin == nil || *params.Bin != want {
		t.Fatalf("readEvalBin(pinned) params = %+v", params)
	}
}
