package cli

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/publicclient"
)

func TestAuditCLITraversesResearchPagesAndExactResult(t *testing.T) {
	cases := []struct {
		args              []string
		path, query, body string
	}{
		{[]string{"tasks", "list", "a", "--hypothesis", "h", "--state", "settled", "--limit", "1", "--cursor", "next"}, "/v1/audits/a/tasks", "cursor=next&hypothesis_id=h&limit=1&state=settled", `{"items":[],"page":{"hasMore":true,"nextCursor":"another"}}`},
		{[]string{"tasks", "attempts", "a", "t", "--limit", "1"}, "/v1/audits/a/tasks/t/attempts", "limit=1", `{"items":[],"page":{"hasMore":false},"auditRevision":4}`},
		{[]string{"hypotheses", "get", "a", "h", "--revision", "2"}, "/v1/audits/a/hypotheses/h", "revision=2", `{"document":{"id":"h","revision":2,"grounds":"Exact selected evidence","sources":[]}}`},
		{[]string{"materials", "list", "a", "--cursor", "next"}, "/v1/audits/a/materials", "cursor=next&limit=50", `{"items":[],"page":{"hasMore":false},"auditRevision":4}`},
		{[]string{"findings", "sources", "p", "--finding", "f", "--cursor", "next"}, "/v1/projects/p/finding-sources", "cursor=next&findingId=f&limit=50", `{"items":[],"page":{"hasMore":true,"nextCursor":"another"},"projectRevision":4}`},
		{[]string{"findings", "history", "p", "receipt", "--cursor", "next"}, "/v1/projects/p/finding-sources/receipt/history", "cursor=next&limit=50", `{"items":[],"page":{"hasMore":false}}`},
		{[]string{"tasks", "result", "a", "t", "--revision", "r2"}, "/v1/audits/a/artifacts", "key=task%2Ft%2Fresult&revision=r2", `{"assessment":"refuted","conditions":{"actor":"reader"}}`},
	}
	for _, tc := range cases {
		t.Run(strings.Join(tc.args[:2], "-"), func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != tc.path || r.URL.Query().Encode() != tc.query || r.Header.Get("Authorization") != "Bearer token" {
					t.Errorf("unexpected request %s", r.URL.String())
				}
				w.Header().Set(publicclient.APIVersionHeader, publicclient.APIVersion)
				w.Header().Set("Content-Type", "application/json")
				io.WriteString(w, tc.body)
			}))
			defer server.Close()
			client, err := publicclient.New(publicclient.Options{Server: server.URL, Token: "token", AllowHTTP: true, Timeout: time.Second})
			if err != nil {
				t.Fatal(err)
			}
			var out bytes.Buffer
			c := New(strings.NewReader(""), &out, io.Discard, nil)
			p, _ := NewPrinter("json", &out)
			if err := c.runAudit(t.Context(), client, p, tc.args); err != nil {
				t.Fatal(err)
			}
			var want, got any
			if json.Unmarshal([]byte(tc.body), &want) != nil || json.Unmarshal(out.Bytes(), &got) != nil {
				t.Fatalf("lost JSON response: %s", out.String())
			}
			a, _ := json.Marshal(want)
			b, _ := json.Marshal(got)
			if !bytes.Equal(a, b) {
				t.Fatalf("lost page/result data: %s", out.String())
			}
		})
	}
}

func TestAuditCLILostReplyKeepsOriginalMutation(t *testing.T) {
	calls := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		body, _ := io.ReadAll(r.Body)
		if r.URL.Path != "/v1/audits/a/start" || r.Header.Get("Idempotency-Key") != "original-key" || r.Header.Get("If-Match") != `"7"` || string(body) != "{}" {
			t.Errorf("changed mutation: %s %q %q %s", r.URL.Path, r.Header.Get("Idempotency-Key"), r.Header.Get("If-Match"), body)
		}
		if calls == 1 {
			conn, _, err := w.(http.Hijacker).Hijack()
			if err != nil {
				t.Error(err)
				return
			}
			conn.Close()
			return
		}
		w.Header().Set(publicclient.APIVersionHeader, publicclient.APIVersion)
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"audit":{"auditId":"a"}}`)
	}))
	defer server.Close()
	client, err := publicclient.New(publicclient.Options{Server: server.URL, Token: "token", AllowHTTP: true, Timeout: time.Second})
	if err != nil {
		t.Fatal(err)
	}
	c := New(strings.NewReader(""), io.Discard, io.Discard, nil)
	p, _ := NewPrinter("json", io.Discard)
	args := []string{"start", "a", "--revision", "7", "--idempotency-key", "original-key"}
	if err := c.runAudit(t.Context(), client, p, args); err != nil {
		t.Fatal(err)
	}
	if calls != 2 {
		t.Fatalf("same-key reconciliation calls=%d", calls)
	}
	if err := c.runAudit(t.Context(), client, p, args); err != nil {
		t.Fatal(err)
	}
	if calls != 3 {
		t.Fatalf("explicit replay calls=%d", calls)
	}

}

func TestAuditCLIRejectsUnboundAndMalformedMutations(t *testing.T) {
	calls := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { calls++; http.Error(w, "unexpected", 500) }))
	defer server.Close()
	client, err := publicclient.New(publicclient.Options{Server: server.URL, Token: "token", AllowHTTP: true, Timeout: time.Second})
	if err != nil {
		t.Fatal(err)
	}
	for _, body := range []string{`null`, `{"work":[],"work":[]}`, `{"Work":[]}`, `{"work":[],"extra":true}`} {
		c := New(strings.NewReader(body), io.Discard, io.Discard, nil)
		p, _ := NewPrinter("json", io.Discard)
		if err := c.runAudit(t.Context(), client, p, []string{"tasks", "create", "a", "--revision", "1", "--idempotency-key", "same", "--request", "-"}); err == nil {
			t.Fatalf("accepted %s", body)
		}
	}
	for _, args := range [][]string{{"start", "a", "--revision", "1"}, {"pause", "a", "--idempotency-key", "key"}, {"delete", "a", "--revision", "1", "--idempotency-key", "key"}, {"tasks", "result", "a", "t"}, {"findings", "sources", "p", "--limit", "65"}} {
		c := New(strings.NewReader(""), io.Discard, io.Discard, nil)
		p, _ := NewPrinter("json", io.Discard)
		if err := c.runAudit(t.Context(), client, p, args); err == nil {
			t.Fatalf("accepted %v", args)
		}
	}
	if calls != 0 {
		t.Fatalf("invalid commands made %d requests", calls)
	}
}

func TestAuditCLIMissingResponseBodyIsAnError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set(publicclient.APIVersionHeader, publicclient.APIVersion)
		w.Header().Set("Content-Type", "text/plain")
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()
	client, err := publicclient.New(publicclient.Options{Server: server.URL, Token: "token", AllowHTTP: true, Timeout: time.Second})
	if err != nil {
		t.Fatal(err)
	}
	for _, args := range [][]string{
		{"profiles"}, {"list"}, {"list", "--project", "p"},
		{"tasks", "list", "a"}, {"tasks", "attempts", "a", "t"},
		{"tasks", "create", "a", "--revision", "1", "--idempotency-key", "key", "--request", "-"},
		{"hypotheses", "list", "a"}, {"materials", "list", "a"},
		{"findings", "list", "p"}, {"findings", "sources", "p"},
	} {
		c := New(strings.NewReader(`{"work":[]}`), io.Discard, io.Discard, nil)
		p, _ := NewPrinter("json", io.Discard)
		if err := c.runAudit(t.Context(), client, p, args); err == nil {
			t.Fatalf("missing body succeeded: %v", args)
		}
	}
}
