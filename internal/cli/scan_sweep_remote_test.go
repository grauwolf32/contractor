package cli

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/grauwolf32/contractor/internal/publicclient"
	publicapi "github.com/grauwolf32/contractor/internal/publicclient/generated"
)

const remoteSweepFixture = `{"sweepId":"sweep-fixture","state":"active","workflow":"nuclei-web@1","snapshotDigest":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","createdAt":"2026-10-23T00:00:00Z","settledAt":null,"counts":{"total":2,"pending":2,"admitted":0,"skipped":0,"failed":0,"cancelled":0,"unsettled":2}}`

func invokeRemoteSweep(ctx context.Context, server string, args ...string) (string, string, error) {
	var out, stderr bytes.Buffer
	command := New(strings.NewReader(""), &out, &stderr, func(name string) string {
		if name == "CONTRACTOR_API_TOKEN" {
			return "fixture-token"
		}
		return ""
	})
	err := command.Run(ctx, append([]string{"--server", server, "--output", "json", "scan-sweep", "remote"}, args...))
	return out.String(), stderr.String(), err
}
func remoteSweepServer(t *testing.T, handler http.HandlerFunc) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set(publicclient.APIVersionHeader, publicclient.APIVersion)
		w.Header().Set("Content-Type", "application/json")
		if r.Header.Get("Authorization") != "Bearer fixture-token" {
			t.Error("missing authentication")
		}
		handler(w, r)
	}))
	t.Cleanup(server.Close)
	return server
}

func TestRemoteSweepLostAcknowledgementKeepsExactBodyAndKey(t *testing.T) {
	var mu sync.Mutex
	requests := map[string]string{}
	posts := 0
	server := remoteSweepServer(t, func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		if r.Method != http.MethodPost || r.URL.Path != "/v1/scan-sweeps" {
			t.Errorf("unexpected mutable read or operation: %s %s", r.Method, r.URL)
			w.WriteHeader(400)
			return
		}
		posts++
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Error(err)
			return
		}
		key := r.Header.Get("Idempotency-Key")
		if key != "create-fixture" {
			t.Errorf("key=%q", key)
		}
		if previous, found := requests[key]; found && previous != string(body) {
			t.Error("changed retry body")
		}
		requests[key] = string(body)
		var request publicapi.CreateScanSweepRequest
		if err := json.Unmarshal(body, &request); err != nil {
			t.Error(err)
			return
		}
		if request.ProjectIds == nil || strings.Join(*request.ProjectIds, ",") != "p1,p2" || request.Rulepack == nil || request.Rulepack.Revision != "revision-fixture" || request.Rulepack.Namespace != "rules" || (*request.Parameters)["template_ids"] != "fixture" {
			t.Errorf("request=%+v", request)
		}
		if posts == 1 {
			connection, _, err := w.(http.Hijacker).Hijack()
			if err != nil {
				t.Error(err)
				return
			}
			connection.Close()
			return
		}
		w.WriteHeader(http.StatusCreated)
		_, _ = io.WriteString(w, remoteSweepFixture)
	})
	args := []string{"create", "--project", "p2", "--project", "p1", "--rulepack", "rules/fixture@revision-fixture", "--param", "template_ids=fixture", "--idempotency-key", "create-fixture"}
	for range 2 {
		out, _, err := invokeRemoteSweep(t.Context(), server.URL, args...)
		if err != nil || !strings.Contains(out, `"sweepId": "sweep-fixture"`) {
			t.Fatalf("out=%s err=%v", out, err)
		}
	}
	mu.Lock()
	defer mu.Unlock()
	if posts != 3 || len(requests) != 1 {
		t.Fatalf("posts=%d identities=%d", posts, len(requests))
	}
}

func TestRemoteSweepGeneratedKeyIsRecoverableAfterAmbiguousFailure(t *testing.T) {
	var keys []string
	server := remoteSweepServer(t, func(w http.ResponseWriter, r *http.Request) {
		keys = append(keys, r.Header.Get("Idempotency-Key"))
		w.Header().Set("Retry-After", "0")
		w.WriteHeader(503)
		_, _ = io.WriteString(w, `{"code":"storage_transaction_conflict","message":"retry","retryable":true}`)
	})
	_, stderr, err := invokeRemoteSweep(t.Context(), server.URL, "create", "--all")
	if err == nil || len(keys) != 3 || keys[0] == "" || keys[0] != keys[1] || keys[0] != keys[2] || !strings.Contains(stderr, keys[0]) || !strings.Contains(stderr, "--idempotency-key") {
		t.Fatalf("keys=%v hint=%s err=%v", keys, stderr, err)
	}
}

func TestRemoteSweepInvalidScopeCannotLaunchTraffic(t *testing.T) {
	requests := 0
	server := remoteSweepServer(t, func(w http.ResponseWriter, r *http.Request) { requests++; w.WriteHeader(500) })
	cases := [][]string{
		{"create"}, {"create", "--all", "--project", "p1"}, {"create", "--project", "p1", "--project", "p1"},
		{"create", "--project", "p1", "--param", "target=https://other.example"},
		{"create", "--all", "--rulepack", "rules/mutable"}, {"create", "--all", "--workflow", "nuclei-web"},
		{"create", "--all", "--idempotency-key", "bad/key"}, {"purge", "sweep-fixture"},
	}
	for _, args := range cases {
		if _, _, err := invokeRemoteSweep(t.Context(), server.URL, args...); err == nil {
			t.Errorf("accepted %v", args)
		}
	}
	if requests != 0 {
		t.Fatalf("invalid input sent %d requests", requests)
	}
}

func remoteMember(ordinal int) map[string]any {
	return map[string]any{"ordinal": ordinal, "projectId": fmt.Sprintf("project-%04d", ordinal), "target": "https://fixture.example/", "state": "admitted", "reason": "", "runId": fmt.Sprintf("run-%04d", ordinal), "runState": "succeeded", "settled": true}
}
func TestRemoteSweepReadsAllThousandMembersAndPreservesRunStates(t *testing.T) {
	requests := 0
	server := remoteSweepServer(t, func(w http.ResponseWriter, r *http.Request) {
		requests++
		if r.URL.Path != "/v1/scan-sweeps/sweep-fixture/members" || r.URL.Query().Get("limit") != "200" {
			t.Errorf("unexpected %s", r.URL)
		}
		start, _ := strconv.Atoi(r.URL.Query().Get("cursor"))
		end := min(start+200, 1000)
		items := []map[string]any{}
		for i := start; i < end; i++ {
			items = append(items, remoteMember(i))
		}
		page := map[string]any{"hasMore": end < 1000}
		if end < 1000 {
			page["nextCursor"] = strconv.Itoa(end)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"items": items, "page": page})
	})
	out, _, err := invokeRemoteSweep(t.Context(), server.URL, "members", "sweep-fixture", "--limit", "200", "--all-pages")
	if err != nil {
		t.Fatal(err)
	}
	var members []publicapi.ScanSweepMember
	if err := json.Unmarshal([]byte(out), &members); err != nil {
		t.Fatal(err)
	}
	if requests != 5 || len(members) != 1000 {
		t.Fatalf("requests=%d members=%d", requests, len(members))
	}
	state, err := members[999].RunState.Get()
	if err != nil || state != "succeeded" {
		t.Fatalf("Run state=%v err=%v", state, err)
	}
	// A default read retains the page/cursor rather than silently truncating a whole collection.
	out, _, err = invokeRemoteSweep(t.Context(), server.URL, "members", "sweep-fixture", "--limit", "200")
	var page publicapi.ScanSweepMemberPage
	if err != nil || json.Unmarshal([]byte(out), &page) != nil || !page.Page.HasMore || page.Page.NextCursor == nil || *page.Page.NextCursor != "200" {
		t.Fatalf("page=%s err=%v", out, err)
	}
}

func TestRemoteSweepRejectsMalformedPaginationAndOversizedResponses(t *testing.T) {
	for _, mode := range []string{"cycle", "duplicate", "empty", "oversized", "invalid-run-state", "oversized-reason", "overflow"} {
		t.Run(mode, func(t *testing.T) {
			requests := 0
			server := remoteSweepServer(t, func(w http.ResponseWriter, r *http.Request) {
				requests++
				if mode == "oversized" {
					_, _ = io.WriteString(w, strings.Repeat(" ", maxSweepResponseBytes+1))
					return
				}
				ordinal := requests - 1
				item := remoteMember(ordinal)
				page := map[string]any{"hasMore": true, "nextCursor": strconv.Itoa(requests)}
				items := []map[string]any{item}
				switch mode {
				case "cycle":
					page["nextCursor"] = "same"
				case "duplicate":
					item["projectId"] = "same-project"
				case "empty":
					items = []map[string]any{}
				case "oversized-reason":
					item["reason"] = strings.Repeat("x", 129)
				case "invalid-run-state":
					item["runState"] = map[string]any{"fake": "succeeded"}
				case "overflow":
					items = nil
					for i := 0; i < 200; i++ {
						items = append(items, remoteMember((requests-1)*200+i))
					}
				}
				_ = json.NewEncoder(w).Encode(map[string]any{"items": items, "page": page})
			})
			if _, _, err := invokeRemoteSweep(t.Context(), server.URL, "members", "sweep-fixture", "--limit", "200", "--all-pages"); err == nil {
				t.Fatal("malformed response accepted")
			}
			if requests > 5 {
				t.Fatalf("unbounded reads: %d", requests)
			}
		})
	}
}

func TestRemoteSweepCancelShowsServerIntentAndPurgeRequiresConfirmation(t *testing.T) {
	deletes := 0
	server := remoteSweepServer(t, func(w http.ResponseWriter, r *http.Request) {
		switch r.Method + " " + r.URL.Path {
		case "POST /v1/scan-sweeps/sweep-fixture/cancel":
			_, _ = io.WriteString(w, strings.Replace(remoteSweepFixture, `"state":"active"`, `"state":"cancelling"`, 1))
		case "DELETE /v1/scan-sweeps/sweep-fixture":
			deletes++
			w.WriteHeader(http.StatusNoContent)
		case "GET /v1/scan-sweeps":
			_, _ = io.WriteString(w, `{"items":[`+remoteSweepFixture+`],"page":{"hasMore":true,"nextCursor":"older"}}`)
		default:
			t.Errorf("unexpected %s %s", r.Method, r.URL)
			w.WriteHeader(400)
		}
	})
	out, _, err := invokeRemoteSweep(t.Context(), server.URL, "cancel", "sweep-fixture")
	if err != nil || !strings.Contains(out, `"state": "cancelling"`) {
		t.Fatalf("cancel=%s err=%v", out, err)
	}
	if _, _, err := invokeRemoteSweep(t.Context(), server.URL, "purge", "sweep-fixture"); err == nil || deletes != 0 {
		t.Fatalf("unconfirmed purge deletes=%d err=%v", deletes, err)
	}
	if _, _, err := invokeRemoteSweep(t.Context(), server.URL, "purge", "sweep-fixture", "--yes"); err != nil || deletes != 1 {
		t.Fatalf("purge deletes=%d err=%v", deletes, err)
	}
	out, _, err = invokeRemoteSweep(t.Context(), server.URL, "list", "--limit", "1")
	if err != nil || !strings.Contains(out, `"nextCursor": "older"`) {
		t.Fatalf("list=%s err=%v", out, err)
	}
}
