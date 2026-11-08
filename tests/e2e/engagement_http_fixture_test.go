//go:build e2e

package e2e

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts/engagementcontract"
	"github.com/jackc/pgx/v5/pgxpool"
)

// The observer counts socket bytes independently of HTTP parsing. A rejected
// partial request therefore cannot masquerade as zero prohibited traffic.
type engagementHTTPObserver struct {
	server      *httptest.Server
	bytes       atomic.Int64
	connections atomic.Int64
	mu          sync.Mutex
	paths       []string
	hosts       []string
	firstByte   chan struct{}
	firstOnce   sync.Once
	handler     http.HandlerFunc
}
type engagementObserveListener struct {
	net.Listener
	owner *engagementHTTPObserver
}
type engagementObserveConn struct {
	net.Conn
	owner *engagementHTTPObserver
}

func (l engagementObserveListener) Accept() (net.Conn, error) {
	c, e := l.Listener.Accept()
	if e != nil {
		return nil, e
	}
	l.owner.connections.Add(1)
	return engagementObserveConn{c, l.owner}, nil
}
func (c engagementObserveConn) Read(p []byte) (int, error) {
	n, e := c.Conn.Read(p)
	if n > 0 {
		c.owner.bytes.Add(int64(n))
		c.owner.firstOnce.Do(func() { close(c.owner.firstByte) })
	}
	return n, e
}
func newEngagementHTTPObserver(t *testing.T, host string) *engagementHTTPObserver {
	t.Helper()
	o := &engagementHTTPObserver{firstByte: make(chan struct{})}
	listener, err := net.Listen("tcp", net.JoinHostPort(host, "0"))
	if err != nil {
		t.Fatal(err)
	}
	o.server = httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		o.mu.Lock()
		if len(o.paths) >= 256 {
			o.mu.Unlock()
			http.Error(w, "fixture request bound exceeded", 500)
			return
		}
		o.paths = append(o.paths, r.RequestURI)
		o.hosts = append(o.hosts, r.Host)
		handler := o.handler
		o.mu.Unlock()
		if handler != nil {
			handler(w, r)
			return
		}
		w.Header().Set("Content-Type", "text/plain")
		_, _ = io.WriteString(w, "observed")
	}))
	_ = o.server.Listener.Close()
	o.server.Listener = engagementObserveListener{listener, o}
	o.server.Start()
	t.Cleanup(o.server.Close)
	return o
}
func (o *engagementHTTPObserver) requests() []string {
	o.mu.Lock()
	defer o.mu.Unlock()
	return append([]string(nil), o.paths...)
}
func (o *engagementHTTPObserver) hostnames() []string {
	o.mu.Lock()
	defer o.mu.Unlock()
	return append([]string(nil), o.hosts...)
}
func engagementFixtureIP(t *testing.T, target *engagementTargetFixture) string {
	t.Helper()
	u, e := url.Parse(target.httpURL)
	if e != nil {
		t.Fatal(e)
	}
	return u.Hostname()
}
func engagementAlias(t *testing.T, address, host string) string {
	t.Helper()
	u, e := url.Parse(address)
	if e != nil {
		t.Fatal(e)
	}
	return u.Scheme + "://" + net.JoinHostPort(host, u.Port())
}
func engagementEnvironment(t *testing.T, dns map[string][][]string, protected ...string) map[string]string {
	t.Helper()
	d, e := json.Marshal(dns)
	if e != nil {
		t.Fatal(e)
	}
	p, e := json.Marshal(protected)
	if e != nil {
		t.Fatal(e)
	}
	return map[string]string{"CONTRACTOR_TEST_TARGET_DNS": string(d), "CONTRACTOR_TEST_PROTECTED_URLS": string(p)}
}
func engagementRoute(origin, prefix string) engagementcontract.Route {
	return engagementcontract.Route{Origin: origin, PathPrefixes: []string{prefix}, Methods: []string{"GET"}, AddressRanges: []string{"0.0.0.0/0", "::/0"}}
}
func startEngagementMatrixAudit(t *testing.T, h *engagementProcessHarness, origins, seeds []string, routes, exclusions []engagementcontract.Route, responseBytes uint32, configure ...func()) string {
	t.Helper()
	setEngagementURLs(h, seeds)
	for _, setup := range configure {
		setup()
	}
	if exclusions == nil {
		exclusions = []engagementcontract.Route{}
	}
	var project projectResourceResponse
	engagementAuditPOST(t, h, "/v1/projects", map[string]string{"kind": "project", "name": "HTTP conformance fixture", "activeChecks": "automatic"}, 0, http.StatusCreated, &project)
	checklistBytes := engagementAssertionSource(t, "Record bounded preflight conformance.")
	checklist := uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "inputs", "checks", "application/json", checklistBytes)
	document := engagementcontract.Document{Schema: engagementcontract.Schema, Egress: engagementcontract.Egress{Routes: routes, Exclusions: exclusions}, Identities: []engagementcontract.Identity{{ID: "anonymous", Kind: "anonymous", Origins: origins}}, Limits: &engagementcontract.HTTPLimits{}}
	document.Limits.ResponseBodyBytes = &responseBytes
	if e := document.Validate(); e != nil {
		t.Fatal(e)
	}
	body, e := json.Marshal(document)
	if e != nil {
		t.Fatal(e)
	}
	engagement := uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "inputs", "engagement", engagementcontract.MediaType, body)
	var draft auditProgramAudit
	engagementAuditPOST(t, h, "/v1/projects/"+project.ProjectID+"/audits", map[string]any{"profile": map[string]string{"name": "engagement-fixture", "version": "1"}, "inputs": map[string]artifactRef{"checklist": checklist, "engagement": engagement}, "scope": map[string]string{"objective": "Verify the independent HTTP conformance matrix"}}, 0, http.StatusCreated, &draft)
	engagementAuditPOST(t, h, "/v1/audits/"+draft.AuditID+"/start", nil, draft.Revision, http.StatusOK, nil)
	return draft.AuditID
}
func engagementPool(t *testing.T, h *engagementProcessHarness) *pgxpool.Pool {
	t.Helper()
	p, e := pgxpool.New(h.ctx, h.databaseURL)
	if e != nil {
		t.Fatal(e)
	}
	t.Cleanup(p.Close)
	return p
}

type engagementHTTPObservation struct {
	Step          int    `json:"step"`
	Kind          string `json:"kind"`
	Reason        string `json:"reason"`
	StatusCode    int    `json:"statusCode"`
	ResponseBytes int    `json:"responseBytes"`
	Truncated     bool   `json:"truncated"`
}

func engagementOutcomes(t *testing.T, h *engagementProcessHarness) map[int][]engagementHTTPObservation {
	t.Helper()
	data, err := os.ReadFile(h.httpObservations)
	if err != nil {
		t.Fatal(err)
	}
	result := map[int][]engagementHTTPObservation{}
	for _, line := range bytes.Split(bytes.TrimSpace(data), []byte("\n")) {
		var outcome engagementHTTPObservation
		if err := json.Unmarshal(line, &outcome); err != nil {
			t.Fatal(err)
		}
		result[outcome.Step] = append(result[outcome.Step], outcome)
	}
	return result
}

func waitEngagementCondition(t *testing.T, ctx context.Context, condition func() bool) {
	t.Helper()
	ticker := time.NewTicker(20 * time.Millisecond)
	defer ticker.Stop()
	for {
		if condition() {
			return
		}
		select {
		case <-ctx.Done():
			t.Fatalf("conformance observation deadline: %v", ctx.Err())
		case <-ticker.C:
		}
	}
}
func engagementRawReply(t *testing.T, w http.ResponseWriter, header string, body []byte) {
	t.Helper()
	connection, _, e := w.(http.Hijacker).Hijack()
	if e != nil {
		t.Error(e)
		return
	}
	defer connection.Close()
	if _, e = io.WriteString(connection, header); e != nil {
		return
	}
	_, _ = connection.Write(body)
}
func engagementHeader(count, total int) string {
	header := "HTTP/1.1 200 OK\r\nContent-Length: 0\r\n"
	for i := 1; i < count; i++ {
		header += fmt.Sprintf("X-%d: x\r\n", i)
	}
	if total > 0 {
		header += "X-Pad: " + strings.Repeat("x", total-len(header)-len("X-Pad: \r\n\r\n")) + "\r\n"
	}
	return header + "\r\n"
}

// Denial may deliberately fail placement after earlier seeds completed. Wait
// for the actual controller and joined allocation, not an assumed happy result.
func waitEngagementAuditClosed(t *testing.T, h *engagementProcessHarness, pool *pgxpool.Pool, audit string) {
	t.Helper()
	waitEngagementCondition(t, h.ctx, func() bool {
		var state string
		var released bool
		err := pool.QueryRow(h.ctx, `SELECT state, NOT EXISTS (SELECT 1 FROM stage_allocations a JOIN stage_executions s USING(stage_execution_id) JOIN audit_executions e ON e.run_id=s.run_id WHERE e.audit_id=$1 AND a.release_completed_at IS NULL) FROM audits WHERE audit_id=$1`, audit).Scan(&state, &released)
		if err != nil {
			t.Fatal(err)
		}
		return (state == "completed" || state == "failed" || state == "cancelled") && released
	})
}

func (o *engagementHTTPObserver) setHandler(handler http.HandlerFunc) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.handler = handler
}
