//go:build e2e

package e2e

import (
	"bytes"
	"context"
	"crypto/tls"
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
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/localpki"
)

const (
	httpCaidoProxyCanary = "HTTP_CAIDO_PROXY_CREDENTIAL_CANARY"
	httpCaidoTokenCanary = "HTTP_CAIDO_CONTROL_TOKEN_CANARY"
)

type httpCaidoTarget struct {
	server *httptest.Server

	mu       sync.Mutex
	requests int
	failures []string
}

func newHTTPCaidoTarget() *httpCaidoTarget {
	target := &httpCaidoTarget{}
	target.server = httptest.NewServer(http.HandlerFunc(target.serveHTTP))
	return target
}

func (t *httpCaidoTarget) URL() string { return t.server.URL }

func (t *httpCaidoTarget) close() { t.server.Close() }

func (t *httpCaidoTarget) serveHTTP(w http.ResponseWriter, r *http.Request) {
	t.mu.Lock()
	t.requests++
	t.mu.Unlock()
	if r.Method != http.MethodPost || r.URL.Path != "/failure" ||
		r.Header.Get("Authorization") != "Bearer "+httpCaidoSessionCanary {
		t.mu.Lock()
		t.failures = append(t.failures, "invalid target request")
		t.mu.Unlock()
		http.Error(w, "invalid target request", http.StatusBadRequest)
		return
	}
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.Header().Set("X-Test-Evidence", "bounded-500")
	w.WriteHeader(http.StatusInternalServerError)
	_, _ = io.WriteString(w, httpCaidoRawCanary+"\n"+strings.Repeat("bounded-body-", 900))
}

func (t *httpCaidoTarget) snapshot() (int, []string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.requests, append([]string(nil), t.failures...)
}

type httpCaidoForwardProxy struct {
	server     *httptest.Server
	targetHost string
	targetURL  string
	upstream   *url.URL
	transport  *http.Transport

	mu       sync.Mutex
	requests int
	failures []string
}

func newHTTPCaidoForwardProxy(targetURL string) *httpCaidoForwardProxy {
	upstream, _ := url.Parse(targetURL)
	proxy := &httpCaidoForwardProxy{
		targetHost: "tool-target.contractor.invalid",
		targetURL:  "http://tool-target.contractor.invalid",
		upstream:   upstream,
		transport:  &http.Transport{Proxy: nil},
	}
	proxy.server = httptest.NewServer(http.HandlerFunc(proxy.serveHTTP))
	return proxy
}

func (p *httpCaidoForwardProxy) URL() string { return p.server.URL }

func (p *httpCaidoForwardProxy) TargetURL() string { return p.targetURL }

func (p *httpCaidoForwardProxy) close() {
	p.server.Close()
	p.transport.CloseIdleConnections()
}

func (p *httpCaidoForwardProxy) serveHTTP(w http.ResponseWriter, r *http.Request) {
	p.mu.Lock()
	p.requests++
	p.mu.Unlock()
	if r.URL.Scheme != "http" || r.URL.Host != p.targetHost ||
		r.Header.Get("Proxy-Authorization") != "Bearer "+httpCaidoProxyCanary {
		p.recordFailure("proxy envelope or target mismatch")
		http.Error(w, "proxy rejected request", http.StatusProxyAuthRequired)
		return
	}
	outgoing := r.Clone(r.Context())
	outgoing.RequestURI = ""
	outgoing.URL.Scheme = p.upstream.Scheme
	outgoing.URL.Host = p.upstream.Host
	outgoing.Host = p.upstream.Host
	outgoing.Header = r.Header.Clone()
	outgoing.Header.Del("Proxy-Authorization")
	outgoing.Header.Del("Proxy-Connection")
	response, err := p.transport.RoundTrip(outgoing)
	if err != nil {
		p.recordFailure("proxy target transport failed")
		http.Error(w, "proxy target unavailable", http.StatusBadGateway)
		return
	}
	defer response.Body.Close()
	copyHTTPHeaders(w.Header(), response.Header)
	w.WriteHeader(response.StatusCode)
	_, _ = io.Copy(w, io.LimitReader(response.Body, 20<<20))
}

func (p *httpCaidoForwardProxy) recordFailure(message string) {
	p.mu.Lock()
	p.failures = append(p.failures, message)
	p.mu.Unlock()
}

func (p *httpCaidoForwardProxy) snapshot() (int, []string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.requests, append([]string(nil), p.failures...)
}

type fakeCaidoControl struct {
	server  *httptest.Server
	name    string
	block   bool
	first   chan struct{}
	release chan struct{}
	once    sync.Once

	mu         sync.Mutex
	operations []string
	failures   []string
}

func newFakeCaidoControl(name string, blockFirst bool) *fakeCaidoControl {
	control := &fakeCaidoControl{
		name: name, block: blockFirst, first: make(chan struct{}), release: make(chan struct{}),
	}
	control.server = httptest.NewServer(http.HandlerFunc(control.serveHTTP))
	return control
}

func (c *fakeCaidoControl) URL() string { return c.server.URL }

func (c *fakeCaidoControl) close() {
	c.releaseFirst()
	c.server.Close()
}

func (c *fakeCaidoControl) releaseFirst() { c.once.Do(func() { close(c.release) }) }

func (c *fakeCaidoControl) serveHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost || r.URL.Path != "/graphql" ||
		r.Header.Get("Authorization") != "Bearer "+httpCaidoTokenCanary ||
		r.Header.Get("Content-Type") != "application/json" {
		c.recordFailure("invalid Caido request envelope")
		http.Error(w, "invalid request", http.StatusBadRequest)
		return
	}
	defer r.Body.Close()
	var request struct {
		OperationName string         `json:"operationName"`
		Query         string         `json:"query"`
		Variables     map[string]any `json:"variables"`
	}
	decoder := json.NewDecoder(io.LimitReader(r.Body, 5<<20))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&request); err != nil || request.OperationName == "" ||
		!strings.HasPrefix(request.Query, map[bool]string{
			true: "mutation ", false: "query ",
		}[request.OperationName == "CreateScope"]) {
		c.recordFailure("invalid static GraphQL request")
		http.Error(w, "invalid request", http.StatusBadRequest)
		return
	}
	c.mu.Lock()
	isFirst := len(c.operations) == 0
	c.operations = append(c.operations, request.OperationName)
	c.mu.Unlock()
	if c.block && isFirst {
		close(c.first)
		select {
		case <-c.release:
		case <-r.Context().Done():
			return
		}
	}

	var data map[string]any
	switch request.OperationName {
	case "RequestsByOffset":
		data = map[string]any{"requestsByOffset": map[string]any{
			"count": map[string]any{"value": 1},
			"nodes": []any{map[string]any{
				"id": "request-" + c.name, "method": "GET", "host": "target.example",
				"path": "/observed", "port": 443, "query": "", "isTls": true,
				"source": "PROXY", "createdAt": "2026-09-01T00:00:00Z",
				"response": map[string]any{
					"statusCode": 500, "length": 12000, "roundtripTime": 7,
				},
			}},
		}}
	case "CreateScope":
		input, _ := request.Variables["input"].(map[string]any)
		name, _ := input["name"].(string)
		allowlist, _ := input["allowlist"].([]any)
		denylist, _ := input["denylist"].([]any)
		data = map[string]any{"createScope": map[string]any{
			"error": nil,
			"scope": map[string]any{
				"id": "scope-" + c.name, "name": name,
				"allowlist": allowlist, "denylist": denylist,
			},
		}}
	default:
		c.recordFailure("unexpected GraphQL operation " + request.OperationName)
		http.Error(w, "unsupported", http.StatusBadRequest)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"data": data})
}

func (c *fakeCaidoControl) recordFailure(message string) {
	c.mu.Lock()
	c.failures = append(c.failures, message)
	c.mu.Unlock()
}

func (c *fakeCaidoControl) snapshot() ([]string, []string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]string(nil), c.operations...), append([]string(nil), c.failures...)
}

type runtimeReleaseLossProxy struct {
	server   *http.Server
	listener net.Listener
	client   *http.Client
	target   string
	dropped  chan struct{}
	once     sync.Once

	mu          sync.Mutex
	dropRelease bool
	dropCount   int
	failures    []string
}

func newRuntimeReleaseLossProxy(
	t *testing.T,
	address, target, caFile string,
	agentIdentity, controlPlaneIdentity localpki.Paths,
) *runtimeReleaseLossProxy {
	t.Helper()
	certificate, err := tls.LoadX509KeyPair(agentIdentity.Certificate, agentIdentity.PrivateKey)
	if err != nil {
		t.Fatal(err)
	}
	plain, err := net.Listen("tcp", address)
	if err != nil {
		t.Fatal(err)
	}
	tlsListener := tls.NewListener(plain, &tls.Config{
		MinVersion: tls.VersionTLS13, Certificates: []tls.Certificate{certificate},
		ClientAuth: tls.RequireAndVerifyClientCert, ClientCAs: certificatePool(t, caFile),
		NextProtos: []string{"http/1.1"},
	})
	client := newMTLSClient(t, caFile, controlPlaneIdentity)
	client.Timeout = 30 * time.Second
	proxy := &runtimeReleaseLossProxy{
		listener: tlsListener, client: client, target: strings.TrimRight(target, "/"),
		dropped: make(chan struct{}), dropRelease: true,
	}
	proxy.server = &http.Server{
		Handler:           proxy,
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       30 * time.Second,
		WriteTimeout:      30 * time.Second,
		IdleTimeout:       15 * time.Second,
	}
	go func() {
		if serveErr := proxy.server.Serve(tlsListener); serveErr != nil &&
			serveErr != http.ErrServerClosed {
			proxy.recordFailure("release-loss proxy server failed")
		}
	}()
	t.Cleanup(func() {
		shutdown, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		_ = proxy.server.Shutdown(shutdown)
		client.CloseIdleConnections()
	})
	return proxy
}

func (p *runtimeReleaseLossProxy) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	defer r.Body.Close()
	body, err := io.ReadAll(io.LimitReader(r.Body, 20<<20))
	if err != nil {
		p.recordFailure("read proxied Runtime request")
		http.Error(w, "proxy read failed", http.StatusBadGateway)
		return
	}
	target := p.target + r.URL.RequestURI()
	outgoing, err := http.NewRequestWithContext(r.Context(), r.Method, target, bytes.NewReader(body))
	if err != nil {
		p.recordFailure("build proxied Runtime request")
		http.Error(w, "proxy request failed", http.StatusBadGateway)
		return
	}
	outgoing.Header = r.Header.Clone()
	response, err := p.client.Do(outgoing)
	if err != nil {
		// Readiness polling starts before the child Runtime has accepted its
		// first connection.  That expected startup edge is retried by the test;
		// transport loss on any lifecycle/A2A request remains a fixture failure.
		if r.URL.Path != "/healthz" {
			p.recordFailure("proxied Runtime transport failed")
		}
		http.Error(w, "Runtime unavailable", http.StatusBadGateway)
		return
	}
	defer response.Body.Close()
	if strings.HasSuffix(r.URL.Path, "/release") && p.shouldDropRelease() {
		_, _ = io.Copy(io.Discard, io.LimitReader(response.Body, 1024))
		p.once.Do(func() { close(p.dropped) })
		if hijacker, ok := w.(http.Hijacker); ok {
			connection, _, hijackErr := hijacker.Hijack()
			if hijackErr == nil {
				_ = connection.Close()
				return
			}
		}
		p.recordFailure("release response could not be dropped")
		http.Error(w, "release acknowledgement lost", http.StatusBadGateway)
		return
	}
	copyHTTPHeaders(w.Header(), response.Header)
	w.WriteHeader(response.StatusCode)
	_, _ = io.Copy(w, io.LimitReader(response.Body, 20<<20))
}

func (p *runtimeReleaseLossProxy) shouldDropRelease() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	if !p.dropRelease {
		return false
	}
	p.dropCount++
	return true
}

func (p *runtimeReleaseLossProxy) allowReleaseAcknowledgement() {
	p.mu.Lock()
	p.dropRelease = false
	p.mu.Unlock()
}

func (p *runtimeReleaseLossProxy) recordFailure(message string) {
	p.mu.Lock()
	p.failures = append(p.failures, message)
	p.mu.Unlock()
}

func (p *runtimeReleaseLossProxy) snapshot() (int, []string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.dropCount, append([]string(nil), p.failures...)
}

func copyHTTPHeaders(destination, source http.Header) {
	for name, values := range source {
		if strings.EqualFold(name, "Connection") || strings.EqualFold(name, "Proxy-Connection") {
			continue
		}
		for _, value := range values {
			destination.Add(name, value)
		}
	}
}

func writeHTTPAnalysisE2EWorkflow(t *testing.T, configRoot string) {
	t.Helper()
	instructions := `Delegate the complete bounded HTTP objective to the explorer exactly once.
Finish from the Worker's semantic result and do not repeat requests.
`
	workflow := `apiVersion: contractor/v1alpha1
kind: Workflow
metadata:
  name: http-analysis-e2e
  version: "1"
spec:
  parameters:
    objective: {required: true}
    target: {required: true}
    authorization_scope: {required: true}
  inputs: {}
  outputs:
    report: {required: true, mediaTypes: [text/markdown]}
  executionConfig:
    workers:
      llmGateway: local-litellm@1
      credential: development-worker
  entryStage: analyze
  stages:
    analyze:
      objective: Perform one bounded authorized HTTP observation and report exact evidence
      instructions: {ref: instructions/http-analysis-e2e-planner.md}
      planner: passthrough@1
      agents:
        explorer: {template: http_explorer@1, namespace: http}
      result:
        artifacts:
          report:
            required: true
            mediaTypes: [text/markdown]
            from: {namespace: http, name: report}
      workflowOutputs: {report: report}
      on:
        succeeded: {succeed: {}}
        failed: {fail: {}}
        interrupted: {fail: {}}
`
	if err := os.WriteFile(
		fmt.Sprintf("%s/instructions/http-analysis-e2e-planner.md", configRoot),
		[]byte(instructions), 0o600,
	); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(
		fmt.Sprintf("%s/workflows/http_analysis_e2e.yaml", configRoot),
		[]byte(workflow), 0o600,
	); err != nil {
		t.Fatal(err)
	}
}
