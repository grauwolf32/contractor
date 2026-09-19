package a2a

// V57-005 is an experiment, not an alternate production client. Keep the
// transport owner visible: SDK Destroy alone does not close idle connections.
import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	sdk "github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/localpki"
	"github.com/grauwolf32/contractor/internal/mtls"
)

type connectionExperiment struct {
	server                      *httptest.Server
	files                       mtls.Files
	principal                   string
	certificate                 *x509.Certificate
	accepts, handshakes, closed atomic.Int64
}

func newConnectionExperiment(t testing.TB, handler http.Handler) *connectionExperiment {
	t.Helper()
	root := t.TempDir()
	g := localpki.Generator{}
	ca, err := g.InitCA(root, false)
	if err != nil {
		t.Fatal(err)
	}
	cp, err := g.IssueControlPlane(root, localpki.ControlPlaneOptions{LeafOptions: localpki.LeafOptions{IPAddresses: []net.IP{net.ParseIP("127.0.0.1")}}, URI: "urn:contractor:control-plane:connection-experiment"})
	if err != nil {
		t.Fatal(err)
	}
	agent, err := g.IssueAgent(root, "connection-experiment", localpki.LeafOptions{IPAddresses: []net.IP{net.ParseIP("127.0.0.1")}})
	if err != nil {
		t.Fatal(err)
	}
	serverConfig, err := mtls.RuntimeAgentServerConfig(mtls.Files{Certificate: agent.Certificate, PrivateKey: agent.PrivateKey, CA: ca.Certificate})
	if err != nil {
		t.Fatal(err)
	}
	cert, err := x509.ParseCertificate(serverConfig.Certificates[0].Certificate[0])
	if err != nil {
		t.Fatal(err)
	}
	principal, err := mtls.RuntimeAgentID(cert)
	if err != nil {
		t.Fatal(err)
	}
	f := &connectionExperiment{files: mtls.Files{Certificate: cp.Certificate, PrivateKey: cp.PrivateKey, CA: ca.Certificate}, certificate: cert, principal: principal}
	verify := serverConfig.VerifyConnection
	serverConfig.VerifyConnection = func(state tls.ConnectionState) error {
		if err := verify(state); err != nil {
			return err
		}
		f.handshakes.Add(1)
		return nil
	}
	f.server = httptest.NewUnstartedServer(handler)
	f.server.TLS = serverConfig
	f.server.EnableHTTP2 = false
	f.server.Config.ErrorLog = log.New(io.Discard, "", 0)
	f.server.Config.ConnState = func(_ net.Conn, state http.ConnState) {
		switch state {
		case http.StateNew:
			f.accepts.Add(1)
		case http.StateClosed, http.StateHijacked:
			f.closed.Add(1)
		}
	}
	f.server.StartTLS()
	t.Cleanup(f.server.Close)
	return f
}

func (f *connectionExperiment) config(t testing.TB) *tls.Config {
	t.Helper()
	config, err := mtls.ControlPlaneEndpointClientConfig(f.files)
	if err != nil {
		t.Fatal(err)
	}
	bound, err := mtls.BindRuntimeAgentPrincipal(config, f.principal)
	if err != nil {
		t.Fatal(err)
	}
	return bound
}

func (f *connectionExperiment) handle() contracts.WorkerHandle {
	h := workerHandle(f.server.URL)
	h.RuntimeAgentID = f.principal
	h.AgentCard["securitySchemes"] = map[string]any{"mutualTLS": map[string]any{"mtlsSecurityScheme": map[string]any{"description": "experiment mTLS"}}}
	h.AgentCard["securityRequirements"] = []any{map[string]any{"schemes": map[string]any{"mutualTLS": map[string]any{}}}}
	return h
}

type experimentRPC struct {
	ID     json.RawMessage `json:"id"`
	Method string          `json:"method"`
	Params json.RawMessage `json:"params"`
}

func experimentResponse(w http.ResponseWriter, id json.RawMessage, result any) {
	data, err := json.Marshal(map[string]any{"jsonrpc": "2.0", "id": id, "result": result})
	if err != nil {
		panic(err)
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Length", strconv.Itoa(len(data)))
	_, _ = w.Write(data)
}

func experimentWorkingTask() *sdk.Task {
	return &sdk.Task{ID: "task-1", ContextID: "context-1", Status: sdk.TaskStatus{State: sdk.TaskStateWorking}}
}

func experimentCompletedTask() *sdk.Task {
	return &sdk.Task{ID: "task-1", ContextID: "context-1", Status: sdk.TaskStatus{State: sdk.TaskStateCompleted, Message: resultMessage(successResult())}}
}

type experimentOwnedClient struct {
	protocolClient
	transport *http.Transport
}

func (c *experimentOwnedClient) Destroy() error {
	err := c.protocolClient.Destroy()
	c.transport.CloseIdleConnections()
	return err
}

// Each build occurs inside Invoke. No connection is shared with a later
// invocation; config hooks exist only for deterministic fault injection.
func newExperimentInvoker(t testing.TB, f *connectionExperiment, reuse bool, poll time.Duration, configure func(*http.Transport)) *Invoker {
	t.Helper()
	boundConfig := f.config(t) // Match NewMTLS: load certificate files before invocation timing.
	return &Invoker{requireHTTPS: true, pollInterval: poll, build: func(ctx context.Context, card *sdk.AgentCard) (protocolClient, error) {
		tr := &http.Transport{TLSClientConfig: boundConfig.Clone(), DisableKeepAlives: !reuse, ForceAttemptHTTP2: false, TLSHandshakeTimeout: 2 * time.Second, ResponseHeaderTimeout: 2 * time.Second}
		if configure != nil {
			configure(tr)
		}
		client, err := sdkClientBuilder(cloneBoundedHTTPClient(&http.Client{Transport: tr, Timeout: 2 * time.Second}))(ctx, card)
		if err != nil {
			tr.CloseIdleConnections()
			return nil, err
		}
		return &experimentOwnedClient{protocolClient: client, transport: tr}, nil
	}}
}

func experimentAwaitClosed(t testing.TB, f *connectionExperiment) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for f.closed.Load() != f.accepts.Load() && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if f.closed.Load() != f.accepts.Load() {
		t.Fatalf("connection leak: accepted=%d closed=%d", f.accepts.Load(), f.closed.Load())
	}
}

func experimentTaskHandler(pollsPerInvocation int64, sends, polls *atomic.Int64) http.Handler {
	var current atomic.Int64
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var rpc experimentRPC
		if r.Method != http.MethodPost || json.NewDecoder(r.Body).Decode(&rpc) != nil {
			http.Error(w, "invalid RPC", 400)
			return
		}
		switch rpc.Method {
		case "SendMessage":
			sends.Add(1)
			current.Store(0)
			experimentResponse(w, rpc.ID, sdk.StreamResponse{Event: experimentWorkingTask()})
		case "GetTask":
			polls.Add(1)
			result := experimentWorkingTask()
			if current.Add(1) >= pollsPerInvocation {
				result = experimentCompletedTask()
			}
			experimentResponse(w, rpc.ID, result)
		default:
			http.Error(w, "unexpected RPC", 400)
		}
	})
}

func TestConnectionReuseExperimentCounts(t *testing.T) {
	for _, strategy := range []string{"production", "owned_baseline", "owned_reuse"} {
		t.Run(strategy, func(t *testing.T) {
			var sends, polls atomic.Int64
			f := newConnectionExperiment(t, experimentTaskHandler(20, &sends, &polls))
			invoker := newExperimentInvoker(t, f, strategy == "owned_reuse", 2*time.Millisecond, nil)
			if strategy == "production" {
				var err error
				invoker, err = NewMTLS(f.files, 2*time.Second, Options{PollInterval: 2 * time.Millisecond})
				if err != nil {
					t.Fatal(err)
				}
			}
			start := time.Now()
			if _, err := invoker.Invoke(context.Background(), "worker", f.handle(), stageRequest()); err != nil {
				t.Fatal(err)
			}
			experimentAwaitClosed(t, f)
			want := int64(21)
			if strategy == "owned_reuse" {
				want = 1
			}
			if f.accepts.Load() != want || f.handshakes.Load() != want || sends.Load() != 1 || polls.Load() != 20 {
				t.Fatalf("counts: connections=%d handshakes=%d sends=%d polls=%d", f.accepts.Load(), f.handshakes.Load(), sends.Load(), polls.Load())
			}
			t.Logf("connections=%d handshakes=%d sends=%d polls=%d elapsed=%s", f.accepts.Load(), f.handshakes.Load(), sends.Load(), polls.Load(), time.Since(start))
		})
	}
}

func TestConnectionReuseExperimentFixedTaskDuration(t *testing.T) {
	// A second workload makes progress depend on elapsed server time instead
	// of number of polls. Slow clients may observe fewer intermediate states.
	for _, reuse := range []bool{false, true} {
		t.Run(fmt.Sprintf("reuse_%t", reuse), func(t *testing.T) {
			var readyAt atomic.Int64
			var sends, polls atomic.Int64
			f := newConnectionExperiment(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var rpc experimentRPC
				if json.NewDecoder(r.Body).Decode(&rpc) != nil {
					return
				}
				if rpc.Method == "SendMessage" {
					sends.Add(1)
					readyAt.Store(time.Now().Add(800 * time.Millisecond).UnixNano())
					experimentResponse(w, rpc.ID, sdk.StreamResponse{Event: experimentWorkingTask()})
					return
				}
				if rpc.Method != "GetTask" {
					http.Error(w, "invalid RPC", 400)
					return
				}
				polls.Add(1)
				result := experimentWorkingTask()
				if time.Now().UnixNano() >= readyAt.Load() {
					result = experimentCompletedTask()
				}
				experimentResponse(w, rpc.ID, result)
			}))
			invoker := newExperimentInvoker(t, f, reuse, 100*time.Millisecond, nil)
			start := time.Now()
			if _, err := invoker.Invoke(context.Background(), "worker", f.handle(), stageRequest()); err != nil {
				t.Fatal(err)
			}
			elapsed := time.Since(start)
			experimentAwaitClosed(t, f)
			want := sends.Load() + polls.Load()
			if reuse {
				want = 1
			}
			if sends.Load() != 1 || polls.Load() < 1 || f.accepts.Load() != want || f.handshakes.Load() != want || elapsed < 800*time.Millisecond {
				t.Fatal("invalid fixed-duration measurements")
			}
			t.Logf("task_ready_after=800ms poll_interval=100ms elapsed=%s sends=%d polls=%d connections=%d handshakes=%d", elapsed, sends.Load(), polls.Load(), f.accepts.Load(), f.handshakes.Load())
		})
	}
}

func TestConnectionReuseExperimentExplicitOwnership(t *testing.T) {
	var sends, polls atomic.Int64
	f := newConnectionExperiment(t, experimentTaskHandler(1, &sends, &polls))
	tr := &http.Transport{TLSClientConfig: f.config(t), ForceAttemptHTTP2: false}
	t.Cleanup(tr.CloseIdleConnections)
	bounded := cloneBoundedHTTPClient(&http.Client{Transport: tr, Timeout: time.Second})
	card, err := decodeCard(f.handle(), true)
	if err != nil {
		t.Fatal(err)
	}
	client, err := sdkClientBuilder(bounded)(context.Background(), card)
	if err != nil {
		t.Fatal(err)
	}
	get := func() {
		t.Helper()
		if _, err := client.GetTask(context.Background(), &sdk.GetTaskRequest{Tenant: "allocation-1", ID: "task-1"}); err != nil {
			t.Fatal(err)
		}
	}
	get()
	if err := client.Destroy(); err != nil {
		t.Fatal(err)
	}
	bounded.CloseIdleConnections()
	get()
	if f.accepts.Load() != 1 {
		t.Fatal("SDK/bounded client unexpectedly closed the connection")
	}
	tr.CloseIdleConnections()
	experimentAwaitClosed(t, f)
	get()
	if f.accepts.Load() != 2 {
		t.Fatal("raw transport cleanup did not close idle connection")
	}
	tr.CloseIdleConnections()
	experimentAwaitClosed(t, f)
	t.Log("SDK Destroy + bounded client CloseIdleConnections retain socket; owned raw transport closes it")
}

// Fixed poll count models identical progress transitions; duration includes
// polling and transport overhead. PKI/server setup is outside timed iterations.
func BenchmarkConnectionReuseExperiment(b *testing.B) {
	for _, interval := range []time.Duration{2 * time.Millisecond, 100 * time.Millisecond} {
		for _, reuse := range []bool{false, true} {
			b.Run(fmt.Sprintf("poll_%s/reuse_%t", interval, reuse), func(b *testing.B) {
				var sends, polls atomic.Int64
				f := newConnectionExperiment(b, experimentTaskHandler(8, &sends, &polls))
				invoker := newExperimentInvoker(b, f, reuse, interval, nil)
				b.ResetTimer()
				for n := 0; n < b.N; n++ {
					if _, err := invoker.Invoke(context.Background(), "worker", f.handle(), stageRequest()); err != nil {
						b.Fatal(err)
					}
				}
				b.StopTimer()
				experimentAwaitClosed(b, f)
				wantConnections := int64(b.N) * 9
				if reuse {
					wantConnections = int64(b.N)
				}
				if f.accepts.Load() != wantConnections || f.handshakes.Load() != wantConnections || sends.Load() != int64(b.N) || polls.Load() != int64(b.N)*8 {
					b.Fatal("unexpected benchmark counters")
				}
				b.ReportMetric(float64(f.accepts.Load())/float64(b.N), "connections/op")
				b.ReportMetric(float64(f.handshakes.Load())/float64(b.N), "handshakes/op")
				b.ReportMetric(float64(sends.Load())/float64(b.N), "sends/op")
				b.ReportMetric(float64(polls.Load())/float64(b.N), "polls/op")
			})
		}
	}
}

type experimentBodyCounter struct {
	io.ReadCloser
	bytes, closes *atomic.Int64
}

func (b *experimentBodyCounter) Read(p []byte) (int, error) {
	n, err := b.ReadCloser.Read(p)
	b.bytes.Add(int64(n))
	return n, err
}
func (b *experimentBodyCounter) Close() error { b.closes.Add(1); return b.ReadCloser.Close() }

type experimentObserveTransport struct {
	base          http.RoundTripper
	bytes, closes *atomic.Int64
}

func (t experimentObserveTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	resp, err := t.base.RoundTrip(r)
	if err == nil {
		resp.Body = &experimentBodyCounter{ReadCloser: resp.Body, bytes: t.bytes, closes: t.closes}
	}
	return resp, err
}

func TestConnectionReuseExperimentResponseBounds(t *testing.T) {
	for _, reuse := range []bool{false, true} {
		for _, scenario := range []string{"known_length", "chunked", "oversize_length", "oversize_chunked", "malformed", "truncated", "valid_prefix_unread_tail"} {
			t.Run(fmt.Sprintf("reuse_%t/%s", reuse, scenario), func(t *testing.T) {
				var bytesRead, closes, calls atomic.Int64
				f := newConnectionExperiment(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					var rpc experimentRPC
					if json.NewDecoder(r.Body).Decode(&rpc) != nil {
						return
					}
					calls.Add(1)
					data, _ := json.Marshal(map[string]any{"jsonrpc": "2.0", "id": rpc.ID, "result": sdk.StreamResponse{Event: resultMessage(successResult())}})
					switch scenario {
					case "known_length":
						w.Header().Set("Content-Length", strconv.Itoa(len(data)))
						_, _ = w.Write(data)
					case "chunked":
						w.(http.Flusher).Flush()
						_, _ = w.Write(data)
					case "oversize_length":
						w.Header().Set("Content-Length", strconv.Itoa(maxA2AResponseBytes+1))
						w.(http.Flusher).Flush()
					case "oversize_chunked":
						w.(http.Flusher).Flush()
						_, _ = io.WriteString(w, `{"padding":"`+strings.Repeat("a", maxA2AResponseBytes+100)+`"}`)
					case "malformed":
						_, _ = io.WriteString(w, "invalid JSON")
					case "truncated":
						_, _ = io.WriteString(w, `{"jsonrpc":"2.0","result":`)
					case "valid_prefix_unread_tail":
						// The entity never finishes; SDK must return after one object,
						// close the body and cancel the server before any tail is sent.
						_, _ = w.Write(data)
						w.(http.Flusher).Flush()
						<-r.Context().Done()
					}
				}))
				tr := &http.Transport{TLSClientConfig: f.config(t), DisableKeepAlives: !reuse, ForceAttemptHTTP2: false}
				defer tr.CloseIdleConnections()
				client := cloneBoundedHTTPClient(&http.Client{Transport: experimentObserveTransport{base: tr, bytes: &bytesRead, closes: &closes}, Timeout: time.Second})
				card, err := decodeCard(f.handle(), true)
				if err != nil {
					t.Fatal(err)
				}
				sdkClient, err := sdkClientBuilder(client)(context.Background(), card)
				if err != nil {
					t.Fatal(err)
				}
				for n := 0; n < 2; n++ {
					before := bytesRead.Load()
					_, err = sdkClient.SendMessage(context.Background(), &sdk.SendMessageRequest{Tenant: "allocation-1", Message: sdk.NewMessage(sdk.MessageRoleUser, sdk.NewDataPart(stageRequest()))})
					wantSuccess := scenario == "known_length" || scenario == "chunked" || scenario == "valid_prefix_unread_tail"
					if (err == nil) != wantSuccess {
						t.Fatalf("success=%t error=%v", wantSuccess, err)
					}
					if delta := bytesRead.Load() - before; delta > maxA2AResponseBytes+1 {
						t.Fatalf("unbounded read: %d", delta)
					}
				}
				_ = sdkClient.Destroy()
				tr.CloseIdleConnections()
				experimentAwaitClosed(t, f)
				if calls.Load() != 2 || closes.Load() != 2 {
					t.Fatalf("calls=%d body closes=%d", calls.Load(), closes.Load())
				}
				wantConnections := int64(2)
				if reuse && (scenario == "known_length" || scenario == "chunked") {
					wantConnections = 1
				}
				// Malformed short entities can be fully buffered and reused; the
				// required bound here is closure, not unconditional pool eviction.
				if scenario != "malformed" && scenario != "truncated" && f.accepts.Load() != wantConnections {
					t.Fatalf("connections=%d want=%d", f.accepts.Load(), wantConnections)
				}
				t.Logf("connections=%d handshakes=%d responses=%d read_bytes=%d body_closes=%d success=%t", f.accepts.Load(), f.handshakes.Load(), calls.Load(), bytesRead.Load(), closes.Load(), err == nil)
			})
		}
	}
}
