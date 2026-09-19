package a2a

import (
	"context"
	"crypto/sha256"
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptrace"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	sdk "github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/grauwolf32/contractor/internal/requestid"
)

// These cases characterize Go's hidden Transport retries through the real SDK
// and the production bounded-response wrapper. Warming SendMessage with GetTask
// is deliberately adversarial: Invoke sends its only SendMessage on a fresh
// invocation-owned connection. The GetTask cases preserve Invoke's actual order.
func TestConnectionReuseExperimentRetry(t *testing.T) {
	tests := []struct {
		name          string
		fault         string
		warm          bool
		getTask       bool
		removeGetBody bool
		header        string
		wantSuccess   bool
		wantEffects   int
		wantDials     int
	}{
		{name: "fresh_send_control", wantSuccess: true, wantEffects: 1, wantDials: 1},
		{name: "reused_send_control_without_getbody", warm: true, removeGetBody: true, wantSuccess: true, wantEffects: 1, wantDials: 1},
		{name: "fresh_send_zero_write", fault: "zero_write", wantDials: 1},
		{name: "reused_send_zero_write", fault: "zero_write", warm: true, wantSuccess: true, wantEffects: 1, wantDials: 2},
		{name: "reused_send_zero_write_without_getbody", fault: "zero_write", warm: true, removeGetBody: true, wantDials: 1},
		{name: "reused_send_zero_write_header_without_getbody", fault: "zero_write", warm: true, removeGetBody: true, header: "Idempotency-Key", wantDials: 1},
		{name: "fresh_send_partial_write", fault: "partial_write", wantDials: 1},
		{name: "reused_send_partial_write", fault: "partial_write", warm: true, wantDials: 1},
		{name: "reused_send_partial_write_without_getbody", fault: "partial_write", warm: true, removeGetBody: true, wantDials: 1},
		{name: "fresh_send_lost_response", fault: "lost_response", wantEffects: 1, wantDials: 1},
		{name: "fresh_send_lost_response_header", fault: "lost_response", header: "Idempotency-Key", wantEffects: 1, wantDials: 1},
		{name: "reused_send_lost_response", fault: "lost_response", warm: true, wantEffects: 1, wantDials: 1},
		{name: "reused_send_lost_response_without_getbody", fault: "lost_response", warm: true, removeGetBody: true, wantEffects: 1, wantDials: 1},
		{name: "reused_send_lost_response_idempotency_key_replays", fault: "lost_response", warm: true, header: "Idempotency-Key", wantSuccess: true, wantEffects: 2, wantDials: 2},
		{name: "reused_send_lost_response_x_idempotency_key_replays", fault: "lost_response", warm: true, header: "X-Idempotency-Key", wantSuccess: true, wantEffects: 2, wantDials: 2},
		{name: "reused_send_lost_response_idempotency_key_without_getbody", fault: "lost_response", warm: true, removeGetBody: true, header: "Idempotency-Key", wantEffects: 1, wantDials: 1},
		{name: "reused_send_lost_response_x_idempotency_key_without_getbody", fault: "lost_response", warm: true, removeGetBody: true, header: "X-Idempotency-Key", wantEffects: 1, wantDials: 1},
		{name: "get_task_after_send_zero_write", fault: "zero_write", warm: true, getTask: true, wantSuccess: true, wantEffects: 1, wantDials: 2},
		{name: "get_task_after_send_lost_response", fault: "lost_response", warm: true, getTask: true, wantEffects: 1, wantDials: 1},
		{name: "get_task_after_send_lost_response_header_replays_read", fault: "lost_response", warm: true, getTask: true, header: "Idempotency-Key", wantSuccess: true, wantEffects: 2, wantDials: 2},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()

			var recordsMu sync.Mutex
			var records []experimentRetryRecord
			var loseResponse atomic.Bool
			fixture := newConnectionExperiment(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, err := io.ReadAll(io.LimitReader(r.Body, 64<<10))
				if err != nil {
					t.Errorf("read fault-fixture request: %v", err)
					return
				}
				var rpc experimentRPC
				if err := json.Unmarshal(body, &rpc); err != nil {
					t.Errorf("decode fault-fixture request: %v", err)
					return
				}
				// Recording is the fixture's semantic effect. There is intentionally
				// no RPC-ID deduplication that could conceal a physical replay.
				recordsMu.Lock()
				records = append(records, experimentRetryRecord{
					method: rpc.Method, id: string(rpc.ID), digest: sha256.Sum256(body),
					httpMethod: r.Method, bodyBytes: len(body), requestID: r.Header.Get(requestid.Header),
					idempotency: r.Header.Get("Idempotency-Key"), xIdempotency: r.Header.Get("X-Idempotency-Key"),
				})
				recordsMu.Unlock()
				if loseResponse.CompareAndSwap(true, false) {
					hijacker, ok := w.(http.Hijacker)
					if !ok {
						t.Error("HTTP/1 TLS fixture must support hijacking")
						return
					}
					conn, _, err := hijacker.Hijack()
					if err != nil {
						t.Errorf("hijack lost-response connection: %v", err)
						return
					}
					_ = conn.Close() // No HTTP response bytes after the recorded effect.
					return
				}
				switch rpc.Method {
				case "SendMessage":
					experimentResponse(w, rpc.ID, sdk.StreamResponse{Event: experimentWorkingTask()})
				case "GetTask":
					experimentResponse(w, rpc.ID, experimentWorkingTask())
				default:
					t.Errorf("unexpected SDK RPC method %q", rpc.Method)
					http.Error(w, "unexpected method", http.StatusBadRequest)
				}
			}))

			fault := &experimentRetryWriteFault{kind: test.fault}
			var dials, clientHandshakes atomic.Int64
			boundTLS := fixture.config(t)
			transport := &http.Transport{
				ForceAttemptHTTP2: false, MaxIdleConns: 1, MaxIdleConnsPerHost: 1,
				TLSClientConfig: boundTLS, ResponseHeaderTimeout: 3 * time.Second,
				DialTLSContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
					dials.Add(1)
					raw, err := (&net.Dialer{}).DialContext(ctx, network, addr)
					if err != nil {
						return nil, err
					}
					config := boundTLS.Clone()
					if config.ServerName == "" {
						host, _, err := net.SplitHostPort(addr)
						if err != nil {
							_ = raw.Close()
							return nil, err
						}
						config.ServerName = host
					}
					tlsConn := tls.Client(raw, config)
					if err := tlsConn.HandshakeContext(ctx); err != nil {
						_ = raw.Close()
						return nil, err
					}
					clientHandshakes.Add(1)
					// Injection above TLS counts plaintext HTTP writes, never handshake
					// records. The wrapper hides *tls.Conn from net/http; principal and
					// hostname checks therefore happen explicitly before it is returned.
					return &experimentRetryConn{Conn: tlsConn, fault: fault}, nil
				},
			}
			t.Cleanup(transport.CloseIdleConnections)
			policy := &experimentRetryPolicy{base: transport, removeGetBody: test.removeGetBody, header: test.header}
			httpClient := cloneBoundedHTTPClient(&http.Client{Transport: policy, Timeout: 4 * time.Second})
			card, err := decodeCard(fixture.handle(), true)
			if err != nil {
				t.Fatal(err)
			}
			client, err := sdkClientBuilder(httpClient)(ctx, card)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = client.Destroy() })

			send := func(ctx context.Context) error {
				_, err := client.SendMessage(ctx, &sdk.SendMessageRequest{
					Tenant:  fixture.handle().AllocationID,
					Message: sdk.NewMessage(sdk.MessageRoleUser, sdk.NewTextPart("offline connection experiment")),
					Config:  &sdk.SendMessageConfig{ReturnImmediately: true},
				})
				return err
			}
			get := func(ctx context.Context) error {
				_, err := client.GetTask(ctx, &sdk.GetTaskRequest{
					Tenant: fixture.handle().AllocationID, ID: experimentWorkingTask().ID,
				})
				return err
			}
			if test.warm {
				warm := get
				if test.getTask {
					warm = send
				}
				if err := warm(ctx); err != nil {
					t.Fatalf("SDK warmup: %v", err)
				}
			}
			if test.fault == "lost_response" {
				loseResponse.Store(true)
			} else if test.fault != "" {
				fault.armed.Store(true)
			}
			var traceMu sync.Mutex
			var reused []bool
			targetCtx := httptrace.WithClientTrace(ctx, &httptrace.ClientTrace{
				GotConn: func(info httptrace.GotConnInfo) {
					traceMu.Lock()
					reused = append(reused, info.Reused)
					traceMu.Unlock()
				},
			})
			target := send
			targetMethod := "SendMessage"
			if test.getTask {
				target, targetMethod = get, "GetTask"
			}
			err = target(targetCtx)
			if (err == nil) != test.wantSuccess {
				t.Fatalf("SDK success=%t, want %t: %v", err == nil, test.wantSuccess, err)
			}
			if ctx.Err() != nil {
				t.Fatalf("fault must finish without timing out: %v", ctx.Err())
			}
			cancel()
			transport.CloseIdleConnections()
			experimentAwaitClosed(t, fixture)

			recordsMu.Lock()
			gotRecords := append([]experimentRetryRecord(nil), records...)
			recordsMu.Unlock()
			var effects []experimentRetryRecord
			var sends, polls int
			for _, record := range gotRecords {
				if record.httpMethod != http.MethodPost || record.bodyBytes == 0 || record.requestID == "" {
					t.Errorf("actual request must be nonempty SDK POST with Contractor request ID: %+v", record)
				}
				if record.method == "SendMessage" {
					sends++
				} else if record.method == "GetTask" {
					polls++
				}
				if record.method == targetMethod {
					effects = append(effects, record)
				}
				wantID, wantXID := "", ""
				if test.header == "Idempotency-Key" {
					wantID = "offline-replay-characterization"
				} else if test.header == "X-Idempotency-Key" {
					wantXID = "offline-replay-characterization"
				}
				if record.idempotency != wantID || record.xIdempotency != wantXID {
					t.Errorf("actual idempotency headers = (%q,%q), want (%q,%q)", record.idempotency, record.xIdempotency, wantID, wantXID)
				}
			}
			if len(effects) != test.wantEffects {
				t.Errorf("target effects=%d, want %d (SendMessage=%d GetTask=%d)", len(effects), test.wantEffects, sends, polls)
			}
			if test.getTask && sends != 1 {
				t.Errorf("poll fault must not repeat the initial SendMessage: got %d", sends)
			}
			if len(effects) == 2 && (effects[0].id != effects[1].id || effects[0].digest != effects[1].digest || effects[0].requestID != effects[1].requestID) {
				t.Error("transport replay must preserve JSON-RPC ID, full body digest, and request ID")
			}
			if int(dials.Load()) != test.wantDials || clientHandshakes.Load() != dials.Load() {
				t.Errorf("dials=%d client TLS handshakes=%d, want %d", dials.Load(), clientHandshakes.Load(), test.wantDials)
			}
			traceMu.Lock()
			gotReused := append([]bool(nil), reused...)
			traceMu.Unlock()
			wantConnections := 1
			if test.warm && test.wantDials == 2 {
				wantConnections = 2
			}
			if len(gotReused) != wantConnections || len(gotReused) == 0 || gotReused[0] != test.warm || (len(gotReused) == 2 && gotReused[1]) {
				t.Errorf("target GotConn.Reused=%v, want first=%t and %d connections with fresh retry", gotReused, test.warm, wantConnections)
			}
			wantSDKRequests := int64(1)
			if test.warm {
				wantSDKRequests++
			}
			if policy.calls.Load() != wantSDKRequests || policy.invalidSDKRequests.Load() != 0 {
				t.Errorf("SDK requests=%d invalid=%d, want %d nonempty POSTs with original GetBody", policy.calls.Load(), policy.invalidSDKRequests.Load(), wantSDKRequests)
			}
			if test.fault == "zero_write" || test.fault == "partial_write" {
				wantBytes := int64(0)
				if test.fault == "partial_write" {
					wantBytes = 1
				}
				if fault.fired.Load() != 1 || fault.forwarded.Load() != wantBytes {
					t.Errorf("injected writes=%d forwarded plaintext bytes=%d, want 1/%d", fault.fired.Load(), fault.forwarded.Load(), wantBytes)
				}
			}
			if loseResponse.Load() {
				t.Error("lost-response fault was not exercised")
			}
			t.Logf("target=%s success=%t dials=%d client_tls=%d server_accepts=%d server_tls=%d sends=%d polls=%d target_effects=%d sdk_requests=%d reused=%v fault_writes=%d fault_bytes=%d",
				targetMethod, err == nil, dials.Load(), clientHandshakes.Load(), fixture.accepts.Load(), fixture.handshakes.Load(), sends, polls, len(effects), policy.calls.Load(), gotReused, fault.fired.Load(), fault.forwarded.Load())
		})
	}
}

type experimentRetryRecord struct {
	method, id, httpMethod, requestID, idempotency, xIdempotency string
	digest                                                       [sha256.Size]byte
	bodyBytes                                                    int
}

type experimentRetryPolicy struct {
	base               http.RoundTripper
	removeGetBody      bool
	header             string
	calls              atomic.Int64
	invalidSDKRequests atomic.Int64
}

func (p *experimentRetryPolicy) RoundTrip(request *http.Request) (*http.Response, error) {
	p.calls.Add(1)
	if request.Method != http.MethodPost || request.GetBody == nil || request.ContentLength <= 0 {
		p.invalidSDKRequests.Add(1)
	}
	outgoing := request.Clone(request.Context())
	outgoing.Header = request.Header.Clone()
	if p.removeGetBody {
		outgoing.GetBody = nil // Test-only retry-policy candidate, preserving the actual body.
	}
	if p.header != "" {
		outgoing.Header.Set(p.header, "offline-replay-characterization")
	}
	return p.base.RoundTrip(outgoing)
}

type experimentRetryWriteFault struct {
	kind      string
	armed     atomic.Bool
	fired     atomic.Int64
	forwarded atomic.Int64
}

type experimentRetryConn struct {
	net.Conn
	fault *experimentRetryWriteFault
}

func (c *experimentRetryConn) Write(p []byte) (int, error) {
	if len(p) == 0 || !c.fault.armed.CompareAndSwap(true, false) {
		return c.Conn.Write(p)
	}
	c.fault.fired.Add(1)
	switch c.fault.kind {
	case "zero_write":
		return 0, errors.New("offline injected zero write")
	case "partial_write":
		n, err := c.Conn.Write(p[:1])
		c.fault.forwarded.Add(int64(n))
		if err != nil {
			return n, err
		}
		return n, errors.New("offline injected partial write")
	default:
		return 0, fmt.Errorf("unknown offline write fault %q", c.fault.kind)
	}
}
