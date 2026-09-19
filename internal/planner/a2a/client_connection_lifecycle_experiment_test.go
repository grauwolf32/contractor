package a2a

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptrace"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	sdk "github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/localpki"
	"github.com/grauwolf32/contractor/internal/mtls"
)

func TestConnectionReuseExperimentLifecycleWrongPrincipal(t *testing.T) {
	for _, reuse := range []bool{false, true} {
		t.Run(fmt.Sprintf("reuse=%t", reuse), func(t *testing.T) {
			var requests atomic.Int64
			f := newConnectionExperiment(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				requests.Add(1)
				http.Error(w, "unexpected protected request", http.StatusForbidden)
			}))
			other := lifecycleExperimentCertificate(t, f, "expected-other-principal")
			principal, err := mtls.RuntimeAgentID(other.Leaf)
			if err != nil {
				t.Fatal(err)
			}
			invoker := newExperimentInvoker(t, f, reuse, time.Millisecond, func(transport *http.Transport) {
				base, err := mtls.ControlPlaneEndpointClientConfig(f.files)
				if err != nil {
					t.Fatal(err)
				}
				transport.TLSClientConfig, err = mtls.BindRuntimeAgentPrincipal(base, principal)
				if err != nil {
					t.Fatal(err)
				}
			})
			_, err = invoker.Invoke(context.Background(), "builder", f.handle(), stageRequest())
			assertPlannerCode(t, err, "worker_unavailable")
			lifecycleExperimentWaitClosed(t, f)
			if requests.Load() != 0 {
				t.Fatalf("wrong principal received %d HTTP requests", requests.Load())
			}
			t.Logf("wrong principal: protected requests=0 accepted=%d closed=%d", f.accepts.Load(), f.closed.Load())
		})
	}
}

func TestConnectionReuseExperimentLifecycleCancellation(t *testing.T) {
	for _, reuse := range []bool{false, true} {
		for _, phase := range []string{
			"poll_wait", "headers", "body", "deadline_headers", "deadline_body",
			"poll_headers", "poll_body", "deadline_poll_headers", "deadline_poll_body",
		} {
			t.Run(fmt.Sprintf("reuse=%t/%s", reuse, phase), func(t *testing.T) {
				activePoll := phase != "poll_wait" && strings.Contains(phase, "poll_")
				ready := make(chan struct{})
				stopped := make(chan struct{})
				var sends, polls, connectionUses, reusedUses atomic.Int64
				f := newConnectionExperiment(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					rpc, ok := lifecycleExperimentRPC(t, w, r)
					if !ok {
						return
					}
					if rpc.Method == "GetTask" {
						polls.Add(1)
						if !activePoll {
							http.Error(w, "unexpected poll", http.StatusInternalServerError)
							return
						}
					} else {
						sends.Add(1)
						if phase == "poll_wait" || activePoll {
							experimentResponse(w, rpc.ID, sdk.StreamResponse{Event: experimentWorkingTask()})
							return
						}
					}
					if strings.HasSuffix(phase, "body") {
						w.Header().Set("Content-Type", "application/json")
						_, _ = w.Write([]byte(`{"jsonrpc":"2.0","id":`))
						w.(http.Flusher).Flush()
					}
					close(ready)
					<-r.Context().Done()
					close(stopped)
				}))
				pollInterval := time.Minute
				if activePoll {
					pollInterval = time.Millisecond
				}
				invoker := newExperimentInvoker(t, f, reuse, pollInterval, nil)
				if phase == "poll_wait" {
					lifecycleExperimentAfterSend(invoker, func() { close(ready) })
				}
				var ctx context.Context
				var cancel context.CancelFunc
				deadline := strings.HasPrefix(phase, "deadline_")
				if deadline {
					ctx, cancel = context.WithTimeout(context.Background(), 600*time.Millisecond)
				} else {
					ctx, cancel = context.WithCancel(context.Background())
				}
				defer cancel()
				ctx = httptrace.WithClientTrace(ctx, &httptrace.ClientTrace{GotConn: func(info httptrace.GotConnInfo) {
					connectionUses.Add(1)
					if info.Reused {
						reusedUses.Add(1)
					}
				}})
				finished := make(chan error, 1)
				go func() {
					_, err := invoker.Invoke(ctx, "builder", f.handle(), stageRequest())
					finished <- err
				}()
				lifecycleExperimentWait(t, ready, "request phase")
				if !deadline {
					cancel()
				}
				select {
				case err := <-finished:
					code := "planner_cancelled"
					if deadline {
						code = "worker_deadline_exceeded"
					}
					assertPlannerCode(t, err, code)
				case <-time.After(3 * time.Second):
					t.Fatal("invocation did not terminate within cancellation/deadline bound")
				}
				if phase != "poll_wait" {
					lifecycleExperimentWait(t, stopped, "server request context cancellation")
				}
				lifecycleExperimentWaitClosed(t, f)
				var wantPolls, wantReused int64
				wantConnections := int64(1)
				if activePoll {
					wantPolls = 1
					if reuse {
						wantReused = 1
					} else {
						wantConnections = 2
					}
				}
				if sends.Load() != 1 || polls.Load() != wantPolls {
					t.Fatalf("cancellation dispatches: SendMessage=%d GetTask=%d", sends.Load(), polls.Load())
				}
				if connectionUses.Load() != 1+wantPolls || reusedUses.Load() != wantReused ||
					f.accepts.Load() != wantConnections || f.handshakes.Load() != wantConnections {
					t.Fatalf("cancellation connection path: GotConn=%d reused=%d accepted=%d handshakes=%d",
						connectionUses.Load(), reusedUses.Load(), f.accepts.Load(), f.handshakes.Load())
				}
				t.Logf("%s: SendMessage=1 GetTask=%d GotConn=%d reused=%d accepted=%d closed=%d",
					phase, polls.Load(), connectionUses.Load(), reusedUses.Load(), f.accepts.Load(), f.closed.Load())
			})
		}
	}
}

func TestConnectionReuseExperimentLifecycleLaterAllocation(t *testing.T) {
	for _, reuse := range []bool{false, true} {
		t.Run(fmt.Sprintf("reuse=%t", reuse), func(t *testing.T) {
			var sends atomic.Int64
			var tenants []string
			f := newConnectionExperiment(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				rpc, ok := lifecycleExperimentRPC(t, w, r)
				if !ok {
					return
				}
				var params struct {
					Tenant string `json:"tenant"`
				}
				if err := json.Unmarshal(rpc.Params, &params); err != nil {
					t.Errorf("decode tenant: %v", err)
					return
				}
				tenants = append(tenants, params.Tenant)
				sends.Add(1)
				experimentResponse(w, rpc.ID, sdk.StreamResponse{Event: experimentCompletedTask()})
			}))
			invoker := newExperimentInvoker(t, f, reuse, time.Millisecond, nil)
			for _, allocation := range []string{"allocation-first", "allocation-later"} {
				_, err := invoker.Invoke(context.Background(), "builder", lifecycleExperimentHandle(t, f, allocation), stageRequest())
				if err != nil {
					t.Fatal(err)
				}
				lifecycleExperimentWaitClosed(t, f)
			}
			if sends.Load() != 2 || f.accepts.Load() != 2 || f.handshakes.Load() != 2 ||
				len(tenants) != 2 || tenants[0] != "allocation-first" || tenants[1] != "allocation-later" {
				t.Fatalf("later allocation reused invocation resources: sends=%d accepts=%d handshakes=%d tenants=%v",
					sends.Load(), f.accepts.Load(), f.handshakes.Load(), tenants)
			}
			t.Log("two allocations on the same endpoint: two independently closed connections/handshakes, exact tenant per invocation")
		})
	}
}

func TestConnectionReuseExperimentLifecycleSimulatedRetirement(t *testing.T) {
	for _, reuse := range []bool{false, true} {
		t.Run(fmt.Sprintf("reuse=%t", reuse), func(t *testing.T) {
			var retired atomic.Bool
			var sends, rejected, polls atomic.Int64
			f := newConnectionExperiment(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				rpc, ok := lifecycleExperimentRPC(t, w, r)
				if !ok {
					return
				}
				var params struct {
					Tenant string `json:"tenant"`
				}
				if err := json.Unmarshal(rpc.Params, &params); err != nil {
					t.Errorf("decode tenant: %v", err)
					return
				}
				// This fixture models the Runtime allocation gate. It does not run
				// the production Registry, release protocol or process watchdog.
				if retired.Load() && params.Tenant == "allocation-retired" {
					rejected.Add(1)
					http.Error(w, "allocation retired", http.StatusGone)
					return
				}
				if rpc.Method == "GetTask" {
					polls.Add(1)
					http.Error(w, "unexpected dispatch", http.StatusInternalServerError)
					return
				}
				sends.Add(1)
				if params.Tenant == "allocation-retired" {
					experimentResponse(w, rpc.ID, sdk.StreamResponse{Event: experimentWorkingTask()})
					return
				}
				experimentResponse(w, rpc.ID, sdk.StreamResponse{Event: experimentCompletedTask()})
			}))
			invoker := newExperimentInvoker(t, f, reuse, time.Millisecond, nil)
			lifecycleExperimentAfterSend(invoker, func() {
				retired.Store(true)
				// Simulate retirement of the old endpoint connections. Reconnecting
				// with the same certificate still cannot continue the old tenant.
				f.server.CloseClientConnections()
			})
			_, err := invoker.Invoke(context.Background(), "builder", lifecycleExperimentHandle(t, f, "allocation-retired"), stageRequest())
			assertPlannerCode(t, err, "worker_unavailable")
			lifecycleExperimentWaitClosed(t, f)
			if sends.Load() != 1 || rejected.Load() != 1 || polls.Load() != 0 {
				t.Fatalf("retired allocation crossed simulated gate: sends=%d rejected=%d dispatched polls=%d", sends.Load(), rejected.Load(), polls.Load())
			}
			fresh := newExperimentInvoker(t, f, reuse, time.Millisecond, nil)
			before := f.accepts.Load()
			_, err = fresh.Invoke(context.Background(), "builder", lifecycleExperimentHandle(t, f, "allocation-replacement"), stageRequest())
			if err != nil {
				t.Fatal(err)
			}
			lifecycleExperimentWaitClosed(t, f)
			if sends.Load() != 2 || f.accepts.Load() != before+1 {
				t.Fatal("replacement allocation did not use a new invocation connection")
			}
			t.Log("same-key simulated process retirement: old tenant reaches HTTP but is rejected before Worker dispatch; replacement uses a new invocation")
		})
	}
}

func TestConnectionReuseExperimentLifecycleCertificateExpiry(t *testing.T) {
	for _, reuse := range []bool{false, true} {
		for _, expiredInitially := range []bool{false, true} {
			t.Run(fmt.Sprintf("reuse=%t/expiredInitially=%t", reuse, expiredInitially), func(t *testing.T) {
				var clock atomic.Int64
				clock.Store(time.Now().UnixNano())
				var requests atomic.Int64
				var f *connectionExperiment
				f = newConnectionExperiment(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					rpc, ok := lifecycleExperimentRPC(t, w, r)
					if !ok {
						return
					}
					requests.Add(1)
					if rpc.Method == "SendMessage" {
						clock.Store(f.certificate.NotAfter.Add(time.Second).UnixNano())
						experimentResponse(w, rpc.ID, sdk.StreamResponse{Event: experimentWorkingTask()})
						return
					}
					experimentResponse(w, rpc.ID, experimentCompletedTask())
				}))
				if expiredInitially {
					clock.Store(f.certificate.NotAfter.Add(time.Second).UnixNano())
				}
				invoker := newExperimentInvoker(t, f, reuse, time.Millisecond, func(transport *http.Transport) {
					transport.TLSClientConfig.Time = func() time.Time { return time.Unix(0, clock.Load()) }
				})
				_, err := invoker.Invoke(context.Background(), "builder", f.handle(), stageRequest())
				lifecycleExperimentWaitClosed(t, f)
				switch {
				case expiredInitially:
					assertPlannerCode(t, err, "worker_unavailable")
					if requests.Load() != 0 {
						t.Fatal("initially expired certificate received protected HTTP bytes")
					}
				case reuse:
					if err != nil || requests.Load() != 2 || f.accepts.Load() != 1 || f.handshakes.Load() != 1 {
						t.Fatalf("existing connection unexpectedly reverified certificate: err=%v requests=%d accepts=%d handshakes=%d", err, requests.Load(), f.accepts.Load(), f.handshakes.Load())
					}
					t.Log("candidate limitation: established TLS connection continues after server-certificate NotAfter in client verification clock")
				default:
					assertPlannerCode(t, err, "worker_unavailable")
					if requests.Load() != 1 || f.accepts.Load() != 2 {
						t.Fatalf("fresh handshake expiry bound: requests=%d accepts=%d", requests.Load(), f.accepts.Load())
					}
					t.Log("baseline: next poll performs fresh handshake and rejects expired server certificate before HTTP")
				}
			})
		}
	}
}

func TestConnectionReuseExperimentLifecycleKeyRotation(t *testing.T) {
	for _, reuse := range []bool{false, true} {
		t.Run(fmt.Sprintf("reuse=%t", reuse), func(t *testing.T) {
			var requests atomic.Int64
			f := newConnectionExperiment(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				rpc, ok := lifecycleExperimentRPC(t, w, r)
				if !ok {
					return
				}
				requests.Add(1)
				experimentResponse(w, rpc.ID, sdk.StreamResponse{Event: experimentCompletedTask()})
			}))
			original := f.server.TLS.Certificates[0]
			var current atomic.Pointer[tls.Certificate]
			current.Store(&original)
			// Configure once before any handshake; only the atomic certificate
			// pointer changes while the server is serving.
			f.server.TLS.Certificates = nil
			f.server.TLS.GetCertificate = func(*tls.ClientHelloInfo) (*tls.Certificate, error) { return current.Load(), nil }
			rotated := lifecycleExperimentCertificate(t, f, "rotated-runtime")
			invoker := newExperimentInvoker(t, f, reuse, time.Millisecond, nil)
			_, err := invoker.Invoke(context.Background(), "builder", f.handle(), stageRequest())
			if err != nil {
				t.Fatal(err)
			}
			lifecycleExperimentWaitClosed(t, f)
			current.Store(&rotated)
			_, err = invoker.Invoke(context.Background(), "builder", f.handle(), stageRequest())
			assertPlannerCode(t, err, "worker_unavailable")
			lifecycleExperimentWaitClosed(t, f)
			if requests.Load() != 1 {
				t.Fatal("rotated endpoint received a request bound to the former principal")
			}
			f.principal, err = mtls.RuntimeAgentID(rotated.Leaf)
			if err != nil {
				t.Fatal(err)
			}
			fresh := newExperimentInvoker(t, f, reuse, time.Millisecond, nil)
			_, err = fresh.Invoke(context.Background(), "builder", f.handle(), stageRequest())
			if err != nil {
				t.Fatal(err)
			}
			lifecycleExperimentWaitClosed(t, f)
			if requests.Load() != 2 || f.accepts.Load() != 3 {
				t.Fatalf("rotation accounting: requests=%d accepts=%d", requests.Load(), f.accepts.Load())
			}
			t.Log("new key on same CA/SAN/endpoint: old principal rejected before HTTP; newly bound invocation accepted")
		})
	}
}

func lifecycleExperimentRPC(t *testing.T, w http.ResponseWriter, r *http.Request) (experimentRPC, bool) {
	t.Helper()
	var rpc experimentRPC
	if err := json.NewDecoder(r.Body).Decode(&rpc); err != nil {
		t.Errorf("decode experiment request: %v", err)
		http.Error(w, "invalid fixture request", http.StatusBadRequest)
		return rpc, false
	}
	return rpc, true
}

func lifecycleExperimentCertificate(t *testing.T, f *connectionExperiment, name string) tls.Certificate {
	t.Helper()
	paths, err := (localpki.Generator{}).IssueAgent(filepath.Dir(f.files.CA), name, localpki.LeafOptions{
		DNSNames: []string{"localhost"}, IPAddresses: []net.IP{net.ParseIP("127.0.0.1")},
	})
	if err != nil {
		t.Fatal(err)
	}
	certificate, err := tls.LoadX509KeyPair(paths.Certificate, paths.PrivateKey)
	if err != nil {
		t.Fatal(err)
	}
	certificate.Leaf, err = x509.ParseCertificate(certificate.Certificate[0])
	if err != nil {
		t.Fatal(err)
	}
	return certificate
}

func lifecycleExperimentHandle(t *testing.T, f *connectionExperiment, allocation string) contracts.WorkerHandle {
	t.Helper()
	handle := f.handle()
	card, err := decodeCard(handle, true)
	if err != nil {
		t.Fatal(err)
	}
	card.SupportedInterfaces[0].URL = strings.ReplaceAll(card.SupportedInterfaces[0].URL, handle.AllocationID, allocation)
	card.SupportedInterfaces[0].Tenant = allocation
	handle.AllocationID = allocation
	encoded, err := json.Marshal(card)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(encoded, &handle.AgentCard); err != nil {
		t.Fatal(err)
	}
	return handle
}

type lifecycleExperimentObservedClient struct {
	protocolClient
	afterSend func()
}

func (client *lifecycleExperimentObservedClient) SendMessage(ctx context.Context, request *sdk.SendMessageRequest) (sdk.SendMessageResult, error) {
	result, err := client.protocolClient.SendMessage(ctx, request)
	if err == nil {
		client.afterSend()
	}
	return result, err
}

func lifecycleExperimentAfterSend(invoker *Invoker, afterSend func()) {
	build := invoker.build
	invoker.build = func(ctx context.Context, card *sdk.AgentCard) (protocolClient, error) {
		client, err := build(ctx, card)
		if err != nil {
			return nil, err
		}
		return &lifecycleExperimentObservedClient{protocolClient: client, afterSend: afterSend}, nil
	}
}

func lifecycleExperimentWait(t *testing.T, signal <-chan struct{}, description string) {
	t.Helper()
	select {
	case <-signal:
	case <-time.After(3 * time.Second):
		t.Fatalf("timed out waiting for %s", description)
	}
}

func lifecycleExperimentWaitClosed(t *testing.T, f *connectionExperiment) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if f.accepts.Load() > 0 && f.closed.Load() == f.accepts.Load() {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("invocation left connections open: accepts=%d closed=%d", f.accepts.Load(), f.closed.Load())
}
