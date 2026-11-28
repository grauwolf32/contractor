package controlplane

import (
	"context"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	pb "github.com/grauwolf32/contractor/api/v1alpha1/runtimepb"
	"github.com/grauwolf32/contractor/internal/mtls"
	"github.com/grauwolf32/contractor/internal/runtimewire"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
)

type controlDeadlineWriter struct {
	http.ResponseWriter
	readDeadlines chan *http.ResponseController
	once          sync.Once
}

func (w *controlDeadlineWriter) Unwrap() http.ResponseWriter { return w.ResponseWriter }
func (w *controlDeadlineWriter) Flush()                      { _ = http.NewResponseController(w.ResponseWriter).Flush() }
func (w *controlDeadlineWriter) SetReadDeadline(deadline time.Time) error {
	controller := http.NewResponseController(w.ResponseWriter)
	// Keep expiry under the test's control even on a loaded machine. The
	// production handler must still request its nonzero read budget first.
	effective := deadline
	if !deadline.IsZero() {
		effective = time.Now().Add(time.Hour)
	}
	if err := controller.SetReadDeadline(effective); err != nil {
		return err
	}
	if !deadline.IsZero() {
		w.once.Do(func() { w.readDeadlines <- controller })
	}
	return nil
}

func TestControlRPCStalledHTTP2BodiesReleaseAdmission(t *testing.T) {
	trusted := generatePKI(t, "control-deadlines")
	handler, err := NewGRPCHandler(newTestRegistry(t, newTestClock()), testPrincipalRegistrar{})
	if err != nil {
		t.Fatal(err)
	}
	deadlines := make(chan *http.ResponseController, runtimewire.MaxConcurrentRPCs)
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		handler.ServeHTTP(&controlDeadlineWriter{ResponseWriter: w, readDeadlines: deadlines}, r)
	}))
	server.EnableHTTP2 = true
	server.TLS, err = mtls.ControlPlaneServerConfig(trusted.controlPlane)
	if err != nil {
		t.Fatal(err)
	}
	server.Config.ErrorLog = log.New(io.Discard, "", 0)
	server.StartTLS()
	defer server.Close()
	clientTLS, err := mtls.ControlPlaneEndpointClientConfig(trusted.agent)
	if err != nil {
		t.Fatal(err)
	}
	transport := &http.Transport{TLSClientConfig: clientTLS, ForceAttemptHTTP2: true}
	defer transport.CloseIdleConnections()
	client := &http.Client{Transport: transport}
	var calls sync.WaitGroup
	var writers []*io.PipeWriter
	returned := make(chan struct{}, runtimewire.MaxConcurrentRPCs)
	ctx, cancel := context.WithCancel(t.Context())
	defer func() {
		cancel()
		for _, w := range writers {
			_ = w.Close()
		}
		calls.Wait()
	}()
	for range runtimewire.MaxConcurrentRPCs {
		reader, writer := io.Pipe()
		writers = append(writers, writer)
		request, err := http.NewRequestWithContext(ctx, http.MethodPost, server.URL+pb.ControlPlane_Heartbeat_FullMethodName, reader)
		if err != nil {
			t.Fatal(err)
		}
		request.Header.Set("Content-Type", "application/grpc")
		request.Header.Set("Te", "trailers")
		// No grpc-timeout: the Server owns the body-read bound.
		calls.Go(func() {
			response, err := client.Do(request)
			if err == nil {
				_, _ = io.Copy(io.Discard, response.Body)
				_ = response.Body.Close()
			}
			returned <- struct{}{}
		})
	}
	controllers := make([]*http.ResponseController, 0, runtimewire.MaxConcurrentRPCs)
	for range runtimewire.MaxConcurrentRPCs {
		select {
		case c := <-deadlines:
			controllers = append(controllers, c)
		case <-returned:
			t.Fatal("stalled RPC returned before controlled expiry")
		case <-ctx.Done():
			t.Fatal("body admission cancelled")
		}
	}
	// Expire each real HTTP/2 read deadline explicitly, without elapsed-time assertions.
	for _, c := range controllers {
		if err := c.SetReadDeadline(time.Unix(1, 0)); err != nil {
			t.Fatal(err)
		}
	}
	for range runtimewire.MaxConcurrentRPCs {
		select {
		case <-returned:
		case <-ctx.Done():
			t.Fatal("deadline did not release stream")
		}
	}
	conn, err := grpc.NewClient(strings.TrimPrefix(server.URL, "https://"), grpc.WithTransportCredentials(credentials.NewTLS(clientTLS)), grpc.WithDisableRetry())
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	// All permits must be reusable by a healthy native RPC after stalled reads.
	if _, err := pb.NewControlPlaneClient(conn).Bootstrap(ctx, &pb.BootstrapRequest{}); err != nil {
		t.Fatal(err)
	}
}

type controlBudgetWriter struct {
	*httptest.ResponseRecorder
	read, write time.Time
}

func (w *controlBudgetWriter) SetReadDeadline(deadline time.Time) error {
	w.read = deadline
	return nil
}
func (w *controlBudgetWriter) SetWriteDeadline(deadline time.Time) error {
	w.write = deadline
	return nil
}

func TestControlRPCIOBudgetsAreSeparateFromDomainWork(t *testing.T) {
	for _, path := range []string{pb.ControlPlane_Bootstrap_FullMethodName, pb.ControlPlane_Register_FullMethodName, pb.ControlPlane_Heartbeat_FullMethodName} {
		t.Run(path, func(t *testing.T) {
			w := &controlBudgetWriter{ResponseRecorder: httptest.NewRecorder()}
			h := &boundedControlRPC{permits: make(chan struct{}, 1)}
			h.next = http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
				if w.read.IsZero() || w.write.IsZero() {
					t.Fatal("missing pre-decode I/O budget")
				}
				_, err := controlRPCDecoded(r.Context(), nil, nil, func(ctx context.Context, _ any) (any, error) {
					if !w.read.IsZero() || !w.write.IsZero() {
						t.Fatal("socket deadline leaked into domain operation")
					}
					if _, ok := ctx.Deadline(); ok {
						t.Fatal("body budget became a domain timeout")
					}
					return &pb.HeartbeatResponse{}, nil
				})
				if err != nil {
					t.Fatal(err)
				}
				if !w.read.IsZero() || w.write.IsZero() {
					t.Fatal("response did not receive a fresh independent budget")
				}
			})
			h.ServeHTTP(w, httptest.NewRequest(http.MethodPost, path, nil))
			if len(h.permits) != 0 {
				t.Fatal("permit leaked")
			}
		})
	}
}
