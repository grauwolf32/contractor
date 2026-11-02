package privateartifacts

import (
	"bufio"
	"bytes"
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/findingintake"
)

// The production HTTP server, not a recorder, must apply deadlines before
// touching the body and release admission before writing either answer.
func TestFindingTransferBoundsSocketAndStorageAndReleasesBeforeResponse(t *testing.T) {
	runtime := blob.NewRuntime(nil, nil)
	base := &fakeFindingIntake{}
	var reads, writes, storage atomic.Int32
	intake := &findingBudgetIntake{FindingIntake: base, check: func(ctx context.Context) {
		if _, ok := ctx.Deadline(); !ok {
			t.Error("finding storage has no independent deadline")
		}
		if reads.Load() == 0 {
			t.Error("finding intake reached without a read deadline")
		}
		storage.Add(1)
	}}
	inner, err := NewHandler(Dependencies{Registry: &fakeRegistry{grant: testGrant("run-a")}, Artifacts: artifacts.NewService(newMemoryRepository()), Findings: intake})
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		certificate := &x509.Certificate{RawSubjectPublicKeyInfo: []byte("artifact-test-key")}
		r.TLS = &tls.ConnectionState{PeerCertificates: []*x509.Certificate{certificate}, VerifiedChains: [][]*x509.Certificate{{certificate}}}
		r.Header.Set(RuntimeInstanceHeader, "runtime-1")
		r = r.WithContext(blob.WithRuntime(r.Context(), runtime))
		inner.ServeHTTP(&findingBudgetWriter{ResponseWriter: w, runtime: runtime, t: t, reads: &reads, writes: &writes}, r)
	}))
	t.Cleanup(server.Close)
	// Use the existing accepted submission fixture; replay and fencing retain
	// their existing service authority and require no different wire shape.
	postFinding(t, http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		body, readErr := io.ReadAll(r.Body)
		if readErr != nil {
			t.Error(readErr)
			return
		}
		for _, want := range []int{http.StatusCreated, http.StatusOK} {
			response, callErr := http.Post(server.URL+r.URL.Path, "application/json", bytes.NewReader(body))
			if callErr != nil {
				t.Error(callErr)
				return
			}
			data, _ := io.ReadAll(response.Body)
			response.Body.Close()
			if response.StatusCode != want {
				t.Errorf("finding = %d %s, want %d", response.StatusCode, data, want)
			}
		}
	}), "deadline")
	before := storage.Load()
	conn, err := net.Dial("tcp", server.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	if _, err = fmt.Fprintf(conn, "POST /private/v1/allocations/allocation-1/finding-proposals HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 20\r\nConnection: close\r\n\r\n{"); err != nil {
		t.Fatal(err)
	}
	if err := conn.(*net.TCPConn).CloseWrite(); err != nil {
		t.Fatal(err)
	}
	response, err := http.ReadResponse(bufio.NewReader(conn), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	var envelope errorResponse
	if err := json.NewDecoder(response.Body).Decode(&envelope); err != nil {
		t.Fatal(err)
	}
	if response.StatusCode != 400 || envelope.Code != "invalid_finding_request" || envelope.Retryable {
		t.Fatalf("interrupted finding = %d %+v", response.StatusCode, envelope)
	}
	if storage.Load() != before {
		t.Fatal("interrupted JSON reached finding intake")
	}
	if reads.Load() < 3 || writes.Load() < 3 {
		t.Fatalf("socket deadlines: reads=%d writes=%d", reads.Load(), writes.Load())
	}
}

type findingBudgetWriter struct {
	http.ResponseWriter
	runtime       *blob.Runtime
	t             *testing.T
	reads, writes *atomic.Int32
}

func (w *findingBudgetWriter) Unwrap() http.ResponseWriter { return w.ResponseWriter }
func (w *findingBudgetWriter) SetReadDeadline(deadline time.Time) error {
	if !deadline.IsZero() {
		w.reads.Add(1)
	}
	return http.NewResponseController(w.ResponseWriter).SetReadDeadline(deadline)
}
func (w *findingBudgetWriter) SetWriteDeadline(deadline time.Time) error {
	if !deadline.IsZero() {
		w.writes.Add(1)
	}
	return http.NewResponseController(w.ResponseWriter).SetWriteDeadline(deadline)
}
func (w *findingBudgetWriter) WriteHeader(status int) {
	ctx := blob.WithRuntime(context.Background(), w.runtime)
	releases := []func(){}
	for range 4 {
		_, release, err := blob.AcquireTransfer(ctx)
		if err != nil {
			w.t.Errorf("finding response retains transfer admission: %v", err)
			break
		}
		releases = append(releases, release)
	}
	for _, release := range releases {
		release()
	}
	w.ResponseWriter.WriteHeader(status)
}

type findingBudgetIntake struct {
	FindingIntake
	check func(context.Context)
}

func (i *findingBudgetIntake) FindReplay(ctx context.Context, g controlplane.AllocationGrant, s findingintake.Submission) (findingintake.Receipt, bool, error) {
	i.check(ctx)
	return i.FindingIntake.FindReplay(ctx, g, s)
}

// Trigger timeout from the body itself after observing the armed deadline,
// without waiting for elapsed wall time or starting storage work.
func TestFindingTransferTimeoutReleasesAllSlots(t *testing.T) {
	runtime := blob.NewRuntime(nil, nil)
	h := handler{dependencies: Dependencies{Registry: &fakeRegistry{grant: testGrant("run-a")}, Artifacts: artifacts.NewService(newMemoryRepository()), Findings: &fakeFindingIntake{}}}
	for range 4 {
		writer := &findingTimeoutWriter{ResponseRecorder: httptest.NewRecorder()}
		r := httptest.NewRequest(http.MethodPost, "/", nil)
		r.Header.Set("Content-Type", "application/json")
		r.SetPathValue("allocationID", "allocation-1")
		r.Body = io.NopCloser(findingTimeoutReader{t: t, writer: writer})
		r.ContentLength = -1
		ctx := context.WithValue(blob.WithRuntime(t.Context(), runtime), authenticatedRuntimeContextKey{}, authenticatedRuntime{principalID: testGrant("run-a").RuntimeAgentID, instanceID: "runtime-1"})
		h.postFindingProposal(writer, r.WithContext(ctx))
		if writer.Code != 400 || !strings.Contains(writer.Body.String(), "invalid_finding_request") {
			t.Fatalf("timeout response = %d %s", writer.Code, writer.Body.String())
		}
	}
	ctx := blob.WithRuntime(t.Context(), runtime)
	releases := []func(){}
	defer func() {
		for _, release := range releases {
			release()
		}
	}()
	for range 4 {
		_, release, err := blob.AcquireTransfer(ctx)
		if err != nil {
			t.Fatal(err)
		}
		releases = append(releases, release)
	}
	_, _, err := blob.AcquireTransfer(ctx)
	if !errors.Is(err, blob.ErrTransferCapacity) {
		t.Fatalf("capacity after four acquisitions: %v", err)
	}
}

type findingTimeoutWriter struct {
	*httptest.ResponseRecorder
	armed bool
}

func (w *findingTimeoutWriter) SetReadDeadline(d time.Time) error { w.armed = !d.IsZero(); return nil }
func (*findingTimeoutWriter) SetWriteDeadline(time.Time) error    { return nil }

type findingTimeoutReader struct {
	t      *testing.T
	writer *findingTimeoutWriter
}

func (r findingTimeoutReader) Read([]byte) (int, error) {
	if !r.writer.armed {
		r.t.Error("finding body read without socket deadline")
	}
	return 0, context.DeadlineExceeded
}
