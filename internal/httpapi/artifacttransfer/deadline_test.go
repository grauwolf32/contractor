package artifacttransfer

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"runtime"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
)

func TestDurationAllowsMaximumPayloadAtMinimumThroughput(t *testing.T) {
	if got := Duration(artifacts.MaxPayloadSize); got != 69*time.Second {
		t.Fatalf("64 MiB transfer deadline = %s, want 69s", got)
	}
	if got := Duration(-1); got != Duration(artifacts.MaxPayloadSize) {
		t.Fatalf("unknown-length transfer deadline = %s", got)
	}
	if got := Duration(1); got != 6*time.Second {
		t.Fatalf("small transfer deadline = %s, want 6s", got)
	}
}

// deadlineRecorder exposes the socket deadline hooks http.ResponseController
// looks for, so tests can observe which deadlines a handler arms.
type deadlineRecorder struct {
	*httptest.ResponseRecorder
	read, write []time.Time
}

func (d *deadlineRecorder) SetReadDeadline(deadline time.Time) error {
	d.read = append(d.read, deadline)
	return nil
}

func (d *deadlineRecorder) SetWriteDeadline(deadline time.Time) error {
	d.write = append(d.write, deadline)
	return nil
}

func TestReleaseBeforeWriteFreesTheSlotAndBoundsTheResponse(t *testing.T) {
	ctx := artifacts.WithBlobRuntime(context.Background(), artifacts.NewBlobRuntime(nil, nil))
	request := httptest.NewRequestWithContext(ctx, http.MethodPut, "/artifact", nil)
	recorders := make([]*deadlineRecorder, 4)
	transfers := make([]*Transfer, 4)
	for index := range transfers {
		recorders[index] = &deadlineRecorder{ResponseRecorder: httptest.NewRecorder()}
		_, transfer, err := Acquire(recorders[index], request)
		if err != nil {
			t.Fatal(err)
		}
		transfers[index] = transfer
	}
	defer func() {
		for _, transfer := range transfers {
			transfer.Close()
		}
	}()
	rejected := &deadlineRecorder{ResponseRecorder: httptest.NewRecorder()}
	if _, _, err := Acquire(rejected, request); !errors.Is(err, artifacts.ErrTransferCapacity) {
		t.Fatalf("fifth transfer = %v, want capacity error", err)
	}
	if len(rejected.read)+len(rejected.write) != 0 {
		t.Fatal("a rejected transfer armed socket deadlines")
	}

	beforeWrite := time.Now()
	transfers[0].ReleaseBeforeWrite(3 << 20)
	written := recorders[0].write[len(recorders[0].write)-1]
	if written.Before(beforeWrite.Add(Duration(3<<20))) || written.After(time.Now().Add(Duration(3<<20))) {
		t.Fatalf("response deadline = %s after the write began, want %s", written.Sub(beforeWrite), Duration(3<<20))
	}
	_, replacement, err := Acquire(rejected, request)
	if err != nil {
		t.Fatalf("slot was not released before the response write: %v", err)
	}
	defer replacement.Close()
	// Repeated release and Close must not free a slot another request now holds.
	transfers[0].ReleaseBeforeWrite(0)
	transfers[0].Close()
	if _, _, err := Acquire(&deadlineRecorder{ResponseRecorder: httptest.NewRecorder()}, request); !errors.Is(err, artifacts.ErrTransferCapacity) {
		t.Fatalf("transfer after repeated release = %v, want capacity error", err)
	}
	for _, deadline := range append(recorders[0].read, recorders[0].write...) {
		if deadline.IsZero() {
			t.Fatal("Close cleared a socket deadline before net/http flushed the response")
		}
	}
}

func TestBoundWriteMeasuresTheResponseFromTheWrite(t *testing.T) {
	ctx := artifacts.WithBlobRuntime(context.Background(), artifacts.NewBlobRuntime(nil, nil))
	recorder := &deadlineRecorder{ResponseRecorder: httptest.NewRecorder()}
	request := httptest.NewRequestWithContext(ctx, http.MethodGet, "/artifact", nil)
	_, transfer, err := Acquire(recorder, request)
	if err != nil {
		t.Fatal(err)
	}
	defer transfer.Close()
	time.Sleep(20 * time.Millisecond)
	beforeWrite := time.Now()
	transfer.BoundWrite(1)
	written := recorder.write[len(recorder.write)-1]
	if written.Before(beforeWrite.Add(Duration(1))) || written.After(time.Now().Add(Duration(1))) {
		t.Fatalf("payload write deadline = %s after the write began, want %s", written.Sub(beforeWrite), Duration(1))
	}

	small := &deadlineRecorder{ResponseRecorder: httptest.NewRecorder()}
	beforeWrite = time.Now()
	BoundWrite(small, 0)
	if len(small.write) != 1 || small.write[0].Before(beforeWrite.Add(TransferGrace)) ||
		small.write[0].After(time.Now().Add(TransferGrace)) {
		t.Fatalf("small response deadline = %v, want %s from the write", small.write, TransferGrace)
	}
}

func TestAcquireLeavesStorageWorkWithoutAClientDeadline(t *testing.T) {
	ctx := artifacts.WithBlobRuntime(context.Background(), artifacts.NewBlobRuntime(nil, nil))
	recorder := &deadlineRecorder{ResponseRecorder: httptest.NewRecorder()}
	admitted, transfer, err := Acquire(recorder, httptest.NewRequestWithContext(ctx, http.MethodPut, "/artifact", nil))
	if err != nil {
		t.Fatal(err)
	}
	defer transfer.Close()
	if deadline, ok := admitted.Context().Deadline(); ok {
		t.Fatalf("admitted request carries a client deadline %s", time.Until(deadline))
	}
	if len(recorder.read)+len(recorder.write) != 0 {
		t.Fatal("Acquire armed socket deadlines before any client I/O")
	}
	started := time.Now()
	storage, cancel := StorageContext(admitted.Context())
	defer cancel()
	deadline, ok := storage.Deadline()
	if !ok || deadline.Before(started.Add(StorageBudget)) || deadline.After(time.Now().Add(StorageBudget)) {
		t.Fatalf("storage deadline = %v (%t), want %s from the start of storage work", deadline, ok, StorageBudget)
	}
	if _, nested, err := artifacts.AcquireTransfer(storage); err != nil {
		t.Fatalf("storage work does not share the transfer lease: %v", err)
	} else {
		nested()
	}
}

// onlyReader hides the concrete reader type, so httptest reports an unknown
// request length.
type onlyReader struct{ io.Reader }

type failingReader struct{}

func (failingReader) Read([]byte) (int, error) { return 0, errors.New("body must not be read") }

func TestReadBodyAllocatesAtMostTheDeclaredBound(t *testing.T) {
	ctx := artifacts.WithBlobRuntime(context.Background(), artifacts.NewBlobRuntime(nil, nil))
	read := func(body io.Reader, declared, limit int64) ([]byte, uint64, *deadlineRecorder, error) {
		t.Helper()
		request := httptest.NewRequestWithContext(ctx, http.MethodPut, "/artifact", body)
		request.ContentLength = declared
		recorder := &deadlineRecorder{ResponseRecorder: httptest.NewRecorder()}
		admitted, transfer, err := Acquire(recorder, request)
		if err != nil {
			t.Fatal(err)
		}
		defer transfer.Close()
		var before, after runtime.MemStats
		runtime.ReadMemStats(&before)
		data, err := transfer.ReadBody(admitted, limit)
		runtime.ReadMemStats(&after)
		return data, after.TotalAlloc - before.TotalAlloc, recorder, err
	}

	const size = 4 << 20
	payload := bytes.Repeat([]byte("x"), size)
	data, allocated, recorder, err := read(bytes.NewReader(payload), size, artifacts.MaxPayloadSize)
	if err != nil || !bytes.Equal(data, payload) {
		t.Fatalf("declared body = %d bytes (%v)", len(data), err)
	}
	if cap(data) != size || allocated > size+size/8 {
		t.Fatalf("declared %d-byte body allocated %d bytes, capacity %d", size, allocated, cap(data))
	}
	if len(recorder.read) != 2 || recorder.read[0].IsZero() || !recorder.read[1].IsZero() {
		t.Fatalf("read deadlines = %v, want armed for the body and cleared after it", recorder.read)
	}
	if len(recorder.write) != 1 || !recorder.write[0].Equal(recorder.read[0]) {
		t.Fatalf("write deadlines = %v, want the body deadline for an interim response", recorder.write)
	}

	const limit = 1 << 20
	exact := bytes.Repeat([]byte("y"), limit)
	data, allocated, _, err = read(onlyReader{bytes.NewReader(exact)}, -1, limit)
	if err != nil || !bytes.Equal(data, exact) {
		t.Fatalf("unknown-length body at the limit = %d bytes (%v)", len(data), err)
	}
	if cap(data) != limit || allocated > 2*limit+limit/8 {
		t.Fatalf("unknown-length body allocated %d bytes, capacity %d, limit %d", allocated, cap(data), limit)
	}
	if _, _, recorder, err = read(onlyReader{bytes.NewReader(append(exact, 'z'))}, -1, limit); !errors.Is(err, artifacts.ErrPayloadTooLarge) {
		t.Fatalf("unknown-length body above the limit = %v", err)
	}
	if len(recorder.read) != 1 || recorder.read[0].IsZero() {
		t.Fatalf("failed read cleared its deadline: %v", recorder.read)
	}
	if _, _, recorder, err = read(failingReader{}, limit+1, limit); !errors.Is(err, artifacts.ErrPayloadTooLarge) {
		t.Fatalf("declared body above the limit = %v", err)
	}
	if len(recorder.read)+len(recorder.write) != 0 {
		t.Fatal("an oversized declared body armed socket deadlines")
	}
	if _, _, _, err = read(bytes.NewReader([]byte("short")), 10, limit); !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("truncated declared body = %v", err)
	}
}
