package artifacttransfer

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
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
		_, transfer, err := Acquire(recorders[index], request, 1)
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
	if _, _, err := Acquire(rejected, request, 1); !errors.Is(err, artifacts.ErrTransferCapacity) {
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
	_, replacement, err := Acquire(rejected, request, 1)
	if err != nil {
		t.Fatalf("slot was not released before the response write: %v", err)
	}
	defer replacement.Close()
	// Repeated release and Close must not free a slot another request now holds.
	transfers[0].ReleaseBeforeWrite(0)
	transfers[0].Close()
	if _, _, err := Acquire(&deadlineRecorder{ResponseRecorder: httptest.NewRecorder()}, request, 1); !errors.Is(err, artifacts.ErrTransferCapacity) {
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
	_, transfer, err := Acquire(recorder, request, artifacts.MaxPayloadSize)
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
