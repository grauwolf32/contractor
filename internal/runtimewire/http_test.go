package runtimewire

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"testing"
)

// The next five-byte header may reveal EOF/cardinality, but its payload cannot
// enter grpc-go's eager queue before application receive credit is returned.
func TestHTTPReceiveCreditBoundsReadAheadAndCloseJoinsBlockedRead(t *testing.T) {
	first := framedTestMessage([]byte("first"))
	second := framedTestMessage(bytes.Repeat([]byte("x"), BulkChunkBytes))
	source := &headerLookaheadSource{Reader: bytes.NewReader(append(first, second...)), boundary: len(first) + 5, reached: make(chan struct{})}
	body := newCreditedRPCBody(io.NopCloser(source), BulkFrameBytes, true)
	actual := make([]byte, len(first))
	if _, err := io.ReadFull(body, actual); err != nil || !bytes.Equal(actual, first) {
		t.Fatalf("first message=%q %v", actual, err)
	}
	type outcome struct {
		n   int
		err error
	}
	done := make(chan outcome, 1)
	go func() { n, err := body.Read(make([]byte, BulkChunkBytes)); done <- outcome{n, err} }()
	select {
	case <-source.reached:
	case <-t.Context().Done():
		t.Fatal(t.Context().Err())
	}
	if err := body.Close(); err != nil {
		t.Fatal(err)
	}
	result := <-done
	if result.n != 0 || !errors.Is(result.err, io.ErrClosedPipe) {
		t.Fatalf("uncredited message was drained: %+v", result)
	}
	if source.read != len(first)+5 {
		t.Fatalf("read ahead %d bytes, want one message plus fixed header", source.read)
	}
}

func TestHTTPReceiveCreditRejectsOversizeWithoutReadingPayload(t *testing.T) {
	first := framedTestMessage(bytes.Repeat([]byte("x"), BulkFrameBytes+1))
	source := bytes.NewReader(first)
	body := newCreditedRPCBody(io.NopCloser(source), BulkFrameBytes, true)
	defer body.Close()
	header := make([]byte, 5)
	if _, err := io.ReadFull(body, header); err != nil {
		t.Fatal(err)
	}
	if binary.BigEndian.Uint32(header[1:]) != BulkFrameBytes+1 || source.Len() != len(first)-5 {
		t.Fatal("oversized payload was drained before grpc-go's message-limit rejection")
	}
}

type headerLookaheadSource struct {
	*bytes.Reader
	boundary, read int
	reached        chan struct{}
}

func (s *headerLookaheadSource) Read(p []byte) (int, error) {
	n, err := s.Reader.Read(p)
	s.read += n
	if s.read == s.boundary {
		close(s.reached)
	}
	return n, err
}
func framedTestMessage(payload []byte) []byte {
	result := make([]byte, len(payload)+5)
	binary.BigEndian.PutUint32(result[1:5], uint32(len(payload)))
	copy(result[5:], payload)
	return result
}
