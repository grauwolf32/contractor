package runtimewire

import (
	"context"
	"encoding/binary"
	"io"
	"net/http"
	"sync"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
)

// grpc-go's HTTP handler drains Request.Body into its receive queue and can
// return on cancellation before the application handler joins. Keep both
// receive credit and application lifetime at our HTTP ownership boundary.
// A reader may hold one bounded message ahead of the application's RecvMsg.
type httpRPCScope struct {
	mu      sync.Mutex
	closing bool
	calls   sync.WaitGroup
	body    *creditedRPCBody
}
type httpRPCScopeKey struct{}

// ServeHTTP bounds read-ahead and joins admitted RPC work before the listener
// can release its admission, HTTP/2 writer or request-owned resources.
func ServeHTTP(w http.ResponseWriter, r *http.Request, next http.Handler, maximum int, clientStreaming bool) {
	scope := &httpRPCScope{body: newCreditedRPCBody(r.Body, maximum, clientStreaming)}
	defer func() {
		scope.mu.Lock()
		scope.closing = true
		scope.mu.Unlock()
		_ = scope.body.Close()
		scope.calls.Wait()
	}()
	ctx := context.WithValue(r.Context(), httpRPCScopeKey{}, scope)
	request := r.WithContext(ctx)
	request.Body = scope.body
	next.ServeHTTP(w, request)
}
func enterHTTPRPC(ctx context.Context) (*httpRPCScope, bool) {
	scope, ok := ctx.Value(httpRPCScopeKey{}).(*httpRPCScope)
	if !ok {
		return nil, false
	}
	scope.mu.Lock()
	defer scope.mu.Unlock()
	if scope.closing {
		return nil, false
	}
	scope.calls.Add(1)
	return scope, true
}

// UnaryHTTP joins application work, including the existing domain interceptor.
func UnaryHTTP(next grpc.UnaryServerInterceptor) grpc.UnaryServerInterceptor {
	return func(ctx context.Context, request any, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (any, error) {
		scope, ok := enterHTTPRPC(ctx)
		if !ok {
			return nil, Error(ctx, codes.Canceled, "cancelled", false)
		}
		defer scope.calls.Done()
		if next != nil {
			return next(ctx, request, info, handler)
		}
		return handler(ctx, request)
	}
}

// StreamHTTP releases receive credit only after the application consumed a
// bounded message, and joins that application before the HTTP owner returns.
func StreamHTTP(service any, stream grpc.ServerStream, info *grpc.StreamServerInfo, next grpc.StreamHandler) error {
	scope, ok := enterHTTPRPC(stream.Context())
	if !ok {
		return Error(stream.Context(), codes.Canceled, "cancelled", false)
	}
	defer scope.calls.Done()
	return next(service, &creditedServerStream{ServerStream: stream, body: scope.body})
}

type creditedServerStream struct {
	grpc.ServerStream
	body *creditedRPCBody
}

func (s *creditedServerStream) RecvMsg(value any) error {
	err := s.ServerStream.RecvMsg(value)
	if err == nil {
		s.body.advance()
	}
	return err
}

type creditedRPCBody struct {
	source       io.ReadCloser
	maximum      uint32
	credit       chan struct{}
	closed       chan struct{}
	once         sync.Once
	header       [5]byte
	headerOffset int
	remaining    uint32
	started      bool
}

func newCreditedRPCBody(source io.ReadCloser, maximum int, clientStreaming bool) *creditedRPCBody {
	credits := 1
	// Non-client-streaming RecvMsg consumes a possible second message itself
	// to reject cardinality before the application interceptor is called.
	if !clientStreaming {
		credits = 2
	}
	body := &creditedRPCBody{source: source, maximum: uint32(maximum), credit: make(chan struct{}, credits), closed: make(chan struct{})}
	for range credits {
		body.credit <- struct{}{}
	}
	return body
}
func (b *creditedRPCBody) advance() {
	select {
	case b.credit <- struct{}{}:
	case <-b.closed:
	}
}
func (b *creditedRPCBody) Read(p []byte) (int, error) {
	if len(p) == 0 {
		return 0, nil
	}
	if !b.started {
		// A bounded five-byte lookahead must be allowed before credit: unary
		// RecvMsg also reads EOF to check cardinality before its interceptor.
		if _, err := io.ReadFull(b.source, b.header[:]); err != nil {
			return 0, err
		}
		select {
		case <-b.credit:
		case <-b.closed:
			return 0, io.ErrClosedPipe
		}
		b.started = true
		b.headerOffset = 0
		b.remaining = binary.BigEndian.Uint32(b.header[1:])
	}
	if b.headerOffset < 5 {
		n := copy(p, b.header[b.headerOffset:])
		b.headerOffset += n
		// Expose an oversized length to grpc-go's own message-limit rejection,
		// without draining or allocating the oversized payload behind it.
		if b.headerOffset == 5 && (b.remaining == 0 || b.remaining > b.maximum) {
			b.started = false
			b.remaining = 0
		}
		return n, nil
	}
	if uint32(len(p)) > b.remaining {
		p = p[:int(b.remaining)]
	}
	n, err := b.source.Read(p)
	b.remaining -= uint32(n)
	if b.remaining == 0 {
		b.started = false
	}
	return n, err
}
func (b *creditedRPCBody) Close() error {
	var err error
	b.once.Do(func() { close(b.closed); err = b.source.Close() })
	return err
}
