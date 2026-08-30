// Package requestid supplies bounded correlation IDs at HTTP trust boundaries.
package requestid

import (
	"bufio"
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"regexp"
	"sync/atomic"
	"time"
)

const Header = "X-Request-ID"

var (
	validPattern    = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$`)
	fallbackCounter atomic.Uint64
)

type contextKey struct{}

type Options struct {
	Generator     func() (string, error)
	Logger        *slog.Logger
	Boundary      string
	TrustIncoming bool
}

// Middleware assigns one safe correlation ID before authentication, exposes it
// in the response, and records it for server-side 5xx diagnostics. An incoming
// value is honored only on explicitly trusted private boundaries.
func Middleware(next http.Handler, options Options) http.Handler {
	if options.Generator == nil {
		options.Generator = New
	}
	if options.Logger == nil {
		options.Logger = slog.New(slog.NewTextHandler(io.Discard, nil))
	}
	if options.Boundary == "" {
		options.Boundary = "http"
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		identifier := ""
		if options.TrustIncoming {
			values := r.Header.Values(Header)
			if len(values) == 1 && Valid(values[0]) {
				identifier = values[0]
			}
		}
		if identifier == "" {
			generated, err := options.Generator()
			if err == nil && Valid(generated) {
				identifier = generated
			} else {
				identifier = fallback()
			}
		}
		w.Header().Set(Header, identifier)
		tracked := &statusWriter{ResponseWriter: w}
		next.ServeHTTP(tracked, r.WithContext(With(r.Context(), identifier)))
		if tracked.status >= http.StatusInternalServerError {
			options.Logger.Error(
				"HTTP request failed",
				"boundary", options.Boundary,
				"request_id", identifier,
				"method", r.Method,
				"status", tracked.status,
			)
		}
	})
}

func New() (string, error) {
	buffer := make([]byte, 16)
	if _, err := rand.Read(buffer); err != nil {
		return "", fmt.Errorf("generate request ID: %w", err)
	}
	return "request_" + hex.EncodeToString(buffer), nil
}

func Valid(value string) bool { return validPattern.MatchString(value) }

func With(ctx context.Context, identifier string) context.Context {
	return context.WithValue(ctx, contextKey{}, identifier)
}

func From(ctx context.Context) string {
	value, _ := ctx.Value(contextKey{}).(string)
	return value
}

// Ensure returns the existing bounded ID or creates a safe ID for an outgoing
// request that did not originate at an HTTP boundary.
func Ensure(ctx context.Context) string {
	if identifier := From(ctx); Valid(identifier) {
		return identifier
	}
	identifier, err := New()
	if err != nil || !Valid(identifier) {
		return fallback()
	}
	return identifier
}

func FromResponse(w http.ResponseWriter) string { return w.Header().Get(Header) }

func fallback() string {
	return fmt.Sprintf("request_fallback_%x_%x", time.Now().UnixNano(), fallbackCounter.Add(1))
}

type statusWriter struct {
	http.ResponseWriter
	status int
}

func (w *statusWriter) WriteHeader(status int) {
	if w.status != 0 {
		return
	}
	w.status = status
	w.ResponseWriter.WriteHeader(status)
}

func (w *statusWriter) Write(data []byte) (int, error) {
	if w.status == 0 {
		w.WriteHeader(http.StatusOK)
	}
	return w.ResponseWriter.Write(data)
}

func (w *statusWriter) Unwrap() http.ResponseWriter { return w.ResponseWriter }

// Hijack preserves WebSocket and other explicit upgrade support through the
// request-correlation wrapper. ResponseController follows nested Unwrap
// implementations without assuming the concrete Server writer type.
func (w *statusWriter) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	connection, buffered, err := http.NewResponseController(w.ResponseWriter).Hijack()
	if err == nil && w.status == 0 {
		w.status = http.StatusSwitchingProtocols
	}
	return connection, buffered, err
}
