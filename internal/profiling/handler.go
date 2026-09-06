// Package profiling exposes an explicitly enabled, loopback-only subset of
// Go's diagnostic profiles. It never uses http.DefaultServeMux.
package profiling

import (
	"bytes"
	"io"
	"net/http"
	httppprof "net/http/pprof"
	"net/url"
	"strconv"
)

const maxSymbolRequestBytes = 64 * 1024

var indexDocument = []byte(`<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Contractor Go profiles</title></head>
<body><h1>Contractor Go profiles</h1><ul>
<li><a href="profile?seconds=30">profile</a></li>
<li><a href="heap">heap</a></li>
<li><a href="allocs">allocs</a></li>
<li><a href="goroutine">goroutine</a></li>
<li><a href="threadcreate">threadcreate</a></li>
<li><a href="trace?seconds=1">trace</a></li>
<li><a href="symbol">symbol</a></li>
</ul></body></html>
`)

type requestSlot chan struct{}

func newRequestSlot() requestSlot { return make(chan struct{}, 1) }

func (s requestSlot) acquire() bool {
	select {
	case s <- struct{}{}:
		return true
	default:
		return false
	}
}

func (s requestSlot) release()     { <-s }
func (s requestSlot) active() bool { return len(s) != 0 }

// Handler is an exact diagnostic allowlist with independent timed-capture and
// snapshot/delta capacity. The zero value is not usable; construct it with NewHandler.
type Handler struct {
	timed    requestSlot
	snapshot requestSlot
}

func NewHandler() *Handler {
	return &Handler{timed: newRequestSlot(), snapshot: newRequestSlot()}
}

func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	switch r.URL.Path {
	case "/debug/pprof", "/debug/pprof/":
		h.serveIndex(w, r)
	case "/debug/pprof/profile":
		h.serveTimed(w, r, 30, 60, httppprof.Profile)
	case "/debug/pprof/trace":
		h.serveTimed(w, r, 1, 10, httppprof.Trace)
	case "/debug/pprof/symbol":
		h.serveSymbol(w, r)
	case "/debug/pprof/heap":
		h.serveSnapshot(w, r, "heap", true)
	case "/debug/pprof/allocs":
		h.serveSnapshot(w, r, "allocs", false)
	case "/debug/pprof/goroutine":
		h.serveSnapshot(w, r, "goroutine", false)
	case "/debug/pprof/threadcreate":
		h.serveSnapshot(w, r, "threadcreate", false)
	default:
		http.NotFound(w, r)
	}
}

func (h *Handler) serveIndex(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodGet) || r.URL.RawQuery != "" {
		if r.Method == http.MethodGet && r.URL.RawQuery != "" {
			writeClientError(w)
		}
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(indexDocument)
}

func (h *Handler) serveTimed(
	w http.ResponseWriter,
	r *http.Request,
	defaultSeconds int,
	maximumSeconds int,
	delegate http.HandlerFunc,
) {
	if !requireMethod(w, r, http.MethodGet) {
		return
	}
	values, ok := exactQuery(r, map[string]bool{"seconds": true})
	if !ok {
		writeClientError(w)
		return
	}
	if _, ok := boundedInteger(values, "seconds", defaultSeconds, 1, maximumSeconds); !ok {
		writeClientError(w)
		return
	}
	if !h.timed.acquire() {
		writeConflict(w)
		return
	}
	defer h.timed.release()
	delegate(w, r)
}

func (h *Handler) serveSnapshot(w http.ResponseWriter, r *http.Request, name string, heap bool) {
	if !requireMethod(w, r, http.MethodGet) {
		return
	}
	allowed := map[string]bool{"seconds": true, "debug": true}
	if heap {
		allowed["gc"] = true
	}
	values, ok := exactQuery(r, allowed)
	if !ok {
		writeClientError(w)
		return
	}
	_, hasSeconds, ok := optionalBoundedInteger(values, "seconds", 1, 60)
	if !ok {
		writeClientError(w)
		return
	}
	debugMaximum := 1
	if name == "goroutine" {
		debugMaximum = 2
	}
	debug, _, ok := optionalBoundedInteger(values, "debug", 0, debugMaximum)
	if !ok || (hasSeconds && debug != 0) {
		writeClientError(w)
		return
	}
	_, hasGC, ok := optionalBoundedInteger(values, "gc", 0, 1)
	if !ok || (hasSeconds && hasGC) {
		writeClientError(w)
		return
	}
	if !h.snapshot.acquire() {
		writeConflict(w)
		return
	}
	defer h.snapshot.release()
	httppprof.Handler(name).ServeHTTP(w, r)
}

func (h *Handler) serveSymbol(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodPost {
		w.Header().Set("Allow", "GET, POST")
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if len(r.URL.RawQuery) > maxSymbolRequestBytes || (r.Method == http.MethodPost && r.URL.RawQuery != "") {
		writeClientError(w)
		return
	}
	if r.Method == http.MethodPost {
		if r.ContentLength > maxSymbolRequestBytes {
			writeTooLarge(w)
			return
		}
		body, err := io.ReadAll(io.LimitReader(r.Body, maxSymbolRequestBytes+1))
		if err != nil || len(body) > maxSymbolRequestBytes {
			writeTooLarge(w)
			return
		}
		r.Body = io.NopCloser(bytes.NewReader(body))
	}
	httppprof.Symbol(w, r)
}

func requireMethod(w http.ResponseWriter, r *http.Request, method string) bool {
	if r.Method == method {
		return true
	}
	w.Header().Set("Allow", method)
	http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	return false
}

func exactQuery(r *http.Request, allowed map[string]bool) (url.Values, bool) {
	values, err := url.ParseQuery(r.URL.RawQuery)
	if err != nil {
		return nil, false
	}
	for key, current := range values {
		if !allowed[key] || len(current) != 1 || current[0] == "" {
			return nil, false
		}
	}
	return values, true
}

func boundedInteger(
	values url.Values,
	name string,
	fallback int,
	minimum int,
	maximum int,
) (int, bool) {
	value, present, ok := optionalBoundedInteger(values, name, minimum, maximum)
	if !ok {
		return 0, false
	}
	if !present {
		return fallback, true
	}
	return value, true
}

func optionalBoundedInteger(
	values url.Values,
	name string,
	minimum int,
	maximum int,
) (value int, present bool, valid bool) {
	encoded, present := values[name]
	if !present {
		return 0, false, true
	}
	if len(encoded) != 1 || encoded[0] == "" {
		return 0, true, false
	}
	for _, character := range encoded[0] {
		if character < '0' || character > '9' {
			return 0, true, false
		}
	}
	parsed, err := strconv.Atoi(encoded[0])
	if err != nil || parsed < minimum || parsed > maximum {
		return 0, true, false
	}
	return parsed, true, true
}

func writeClientError(w http.ResponseWriter) {
	http.Error(w, "invalid profiling request", http.StatusBadRequest)
}

func writeConflict(w http.ResponseWriter) {
	http.Error(w, "profiling capacity is already in use", http.StatusConflict)
}

func writeTooLarge(w http.ResponseWriter) {
	http.Error(w, "profiling request is too large", http.StatusRequestEntityTooLarge)
}
