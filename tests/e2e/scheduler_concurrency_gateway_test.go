//go:build e2e

package e2e

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"time"
)

// schedulerConcurrencyGateway is deliberately stateless with respect to an
// individual conversation. It derives the next copy step from tool results in
// the request, which lets several real ADK Workers use it concurrently without
// sharing a model-side session counter.
type schedulerConcurrencyGateway struct {
	server *httptest.Server
	token  string

	mu              sync.Mutex
	barrier         chan struct{}
	barrierReleased bool
	initialCalls    int
	blockedCalls    int
	maximumBlocked  int
	requests        int
	failures        []string
}

func newSchedulerConcurrencyGateway(token string) *schedulerConcurrencyGateway {
	gateway := &schedulerConcurrencyGateway{token: token, barrier: make(chan struct{})}
	gateway.server = httptest.NewServer(http.HandlerFunc(gateway.serveHTTP))
	return gateway
}

func (g *schedulerConcurrencyGateway) close() { g.server.Close() }

func (g *schedulerConcurrencyGateway) URL() string { return g.server.URL + "/v1" }

func (g *schedulerConcurrencyGateway) Failures() []string {
	g.mu.Lock()
	defer g.mu.Unlock()
	return append([]string(nil), g.failures...)
}

func (g *schedulerConcurrencyGateway) snapshot() (initial, blocked, maximum int) {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.initialCalls, g.blockedCalls, g.maximumBlocked
}

func (g *schedulerConcurrencyGateway) releaseBarrier() {
	g.mu.Lock()
	defer g.mu.Unlock()
	if !g.barrierReleased {
		close(g.barrier)
		g.barrierReleased = true
	}
}

func (g *schedulerConcurrencyGateway) resetBarrier() error {
	g.mu.Lock()
	defer g.mu.Unlock()
	if !g.barrierReleased || g.blockedCalls != 0 {
		return errors.New("cannot reset an active Scheduler-concurrency Gateway barrier")
	}
	g.barrier = make(chan struct{})
	g.barrierReleased = false
	g.initialCalls = 0
	g.maximumBlocked = 0
	return nil
}

func (g *schedulerConcurrencyGateway) serveHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost || !strings.HasSuffix(r.URL.Path, "/chat/completions") {
		g.fail(w, http.StatusNotFound, "unsupported Scheduler-concurrency Gateway endpoint")
		return
	}
	if r.Header.Get("Authorization") != "Bearer "+g.token {
		g.fail(w, http.StatusUnauthorized, "invalid Scheduler-concurrency Gateway token")
		return
	}
	defer r.Body.Close()
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 4<<20))
	decoder.UseNumber()
	var request map[string]any
	if err := decoder.Decode(&request); err != nil {
		g.fail(w, http.StatusBadRequest, "invalid Scheduler-concurrency Gateway request")
		return
	}
	if trailing := decoder.Decode(&struct{}{}); !errors.Is(trailing, io.EOF) {
		g.fail(w, http.StatusBadRequest, "Scheduler-concurrency Gateway request has trailing JSON")
		return
	}

	finalizerInput, finalizer, err := decodeWorkerResultFinalizerRequest(request)
	if err != nil {
		g.fail(w, http.StatusBadRequest, err.Error())
		return
	}
	if finalizer {
		if !requestHasNoModelTools(request) || finalizerInput.ResultText != "Source artifact copied byte-for-byte" {
			g.fail(w, http.StatusBadRequest, "invalid Scheduler-concurrency Worker result finalizer")
			return
		}
		message, encodeErr := workerResultFinalizerMessage(finalizerInput)
		if encodeErr != nil {
			g.fail(w, http.StatusBadRequest, encodeErr.Error())
			return
		}
		g.respond(w, request, message, "stop")
		return
	}
	if !selectedTools(request, "read_artifact", "write_artifact") {
		g.fail(w, http.StatusBadRequest, "unexpected Scheduler-concurrency model-visible tools")
		return
	}

	if _, ok := lastExactArtifact(request, "builder", "copied"); ok {
		g.respond(w, request, map[string]any{
			"role": "assistant", "content": "Source artifact copied byte-for-byte",
		}, "stop")
		return
	}
	if data, ok := lastStringValue(request, "dataBase64"); ok && data != "" {
		g.respond(w, request, toolCallMessage(
			fmt.Sprintf("scheduler-write-%d", time.Now().UnixNano()), "write_artifact", map[string]any{
				"namespace": "builder", "name": "copied", "media_type": "text/plain",
				"data_base64": data, "expected_revision": nil,
			},
		), "tool_calls")
		return
	}

	g.mu.Lock()
	g.initialCalls++
	g.blockedCalls++
	if g.blockedCalls > g.maximumBlocked {
		g.maximumBlocked = g.blockedCalls
	}
	barrier := g.barrier
	g.mu.Unlock()
	select {
	case <-r.Context().Done():
		g.mu.Lock()
		g.blockedCalls--
		g.mu.Unlock()
		return
	case <-barrier:
	}
	g.mu.Lock()
	g.blockedCalls--
	g.mu.Unlock()
	g.respond(w, request, toolCallMessage(
		fmt.Sprintf("scheduler-read-%d", time.Now().UnixNano()), "read_artifact", map[string]any{
			"namespace": "inputs", "name": "source", "revision": nil,
		},
	), "tool_calls")
}

func (g *schedulerConcurrencyGateway) respond(
	w http.ResponseWriter,
	request map[string]any,
	message map[string]any,
	finishReason string,
) {
	model, _ := request["model"].(string)
	g.mu.Lock()
	g.requests++
	requestNumber := g.requests
	g.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"id":      fmt.Sprintf("chatcmpl-scheduler-concurrency-%d", requestNumber),
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   model,
		"choices": []any{map[string]any{
			"index": 0, "message": message, "finish_reason": finishReason,
		}},
		"usage": map[string]any{
			"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10,
		},
	})
}

func (g *schedulerConcurrencyGateway) fail(w http.ResponseWriter, status int, message string) {
	g.mu.Lock()
	g.failures = append(g.failures, message)
	g.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{"message": message, "type": "invalid_request_error"},
	})
}
