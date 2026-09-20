//go:build e2e

package e2e

import (
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"
)

const mcpReleaseSecret = "V61_LOCAL_MCP_SECRET_CANARY"

type mcpReleaseFixture struct {
	server      *httptest.Server
	mu          sync.Mutex
	sessions    map[string]bool
	calls       map[string][]string
	modes       map[string]string
	deleted     map[string]bool
	connections map[net.Conn]http.ConnState
	firstCall   chan struct{}
	unblock     chan struct{}
	once        sync.Once
	release     sync.Once
}

func newMCPReleaseFixture(t *testing.T, block bool) *mcpReleaseFixture {
	t.Helper()
	f := &mcpReleaseFixture{sessions: map[string]bool{}, calls: map[string][]string{}, modes: map[string]string{}, deleted: map[string]bool{}, connections: map[net.Conn]http.ConnState{}, firstCall: make(chan struct{}), unblock: make(chan struct{})}
	if !block {
		f.release.Do(func() { close(f.unblock) })
	}
	f.server = httptest.NewUnstartedServer(http.HandlerFunc(f.serve))
	f.server.Listener.Close()
	listener, err := net.Listen("tcp", "127.0.0.2:0")
	if err != nil {
		t.Fatal(err)
	}
	f.server.Listener = listener
	f.server.Config.ConnState = func(conn net.Conn, state http.ConnState) {
		f.mu.Lock()
		defer f.mu.Unlock()
		if state == http.StateClosed || state == http.StateHijacked {
			delete(f.connections, conn)
		} else {
			f.connections[conn] = state
		}
	}
	f.server.Start()
	t.Cleanup(func() { f.release.Do(func() { close(f.unblock) }); f.server.Close() })
	return f
}
func (f *mcpReleaseFixture) endpoint() string { return f.server.URL + "/mcp" }
func (f *mcpReleaseFixture) serve(w http.ResponseWriter, r *http.Request) {
	if r.Header.Get("Authorization") != "Bearer "+mcpReleaseSecret || strings.HasSuffix(r.URL.Path, "/unauthorized") {
		http.Error(w, "private authentication body", 401)
		return
	}
	if r.Method == "GET" {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		w.(http.Flusher).Flush()
		<-r.Context().Done()
		return
	}
	session := r.Header.Get("Mcp-Session-Id")
	if r.Method == "DELETE" {
		f.mu.Lock()
		f.deleted[session] = true
		f.mu.Unlock()
		w.WriteHeader(200)
		return
	}
	var body struct {
		ID     any            `json:"id"`
		Method string         `json:"method"`
		Params map[string]any `json:"params"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 2<<20)).Decode(&body); err != nil {
		http.Error(w, "invalid request", 400)
		return
	}
	var result any
	switch body.Method {
	case "initialize":
		f.mu.Lock()
		session = fmt.Sprintf("private-mcp-session-%d", len(f.sessions)+1)
		f.sessions[session] = true
		f.modes[session] = r.URL.Path
		f.mu.Unlock()
		w.Header().Set("Mcp-Session-Id", session)
		result = map[string]any{"protocolVersion": "2025-11-25", "capabilities": map[string]any{"tools": map[string]any{"listChanged": true}}, "serverInfo": map[string]any{"name": "local-release-fixture", "version": "1"}}
	case "notifications/initialized":
		w.WriteHeader(202)
		return
	case "tools/list":
		schema := map[string]any{"type": "object", "properties": map[string]any{"query": map[string]any{"type": "string"}}, "required": []string{"query"}, "additionalProperties": false}
		if strings.HasSuffix(r.URL.Path, "/schema") {
			schema["$ref"] = "https://unsupported.example/schema"
		}
		tool := func(name string) any {
			return map[string]any{"name": name, "description": "Controlled " + name, "inputSchema": schema}
		}
		if body.Params["cursor"] == "page-2" {
			result = map[string]any{"tools": []any{tool("fetch"), tool("delete_document")}}
		} else {
			name := "search"
			if strings.HasSuffix(r.URL.Path, "/missing") {
				name = "unselected"
			}
			result = map[string]any{"tools": []any{tool(name)}, "nextCursor": "page-2"}
		}
	case "tools/call":
		name, _ := body.Params["name"].(string)
		f.mu.Lock()
		f.calls[session] = append(f.calls[session], name)
		f.mu.Unlock()
		f.once.Do(func() { close(f.firstCall) })
		select {
		case <-f.unblock:
		case <-r.Context().Done():
			return
		}
		if strings.HasSuffix(r.URL.Path, "/drop") {
			conn, _, err := w.(http.Hijacker).Hijack()
			if err == nil {
				conn.Close()
			}
			return
		}
		if strings.HasSuffix(r.URL.Path, "/content") {
			result = map[string]any{"content": []any{map[string]any{"type": "image", "data": "YQ==", "mimeType": "image/png"}}}
		} else {
			result = map[string]any{"content": []any{map[string]any{"type": "text", "text": "found"}}, "structuredContent": map[string]any{"found": true}, "isError": false}
		}
	default:
		http.Error(w, "unexpected MCP method", 400)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"jsonrpc": "2.0", "id": body.ID, "result": result})
}
func (f *mcpReleaseFixture) counts() (int, int, int, int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	calls := 0
	for _, names := range f.calls {
		calls += len(names)
	}
	return len(f.sessions), calls, len(f.deleted), len(f.connections)
}

type mcpScriptedGateway struct {
	server     *httptest.Server
	mu         sync.Mutex
	selections [][]string
	errors     []string
	bodies     [][]byte
}

func newMCPScriptedGateway(t *testing.T) *mcpScriptedGateway {
	g := &mcpScriptedGateway{}
	g.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(io.LimitReader(r.Body, 2<<20))
		var request map[string]any
		if json.Unmarshal(raw, &request) != nil {
			http.Error(w, "invalid", 400)
			return
		}
		g.mu.Lock()
		g.bodies = append(g.bodies, raw)
		g.mu.Unlock()
		var message map[string]any
		finish := "stop"
		input, isFinalizer, err := decodeWorkerResultFinalizerRequest(request)
		if err == nil && isFinalizer {
			if !requestHasNoModelTools(request) {
				err = fmt.Errorf("finalizer exposed tools")
			} else {
				message, err = workerResultFinalizerMessage(input)
			}
		} else if err == nil {
			var names []string
			for _, entry := range request["tools"].([]any) {
				fn := entry.(map[string]any)["function"].(map[string]any)
				names = append(names, fn["name"].(string))
			}
			slices.Sort(names)
			g.mu.Lock()
			g.selections = append(g.selections, names)
			g.mu.Unlock()
			if len(names) != 1 || !slices.Contains([]string{"search", "fetch", "list_artifacts"}, names[0]) {
				err = fmt.Errorf("unexpected allowlist")
			} else {
				hasTool := false
				for _, entry := range request["messages"].([]any) {
					if entry.(map[string]any)["role"] == "tool" {
						hasTool = true
					}
				}
				if hasTool {
					message = map[string]any{"role": "assistant", "content": "Completed bounded tool call"}
				} else {
					args := map[string]any{"query": "release query"}
					if names[0] == "list_artifacts" {
						args = map[string]any{"namespace": nil}
					}
					message = toolCallMessage("selected-once", names[0], args)
					finish = "tool_calls"
				}
			}
		}
		if err != nil {
			g.mu.Lock()
			g.errors = append(g.errors, err.Error())
			g.mu.Unlock()
			http.Error(w, "scripted model rejected request", 400)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"id": "chatcmpl-mcp-release", "object": "chat.completion", "created": time.Now().Unix(), "model": "worker-model", "choices": []any{map[string]any{"index": 0, "message": message, "finish_reason": finish}}, "usage": map[string]any{"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}})
	}))
	t.Cleanup(g.server.Close)
	return g
}
