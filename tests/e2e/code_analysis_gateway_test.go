//go:build e2e

package e2e

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
	"sync"
	"time"
)

const codeAnalysisReport = `# Code analysis fixture

- Portable structural discovery completed.
- Graph relationships and bounded paths completed when selected.
- Workspace changes were observed through fresh analysis state.
`

const (
	codeAnalysisDuplicateSymbol   = "SensitiveGraphQueryCanary"
	codeAnalysisReplacementSymbol = "ReplacementGraphCanary"
	codeAnalysisEditedPath        = "private-path-canary/dup_000.py"
)

var codeAnalysisShallowTools = []string{
	"list_symbols", "search_def", "write_text_artifact",
}

var codeAnalysisGraphTools = []string{
	"attack_surface", "complexity_hotspots", "edit", "entrypoint_paths_to",
	"find_callees", "find_callers", "find_symbol", "functions_that_raise",
	"graph_summary", "list_symbols", "paths_between", "search_def",
	"write_text_artifact",
}

type codeAnalysisGateway struct {
	server *httptest.Server
	token  string

	mu        sync.Mutex
	scenarios []codeAnalysisScenario
	scenario  int
	step      int
	calls     int
	failures  []string
}

type codeAnalysisScenario struct {
	name  string
	tools []string
	steps []codeAnalysisGatewayStep
}

type codeAnalysisGatewayStep struct {
	tool      string
	arguments func(map[string]any) (map[string]any, error)
	validate  func(map[string]any) error
	final     bool
}

type codeAnalysisGraphScript struct {
	shallowCursor string
	oldSymbolID   string
	entryID       string
	targetID      string
}

func newCodeAnalysisGateway(token string) *codeAnalysisGateway {
	gateway := &codeAnalysisGateway{
		token: token,
		scenarios: []codeAnalysisScenario{
			codeAnalysisShallowScenario("shallow-initial"),
			codeAnalysisGraphScenario("graph-initial", true),
			codeAnalysisShallowScenario("shallow-unrelated"),
			codeAnalysisGraphScenario("graph-reuse", false),
		},
	}
	gateway.server = httptest.NewServer(http.HandlerFunc(gateway.serveHTTP))
	return gateway
}

func (g *codeAnalysisGateway) close() { g.server.Close() }

func (g *codeAnalysisGateway) URL() string { return g.server.URL + "/v1" }

func (g *codeAnalysisGateway) Failures() []string {
	g.mu.Lock()
	defer g.mu.Unlock()
	return append([]string(nil), g.failures...)
}

func (g *codeAnalysisGateway) CompletedScenarios() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.scenario
}

func (g *codeAnalysisGateway) Calls() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.calls
}

func (g *codeAnalysisGateway) serveHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost || !strings.HasSuffix(r.URL.Path, "/chat/completions") {
		g.fail(w, http.StatusNotFound, "unsupported code-analysis Gateway endpoint")
		return
	}
	if r.Header.Get("Authorization") != "Bearer "+g.token {
		g.fail(w, http.StatusUnauthorized, "invalid code-analysis Gateway token")
		return
	}
	defer r.Body.Close()
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 8<<20))
	decoder.UseNumber()
	var request map[string]any
	if err := decoder.Decode(&request); err != nil {
		g.fail(w, http.StatusBadRequest, "invalid code-analysis OpenAI request")
		return
	}
	message, reason, call, err := g.next(request)
	if err != nil {
		g.fail(w, http.StatusBadRequest, err.Error())
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"id":      fmt.Sprintf("chatcmpl-code-analysis-%d", call),
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   "worker-model",
		"choices": []any{map[string]any{
			"index": 0, "message": message, "finish_reason": reason,
		}},
		"usage": map[string]any{
			"prompt_tokens": 13, "completion_tokens": 7, "total_tokens": 20,
		},
	})
}

func (g *codeAnalysisGateway) next(
	request map[string]any,
) (map[string]any, string, int, error) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.scenario >= len(g.scenarios) {
		return nil, "", 0, errors.New("unexpected additional code-analysis invocation")
	}
	scenario := g.scenarios[g.scenario]
	if g.step >= len(scenario.steps) {
		return nil, "", 0, fmt.Errorf("invalid code-analysis script position for %s", scenario.name)
	}
	tools, err := requestToolNames(request)
	if err != nil {
		return nil, "", 0, fmt.Errorf("%s: %w", scenario.name, err)
	}
	want := append([]string(nil), scenario.tools...)
	sort.Strings(want)
	if !equalStringSlices(tools, want) {
		return nil, "", 0, fmt.Errorf("%s exposed tools %v, want %v", scenario.name, tools, want)
	}
	step := scenario.steps[g.step]
	if step.validate != nil {
		if err := step.validate(request); err != nil {
			return nil, "", 0, fmt.Errorf("%s step %d: %w", scenario.name, g.step+1, err)
		}
	}

	var message map[string]any
	reason := "tool_calls"
	if step.final {
		_, ok := lastExactArtifact(request, "analysis", "report")
		if !ok {
			return nil, "", 0, fmt.Errorf("%s final response did not observe analysis/report", scenario.name)
		}
		message, err = workerModelResultMessage(request, "Code analysis fixture completed")
		if err != nil {
			return nil, "", 0, err
		}
		reason = "stop"
	} else {
		arguments, err := step.arguments(request)
		if err != nil {
			return nil, "", 0, fmt.Errorf("%s step %d: %w", scenario.name, g.step+1, err)
		}
		message = toolCallMessage(
			fmt.Sprintf("code-analysis-%d-%d", g.scenario+1, g.step+1),
			step.tool,
			arguments,
		)
	}
	g.calls++
	call := g.calls
	g.step++
	if g.step == len(scenario.steps) {
		g.scenario++
		g.step = 0
	}
	return message, reason, call, nil
}

func (g *codeAnalysisGateway) fail(w http.ResponseWriter, status int, message string) {
	g.mu.Lock()
	g.failures = append(g.failures, message)
	g.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{"message": message, "type": "invalid_request_error"},
	})
}

func codeAnalysisShallowScenario(name string) codeAnalysisScenario {
	return codeAnalysisScenario{name: name, tools: codeAnalysisShallowTools, steps: []codeAnalysisGatewayStep{
		codeAnalysisCall("list_symbols", fixedArguments(map[string]any{"limit": 3})),
		{
			tool: "search_def",
			validate: func(request map[string]any) error {
				if !hasShallowCollection(request) {
					return errors.New("list_symbols did not return structural definitions")
				}
				return nil
			},
			arguments: fixedArguments(map[string]any{
				"symbol": "entry", "language": "python", "limit": 10,
			}),
		},
		{
			tool: "write_text_artifact",
			validate: func(request map[string]any) error {
				if !hasCollectionItem(request, "entry", false) {
					return errors.New("search_def did not return structural evidence")
				}
				return nil
			},
			arguments: fixedArguments(map[string]any{
				"name": "report", "text": codeAnalysisReport,
				"media_type": "text/markdown", "expected_revision": nil,
			}),
		},
		{final: true},
	}}
}

func codeAnalysisGraphScenario(name string, mutate bool) codeAnalysisScenario {
	if !mutate {
		return codeAnalysisScenario{name: name, tools: codeAnalysisGraphTools, steps: []codeAnalysisGatewayStep{
			codeAnalysisCall("graph_summary", fixedArguments(nil)),
			{
				tool: "find_symbol",
				validate: func(request map[string]any) error {
					if !hasGraphSummary(request) {
						return errors.New("clean reused slot did not build a graph")
					}
					return nil
				},
				arguments: fixedArguments(map[string]any{"query": "entry", "limit": 10}),
			},
			{
				tool: "write_text_artifact",
				validate: func(request map[string]any) error {
					if !hasCollectionItem(request, "entry", true) {
						return errors.New("clean reused slot did not resolve an exact graph symbol")
					}
					return nil
				},
				arguments: fixedArguments(map[string]any{
					"name": "report", "text": codeAnalysisReport,
					"media_type": "text/markdown", "expected_revision": nil,
				}),
			},
			{final: true},
		}}
	}

	state := &codeAnalysisGraphScript{}
	return codeAnalysisScenario{name: name, tools: codeAnalysisGraphTools, steps: []codeAnalysisGatewayStep{
		codeAnalysisCall("list_symbols", fixedArguments(map[string]any{"limit": 1})),
		{
			tool: "find_symbol",
			arguments: func(request map[string]any) (map[string]any, error) {
				cursor, ok := shallowCollectionCursor(request)
				if !ok {
					return nil, errors.New("shallow result did not provide a resumable cursor")
				}
				state.shallowCursor = cursor
				return map[string]any{"query": codeAnalysisDuplicateSymbol, "limit": 1}, nil
			},
		},
		{
			tool: "find_symbol",
			arguments: func(request map[string]any) (map[string]any, error) {
				id, ok := graphSymbolID(request, codeAnalysisDuplicateSymbol)
				if !ok {
					return nil, errors.New("duplicate graph symbol was not returned")
				}
				state.oldSymbolID = id
				return map[string]any{"query": "entry", "limit": 10}, nil
			},
		},
		{
			tool: "find_symbol",
			arguments: func(request map[string]any) (map[string]any, error) {
				id, ok := graphSymbolID(request, "entry")
				if !ok {
					return nil, errors.New("entry graph symbol was not returned")
				}
				state.entryID = id
				return map[string]any{"query": "sink", "limit": 10}, nil
			},
		},
		{
			tool: "paths_between",
			arguments: func(request map[string]any) (map[string]any, error) {
				id, ok := graphSymbolID(request, "sink")
				if !ok {
					return nil, errors.New("sink graph symbol was not returned")
				}
				state.targetID = id
				return map[string]any{
					"source_id": state.entryID, "target_id": state.targetID,
					"max_depth": 6, "limit": 2,
				}, nil
			},
		},
		{
			tool: "find_callees",
			validate: func(request map[string]any) error {
				if !hasTruncatedPathResult(request) {
					return errors.New("bounded high-branching traversal did not truncate")
				}
				return nil
			},
			arguments: func(map[string]any) (map[string]any, error) {
				return map[string]any{"symbol_id": state.entryID, "limit": 10}, nil
			},
		},
		{
			tool: "edit",
			validate: func(request map[string]any) error {
				if !hasCollectionItem(request, "branch_a", true) {
					return errors.New("find_callees did not return a real relationship")
				}
				return nil
			},
			arguments: fixedArguments(map[string]any{
				"path": codeAnalysisEditedPath, "old": "def " + codeAnalysisDuplicateSymbol + "():",
				"new": "def " + codeAnalysisReplacementSymbol + "():", "replace_all": false,
			}),
		},
		{
			tool: "list_symbols",
			validate: func(request map[string]any) error {
				if !hasChangedResult(request) {
					return errors.New("overlay edit was not confirmed")
				}
				return nil
			},
			arguments: func(map[string]any) (map[string]any, error) {
				return map[string]any{"cursor": state.shallowCursor, "limit": 1}, nil
			},
		},
		{
			tool: "find_callers",
			validate: func(request map[string]any) error {
				if !containsCodeAnalysisValue(request, "code_analysis_workspace_changed") {
					return errors.New("stale shallow cursor was not rejected")
				}
				return nil
			},
			arguments: func(map[string]any) (map[string]any, error) {
				return map[string]any{"symbol_id": state.oldSymbolID, "limit": 10}, nil
			},
		},
		{
			tool: "find_symbol",
			validate: func(request map[string]any) error {
				if !containsCodeAnalysisValue(request, "code_analysis_stale_symbol") {
					return errors.New("stale graph symbol ID was not rejected")
				}
				return nil
			},
			arguments: fixedArguments(map[string]any{"query": codeAnalysisReplacementSymbol, "limit": 10}),
		},
		{
			tool: "graph_summary",
			validate: func(request map[string]any) error {
				if !hasCollectionItem(request, codeAnalysisReplacementSymbol, true) {
					return errors.New("fresh graph query did not observe the overlay edit")
				}
				return nil
			},
			arguments: fixedArguments(nil),
		},
		{
			tool: "write_text_artifact",
			validate: func(request map[string]any) error {
				if !hasGraphSummary(request) {
					return errors.New("rebuilt graph summary is absent")
				}
				return nil
			},
			arguments: fixedArguments(map[string]any{
				"name": "report", "text": codeAnalysisReport,
				"media_type": "text/markdown", "expected_revision": nil,
			}),
		},
		{final: true},
	}}
}

func codeAnalysisCall(
	tool string,
	arguments func(map[string]any) (map[string]any, error),
) codeAnalysisGatewayStep {
	return codeAnalysisGatewayStep{tool: tool, arguments: arguments}
}

func shallowCollectionCursor(value any) (string, bool) {
	var result string
	walkDomainJSON(value, func(object map[string]any) {
		items, ok := object["items"].([]any)
		cursor, cursorOK := object["nextCursor"].(string)
		if !ok || len(items) == 0 || !cursorOK || cursor == "" {
			return
		}
		first, ok := items[0].(map[string]any)
		if ok {
			if _, graph := first["symbolId"]; !graph {
				result = cursor
			}
		}
	})
	return result, result != ""
}

func graphSymbolID(value any, name string) (string, bool) {
	var result string
	walkDomainJSON(value, func(object map[string]any) {
		if object["name"] != name {
			return
		}
		if id, ok := object["symbolId"].(string); ok && id != "" {
			result = id
		}
	})
	return result, result != ""
}

func hasCollectionItem(value any, name string, requireGraphID bool) bool {
	found := false
	walkDomainJSON(value, func(object map[string]any) {
		if object["name"] != name {
			return
		}
		_, graph := object["symbolId"].(string)
		if graph == requireGraphID {
			found = true
		}
	})
	return found
}

func hasShallowCollection(value any) bool {
	found := false
	walkDomainJSON(value, func(object map[string]any) {
		items, ok := object["items"].([]any)
		if !ok || len(items) == 0 {
			return
		}
		first, ok := items[0].(map[string]any)
		if !ok {
			return
		}
		_, graph := first["symbolId"]
		_, path := first["path"].(string)
		_, line := first["line"]
		if !graph && path && line {
			found = true
		}
	})
	return found
}

func hasGraphSummary(value any) bool {
	found := false
	walkDomainJSON(value, func(object map[string]any) {
		_, nodes := object["nodeCount"]
		_, edges := object["callEdgeCount"]
		_, languages := object["languages"]
		if nodes && edges && languages {
			found = true
		}
	})
	return found
}

func hasChangedResult(value any) bool {
	found := false
	walkDomainJSON(value, func(object map[string]any) {
		if changed, ok := object["changed"].(bool); ok && changed {
			found = true
		}
	})
	return found
}

func hasTruncatedPathResult(value any) bool {
	found := false
	walkDomainJSON(value, func(object map[string]any) {
		items, itemsOK := object["items"].([]any)
		truncated, truncatedOK := object["truncated"].(bool)
		if !itemsOK || len(items) != 2 || !truncatedOK || !truncated {
			return
		}
		if _, path := items[0].([]any); path {
			found = true
		}
	})
	return found
}

func containsCodeAnalysisValue(value any, want string) bool {
	encoded, err := json.Marshal(value)
	return err == nil && strings.Contains(string(encoded), want)
}
