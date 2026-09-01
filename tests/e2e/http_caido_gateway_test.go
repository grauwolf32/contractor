//go:build e2e

package e2e

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"sort"
)

const (
	httpCaidoSessionCanary = "HTTP_CAIDO_SESSION_SECRET_CANARY"
	httpCaidoRawCanary     = "HTTP_CAIDO_RAW_BODY_CANARY"
	httpCaidoHTTPReport    = "# HTTP analysis\n\nThe bounded 500 response body is retained by exact artifact reference.\n"
	httpCaidoCaidoReport   = "# Caido analysis\n\nHistory and one bounded scope mutation completed against the pinned instance.\n"
)

func newHTTPCaidoGateway(token, targetURL string) *domainGateway {
	gateway := &domainGateway{
		token: token,
		stages: []domainGatewayStage{
			httpCaidoHTTPGatewayStage("http/initial", targetURL),
			httpCaidoCaidoGatewayStage("caido/pinned-old", "target-old"),
			httpCaidoHTTPGatewayStage("http/unrelated-during-release-recovery", targetURL),
			httpCaidoCaidoGatewayStage("caido/agent-label-new", "target-new"),
		},
	}
	gateway.server = httptest.NewServer(http.HandlerFunc(gateway.serveHTTP))
	return gateway
}

func httpCaidoHTTPGatewayStage(name, targetURL string) domainGatewayStage {
	tools := []string{
		"http_history", "http_read_body", "http_request", "http_session_clear",
		"http_session_get", "http_session_set", "read_text_artifact", "write_text_artifact",
	}
	sort.Strings(tools)
	return domainGatewayStage{
		name:  name,
		tools: tools,
		steps: []domainGatewayStep{
			{
				tool: "http_session_get", validate: requireNoToolOutput(httpCaidoSessionCanary),
				arguments: fixedArguments(map[string]any{}),
			},
			{
				tool: "http_session_set",
				validate: composeGatewayValidation(
					requireToolOutput("auth_kind"), requireToolOutput("none"),
				),
				arguments: fixedArguments(map[string]any{
					"auth":            map[string]any{"kind": "bearer", "token": httpCaidoSessionCanary},
					"replace_cookies": true, "replace_headers": true,
				}),
			},
			{
				tool: "http_request", validate: requireToolOutput("bearer"),
				arguments: fixedArguments(map[string]any{
					"url": targetURL + "/failure", "method": "POST",
					"body_type": "none", "follow_redirects": false,
				}),
			},
			{
				tool:     "http_read_body",
				validate: requireToolOutput("500"),
				arguments: fixedArguments(map[string]any{
					"request_id": 1, "offset": 0, "length": 8192,
				}),
			},
			{
				tool: "http_history", validate: requireToolOutput(httpCaidoRawCanary),
				arguments: fixedArguments(map[string]any{"limit": 10}),
			},
			toolGatewayStep("write_text_artifact", fixedArguments(map[string]any{
				"name": "report", "text": httpCaidoHTTPReport,
				"media_type": "text/markdown", "expected_revision": nil,
			})),
			toolGatewayStep("http_session_clear", fixedArguments(map[string]any{})),
			{
				validate: composeGatewayValidation(
					requireToolOutput("http_session_clear"), requireToolOutput("history_count"),
				),
				summary: "HTTP evidence report published and session cleared",
				artifacts: map[string]domainArtifactBinding{
					"report": {namespace: "http", name: "report"},
				},
			},
		},
	}
}

func httpCaidoCaidoGatewayStage(name, scopeName string) domainGatewayStage {
	tools := []string{
		"caido_automate_results", "caido_automate_run", "caido_history", "caido_replay",
		"caido_request_detail", "caido_scope", "caido_sitemap", "caido_workflow_findings",
		"caido_workflow_list", "caido_workflow_run", "http_history", "http_read_body",
		"http_request", "list_skills", "load_skill", "load_skill_resource",
		"read_text_artifact", "write_text_artifact",
	}
	sort.Strings(tools)
	return domainGatewayStage{
		name:  name,
		tools: tools,
		steps: []domainGatewayStep{
			toolGatewayStep("list_skills", fixedArguments(map[string]any{})),
			{
				tool: "load_skill",
				validate: composeGatewayValidation(
					requireToolOutput("available_skills"), requireToolOutput("caido"),
				),
				arguments: fixedArguments(map[string]any{"skill_name": "caido"}),
			},
			{
				tool: "caido_history", validate: requireToolOutput("Caido analysis"),
				arguments: fixedArguments(map[string]any{
					"filter": "", "limit": 5, "offset": 0,
				}),
			},
			{
				tool: "caido_scope", validate: requireToolOutput("requests"),
				arguments: fixedArguments(map[string]any{
					"action": "create", "name": scopeName,
					"allowlist": []any{"target.example"}, "denylist": []any{},
				}),
			},
			{
				tool: "write_text_artifact", validate: requireToolOutput("created"),
				arguments: fixedArguments(map[string]any{
					"name": "report", "text": httpCaidoCaidoReport,
					"media_type": "text/markdown", "expected_revision": nil,
				}),
			},
			finalGatewayStep("Caido evidence report published", map[string]domainArtifactBinding{
				"report": {namespace: "security", name: "report"},
			}),
		},
	}
}

func requireNoToolOutput(fragment string) func(map[string]any) error {
	return func(request map[string]any) error {
		messages, ok := request["messages"].([]any)
		if ok && containsStringFragment(messages, fragment) {
			return errors.New("unexpected prior-allocation tool output is visible")
		}
		return nil
	}
}
