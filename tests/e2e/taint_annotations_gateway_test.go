//go:build e2e

package e2e

import (
	"errors"
	"fmt"
	"sort"
)

const (
	taintSourceCanary   = "TAINT_SOURCE_CONTENT_CANARY"
	taintTargetCanary   = "TAINT_TARGET_VALUE_CANARY"
	taintOTLPCredential = "TAINT_OTLP_CREDENTIAL_CANARY"
	taintSourcePath     = "private-taint-path-canary/app.py"
	taintDuplicatePath  = "private-taint-path-canary/duplicates.py"
	taintCleanPath      = "clean-project/clean.py"
	taintTraceReport    = "# Taint trace fixture\n\nThe assigned path was traced, annotated, and verified against source evidence.\n"
	taintReuseReport    = "# Clean reuse fixture\n\nThe unrelated workspace was inspected without carrying prior allocation state.\n"
)

var taintAnnotationModelTools = []string{
	"annotate_sink", "annotate_trace", "annotate_validate",
	"attack_surface", "changed_paths", "complexity_hotspots", "diff",
	"entrypoint_paths_to", "find_callees", "find_callers", "find_symbol",
	"functions_that_raise", "glob", "graph_summary", "grep", "list_skills",
	"list_symbols", "load_skill", "load_skill_resource", "ls", "paths_between",
	"read_file", "read_text_artifact", "rollback_changes", "search_def",
	"write_text_artifact",
}

func newTaintAnnotationGateway(token string) *domainGateway {
	gateway := newBlockedDomainGateway(token, []domainGatewayStage{
		taintMutationGatewayStage(),
		taintCleanReuseGatewayStage(),
	})
	gateway.releaseBlockedRequest()
	return gateway
}

func taintMutationGatewayStage() domainGatewayStage {
	return domainGatewayStage{
		name:  "taint/mutate",
		tools: append([]string(nil), taintAnnotationModelTools...),
		steps: []domainGatewayStep{
			toolGatewayStep("list_skills", fixedArguments(map[string]any{})),
			{
				tool: "load_skill", validate: requireToolOutput("trace"),
				arguments: fixedArguments(map[string]any{"skill_name": "trace"}),
			},
			{
				tool: "load_skill_resource", validate: requireToolOutput("# Trace Annotation Skill"),
				arguments: fixedArguments(map[string]any{
					"skill_name": "trace", "file_path": "references/annotations.md",
				}),
			},
			{
				tool: "graph_summary", validate: requireToolOutput("# Annotation Discipline"),
				arguments: fixedArguments(map[string]any{}),
			},
			{
				tool: "search_def", validate: requireToolOutput("nodeCount"),
				arguments: fixedArguments(map[string]any{
					"symbol": "handler", "language": "python", "path": taintSourcePath,
					"limit": 10,
				}),
			},
			{
				tool: "annotate_trace", validate: requireStructuralItem("handler", taintSourcePath),
				arguments: fixedArguments(map[string]any{
					"path": taintSourcePath, "symbol": "handler", "target": taintTargetCanary,
					"args": "req:tainted", "calls": "validate_input,query_db",
				}),
			},
			{
				tool: "annotate_validate", validate: requireAnnotationResult("trace", "handler", true),
				arguments: fixedArguments(map[string]any{
					"path": taintSourcePath, "symbol": "validate_input",
					"arg": "value", "kind": "schema",
				}),
			},
			{
				tool: "annotate_sink", validate: requireAnnotationResult("validate", "validate_input", true),
				arguments: fixedArguments(map[string]any{
					"path": taintSourcePath, "symbol": "query_db",
					"kind": "db.query", "arg": "value",
				}),
			},
			{
				tool: "annotate_trace", validate: requireAnnotationResult("sink", "query_db", true),
				arguments: fixedArguments(map[string]any{
					"path": taintDuplicatePath, "symbol": "duplicate",
					"target": taintTargetCanary,
				}),
			},
			{
				tool: "search_def", validate: requireToolOutput("taint_annotation_target_ambiguous"),
				arguments: fixedArguments(map[string]any{
					"symbol": "duplicate", "language": "python", "path": taintDuplicatePath,
					"limit": 10,
				}),
			},
			{
				tool: "annotate_trace", validate: requireStructuralItem("duplicate", taintDuplicatePath),
				arguments: func(request map[string]any) (map[string]any, error) {
					line, err := highestStructuralLine(request, "duplicate", taintDuplicatePath)
					if err != nil {
						return nil, err
					}
					return map[string]any{
						"path": taintDuplicatePath, "symbol": "duplicate",
						"target": taintTargetCanary, "definition_line": line,
					}, nil
				},
			},
			{
				tool: "changed_paths", validate: requireAnnotationResult("trace", "duplicate", true),
				arguments: fixedArguments(map[string]any{}),
			},
			{
				tool: "diff", validate: requireChangedPaths(taintSourcePath, taintDuplicatePath),
				arguments: fixedArguments(map[string]any{"max_bytes": 65536}),
			},
			{
				tool: "write_text_artifact", validate: requireTaintDiff(),
				arguments: fixedArguments(map[string]any{
					"name": "report", "text": taintTraceReport,
					"media_type": "text/markdown", "expected_revision": nil,
				}),
			},
			{
				plain: true, summary: "Taint trace fixture completed",
				artifacts: map[string]domainArtifactBinding{
					"report": {namespace: "analysis", name: "report"},
				},
			},
		},
	}
}

func taintCleanReuseGatewayStage() domainGatewayStage {
	return domainGatewayStage{
		name:  "taint/clean-reuse",
		tools: append([]string(nil), taintAnnotationModelTools...),
		steps: []domainGatewayStep{
			toolGatewayStep("list_skills", fixedArguments(map[string]any{})),
			{
				tool: "load_skill", validate: requireToolOutput("trace"),
				arguments: fixedArguments(map[string]any{"skill_name": "trace"}),
			},
			{
				tool: "graph_summary", validate: rejectToolOutput(
					taintSourceCanary, taintTargetCanary, taintSourcePath, taintDuplicatePath,
				),
				arguments: fixedArguments(map[string]any{}),
			},
			{
				tool: "search_def", validate: requireToolOutput("nodeCount"),
				arguments: fixedArguments(map[string]any{
					"symbol": "clean_handler", "language": "python", "path": taintCleanPath,
					"limit": 10,
				}),
			},
			{
				tool: "changed_paths", validate: requireStructuralItem("clean_handler", taintCleanPath),
				arguments: fixedArguments(map[string]any{}),
			},
			{
				tool: "diff", validate: requireNoWorkspaceChanges(),
				arguments: fixedArguments(map[string]any{"max_bytes": 65536}),
			},
			{
				tool: "write_text_artifact", validate: requireEmptyWorkspaceDiff(),
				arguments: fixedArguments(map[string]any{
					"name": "report", "text": taintReuseReport,
					"media_type": "text/markdown", "expected_revision": nil,
				}),
			},
			{
				plain: true, summary: "Clean unrelated workspace completed",
				artifacts: map[string]domainArtifactBinding{
					"report": {namespace: "analysis", name: "report"},
				},
			},
		},
	}
}

func requireStructuralItem(name, path string) func(map[string]any) error {
	return func(request map[string]any) error {
		found := false
		walkDomainJSON(request, func(object map[string]any) {
			if object["name"] == name && object["path"] == path {
				if _, ok := object["line"]; ok {
					found = true
				}
			}
		})
		if !found {
			return fmt.Errorf("structural result omitted %s in %s", name, path)
		}
		return nil
	}
}

func requireAnnotationResult(kind, symbol string, changed bool) func(map[string]any) error {
	return func(request map[string]any) error {
		found := false
		walkDomainJSON(request, func(object map[string]any) {
			if object["kind"] == kind && object["symbol"] == symbol && object["changed"] == changed {
				found = true
			}
		})
		if !found {
			return fmt.Errorf("annotation result omitted %s/%s changed=%t", kind, symbol, changed)
		}
		return nil
	}
}

func highestStructuralLine(request map[string]any, name, path string) (int, error) {
	lines := make([]int, 0, 2)
	walkDomainJSON(request, func(object map[string]any) {
		if object["name"] != name || object["path"] != path {
			return
		}
		switch line := object["line"].(type) {
		case jsonNumber:
			if value, err := line.Int64(); err == nil {
				lines = append(lines, int(value))
			}
		case float64:
			lines = append(lines, int(line))
		}
	})
	if len(lines) != 2 {
		return 0, fmt.Errorf("duplicate structural search returned %d definitions", len(lines))
	}
	sort.Ints(lines)
	return lines[len(lines)-1], nil
}

// jsonNumber is the concrete type produced by the fake Gateway decoder while
// keeping this helper independent of representation details elsewhere.
type jsonNumber interface {
	Int64() (int64, error)
}

func requireChangedPaths(paths ...string) func(map[string]any) error {
	return func(request map[string]any) error {
		found := make(map[string]bool, len(paths))
		walkDomainJSON(request, func(object map[string]any) {
			path, pathOK := object["path"].(string)
			change, changeOK := object["change"].(string)
			if pathOK && changeOK && change == "modified" {
				found[path] = true
			}
		})
		for _, path := range paths {
			if !found[path] {
				return fmt.Errorf("changed_paths omitted %s", path)
			}
		}
		return nil
	}
}

func requireTaintDiff() func(map[string]any) error {
	return func(request map[string]any) error {
		for _, fragment := range []string{
			"--- a/" + taintSourcePath,
			"# @trace target=" + taintTargetCanary,
			"# @validate arg=value kind=schema",
			"# @sink kind=db.query arg=value",
			"--- a/" + taintDuplicatePath,
		} {
			if !containsStringFragment(request, fragment) {
				return fmt.Errorf("workspace diff omitted %q", fragment)
			}
		}
		return nil
	}
}

func rejectToolOutput(fragments ...string) func(map[string]any) error {
	return func(request map[string]any) error {
		for _, fragment := range fragments {
			if containsStringFragment(request, fragment) {
				return fmt.Errorf("clean allocation retained prior fragment %q", fragment)
			}
		}
		return nil
	}
}

func requireNoWorkspaceChanges() func(map[string]any) error {
	return func(request map[string]any) error {
		found := false
		walkDomainJSON(request, func(object map[string]any) {
			changes, ok := object["changes"].([]any)
			if ok && len(changes) == 0 && object["truncated"] == false {
				found = true
			}
		})
		if !found {
			return errors.New("clean workspace unexpectedly reports changes")
		}
		return nil
	}
}

func requireEmptyWorkspaceDiff() func(map[string]any) error {
	return func(request map[string]any) error {
		found := false
		walkDomainJSON(request, func(object map[string]any) {
			if object["text"] == "" && object["authoritative"] == false {
				found = true
			}
		})
		if !found {
			return errors.New("clean workspace diff is not empty")
		}
		return nil
	}
}
