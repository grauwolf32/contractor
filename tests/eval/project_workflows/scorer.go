package projectworkflows

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"go.yaml.in/yaml/v4"
)

const maxValidatorOutput = 1 << 20

type Failure struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

type Score struct {
	Failures []Failure `json:"failures"`
}

func (s Score) Passed() bool { return len(s.Failures) == 0 }

func (s *Score) require(condition bool, code, message string) {
	if !condition {
		s.Failures = append(s.Failures, Failure{Code: code, Message: message})
	}
}

func ScoreDiscovery(dependencyReport, projectReport []byte) Score {
	var score Score
	dependency := strings.ToLower(string(dependencyReport))
	project := strings.ToLower(string(projectReport))
	score.require(strings.Contains(dependency, "asyncpg"), "dependency.asyncpg", "dependency report omits the database client")
	score.require(strings.Contains(dependency, "httpx"), "dependency.httpx", "dependency report omits the outbound HTTP client")
	score.require(strings.Contains(dependency, "fastapi"), "dependency.fastapi", "dependency report omits the inbound HTTP framework")
	score.require(strings.Contains(project, "/widgets/{widget_id}"), "project.get_route", "project report omits the GET widget route")
	score.require(strings.Contains(project, "/widgets"), "project.post_route", "project report omits the POST widget route")
	score.require(strings.Contains(project, "postgres") || strings.Contains(project, "database"), "project.datastore", "project report omits the datastore")
	score.require(strings.Contains(project, "inventory"), "project.external", "project report omits the Inventory integration")
	evidence := regexp.MustCompile(`(?m)\b(?:app\.py|pyproject\.toml):\d+(?:-\d+)?\b`)
	score.require(len(evidence.FindAll(dependencyReport, -1)) >= 3, "dependency.evidence", "dependency report has fewer than three source citations")
	score.require(len(evidence.FindAll(projectReport, -1)) >= 5, "project.evidence", "project report has fewer than five source citations")
	return score
}

func ScoreOpenAPI(data []byte) Score {
	var score Score
	var document map[string]any
	if err := yaml.Unmarshal(data, &document); err != nil {
		score.require(false, "openapi.parse", "OpenAPI output is not valid YAML")
		return score
	}
	version, _ := document["openapi"].(string)
	score.require(strings.HasPrefix(version, "3.0.") || strings.HasPrefix(version, "3.1."), "openapi.version", "OpenAPI output does not declare version 3.0 or 3.1")
	paths, pathsOK := stringMap(document["paths"])
	score.require(pathsOK, "openapi.paths", "OpenAPI output has no paths object")
	securitySchemes := nestedMap(document, "components", "securitySchemes")
	score.require(len(securitySchemes) > 0, "openapi.security_scheme", "OpenAPI output omits the enforced authentication scheme")
	globalSecurity := nonemptyList(document["security"])
	for pathName, methods := range ExpectedRoutes {
		pathItem, pathOK := stringMap(paths[pathName])
		score.require(pathOK, "openapi.route."+pathName, "OpenAPI output omits an implemented route")
		if !pathOK {
			continue
		}
		provenance := nonemptyStringList(pathItem["x-path-files"])
		for _, method := range methods {
			operation, operationOK := stringMap(pathItem[method])
			code := "openapi.operation." + method + "." + pathName
			score.require(operationOK, code, "OpenAPI output omits an implemented route method")
			if !operationOK {
				continue
			}
			responses, responsesOK := stringMap(operation["responses"])
			score.require(responsesOK && len(responses) > 0, code+".responses", "OpenAPI operation has no responses")
			operationProvenance := provenance || nonemptyStringList(operation["x-operation-files"])
			score.require(operationProvenance, code+".provenance", "OpenAPI operation has no source provenance")
			score.require(globalSecurity || nonemptyList(operation["security"]), code+".security", "OpenAPI operation omits enforced authentication")
		}
	}
	return score
}

func ScoreLikeC4(data []byte) Score {
	var score Score
	content := string(data)
	lower := strings.ToLower(content)
	score.require(containsAny(lower, "widget service", "widget application", "widget backend"), "likec4.application", "LikeC4 output omits the application")
	score.require(containsAny(lower, "api client", "client", "user"), "likec4.actor", "LikeC4 output omits an external actor")
	score.require(containsAny(lower, "postgresql", "postgres", "database", "data store", "datastore"), "likec4.datastore", "LikeC4 output omits the datastore")
	score.require(strings.Contains(lower, "inventory"), "likec4.external_service", "LikeC4 output omits the Inventory service")
	score.require(regexp.MustCompile(`(?m)(?:^|\{)\s*view\s+[A-Za-z_]`).MatchString(content), "likec4.view", "LikeC4 output has no view")

	relationships := make([]string, 0)
	for _, line := range strings.Split(lower, "\n") {
		if strings.Contains(line, "->") {
			relationships = append(relationships, line)
		}
	}
	score.require(len(relationships) >= 3, "likec4.relationship_count", "LikeC4 output has fewer than three relationships")
	score.require(anyLine(relationships, []string{"client", "user"}, []string{"api", "widget", "service"}), "likec4.inbound", "LikeC4 output omits the inbound relationship")
	score.require(anyLine(relationships, []string{"api", "widget", "service"}, []string{"postgres", "database", "store", "db"}), "likec4.persistence", "LikeC4 output omits the persistence relationship")
	score.require(anyLine(relationships, []string{"api", "widget", "service"}, []string{"inventory"}), "likec4.outbound", "LikeC4 output omits the outbound relationship")
	evidence := regexp.MustCompile(`(?m)\bapp\.py:\d+(?:-\d+)?\b`)
	score.require(len(evidence.FindAll(data, -1)) >= 3, "likec4.evidence", "LikeC4 output has fewer than three source citations")
	return score
}

func ValidateOpenAPI(ctx context.Context, data []byte) error {
	executable, err := exec.LookPath("vacuum")
	if err != nil {
		return errors.New("Vacuum executable is unavailable")
	}
	stdout := &boundedBuffer{limit: maxValidatorOutput}
	command := exec.CommandContext(ctx, executable, "spectral-report", "-i", "-o")
	command.Stdin = bytes.NewReader(data)
	command.Stdout = stdout
	command.Stderr = io.Discard
	if err := command.Run(); err != nil {
		var exit *exec.ExitError
		if !errors.As(err, &exit) || exit.ExitCode() != 1 {
			return errors.New("Vacuum validation could not be executed")
		}
	}
	if stdout.overflow {
		return errors.New("Vacuum validation output exceeded the bound")
	}
	var issues []map[string]any
	if json.Unmarshal(stdout.Bytes(), &issues) != nil {
		return errors.New("Vacuum validation returned invalid JSON")
	}
	for _, issue := range issues {
		severity, ok := jsonNumber(issue["severity"])
		if ok && (severity == 0 || severity == 1) {
			return errors.New("Vacuum validation reported a serious issue")
		}
	}
	return nil
}

func ValidateLikeC4(ctx context.Context, data []byte) error {
	executable, err := exec.LookPath("likec4")
	if err != nil {
		return errors.New("LikeC4 executable is unavailable")
	}
	root, err := os.MkdirTemp("", "contractor-likec4-eval-")
	if err != nil {
		return errors.New("LikeC4 validation workspace could not be created")
	}
	defer os.RemoveAll(root)
	file := filepath.Join(root, "main.c4")
	if err := os.WriteFile(file, data, 0o600); err != nil {
		return errors.New("LikeC4 validation input could not be created")
	}
	stdout := &boundedBuffer{limit: maxValidatorOutput}
	command := exec.CommandContext(
		ctx, executable, "validate", "--json", "--no-layout", "--file", file, root,
	)
	command.Dir = root
	command.Stdin = nil
	command.Stdout = stdout
	command.Stderr = io.Discard
	command.Env = []string{
		"PATH=" + os.Getenv("PATH"), "LANG=C.UTF-8", "CI=1", "NO_COLOR=1", "NO_UPDATE_NOTIFIER=1",
	}
	if err := command.Run(); err != nil {
		var exit *exec.ExitError
		if !errors.As(err, &exit) || exit.ExitCode() != 1 {
			return errors.New("LikeC4 validation could not be executed")
		}
	}
	if stdout.overflow {
		return errors.New("LikeC4 validation output exceeded the bound")
	}
	parsed, err := extractJSON(stdout.Bytes())
	if err != nil {
		return errors.New("LikeC4 validation returned invalid JSON")
	}
	var diagnostics []any
	switch value := parsed.(type) {
	case []any:
		diagnostics = value
	case map[string]any:
		var ok bool
		diagnostics, ok = value["errors"].([]any)
		if !ok {
			return errors.New("LikeC4 validation returned an unexpected JSON shape")
		}
		if valid, exists := value["valid"].(bool); exists && !valid {
			return errors.New("LikeC4 validation reported an invalid document")
		}
	default:
		return errors.New("LikeC4 validation returned an unexpected JSON shape")
	}
	if len(diagnostics) > 0 {
		return errors.New("LikeC4 validation reported DSL errors")
	}
	return nil
}

func MergeScores(scores ...Score) Score {
	var result Score
	for _, score := range scores {
		result.Failures = append(result.Failures, score.Failures...)
	}
	sort.Slice(result.Failures, func(i, j int) bool {
		return result.Failures[i].Code < result.Failures[j].Code
	})
	return result
}

func stringMap(value any) (map[string]any, bool) {
	result, ok := value.(map[string]any)
	return result, ok
}

func nestedMap(root map[string]any, path ...string) map[string]any {
	var current any = root
	for _, item := range path {
		object, ok := stringMap(current)
		if !ok {
			return nil
		}
		current = object[item]
	}
	result, _ := stringMap(current)
	return result
}

func nonemptyList(value any) bool {
	items, ok := value.([]any)
	return ok && len(items) > 0
}

func nonemptyStringList(value any) bool {
	items, ok := value.([]any)
	if !ok || len(items) == 0 {
		return false
	}
	for _, item := range items {
		text, ok := item.(string)
		if !ok || strings.TrimSpace(text) == "" {
			return false
		}
	}
	return true
}

func containsAny(value string, terms ...string) bool {
	for _, term := range terms {
		if strings.Contains(value, term) {
			return true
		}
	}
	return false
}

func anyLine(lines []string, left, right []string) bool {
	for _, line := range lines {
		if containsAny(line, left...) && containsAny(line, right...) {
			return true
		}
	}
	return false
}

func jsonNumber(value any) (int, bool) {
	switch item := value.(type) {
	case float64:
		return int(item), item == float64(int(item))
	case json.Number:
		integer, err := item.Int64()
		return int(integer), err == nil
	default:
		return 0, false
	}
}

func extractJSON(data []byte) (any, error) {
	decoder := json.NewDecoder(bytes.NewReader(bytes.TrimSpace(data)))
	decoder.UseNumber()
	var value any
	if decoder.Decode(&value) == nil {
		return value, nil
	}
	for index, character := range data {
		if character != '{' && character != '[' {
			continue
		}
		decoder = json.NewDecoder(bytes.NewReader(data[index:]))
		decoder.UseNumber()
		if decoder.Decode(&value) == nil {
			return value, nil
		}
	}
	return nil, fmt.Errorf("no JSON value")
}

type boundedBuffer struct {
	data     []byte
	limit    int
	overflow bool
}

func (b *boundedBuffer) Write(value []byte) (int, error) {
	written := len(value)
	remaining := b.limit + 1 - len(b.data)
	if remaining > 0 {
		if len(value) > remaining {
			value = value[:remaining]
		}
		b.data = append(b.data, value...)
	}
	if len(b.data) > b.limit {
		b.overflow = true
		b.data = b.data[:b.limit]
	}
	return written, nil
}

func (b *boundedBuffer) Bytes() []byte { return append([]byte(nil), b.data...) }
