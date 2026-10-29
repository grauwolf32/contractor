package projectsitemap

import (
	"context"
	"fmt"
	"maps"
	"net/url"
	"slices"
	"strings"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/scanplan"
)

// JoinOptions identifies the API contract and explicit deployment base URL.
// Contract identity is independent of artifact binding names or producer Runs.
type JoinOptions struct {
	Origin     string `json:"origin"`
	Contract   string `json:"contract"`
	Deployment string `json:"deployment,omitempty"`
}

type JoinDifference struct {
	Kind         string   `json:"kind"`
	Method       string   `json:"method"`
	Path         string   `json:"path"`
	Missing      []string `json:"missing,omitempty"`
	Extra        []string `json:"extra,omitempty"`
	Statuses     []int    `json:"statuses,omitempty"`
	OmittedNames int      `json:"omittedNames,omitempty"`
}

// JoinSummary compares retained behavior with declarations. Counts describe
// members/operations, never physical requests or security coverage.
type JoinSummary struct {
	Declared           int              `json:"declared"`
	Observed           int              `json:"observed"`
	Matched            int              `json:"matched"`
	DeclaredOnly       int              `json:"declaredOnly"`
	ObservedOnly       int              `json:"observedOnly"`
	Ambiguous          int              `json:"ambiguous"`
	Unresolved         int              `json:"unresolvedDeclarations"`
	Differences        []JoinDifference `json:"differences"`
	OmittedDifferences int              `json:"omittedDifferences"`
	Limitations        []string         `json:"limitations"`
}

type joinedContract struct {
	document  Document
	payload   []byte
	mutation  Mutation
	authority Authority
	summary   JoinSummary
}

type contractOperation struct {
	rule     Rule
	query    []string
	statuses []string
}

func joinOpenAPI(ctx context.Context, previous Document, data []byte, mediaType string, input contracts.ExactArtifact, options JoinOptions) (joinedContract, error) {
	if !contracts.ValidName(options.Contract) || (options.Deployment != "" && !contracts.ValidResourceID(options.Deployment)) {
		return joinedContract{}, ErrInvalid
	}
	base, err := url.Parse(options.Origin)
	if err != nil || base.User != nil || base.RawQuery != "" || base.Fragment != "" || strings.ContainsAny(options.Origin, "{}") {
		return joinedContract{}, ErrInvalid
	}
	origin, _, _, err := projectURL(options.Origin)
	if err != nil {
		return joinedContract{}, err
	}
	basePath := strings.TrimSuffix(base.EscapedPath(), "/")
	root, err := scanplan.ParseDocument(data, mediaType)
	if err != nil {
		return joinedContract{}, ErrInvalid
	}
	version, _ := root["openapi"].(string)
	paths, ok := root["paths"].(map[string]any)
	if !ok || !scanplan.SupportsOpenAPIVersion(version) {
		return joinedContract{}, ErrInvalid
	}
	id := key("openapi", options.Contract, options.Deployment)
	authority := Authority{SourceID: id, Kind: "openapi", Deployment: options.Deployment}
	update := &SourceUpdate{SourceID: id, Artifact: &input, Rules: []Rule{}}
	for _, source := range previous.Sources {
		if source.Provenance.SourceID == id {
			expected := source
			update.Expected = &expected
		}
	}
	operations := []contractOperation{}
	budget := &scanplan.ReferenceBudget{}
	for _, path := range slices.Sorted(maps.Keys(paths)) {
		if err := ctx.Err(); err != nil {
			return joinedContract{}, err
		}
		if strings.HasPrefix(path, "x-") {
			continue
		}
		if !strings.HasPrefix(path, "/") || strings.ContainsAny(path, "?#\\") {
			return joinedContract{}, ErrInvalid
		}
		item, code := scanplan.ResolveLocalObject(root, paths[path], budget)
		if code != "" {
			return joinedContract{}, ErrInvalid
		}
		for _, method := range []string{"delete", "get", "head", "options", "patch", "post", "put", "trace"} {
			raw, found := item[method]
			if !found {
				continue
			}
			op, code := scanplan.ResolveLocalObject(root, raw, budget)
			if code != "" {
				return joinedContract{}, ErrInvalid
			}
			pointer := "#/paths/" + strings.ReplaceAll(strings.ReplaceAll(path, "~", "~0"), "/", "~1") + "/" + method
			rule := Rule{SourceID: id, Origin: origin, Method: strings.ToUpper(method), Path: basePath + path, Pointer: pointer, Parameters: []string{}, Structural: []StructuralParameter{}}
			query, names, err := contractParameters(root, item, op, budget)
			if err != nil {
				return joinedContract{}, err
			}
			rule.Parameters = names
			if validateRule(rule) != nil {
				return joinedContract{}, ErrInvalid
			}
			statuses, err := contractStatuses(root, op, budget)
			if err != nil {
				return joinedContract{}, err
			}
			operations = append(operations, contractOperation{rule: rule, query: query, statuses: statuses})
			update.Rules = append(update.Rules, rule)
			if len(update.Rules) > MaxRules {
				return joinedContract{}, ErrInvalid
			}
		}
	}
	mutation := Mutation{Operation: "replace-source", Source: update}
	if len(jsonBytes(mutation)) > MaxBatchBytes {
		return joinedContract{}, ErrInvalid
	}
	joined, payload, err := Reduce(previous, mutation, authority)
	if err != nil {
		return joinedContract{}, err
	}
	summary, err := compareContract(ctx, joined, operations, origin)
	if err != nil {
		return joinedContract{}, err
	}
	return joinedContract{document: joined, payload: payload, mutation: mutation, authority: authority, summary: summary}, nil
}

func contractParameters(root, item, operation map[string]any, budget *scanplan.ReferenceBudget) ([]string, []string, error) {
	parameters := map[string]string{}
	for _, object := range []map[string]any{item, operation} {
		raw, exists := object["parameters"]
		if !exists {
			continue
		}
		values, ok := raw.([]any)
		if !ok || len(values) > MaxBatch {
			return nil, nil, ErrInvalid
		}
		seen := map[string]bool{}
		for _, value := range values {
			parameter, code := scanplan.ResolveLocalObject(root, value, budget)
			if code != "" {
				return nil, nil, ErrInvalid
			}
			name, _ := parameter["name"].(string)
			location, _ := parameter["in"].(string)
			if !validLabel(name) || !slices.Contains([]string{"path", "query", "header", "cookie"}, location) {
				return nil, nil, ErrInvalid
			}
			identity := location + ":" + name
			if location == "header" {
				identity = strings.ToLower(identity)
			}
			if seen[identity] {
				return nil, nil, ErrInvalid
			}
			seen[identity] = true
			parameters[identity] = name
		}
	}
	names, query := []string{}, []string{}
	for identity, name := range parameters {
		names = append(names, name)
		if strings.HasPrefix(identity, "query:") {
			query = append(query, name)
		}
	}
	slices.Sort(names)
	names = slices.Compact(names)
	slices.Sort(query)
	if len(names) > MaxBatch {
		return nil, nil, ErrInvalid
	}
	return query, names, nil
}

func contractStatuses(root, operation map[string]any, budget *scanplan.ReferenceBudget) ([]string, error) {
	raw, exists := operation["responses"]
	if !exists {
		return []string{}, nil
	}
	responses, ok := raw.(map[string]any)
	if !ok {
		return nil, ErrInvalid
	}
	statuses := []string{}
	for _, status := range slices.Sorted(maps.Keys(responses)) {
		if strings.HasPrefix(status, "x-") {
			continue
		}
		if status != "default" && !(len(status) == 3 && status[0] >= '1' && status[0] <= '5' && ((status[1] == 'X' && status[2] == 'X') || (status[1] >= '0' && status[1] <= '9' && status[2] >= '0' && status[2] <= '9'))) {
			return nil, ErrInvalid
		}
		if _, code := scanplan.ResolveLocalObject(root, responses[status], budget); code != "" {
			return nil, ErrInvalid
		}
		statuses = append(statuses, status)
	}
	return statuses, nil
}

func compareContract(ctx context.Context, document Document, operations []contractOperation, origin string) (JoinSummary, error) {
	summary := JoinSummary{Declared: len(operations), Differences: []JoinDifference{}, Limitations: []string{"retained-members-only", "observed-media-types-not-recorded", "body-shapes-and-security-not-compared", "declarations-do-not-prove-reachability"}}
	declared := map[string]contractOperation{}
	seen := map[string]bool{}
	uncertain := map[string]bool{}
	rules := make([]Rule, 0, len(operations))
	for _, operation := range operations {
		rules = append(rules, operation.rule)
		declared[operation.rule.Method+" "+operation.rule.Path] = operation
	}
	add := func(difference JoinDifference) {
		if len(summary.Differences) < 32 {
			summary.Differences = append(summary.Differences, difference)
		} else {
			summary.OmittedDifferences++
		}
	}
	for _, group := range document.Groups {
		if err := ctx.Err(); err != nil {
			return JoinSummary{}, err
		}
		if group.Origin != origin {
			continue
		}
		for _, member := range group.Members {
			if len(member.Summaries) == 0 {
				continue
			}
			summary.Observed++
			route, ambiguous := routeFor(rules, group.Origin, group.Method, member.Path)
			if ambiguous {
				summary.Ambiguous++
				for _, operation := range operations {
					candidate, _ := routeFor([]Rule{operation.rule}, group.Origin, group.Method, member.Path)
					if candidate == operation.rule.Path {
						uncertain[operation.rule.Method+" "+operation.rule.Path] = true
					}
				}
				add(JoinDifference{Kind: "ambiguous", Method: group.Method, Path: member.Path})
				continue
			}
			key := group.Method + " " + route
			operation, found := declared[key]
			if !found {
				summary.ObservedOnly++
				add(JoinDifference{Kind: "observed-only", Method: group.Method, Path: member.Path})
				continue
			}
			seen[key] = true
			summary.Matched++
			parameters := []string{}
			statuses := []int{}
			for _, behavior := range member.Summaries {
				parameters = append(parameters, behavior.Parameters...)
				if behavior.Outcome == "response" && len(operation.statuses) > 0 && !slices.Contains(operation.statuses, "default") && !slices.Contains(operation.statuses, fmt.Sprint(behavior.Status)) && !slices.Contains(operation.statuses, fmt.Sprintf("%dXX", behavior.Status/100)) {
					statuses = append(statuses, behavior.Status)
				}
			}
			slices.Sort(parameters)
			parameters = slices.Compact(parameters)
			slices.Sort(statuses)
			statuses = slices.Compact(statuses)
			missing, extra := []string{}, []string{}
			for _, name := range operation.query {
				if !slices.Contains(parameters, name) {
					missing = append(missing, name)
				}
			}
			for _, name := range parameters {
				if !slices.Contains(operation.query, name) {
					extra = append(extra, name)
				}
			}
			if len(missing)+len(extra) > 0 {
				add(JoinDifference{Kind: "parameters", Method: group.Method, Path: member.Path, Missing: missing[:min(len(missing), MaxBatch)], Extra: extra[:min(len(extra), MaxBatch)], OmittedNames: max(0, len(missing)-MaxBatch) + max(0, len(extra)-MaxBatch)})
			}
			if len(statuses) > 0 {
				add(JoinDifference{Kind: "statuses", Method: group.Method, Path: member.Path, Statuses: statuses})
			}
		}
	}
	for _, operation := range operations {
		if !seen[operation.rule.Method+" "+operation.rule.Path] {
			if uncertain[operation.rule.Method+" "+operation.rule.Path] {
				summary.Unresolved++
				add(JoinDifference{Kind: "unresolved-declaration", Method: operation.rule.Method, Path: operation.rule.Path})
			} else {
				summary.DeclaredOnly++
				add(JoinDifference{Kind: "declared-only", Method: operation.rule.Method, Path: operation.rule.Path})
			}
		}
	}
	for len(jsonBytes(summary)) > MaxBatchBytes/2 && len(summary.Differences) > 0 {
		summary.Differences = summary.Differences[:len(summary.Differences)-1]
		summary.OmittedDifferences++
	}
	return summary, ctx.Err()
}
