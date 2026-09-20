package scanplan

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"sort"
	"strings"

	"github.com/grauwolf32/contractor/internal/contracts"
)

type preparer struct {
	root            map[string]any
	options         Options
	refs            int
	exhausted       bool
	schemaNodes     int
	schemaExhausted bool
	gaps            []contracts.PreparationGap
	gapKeys         map[contracts.PreparationGap]bool
	gapExhausted    bool
}

// Prepare consumes already-authorized exact artifact bytes. It performs no I/O.
func Prepare(data []byte, mediaType string, source contracts.ArtifactRef, options Options) (contracts.HTTPRequestSet, error) {
	empty := contracts.HTTPRequestSet{}
	if source.ValidateExact() != nil {
		return empty, failure("invalid_source_ref")
	}
	// Retain the exact revision value without aliasing a caller-owned pointer.
	revision := *source.Revision
	source.Revision = &revision
	root, err := parseDocument(data, mediaType)
	if err != nil {
		return empty, err
	}
	version, _ := root["openapi"].(string)
	if !supportedOpenAPIVersion(version) {
		return empty, failure("unsupported_openapi_version")
	}
	paths, ok := root["paths"].(map[string]any)
	if !ok {
		return empty, failure("invalid_document")
	}
	options, err = normalizeOptions(options)
	if err != nil {
		return empty, err
	}
	p := &preparer{root: root, options: options, gaps: []contracts.PreparationGap{}, gapKeys: map[contracts.PreparationGap]bool{}}
	set := contracts.HTTPRequestSet{
		SchemaVersion: 1,
		Source:        contracts.RequestSetSource{Artifact: source, ContentDigest: digest(data)},
		Requests:      []contracts.RequestSetEntry{},
	}
	basis, err := contracts.MarshalPrivateCanonical(struct {
		PolicyVersion int                        `json:"policyVersion"`
		Source        contracts.RequestSetSource `json:"source"`
		MediaType     string                     `json:"mediaType"`
		Options       Options                    `json:"options"`
	}{2, set.Source, mediaType, options})
	if err != nil {
		return empty, failure("invalid_options")
	}
	set.PreparationDigest = digest(basis)
	seenBindings := map[string]bool{}
	byDigest := map[string]int{}
	outputBytes := 0
	for _, path := range keys(paths) {
		if strings.HasPrefix(path, "x-") {
			continue
		}
		if !strings.HasPrefix(path, "/") || len(path) > 8192 {
			return empty, failure("invalid_document")
		}
		pointer := "#/paths/" + escapePointer(path)
		if len(pointer)+8 > 8192 {
			return empty, failure("invalid_document")
		}
		item, code := p.resolve(paths[path], nil)
		if code != "" {
			p.gap(pointer, code)
			continue
		}
		for _, key := range keys(item) {
			if !strings.HasPrefix(key, "x-") && !knownPathField(key) {
				p.gap(pointer, "unsupported_path_item_field")
			}
		}
		for _, method := range []string{"delete", "get", "head", "options", "patch", "post", "put", "trace"} {
			raw, exists := item[method]
			if !exists {
				continue
			}
			set.Coverage.Operations++
			if set.Coverage.Operations > MaxOperations {
				return empty, failure("operation_limit_exceeded")
			}
			opPointer := pointer + "/" + method
			seenBindings[opPointer] = true
			if method == "trace" {
				p.gap(opPointer, "unsupported_method")
				continue
			}
			op, ok := raw.(map[string]any)
			if !ok {
				p.gap(opPointer, "invalid_operation")
				continue
			}
			if _, exists := op["$ref"]; exists {
				p.gap(opPointer, "unsupported_operation_reference")
				continue
			}
			unknown := false
			for _, key := range keys(op) {
				if !strings.HasPrefix(key, "x-") && !knownOperationField(key) {
					unknown = true
				}
			}
			if unknown {
				p.gap(opPointer, "unsupported_operation_field")
			}
			if _, exists := op["callbacks"]; exists {
				p.gap(opPointer, "unsupported_callbacks")
			}
			request, code := p.operation(path, strings.ToUpper(method), item, op, opPointer)
			if code != "" {
				p.gap(opPointer, code)
				continue
			}
			contentDigest, err := contracts.RequestContentDigest(request)
			if err != nil {
				p.gap(opPointer, "invalid_prepared_request")
				continue
			}
			if index, exists := byDigest[contentDigest]; exists {
				set.Requests[index].Origins = append(set.Requests[index].Origins, contracts.RequestOrigin{Pointer: opPointer})
				encoded, _ := contracts.MarshalPrivateCanonical(contracts.RequestOrigin{Pointer: opPointer})
				outputBytes += len(encoded)
			} else {
				if len(set.Requests) >= options.MaxRequests {
					p.gap(opPointer, "request_limit_exceeded")
					continue
				}
				byDigest[contentDigest] = len(set.Requests)
				set.Requests = append(set.Requests, contracts.RequestSetEntry{ID: "request-" + strings.TrimPrefix(contentDigest, "sha256:"), ContentDigest: contentDigest, Request: request, Origins: []contracts.RequestOrigin{{Pointer: opPointer}}})
				encoded, _ := contracts.MarshalPrivateCanonical(set.Requests[len(set.Requests)-1])
				outputBytes += len(encoded)
			}
			if outputBytes > contracts.MaxHTTPRequestSetBytes {
				return empty, failure("request_set_limit_exceeded")
			}
			set.Coverage.Prepared++
		}
	}
	if p.exhausted {
		return empty, failure("reference_limit_exceeded")
	}
	if p.schemaExhausted {
		return empty, failure("schema_work_limit_exceeded")
	}
	for pointer := range options.Operations {
		if !seenBindings[pointer] {
			return empty, failure("unknown_operation_binding")
		}
	}
	if _, exists := root["webhooks"]; exists {
		p.gap("#", "unsupported_webhooks")
	}
	if p.gapExhausted {
		return empty, failure("gap_limit_exceeded")
	}
	sort.Slice(set.Requests, func(i, j int) bool { return set.Requests[i].ID < set.Requests[j].ID })
	for i := range set.Requests {
		sort.Slice(set.Requests[i].Origins, func(a, b int) bool { return set.Requests[i].Origins[a].Pointer < set.Requests[i].Origins[b].Pointer })
	}
	sort.Slice(p.gaps, func(i, j int) bool {
		if p.gaps[i].Pointer == p.gaps[j].Pointer {
			return p.gaps[i].Code < p.gaps[j].Code
		}
		return p.gaps[i].Pointer < p.gaps[j].Pointer
	})
	set.Gaps = []contracts.PreparationGap{}
	for _, gap := range p.gaps {
		if len(set.Gaps) == 0 || set.Gaps[len(set.Gaps)-1] != gap {
			set.Gaps = append(set.Gaps, gap)
		}
	}
	set.Coverage.Skipped = set.Coverage.Operations - set.Coverage.Prepared
	set.Coverage.Complete = set.Coverage.Skipped == 0 && len(set.Gaps) == 0
	if _, err := contracts.MarshalHTTPRequestSet(set); err != nil {
		return empty, failure("request_set_limit_or_contract_error")
	}
	return set, nil
}

func supportedOpenAPIVersion(version string) bool {
	parts := strings.Split(version, ".")
	if len(parts) != 3 || parts[0] != "3" || (parts[1] != "0" && parts[1] != "1") || parts[2] == "" {
		return false
	}
	for _, digit := range parts[2] {
		if digit < '0' || digit > '9' {
			return false
		}
	}
	return true
}

func normalizeOptions(options Options) (Options, error) {
	if !validateOptionsValues(options) {
		return Options{}, failure("invalid_options")
	}
	encoded, err := json.Marshal(options)
	if err != nil || len(encoded) > MaxOptionsBytes {
		return Options{}, failure("invalid_options")
	}
	// The same bounded data model prevents unsafe numbers, excessive depth and
	// custom Go values from acquiring a different meaning during canonicalization.
	if _, err = parseDocument(encoded, "application/json"); err != nil {
		return Options{}, failure("invalid_options")
	}
	var normalized Options
	if err = json.Unmarshal(encoded, &normalized); err != nil {
		return Options{}, failure("invalid_options")
	}
	options = normalized
	if options.MaxRequests == 0 {
		options.MaxRequests = MaxRequests
	}
	if options.MaxRequests < 1 || options.MaxRequests > MaxRequests {
		return Options{}, failure("invalid_options")
	}
	if options.ServerVariables == nil {
		options.ServerVariables = map[string]string{}
	}
	if options.Authentication == nil {
		options.Authentication = map[string]contracts.SecretString{}
	}
	if options.Operations == nil {
		options.Operations = map[string]OperationInput{}
	}
	for key, op := range options.Operations {
		parameters := map[string]any{}
		for name, value := range op.Parameters {
			if strings.HasPrefix(name, "header:") {
				name = strings.ToLower(name)
			}
			if _, exists := parameters[name]; exists {
				return Options{}, failure("duplicate_parameter_binding")
			}
			parameters[name] = value
		}
		op.Parameters = parameters
		options.Operations[key] = op
	}
	return options, nil
}

func (p *preparer) gap(pointer, code string) {
	gap := contracts.PreparationGap{Pointer: pointer, Code: code}
	if p.gapKeys[gap] {
		return
	}
	if len(p.gaps) >= contracts.MaxHTTPRequestSetGaps {
		p.gapExhausted = true
		return
	}
	p.gapKeys[gap] = true
	p.gaps = append(p.gaps, gap)
}
func digest(data []byte) string {
	sum := sha256.Sum256(data)
	return "sha256:" + hex.EncodeToString(sum[:])
}
func keys[V any](value map[string]V) []string {
	result := make([]string, 0, len(value))
	for key := range value {
		result = append(result, key)
	}
	sort.Strings(result)
	return result
}
func escapePointer(value string) string {
	return strings.ReplaceAll(strings.ReplaceAll(value, "~", "~0"), "/", "~1")
}

func knownPathField(key string) bool {
	switch key {
	case "summary", "description", "servers", "parameters", "delete", "get", "head", "options", "patch", "post", "put", "trace":
		return true
	default:
		return false
	}
}

func knownOperationField(key string) bool {
	switch key {
	case "tags", "summary", "description", "externalDocs", "operationId", "parameters", "requestBody", "responses", "callbacks", "deprecated", "security", "servers":
		return true
	default:
		return false
	}
}
