package auditdomain

import (
	"fmt"
	"maps"
	"net/url"
	"path"
	"regexp"
	"slices"
	"sort"
	"strconv"
	"strings"

	"github.com/grauwolf32/contractor/internal/scanplan"
)

var openAPIMethodOrder = []string{"get", "put", "post", "delete", "options", "head", "patch", "trace"}
var openAPIVersionPattern = regexp.MustCompile(`^3\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?$`)

// parseOpenAPIDocument uses the same syntax and source limits as scan
// preparation. Checklist parsing keeps its separate Audit-internal limits.
func parseOpenAPIDocument(data []byte, mediaType string) (map[string]any, error) {
	root, err := scanplan.ParseDocument(data, normalizedMediaType(mediaType))
	if err != nil {
		return nil, invalid(CodeInvalid, "document")
	}
	return root, nil
}

type openAPISource struct {
	raw        []byte
	mediaType  string
	entryPoint string
	documents  map[string]map[string]any
}

type openAPIOperation struct {
	path        string
	method      string
	operationID string
	resolved    map[string]any
	gaps        []string
}

func openAPISourceFromPackage(payload []byte) (openAPISource, error) {
	validated, err := ValidatePackage(payload)
	if err != nil {
		return openAPISource{}, err
	}
	if validated.Manifest.Kind != PackageKindOpenAPISource {
		return openAPISource{}, invalid(CodeInventoryInvalid, "package.kind")
	}
	documents := make(map[string]map[string]any)
	for _, member := range validated.Members() {
		metadata := member.Metadata()
		switch normalizedMediaType(metadata.MediaType) {
		case "application/json", "application/yaml", "application/x-yaml", "text/yaml":
			document, parseErr := parseOpenAPIDocument(member.Data(), metadata.MediaType)
			if parseErr != nil {
				return openAPISource{}, invalid(CodeInventoryInvalid, "package.document")
			}
			documents[metadata.Path] = document
		}
	}
	if _, exists := documents[validated.Manifest.EntryPoint]; !exists {
		return openAPISource{}, invalid(CodeInventoryInvalid, "package.entrypoint")
	}
	return openAPISource{
		raw: payload, mediaType: PackageMediaType,
		entryPoint: validated.Manifest.EntryPoint, documents: documents,
	}, nil
}

func openAPISubjects(source openAPISource, maxItems int) (inventoryBasis, []inventorySubject, error) {
	root := source.documents[source.entryPoint]
	for _, document := range source.documents {
		if hasUnsupportedReferenceKeyword(document) {
			return inventoryBasis{}, nil, invalid(CodeReferenceInvalid, "document.reference_semantics")
		}
	}
	version, err := stringValue(root["openapi"], "openapi")
	if err != nil || !openAPIVersionPattern.MatchString(version) {
		return inventoryBasis{}, nil, invalid(CodeInventoryInvalid, "openapi")
	}
	paths, err := mapValue(root["paths"], "paths")
	if err != nil {
		return inventoryBasis{}, nil, err
	}
	resolver := &openAPIResolver{documents: source.documents}
	globalGaps := make([]string, 0)
	if webhooks, exists := root["webhooks"]; exists {
		webhookMap, ok := webhooks.(map[string]any)
		if !ok {
			return inventoryBasis{}, nil, invalid(CodeInventoryInvalid, "webhooks")
		}
		for _, name := range slices.Sorted(maps.Keys(webhookMap)) {
			globalGaps = append(globalGaps, formatGap("unsupported-webhook", name))
		}
	}

	operations := make([]openAPIOperation, 0)
	for _, pathTemplate := range slices.Sorted(maps.Keys(paths)) {
		if strings.HasPrefix(strings.ToLower(pathTemplate), "x-") {
			continue
		}
		if !strings.HasPrefix(pathTemplate, "/") || strings.ContainsRune(pathTemplate, 0) {
			return inventoryBasis{}, nil, invalid(CodeInventoryInvalid, "paths")
		}
		pathItem, ok := paths[pathTemplate].(map[string]any)
		if !ok {
			return inventoryBasis{}, nil, invalid(CodeInventoryInvalid, "paths")
		}
		pathItem, pathItemDocument, err := resolver.resolveReferenceObject(pathItem, source.entryPoint, nil, 1)
		if err != nil {
			return inventoryBasis{}, nil, err
		}
		for _, method := range openAPIMethodOrder {
			rawOperation, exists := pathItem[method]
			if !exists {
				continue
			}
			operation, ok := rawOperation.(map[string]any)
			if !ok {
				return inventoryBasis{}, nil, invalid(CodeInventoryInvalid, "paths.operation")
			}
			built, buildErr := resolver.buildOperation(root, pathItem, operation, source.entryPoint, pathItemDocument, pathTemplate, method)
			if buildErr != nil {
				return inventoryBasis{}, nil, buildErr
			}
			operations = append(operations, built)
			if len(operations) > maxItems {
				return inventoryBasis{}, nil, invalid(CodeLimitExceeded, "operations")
			}
		}
	}
	for documentPath, document := range source.documents {
		resolver.collectUnselectedRemoteGaps(document, documentPath, "", &globalGaps, 1)
	}
	for _, operation := range operations {
		globalGaps = append(globalGaps, operation.gaps...)
	}
	if len(globalGaps) > MaximumCoverageValues {
		return inventoryBasis{}, nil, invalid(CodeLimitExceeded, "gaps")
	}
	globalGaps = sortedUnique(globalGaps)

	basisSubjects := make([]map[string]any, 0, len(operations))
	for _, operation := range operations {
		basisSubjects = append(basisSubjects, map[string]any{
			"path": operation.path, "method": operation.method,
			"resolved": operation.resolved, "gaps": operation.gaps,
		})
	}
	basis := inventoryBasis{
		Schema: InventoryBasisSchema, Kind: "openapi-operations", Subjects: basisSubjects, Gaps: globalGaps,
	}
	canonical, err := canonicalInventoryJSON(basis)
	if err != nil {
		return inventoryBasis{}, nil, invalid(CodeInventoryInvalid, "inventory")
	}
	canonicalDigest := DigestBytes(canonical)
	subjects := make([]inventorySubject, 0, len(operations))
	for _, operation := range operations {
		operationKey, err := openAPIOperationKey(canonicalDigest, operation.path, operation.method)
		if err != nil {
			return inventoryBasis{}, nil, err
		}
		task := &OperationTask{
			Path: operation.path, Method: operation.method, OperationID: operation.operationID,
			Resolved: cloneMap(operation.resolved), Gaps: slices.Clone(operation.gaps),
		}
		subjects = append(subjects, inventorySubject{
			itemKey: operationKey, kind: "operation-trace", subjectKey: operationKey,
			operation: task, requested: []string{"operation-resolution"}, gaps: slices.Clone(operation.gaps),
		})
	}
	return basis, subjects, nil
}

type openAPIResolver struct {
	documents   map[string]map[string]any
	resolutions int
}

func (r *openAPIResolver) buildOperation(
	root map[string]any,
	pathItem map[string]any,
	operation map[string]any,
	rootDocumentPath, pathItemDocumentPath, pathTemplate, method string,
) (openAPIOperation, error) {
	r.resolutions = 0
	resolvedReference, operationDocumentPath, err := r.resolveReferenceObject(operation, pathItemDocumentPath, nil, 1)
	if err != nil {
		return openAPIOperation{}, err
	}
	operation = resolvedReference
	result := openAPIOperation{path: pathTemplate, method: method, gaps: []string{}}
	if operationID, exists := operation["operationId"]; exists {
		text, ok := operationID.(string)
		if !ok || validateOpenAPISourceText(text, "operationId", false) != nil {
			return openAPIOperation{}, invalid(CodeInventoryInvalid, "operationId")
		}
		result.operationID = text
	}
	selectedOperation := cloneMap(operation)
	delete(selectedOperation, "operationId")
	if callbacks, exists := selectedOperation["callbacks"]; exists {
		callbackMap, ok := callbacks.(map[string]any)
		if !ok {
			return openAPIOperation{}, invalid(CodeInventoryInvalid, "callbacks")
		}
		for _, name := range slices.Sorted(maps.Keys(callbackMap)) {
			result.gaps = append(result.gaps, formatGap("unsupported-callback", method+" "+pathTemplate+" "+name))
		}
		delete(selectedOperation, "callbacks")
	}
	resolvedOperation, err := r.resolveValue(selectedOperation, operationDocumentPath, make(map[string]bool), 1)
	if err != nil {
		return openAPIOperation{}, err
	}
	operationMap, ok := resolvedOperation.(map[string]any)
	if !ok {
		return openAPIOperation{}, invalid(CodeInventoryInvalid, "operation")
	}
	resolved := map[string]any{"operation": operationMap}
	for _, field := range []string{"parameters", "servers"} {
		if value, exists := pathItem[field]; exists {
			resolvedValue, resolveErr := r.resolveValue(value, pathItemDocumentPath, make(map[string]bool), 1)
			if resolveErr != nil {
				return openAPIOperation{}, resolveErr
			}
			resolved["path_"+field] = resolvedValue
		}
	}
	security, exists := operationMap["security"]
	if !exists {
		security = root["security"]
	}
	if security != nil {
		resolvedSecurity, schemes, securityErr := r.resolveSecurity(root, security, rootDocumentPath)
		if securityErr != nil {
			return openAPIOperation{}, securityErr
		}
		resolved["security"] = resolvedSecurity
		resolved["security_schemes"] = schemes
	}
	result.resolved = resolved
	result.gaps = sortedUnique(result.gaps)
	return result, nil
}

func (r *openAPIResolver) resolveSecurity(root map[string]any, raw any, documentPath string) (any, map[string]any, error) {
	resolved, err := r.resolveValue(raw, documentPath, make(map[string]bool), 1)
	if err != nil {
		return nil, nil, err
	}
	items, ok := resolved.([]any)
	if !ok {
		return nil, nil, invalid(CodeInventoryInvalid, "security")
	}
	names := make(map[string]struct{})
	for _, item := range items {
		requirement, ok := item.(map[string]any)
		if !ok {
			return nil, nil, invalid(CodeInventoryInvalid, "security")
		}
		for name := range requirement {
			names[name] = struct{}{}
		}
	}
	if len(names) == 0 {
		return resolved, map[string]any{}, nil
	}
	components, ok := root["components"].(map[string]any)
	if !ok {
		return nil, nil, invalid(CodeReferenceInvalid, "security_schemes")
	}
	securitySchemes, ok := components["securitySchemes"].(map[string]any)
	if !ok {
		return nil, nil, invalid(CodeReferenceInvalid, "security_schemes")
	}
	result := make(map[string]any, len(names))
	ordered := make([]string, 0, len(names))
	for name := range names {
		ordered = append(ordered, name)
	}
	sort.Strings(ordered)
	for _, name := range ordered {
		scheme, exists := securitySchemes[name]
		if !exists {
			return nil, nil, invalid(CodeReferenceInvalid, "security_schemes")
		}
		resolvedScheme, resolveErr := r.resolveValue(scheme, documentPath, make(map[string]bool), 1)
		if resolveErr != nil {
			return nil, nil, resolveErr
		}
		result[name] = resolvedScheme
	}
	return resolved, result, nil
}

// resolveReferenceObject resolves only the Path Item's own reference chain.
// Its operation children are resolved independently so unsupported callbacks
// can be removed before dependency traversal.
func (r *openAPIResolver) resolveReferenceObject(value map[string]any, documentPath string, stack map[string]bool, depth int) (map[string]any, string, error) {
	if depth > MaximumReferenceDepth {
		return nil, "", invalid(CodeLimitExceeded, "refs.depth")
	}
	if stack == nil {
		stack = make(map[string]bool)
	}
	rawRef, exists := value["$ref"]
	if !exists {
		return cloneMap(value), documentPath, nil
	}
	ref, ok := rawRef.(string)
	if !ok {
		return nil, "", invalid(CodeReferenceInvalid, "$ref")
	}
	targetDocument, pointer, remote, err := r.referenceTarget(documentPath, ref)
	if remote {
		return nil, "", invalid(CodeRemoteReference, "$ref")
	}
	if err != nil {
		return nil, "", err
	}
	identity := targetDocument + "#" + pointer
	if stack[identity] {
		return nil, "", invalid(CodeReferenceInvalid, "refs.cycle")
	}
	stack[identity] = true
	defer delete(stack, identity)
	target, err := r.lookup(targetDocument, pointer)
	if err != nil {
		return nil, "", err
	}
	targetMap, ok := target.(map[string]any)
	if !ok {
		return nil, "", invalid(CodeReferenceInvalid, "$ref")
	}
	result, resultDocument, err := r.resolveReferenceObject(targetMap, targetDocument, stack, depth+1)
	if err != nil {
		return nil, "", err
	}
	for key, child := range value {
		if key != "$ref" {
			if resultDocument != documentPath && key != "summary" && key != "description" {
				return nil, "", invalid(CodeReferenceInvalid, "$ref.siblings")
			}
			result[key] = cloneJSONValue(child)
		}
	}
	return result, resultDocument, nil
}

func (r *openAPIResolver) resolveValue(value any, documentPath string, stack map[string]bool, depth int) (any, error) {
	if depth > MaximumJSONDepth {
		return nil, invalid(CodeLimitExceeded, "refs.depth")
	}
	r.resolutions++
	if r.resolutions > MaximumJSONNodes {
		return nil, invalid(CodeLimitExceeded, "refs.count")
	}
	switch typed := value.(type) {
	case map[string]any:
		if hasUnsupportedReferenceKeyword(typed) {
			return nil, invalid(CodeReferenceInvalid, "refs.unsupported")
		}
		if rawRef, exists := typed["$ref"]; exists {
			ref, ok := rawRef.(string)
			if !ok {
				return nil, invalid(CodeReferenceInvalid, "$ref")
			}
			targetDocument, pointer, remote, err := r.referenceTarget(documentPath, ref)
			if remote {
				return nil, invalid(CodeRemoteReference, "$ref")
			}
			if err != nil {
				return nil, err
			}
			identity := targetDocument + "#" + pointer
			if len(stack) >= MaximumReferenceDepth {
				return nil, invalid(CodeLimitExceeded, "refs.depth")
			}
			if stack[identity] {
				return nil, invalid(CodeReferenceInvalid, "refs.cycle")
			}
			stack[identity] = true
			target, lookupErr := r.lookup(targetDocument, pointer)
			if lookupErr != nil {
				delete(stack, identity)
				return nil, lookupErr
			}
			resolved, resolveErr := r.resolveValue(target, targetDocument, stack, depth+1)
			delete(stack, identity)
			if resolveErr != nil {
				return nil, resolveErr
			}
			result, ok := resolved.(map[string]any)
			if !ok {
				if len(typed) != 1 {
					return nil, invalid(CodeReferenceInvalid, "$ref")
				}
				return resolved, nil
			}
			result = cloneMap(result)
			for _, key := range slices.Sorted(maps.Keys(typed)) {
				if key == "$ref" {
					continue
				}
				child, childErr := r.resolveValue(typed[key], documentPath, stack, depth+1)
				if childErr != nil {
					return nil, childErr
				}
				result[key] = child
			}
			return result, nil
		}
		result := make(map[string]any, len(typed))
		for _, key := range slices.Sorted(maps.Keys(typed)) {
			child, err := r.resolveValue(typed[key], documentPath, stack, depth+1)
			if err != nil {
				return nil, err
			}
			result[key] = child
		}
		return result, nil
	case []any:
		result := make([]any, len(typed))
		for index, child := range typed {
			resolved, err := r.resolveValue(child, documentPath, stack, depth+1)
			if err != nil {
				return nil, err
			}
			result[index] = resolved
		}
		return result, nil
	default:
		return typed, nil
	}
}

func (r *openAPIResolver) referenceTarget(currentDocument, raw string) (string, string, bool, error) {
	if raw == "" || strings.Contains(raw, "\\") {
		return "", "", false, invalid(CodeReferenceInvalid, "$ref")
	}
	parsed, err := url.Parse(raw)
	if err != nil {
		return "", "", false, invalid(CodeReferenceInvalid, "$ref")
	}
	if parsed.IsAbs() || parsed.Host != "" || strings.HasPrefix(raw, "//") || strings.HasPrefix(parsed.Path, "/") {
		return "", "", true, nil
	}
	if parsed.User != nil || parsed.RawQuery != "" || parsed.ForceQuery {
		return "", "", false, invalid(CodeReferenceInvalid, "$ref")
	}
	targetDocument := currentDocument
	if parsed.Path != "" {
		targetDocument = path.Clean(path.Join(path.Dir(currentDocument), parsed.Path))
		if strings.HasPrefix(targetDocument, "../") || targetDocument == ".." {
			return "", "", false, invalid(CodeReferenceInvalid, "$ref")
		}
		if _, err := validatePackagePath(targetDocument); err != nil {
			return "", "", false, invalid(CodeReferenceInvalid, "$ref")
		}
	}
	if _, exists := r.documents[targetDocument]; !exists {
		return "", "", false, invalid(CodeReferenceInvalid, "$ref")
	}
	pointer := parsed.Fragment
	if pointer != "" && !strings.HasPrefix(pointer, "/") {
		return "", "", false, invalid(CodeReferenceInvalid, "$ref")
	}
	return targetDocument, pointer, false, nil
}

func (r *openAPIResolver) lookup(documentPath, pointer string) (any, error) {
	document, exists := r.documents[documentPath]
	if !exists {
		return nil, invalid(CodeReferenceInvalid, "$ref")
	}
	var current any = document
	if pointer == "" {
		return current, nil
	}
	for _, encoded := range strings.Split(strings.TrimPrefix(pointer, "/"), "/") {
		token, err := decodeJSONPointerToken(encoded)
		if err != nil {
			return nil, invalid(CodeReferenceInvalid, "$ref")
		}
		switch typed := current.(type) {
		case map[string]any:
			current, exists = typed[token]
			if !exists {
				return nil, invalid(CodeReferenceInvalid, "$ref")
			}
		case []any:
			index, parseErr := strconv.Atoi(token)
			if parseErr != nil || index < 0 || index >= len(typed) || strconv.Itoa(index) != token {
				return nil, invalid(CodeReferenceInvalid, "$ref")
			}
			current = typed[index]
		default:
			return nil, invalid(CodeReferenceInvalid, "$ref")
		}
	}
	return current, nil
}

func decodeJSONPointerToken(value string) (string, error) {
	var result strings.Builder
	for index := 0; index < len(value); index++ {
		if value[index] != '~' {
			result.WriteByte(value[index])
			continue
		}
		if index+1 >= len(value) {
			return "", fmt.Errorf("invalid JSON pointer")
		}
		index++
		switch value[index] {
		case '0':
			result.WriteByte('~')
		case '1':
			result.WriteByte('/')
		default:
			return "", fmt.Errorf("invalid JSON pointer")
		}
	}
	return result.String(), nil
}

func (r *openAPIResolver) collectUnselectedRemoteGaps(value any, documentPath, location string, gaps *[]string, depth int) {
	if depth > MaximumJSONDepth || len(*gaps) > MaximumCoverageValues {
		return
	}
	switch typed := value.(type) {
	case map[string]any:
		if hasUnsupportedReferenceKeyword(typed) {
			*gaps = append(*gaps, formatGap("unselected-unsupported-ref", documentPath+"#"+location))
		}
		if rawRef, exists := typed["$ref"]; exists {
			ref, isString := rawRef.(string)
			if !isString {
				*gaps = append(*gaps, formatGap("unselected-invalid-ref", documentPath+"#"+location))
			} else {
				_, _, remote, referenceErr := r.referenceTarget(documentPath, ref)
				if remote {
					*gaps = append(*gaps, formatGap("unselected-remote-ref", documentPath+"#"+location))
				} else if referenceErr != nil {
					*gaps = append(*gaps, formatGap("unselected-invalid-ref", documentPath+"#"+location))
				}
			}
		}
		for _, key := range slices.Sorted(maps.Keys(typed)) {
			r.collectUnselectedRemoteGaps(typed[key], documentPath, location+"/"+escapeJSONPointerToken(key), gaps, depth+1)
		}
	case []any:
		for index, child := range typed {
			r.collectUnselectedRemoteGaps(child, documentPath, location+"/"+strconv.Itoa(index), gaps, depth+1)
		}
	}
}

func hasUnsupportedReferenceKeyword(value map[string]any) bool {
	for _, keyword := range []string{"$id", "$anchor", "$dynamicRef", "$dynamicAnchor", "$recursiveRef", "$recursiveAnchor"} {
		if _, exists := value[keyword]; exists {
			return true
		}
	}
	return false
}

func escapeJSONPointerToken(value string) string {
	return strings.ReplaceAll(strings.ReplaceAll(value, "~", "~0"), "/", "~1")
}
