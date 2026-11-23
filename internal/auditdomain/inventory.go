package auditdomain

import (
	"bytes"
	"strings"
)

type inventorySubject struct {
	itemKey    string
	kind       string
	subjectKey string

	operation *OperationTask
	scan      *ScanTask

	requested []string
	gaps      []string
}

func canonicalInventoryJSON(basis inventoryBasis) ([]byte, error) {
	if basis.Kind == "openapi-operations" {
		return canonicalOpenAPIJSON(basis)
	}
	return canonicalJSON(basis)
}

func sameCanonicalValue(left, right any) bool {
	leftBytes, leftErr := canonicalJSON(left)
	rightBytes, rightErr := canonicalJSON(right)
	return leftErr == nil && rightErr == nil && bytes.Equal(leftBytes, rightBytes)
}

func cloneMap(value map[string]any) map[string]any {
	if value == nil {
		return nil
	}
	result := make(map[string]any, len(value))
	for key, child := range value {
		result[key] = cloneJSONValue(child)
	}
	return result
}

func cloneJSONValue(value any) any {
	switch typed := value.(type) {
	case map[string]any:
		return cloneMap(typed)
	case []any:
		result := make([]any, len(typed))
		for index, child := range typed {
			result[index] = cloneJSONValue(child)
		}
		return result
	default:
		return typed
	}
}

func arrayStrings(values []string) []string {
	return append([]string{}, values...)
}

func openAPIOperationKey(inventoryDigest, pathTemplate, method string) (string, error) {
	tuple, err := canonicalOpenAPIJSON([]any{
		"contractor.audit.openapi-operation.v1", inventoryDigest, pathTemplate, method,
	})
	if err != nil {
		return "", invalid(CodeInventoryInvalid, "operation_key")
	}
	return "op-" + strings.TrimPrefix(DigestBytes(tuple), "sha256:"), nil
}

func mapValue(value any, field string) (map[string]any, error) {
	result, ok := value.(map[string]any)
	if !ok {
		return nil, invalid(CodeInventoryInvalid, field)
	}
	return result, nil
}

func stringValue(value any, field string) (string, error) {
	result, ok := value.(string)
	if !ok || validateText(result, field, true) != nil {
		return "", invalid(CodeInventoryInvalid, field)
	}
	return result, nil
}

func formatGap(kind, location string) string {
	return kind + ":" + location
}
