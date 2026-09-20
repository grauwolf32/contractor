package scanplan

import (
	"bytes"
	"encoding/json"
	"sort"
	"strings"

	"github.com/grauwolf32/contractor/internal/contracts"
)

const AuditScanSettingsSchema = "contractor.audit.openapi-scan-settings.v1"

// AuditScanSettings is a validated, immutable selection. It owns its decoded
// bindings; accessors never expose maps containing credentials or request data.
// Template selection and execution budgets belong to the pinned Workflow.
type AuditScanSettings struct {
	document auditScanSettingsDocument
}

type auditScanSettingsDocument struct {
	Schema         string                            `json:"schema"`
	Scanner        string                            `json:"scanner"`
	Server         string                            `json:"server"`
	Authentication map[string]contracts.SecretString `json:"authentication,omitempty"`
	Operations     map[string]OperationInput         `json:"operations"`
	TestParameters []string                          `json:"testParameters,omitempty"`
}

// DecodeAuditScanSettings accepts only bounded JSON with exact field names.
// No parser error or supplied credential is returned in diagnostics.
func DecodeAuditScanSettings(data []byte) (AuditScanSettings, error) {
	bad := func() (AuditScanSettings, error) { return AuditScanSettings{}, failure("invalid_audit_scan_settings") }
	if len(data) > MaxOptionsBytes {
		return bad()
	}
	root, err := parseDocument(data, "application/json")
	if err != nil || !auditSettingsFields(root) {
		return bad()
	}
	var value auditScanSettingsDocument
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if decoder.Decode(&value) != nil || value.Schema != AuditScanSettingsSchema ||
		(value.Scanner != "sqlmap" && value.Scanner != "nuclei") ||
		len(value.Operations) == 0 || len(value.Operations) > MaxOperations {
		return bad()
	}
	server, code := scanURL(value.Server)
	if code != "" || server.RawQuery != "" || server.ForceQuery || strings.ContainsAny(value.Server, "{}") {
		return bad()
	}
	if contracts.ValidateScanTestParameters(value.TestParameters) != nil || (value.Scanner == "sqlmap") != (len(value.TestParameters) > 0) {
		return bad()
	}
	for pointer, input := range value.Operations {
		if !(operationSelection{pointer: pointer, mode: "request"}).valid() {
			return bad()
		}
		for name := range input.Parameters {
			parts := strings.SplitN(name, ":", 2)
			if len(parts) != 2 || parts[1] == "" || len(parts[1]) > 128 ||
				(parts[0] != "path" && parts[0] != "query" && parts[0] != "header" && parts[0] != "cookie") ||
				value.Scanner == "nuclei" && parts[0] != "path" && parts[0] != "query" {
				return bad()
			}
		}
		if input.Body != nil && input.Body.MediaType == "" || value.Scanner == "nuclei" && input.Body != nil {
			return bad()
		}
	}
	if value.Scanner == "nuclei" && len(value.Authentication) != 0 {
		return bad()
	}
	for name, secret := range value.Authentication {
		if name == "" || len(name) > 128 || secret.Reveal() == "" {
			return bad()
		}
	}
	options, err := normalizeOptions(Options{Server: value.Server, Authentication: value.Authentication, Operations: value.Operations, MaxRequests: 1})
	if err != nil {
		return bad()
	}
	value.Authentication, value.Operations = options.Authentication, options.Operations
	sort.Strings(value.TestParameters)
	return AuditScanSettings{document: value}, nil
}

// encoding/json accepts case-insensitive aliases and null scalar fields. Reject
// both before decoding, including inside operation/body objects. Parameter and
// body values remain arbitrary bounded JSON data.
func auditSettingsFields(root map[string]any) bool {
	allowed := map[string]bool{"schema": true, "scanner": true, "server": true, "authentication": true, "operations": true, "testParameters": true}
	for key, value := range root {
		if !allowed[key] || value == nil {
			return false
		}
	}
	operations, ok := root["operations"].(map[string]any)
	if !ok {
		return false
	}
	if auth, exists := root["authentication"]; exists {
		values, ok := auth.(map[string]any)
		if !ok {
			return false
		}
		for _, value := range values {
			if _, ok := value.(string); !ok {
				return false
			}
		}
	}
	for _, raw := range operations {
		input, ok := raw.(map[string]any)
		if !ok {
			return false
		}
		for key, value := range input {
			switch key {
			case "parameters":
				if _, ok := value.(map[string]any); !ok {
					return false
				}
			case "body":
				body, ok := value.(map[string]any)
				if !ok || len(body) != 2 {
					return false
				}
				if _, ok := body["mediaType"].(string); !ok {
					return false
				}
				if _, ok := body["value"]; !ok {
					return false
				}
			default:
				return false
			}
		}
	}
	return true
}

func (s AuditScanSettings) Scanner() string      { return s.document.Scanner }
func (s AuditScanSettings) Operations() []string { return keys(s.document.Operations) }
func (s AuditScanSettings) TestParameters() []string {
	return append([]string{}, s.document.TestParameters...)
}

// PreparedAuditOperation holds the concrete input without conflating a Nuclei
// target with a replayable HTTP request. A false Runnable always has gaps.
type PreparedAuditOperation struct {
	Operation         string
	PreparationDigest string
	Runnable          bool
	RequestSet        *contracts.HTTPRequestSet
	Target            *OperationTarget
	Gaps              []contracts.PreparationGap
}

func (s AuditScanSettings) PrepareOperation(data []byte, mediaType string, source contracts.ArtifactRef, pointer string) (PreparedAuditOperation, error) {
	input, exists := s.document.Operations[pointer]
	if !exists {
		return PreparedAuditOperation{}, failure("unassigned_operation_binding")
	}
	options := Options{Server: s.document.Server, Authentication: s.document.Authentication,
		Operations: map[string]OperationInput{pointer: input}, MaxRequests: 1}
	result := PreparedAuditOperation{Operation: pointer}
	if s.document.Scanner == "nuclei" {
		target, err := PrepareOperationTarget(data, mediaType, source, options, pointer)
		if err != nil {
			return PreparedAuditOperation{}, err
		}
		result.Target, result.PreparationDigest, result.Gaps = &target, target.PreparationDigest, append([]contracts.PreparationGap{}, target.Gaps...)
		result.Runnable = target.URL != ""
		if result.Runnable {
			if _, code := scanURL(target.URL); code != "" {
				result.Runnable = false
				result.Gaps = append(result.Gaps, contracts.PreparationGap{Pointer: pointer, Code: code})
			}
		}
	} else {
		set, err := PrepareOperation(data, mediaType, source, options, pointer)
		if err != nil {
			return PreparedAuditOperation{}, err
		}
		result.RequestSet, result.PreparationDigest, result.Gaps = &set, set.PreparationDigest, append([]contracts.PreparationGap{}, set.Gaps...)
		if len(set.Requests) == 1 {
			_, code := prepareSQLMapRequest(set.Requests[0].Request, s.document.TestParameters)
			result.Runnable = code == ""
			if code != "" {
				result.Gaps = append(result.Gaps, contracts.PreparationGap{Pointer: pointer, Code: code})
			}
		}
	}
	return result, nil
}
