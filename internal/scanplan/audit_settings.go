package scanplan

import (
	"bytes"
	"encoding/json"
	"iter"
	"maps"
	"slices"
	"sort"
	"strings"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/scan"
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
	if scan.ValidateScanTestParameters(value.TestParameters) != nil || (value.Scanner == "sqlmap") != (len(value.TestParameters) > 0) {
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

func (s AuditScanSettings) Scanner() string { return s.document.Scanner }
func (s AuditScanSettings) Operations() []string {
	return slices.Sorted(maps.Keys(s.document.Operations))
}
func (s AuditScanSettings) TestParameters() []string {
	return append([]string{}, s.document.TestParameters...)
}

// PreparedAuditOperation holds the concrete input without conflating a Nuclei
// target with a replayable HTTP request. A false Runnable always has gaps.
type PreparedAuditOperation struct {
	Operation         string
	PreparationDigest string
	Runnable          bool
	RequestSet        *scan.HTTPRequestSet
	Target            *OperationTarget
	Gaps              []scan.PreparationGap
}

// PrepareOperation prepares one selected operation from the exact source of
// this call, which it parses. Dispatch-time re-preparation of one accepted
// task uses it.
func (s AuditScanSettings) PrepareOperation(data []byte, mediaType string, source contracts.ArtifactRef, pointer string) (PreparedAuditOperation, error) {
	options, err := s.operationOptions(pointer)
	if err != nil {
		return PreparedAuditOperation{}, err
	}
	if s.document.Scanner == "nuclei" {
		target, err := PrepareOperationTarget(data, mediaType, source, options, pointer)
		if err != nil {
			return PreparedAuditOperation{}, err
		}
		return preparedTarget(pointer, target), nil
	}
	set, err := PrepareOperation(data, mediaType, source, options, pointer)
	if err != nil {
		return PreparedAuditOperation{}, err
	}
	return s.preparedRequest(pointer, set), nil
}

// PrepareOperations prepares every selected operation, in Operations order,
// from one parse and one content digest of the exact source of this call
// (SCN-R254). Each result and error equals PrepareOperation's for the same
// source and operation: a source failure comes first, as it does there,
// because decoded settings hold no binding that a URL target rejects. The
// sequence stops after the first error. The parsed source lives only while
// the sequence runs; nothing retains it on the settings or across calls.
func (s AuditScanSettings) PrepareOperations(data []byte, mediaType string, source contracts.ArtifactRef) iter.Seq2[PreparedAuditOperation, error] {
	return func(yield func(PreparedAuditOperation, error) bool) {
		document, err := parseSource(data, mediaType, source)
		if err != nil {
			yield(PreparedAuditOperation{}, err)
			return
		}
		for _, pointer := range s.Operations() {
			prepared, err := s.prepareParsed(document, pointer)
			if !yield(prepared, err) || err != nil {
				return
			}
		}
	}
}

func (s AuditScanSettings) prepareParsed(document *sourceDocument, pointer string) (PreparedAuditOperation, error) {
	options, err := s.operationOptions(pointer)
	if err != nil {
		return PreparedAuditOperation{}, err
	}
	if s.document.Scanner == "nuclei" {
		target, err := document.prepareTarget(options, pointer)
		if err != nil {
			return PreparedAuditOperation{}, err
		}
		return preparedTarget(pointer, target), nil
	}
	set, err := document.prepareRequest(options, pointer)
	if err != nil {
		return PreparedAuditOperation{}, err
	}
	return s.preparedRequest(pointer, set), nil
}

// operationOptions are the preparation options of one selected operation:
// only its own bindings, so other operations' bindings never apply to it.
func (s AuditScanSettings) operationOptions(pointer string) (Options, error) {
	input, exists := s.document.Operations[pointer]
	if !exists {
		return Options{}, failure("unassigned_operation_binding")
	}
	return Options{Server: s.document.Server, Authentication: s.document.Authentication,
		Operations: map[string]OperationInput{pointer: input}, MaxRequests: 1}, nil
}

func preparedTarget(pointer string, target OperationTarget) PreparedAuditOperation {
	result := PreparedAuditOperation{Operation: pointer, PreparationDigest: target.PreparationDigest,
		Target: &target, Gaps: append([]scan.PreparationGap{}, target.Gaps...), Runnable: target.URL != ""}
	if result.Runnable {
		if _, code := scanURL(target.URL); code != "" {
			result.Runnable = false
			result.Gaps = append(result.Gaps, scan.PreparationGap{Pointer: pointer, Code: code})
		}
	}
	return result
}

func (s AuditScanSettings) preparedRequest(pointer string, set scan.HTTPRequestSet) PreparedAuditOperation {
	result := PreparedAuditOperation{Operation: pointer, PreparationDigest: set.PreparationDigest,
		RequestSet: &set, Gaps: append([]scan.PreparationGap{}, set.Gaps...)}
	if len(set.Requests) == 1 {
		_, code := prepareSQLMapRequest(set.Requests[0].Request, s.document.TestParameters)
		result.Runnable = code == ""
		if code != "" {
			result.Gaps = append(result.Gaps, scan.PreparationGap{Pointer: pointer, Code: code})
		}
	}
	return result
}
