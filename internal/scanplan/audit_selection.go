package scanplan

import (
	"fmt"
	"io"
	"iter"
	"maps"
	"slices"
	"sort"
	"strings"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/scan"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

// AuditSelection is compiled from ordinary Workflow parameters and optional
// request values. Credentials are referenced by ID and applied only at egress.
type AuditSelection struct{ document auditSelectionDocument }

func (AuditSelection) Format(state fmt.State, _ rune) {
	_, _ = io.WriteString(state, "scanplan.AuditSelection([REDACTED])")
}

type auditSelectionDocument struct {
	Scanner        string                    `json:"scanner"`
	Server         string                    `json:"server"`
	CredentialID   string                    `json:"credential_id"`
	Operations     map[string]OperationInput `json:"operations"`
	TestParameters []string                  `json:"test_parameters"`
}

// CompileAuditSelection rejects unknown/unselected request data. The material
// contains only operation parameter/body values, never execution or auth policy.
func CompileAuditSelection(scanner string, parameters map[string]string, data []byte) (AuditSelection, error) {
	bad := func() (AuditSelection, error) { return AuditSelection{}, failure("invalid_audit_scan_selection") }
	value := auditSelectionDocument{Scanner: scanner, Server: parameters["server"], CredentialID: parameters["credential_id"], Operations: map[string]OperationInput{}, TestParameters: []string{}}
	if scanner != "sqlmap" && scanner != "nuclei" || len(data) > MaxOptionsBytes || value.CredentialID != "" && !contracts.ValidConfigID(value.CredentialID) {
		return bad()
	}
	server, code := scanURL(value.Server)
	if code != "" || server.RawQuery != "" || server.ForceQuery || strings.ContainsAny(value.Server, "{}") {
		return bad()
	}
	var pointers []string
	if strictjson.DecodeNonNull([]byte(parameters["operations"]), &pointers) != nil || len(pointers) == 0 || len(pointers) > MaxOperations {
		return bad()
	}
	for _, pointer := range pointers {
		if !ValidOperationPointer(pointer) {
			return bad()
		}
		if _, exists := value.Operations[pointer]; exists {
			return bad()
		}
		value.Operations[pointer] = OperationInput{}
	}
	if raw := parameters["test_parameters"]; raw != "" {
		if strictjson.DecodeNonNull([]byte(raw), &value.TestParameters) != nil {
			return bad()
		}
	}
	if scan.ValidateScanTestParameters(value.TestParameters) != nil || (scanner == "sqlmap") != (len(value.TestParameters) > 0) {
		return bad()
	}
	if len(data) > 0 {
		root, err := parseDocument(data, "application/json")
		if err != nil || !requestValuesFields(root) {
			return bad()
		}
		var values map[string]OperationInput
		if strictjson.Decode(data, &values) != nil {
			return bad()
		}
		for pointer, input := range values {
			if _, exists := value.Operations[pointer]; !exists {
				return bad()
			}
			value.Operations[pointer] = input
		}
	}
	for _, input := range value.Operations {
		for name := range input.Parameters {
			parts := strings.SplitN(name, ":", 2)
			if len(parts) != 2 || parts[1] == "" || len(parts[1]) > 128 || (parts[0] != "path" && parts[0] != "query" && parts[0] != "header" && parts[0] != "cookie") || scanner == "nuclei" && parts[0] != "path" && parts[0] != "query" {
				return bad()
			}
		}
		if input.Body != nil && (input.Body.MediaType == "" || scanner == "nuclei") {
			return bad()
		}
	}
	options, err := normalizeOptions(Options{Server: value.Server, EngagementCredential: value.CredentialID, Operations: value.Operations, MaxRequests: 1})
	if err != nil {
		return bad()
	}
	value.Operations = options.Operations
	sort.Strings(value.TestParameters)
	return AuditSelection{document: value}, nil
}

func requestValuesFields(root map[string]any) bool {
	for _, raw := range root {
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

func (s AuditSelection) Digest() string {
	data, _ := strictjson.Canonical(s.document)
	return contentdigest.Bytes(data)
}

func (s AuditSelection) Scanner() string { return s.document.Scanner }
func (s AuditSelection) Operations() []string {
	return slices.Sorted(maps.Keys(s.document.Operations))
}
func (s AuditSelection) TestParameters() []string {
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
func (s AuditSelection) PrepareOperation(data []byte, mediaType string, source contracts.ArtifactRef, pointer string) (PreparedAuditOperation, error) {
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
// (SCN-R256). Each result and error equals PrepareOperation's for the same
// source and operation: a source failure comes first, as it does there,
// because compiled selections hold no binding that a URL target rejects. The
// sequence stops after the first error. The parsed source lives only while
// the sequence runs; nothing retains it on the selection or across calls.
func (s AuditSelection) PrepareOperations(data []byte, mediaType string, source contracts.ArtifactRef) iter.Seq2[PreparedAuditOperation, error] {
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

func (s AuditSelection) prepareParsed(document *sourceDocument, pointer string) (PreparedAuditOperation, error) {
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
func (s AuditSelection) operationOptions(pointer string) (Options, error) {
	input, exists := s.document.Operations[pointer]
	if !exists {
		return Options{}, failure("unassigned_operation_binding")
	}
	return Options{Server: s.document.Server, EngagementCredential: s.document.CredentialID,
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

func (s AuditSelection) preparedRequest(pointer string, set scan.HTTPRequestSet) PreparedAuditOperation {
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
