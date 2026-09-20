package scanplan

import (
	"strings"

	"github.com/grauwolf32/contractor/internal/contracts"
)

type operationSelection struct {
	pointer string
	mode    string
}

// ValidOperationPointer reports whether pointer names one canonical path/method
// selection. Existence is checked against the exact source during preparation.
func ValidOperationPointer(pointer string) bool {
	return (operationSelection{pointer: pointer, mode: "request"}).valid()
}

func (s operationSelection) pathPointer() string {
	index := strings.LastIndexByte(s.pointer, '/')
	if index < 0 {
		return ""
	}
	return s.pointer[:index]
}

func (s operationSelection) valid() bool {
	if len(s.pointer) > 8192 || !strings.HasPrefix(s.pointer, "#/paths/") || (s.mode != "request" && s.mode != "url-target") {
		return false
	}
	parts := strings.Split(strings.TrimPrefix(s.pointer, "#/paths/"), "/")
	if len(parts) != 2 || parts[0] == "" {
		return false
	}
	// Only canonical JSON Pointer escapes are accepted; fragments and encoded
	// slashes must not select a different operation through a second decoder.
	for i := 0; i < len(parts[0]); i++ {
		if parts[0][i] == '~' {
			if i+1 == len(parts[0]) || (parts[0][i+1] != '0' && parts[0][i+1] != '1') {
				return false
			}
			i++
		}
	}
	switch parts[1] {
	case "delete", "get", "head", "options", "patch", "post", "put", "trace":
		return true
	default:
		return false
	}
}

// PrepareOperation prepares only the assigned operation from the original exact
// source. Other operations neither consume its request budget nor contribute gaps.
// Bindings for other operations are rejected, never silently broadened.
func PrepareOperation(data []byte, mediaType string, source contracts.ArtifactRef, options Options, pointer string) (contracts.HTTPRequestSet, error) {
	return prepare(data, mediaType, source, options, &operationSelection{pointer: pointer, mode: "request"})
}

// OperationTarget is an in-process URL preparation result, not a persisted scan
// plan or an HTTP request. An empty URL with gaps means no target was prepared.
// The caller retains the source/operation/preparation identity alongside the
// exact target artifact passed to the ordinary Nuclei planner.
type OperationTarget struct {
	Source            contracts.RequestSetSource
	Operation         string
	PreparationDigest string
	URL               string
	Gaps              []contracts.PreparationGap
}

// PrepareOperationTarget fixes a concrete URL for Nuclei's target-only interface.
// It uses path/query values but never invents them or projects operation bodies,
// cookies, headers or authentication into the target. Unsupported explicit
// bindings fail; declared request requirements remain visible as limitations.
func PrepareOperationTarget(data []byte, mediaType string, source contracts.ArtifactRef, options Options, pointer string) (OperationTarget, error) {
	if len(options.Authentication) != 0 {
		return OperationTarget{}, failure("unsupported_url_target_binding")
	}
	for _, input := range options.Operations {
		if input.Body != nil {
			return OperationTarget{}, failure("unsupported_url_target_binding")
		}
		for name := range input.Parameters {
			if !strings.HasPrefix(name, "path:") && !strings.HasPrefix(name, "query:") {
				return OperationTarget{}, failure("unsupported_url_target_binding")
			}
		}
	}
	set, err := prepare(data, mediaType, source, options, &operationSelection{pointer: pointer, mode: "url-target"})
	if err != nil {
		return OperationTarget{}, err
	}
	result := OperationTarget{Source: set.Source, Operation: pointer, PreparationDigest: set.PreparationDigest, Gaps: set.Gaps}
	if len(set.Requests) == 1 {
		result.URL = set.Requests[0].Request.URL
	}
	return result, nil
}
