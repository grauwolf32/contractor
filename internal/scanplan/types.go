// Package scanplan prepares bounded HTTP inputs without network or scanner I/O.
package scanplan

import "github.com/grauwolf32/contractor/internal/contracts"

// MaxSourceBytes, MaxDepth and MaxNodes bound the whole source document
// (LIM-SCN-39, LIM-SCN-41, LIM-SCN-42): its stored bytes, its depth with the
// top-level value at depth 1, and its nodes, member names included. The Runtime
// OpenAPI authoring Toolset enforces the same bounds, so no document it writes
// exceeds them (SCN-R253); api/testdata/openapi-document-bounds.json holds
// all three readers, including the Audit OpenAPI inventory, to the same verdicts.
const (
	MaxSourceBytes    = 2 * 1024 * 1024
	MaxOptionsBytes   = 256 * 1024
	MaxDepth          = 64
	MaxNodes          = 100000
	MaxOperations     = 1000
	MaxReferences     = 4096
	MaxReferenceDepth = 32
	MaxRequests       = 1000
)

// PreparationError contains a stable code, never supplied values or parser text.
type PreparationError struct{ Code string }

func (e *PreparationError) Error() string { return "request preparation: " + e.Code }
func failure(code string) error           { return &PreparationError{Code: code} }

type Options struct {
	EngagementCredential string                            `json:"engagementCredential,omitempty"`
	Server               string                            `json:"server"`
	ServerVariables      map[string]string                 `json:"serverVariables"`
	Authentication       map[string]contracts.SecretString `json:"authentication"`
	Operations           map[string]OperationInput         `json:"operations"`
	MaxRequests          int                               `json:"maxRequests"`
}

type OperationInput struct {
	Parameters map[string]any `json:"parameters"`
	Body       *BodyInput     `json:"body"`
}

type BodyInput struct {
	MediaType string `json:"mediaType"`
	Value     any    `json:"value"`
}
