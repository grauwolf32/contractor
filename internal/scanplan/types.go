// Package scanplan prepares bounded HTTP inputs without network or scanner I/O.
package scanplan

import "github.com/grauwolf32/contractor/internal/contracts"

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
	Server          string                            `json:"server"`
	ServerVariables map[string]string                 `json:"serverVariables"`
	Authentication  map[string]contracts.SecretString `json:"authentication"`
	Operations      map[string]OperationInput         `json:"operations"`
	MaxRequests     int                               `json:"maxRequests"`
}

type OperationInput struct {
	Parameters map[string]any `json:"parameters"`
	Body       *BodyInput     `json:"body"`
}

type BodyInput struct {
	MediaType string `json:"mediaType"`
	Value     any    `json:"value"`
}
