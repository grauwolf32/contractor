package contracts

// Private protocol envelope: the error classes carried across the
// Control Plane/Runtime boundary. The DTOs themselves live beside their
// topic: the control, runtimesettings (which also holds the Runtime adapter
// and credential kinds) and reporting packages, and workspace.go and
// worker_completion.go here. Strict decoding and canonical encoding live in
// private_codec.go.

import (
	"fmt"
	"io"
)

const (
	PrivateProtocolErrorVersion   PrivateProtocolErrorClass = "version"
	PrivateProtocolErrorDuplicate PrivateProtocolErrorClass = "duplicate_key"
	PrivateProtocolErrorSchema    PrivateProtocolErrorClass = "schema"
	PrivateProtocolErrorInvariant PrivateProtocolErrorClass = "invariant"
)

// PrivateProtocolError is deliberately detail-free: malformed private input
// can contain credentials and must not be copied into errors or logs.
type PrivateProtocolErrorClass string

type PrivateProtocolError struct {
	Class PrivateProtocolErrorClass
}

func (e *PrivateProtocolError) Error() string {
	return "private protocol " + string(e.Class) + " error"
}

func (e *PrivateProtocolError) Format(state fmt.State, _ rune) {
	_, _ = io.WriteString(state, e.Error())
}
