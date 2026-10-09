package contracts

// Private protocol envelope: the error classes carried across the
// Control Plane/Runtime boundary and the Runtime adapter and credential
// kinds that private DTOs are keyed by. The DTOs themselves live beside
// their topic: the control, runtimesettings and reporting packages, and
// workspace.go and worker_completion.go here. Strict decoding and
// canonical encoding live in private_codec.go.

import (
	"fmt"
	"io"
)

const (
	RuntimeAdapterOTLPHTTP     RuntimeAdapterRef = "otlp-http@1"
	RuntimeAdapterHTTPProxy    RuntimeAdapterRef = "http-proxy@1"
	RuntimeAdapterCaidoGraphQL RuntimeAdapterRef = "caido-graphql@1"

	RuntimeCredentialOTLPHeaders  RuntimeCredentialKind = "otlp-headers@1"
	RuntimeCredentialProxyBasic   RuntimeCredentialKind = "http-proxy-basic@1"
	RuntimeCredentialProxyBearer  RuntimeCredentialKind = "http-proxy-bearer@1"
	RuntimeCredentialCaidoBearer  RuntimeCredentialKind = "caido-bearer@1"
	RuntimeCredentialOriginBasic  RuntimeCredentialKind = "http-origin-basic@1"
	RuntimeCredentialOriginBearer RuntimeCredentialKind = "http-origin-bearer@1"
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

type RuntimeAdapterRef string

func (r RuntimeAdapterRef) Validate() error {
	switch r {
	case RuntimeAdapterOTLPHTTP, RuntimeAdapterHTTPProxy, RuntimeAdapterCaidoGraphQL:
		return nil
	default:
		return Invalidf("unknown RuntimeAdapter ref")
	}
}

type RuntimeCredentialKind string

func (k RuntimeCredentialKind) Validate() error {
	switch k {
	case RuntimeCredentialOTLPHeaders, RuntimeCredentialProxyBasic, RuntimeCredentialProxyBearer,
		RuntimeCredentialCaidoBearer, RuntimeCredentialOriginBasic, RuntimeCredentialOriginBearer:
		return nil
	default:
		return Invalidf("unknown Runtime credential kind")
	}
}

// ValidateRuntimeAdapterRefs requires a non-null, bounded, sorted and unique
// list of known RuntimeAdapter refs.
func ValidateRuntimeAdapterRefs(values []RuntimeAdapterRef) error {
	if values == nil || len(values) > 64 {
		return Invalidf("RuntimeAdapter refs must be a non-null bounded array")
	}
	previous := ""
	for _, value := range values {
		if err := value.Validate(); err != nil {
			return err
		}
		if string(value) <= previous {
			return Invalidf("RuntimeAdapter refs must be sorted and unique")
		}
		previous = string(value)
	}
	return nil
}
