package contracts

// Private protocol envelope: the error classes carried across the
// Control Plane/Runtime boundary, the adapter and credential refs every
// private DTO is keyed by, and the validators shared by those DTOs.
// The DTOs themselves live beside their topic: registration.go,
// workspace.go, runtime_settings.go, allocation.go and telemetry.go.
// Strict decoding and canonical encoding live in private_codec.go.
//
// File layout mirrors the Python runtime's contracts package so both
// sides of one wire contract stay comparable file by file.

import (
	"fmt"
	"io"
	"net"
	"net/url"
	"strconv"
	"strings"
	"unicode"

	"github.com/grauwolf32/contractor/internal/cabundle"
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

func validateRuntimeEndpoint(field, value string) error {
	if len(value) == 0 || len([]byte(value)) > 2048 || value != strings.TrimSpace(value) {
		return Invalidf("%s is invalid", field)
	}
	parsed, err := url.Parse(value)
	if err != nil || parsed.Host == "" || parsed.Hostname() == "" || parsed.User != nil ||
		strings.Contains(value, "#") || parsed.RawQuery != "" || parsed.ForceQuery ||
		(parsed.Scheme != "http" && parsed.Scheme != "https") {
		return Invalidf("%s must be an absolute HTTP(S) URL without userinfo, query, or fragment", field)
	}
	host := parsed.Hostname()
	if strings.Contains(host, "%") || strings.ContainsAny(host, "<>\\^|`{}") ||
		strings.IndexFunc(host, unicode.IsSpace) >= 0 || strings.HasSuffix(parsed.Host, ":") {
		return Invalidf("%s host is invalid", field)
	}
	if port := parsed.Port(); port != "" {
		portNumber, err := strconv.ParseUint(port, 10, 16)
		if err != nil || portNumber == 0 {
			return Invalidf("%s port is invalid", field)
		}
	}
	if net.ParseIP(host) == nil {
		if strings.Contains(host, ":") {
			return Invalidf("%s host is invalid", field)
		}
		labels := strings.Split(strings.TrimSuffix(host, "."), ".")
		last := labels[len(labels)-1]
		if last == "" || allDecimalDigits(last) {
			return Invalidf("%s host is invalid", field)
		}
	}
	return nil
}

func allDecimalDigits(value string) bool {
	if value == "" {
		return false
	}
	for _, digit := range value {
		if digit < '0' || digit > '9' {
			return false
		}
	}
	return true
}

func validateCABundle(owner, value string) error {
	if err := cabundle.Validate(value); err != nil {
		return Invalidf("%s CA bundle is invalid", owner)
	}
	return nil
}

func validateProvenanceBindings(field string, values []RuntimeLabelBindingProvenance) error {
	if values == nil || len(values) > 32 {
		return Invalidf("provenance %s must be a non-null bounded array", field)
	}
	previous := ""
	for _, value := range values {
		if len(value.Label) == 0 || len(value.Label) > 63 || value.Label == "default" ||
			!ValidIdentifier(value.Label) || value.Label <= previous || value.BindingRevision == 0 {
			return Invalidf("provenance %s contains invalid or unsorted bindings", field)
		}
		if err := value.Config.Validate(); err != nil {
			return err
		}
		previous = value.Label
	}
	return nil
}
