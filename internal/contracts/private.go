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
	"bytes"
	"crypto/x509"
	"encoding/pem"
	"fmt"
	"io"
	"net/url"
	"strings"
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
		return invalidf("unknown RuntimeAdapter ref")
	}
}

type RuntimeCredentialKind string

func (k RuntimeCredentialKind) Validate() error {
	switch k {
	case RuntimeCredentialOTLPHeaders, RuntimeCredentialProxyBasic, RuntimeCredentialProxyBearer,
		RuntimeCredentialCaidoBearer, RuntimeCredentialOriginBasic, RuntimeCredentialOriginBearer:
		return nil
	default:
		return invalidf("unknown Runtime credential kind")
	}
}

func validateSortedLabels(field string, values []string, maximum int, allowDefault bool) error {
	if values == nil || len(values) > maximum {
		return invalidf("%s must be a non-null bounded array", field)
	}
	previous := ""
	for _, value := range values {
		if len(value) == 0 || len(value) > 63 || !idPattern.MatchString(value) ||
			(!allowDefault && value == "default") {
			return invalidf("%s contains an invalid label", field)
		}
		if value <= previous {
			return invalidf("%s must be sorted and unique", field)
		}
		previous = value
	}
	return nil
}

func validateSortedRuntimeAdapters(values []RuntimeAdapterRef) error {
	if values == nil || len(values) > 64 {
		return invalidf("RuntimeAdapter refs must be a non-null bounded array")
	}
	previous := ""
	for _, value := range values {
		if err := value.Validate(); err != nil {
			return err
		}
		if string(value) <= previous {
			return invalidf("RuntimeAdapter refs must be sorted and unique")
		}
		previous = string(value)
	}
	return nil
}

func validateRuntimeEndpoint(field, value string) error {
	if len(value) == 0 || len([]byte(value)) > 2048 || value != strings.TrimSpace(value) {
		return invalidf("%s is invalid", field)
	}
	parsed, err := url.Parse(value)
	if err != nil || parsed.Host == "" || parsed.Hostname() == "" || parsed.User != nil ||
		parsed.Fragment != "" || parsed.RawQuery != "" || parsed.ForceQuery ||
		(parsed.Scheme != "http" && parsed.Scheme != "https") {
		return invalidf("%s must be an absolute HTTP(S) URL without userinfo, query, or fragment", field)
	}
	return nil
}

func validateCABundle(owner, value string) error {
	if len(value) == 0 || len([]byte(value)) > 64*1024 || strings.Contains(value, "PRIVATE KEY") {
		return invalidf("%s CA bundle is invalid", owner)
	}
	rest := []byte(value)
	count := 0
	for len(bytes.TrimSpace(rest)) > 0 {
		block, remaining := pem.Decode(rest)
		if block == nil || block.Type != "CERTIFICATE" {
			return invalidf("%s CA bundle is invalid", owner)
		}
		if _, err := x509.ParseCertificate(block.Bytes); err != nil {
			return invalidf("%s CA bundle is invalid", owner)
		}
		count++
		if count > 8 {
			return invalidf("%s CA bundle contains too many certificates", owner)
		}
		rest = remaining
	}
	if count == 0 {
		return invalidf("%s CA bundle must contain a certificate", owner)
	}
	return nil
}

func validateProvenanceBindings(field string, values []RuntimeLabelBindingProvenance) error {
	if values == nil || len(values) > 32 {
		return invalidf("provenance %s must be a non-null bounded array", field)
	}
	previous := ""
	for _, value := range values {
		if len(value.Label) == 0 || len(value.Label) > 63 || value.Label == "default" ||
			!idPattern.MatchString(value.Label) || value.Label <= previous || value.BindingRevision == 0 {
			return invalidf("provenance %s contains invalid or unsorted bindings", field)
		}
		if err := value.Config.Validate(); err != nil {
			return err
		}
		previous = value.Label
	}
	return nil
}
