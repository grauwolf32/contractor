package runtimesettings

import "github.com/grauwolf32/contractor/internal/contracts"

// The Runtime adapter and credential kinds that RuntimeSettings, Runtime
// registration and Runtime reports are keyed by.
const (
	RuntimeAdapterOTLPHTTP       RuntimeAdapterRef = "otlp-http@1"
	RuntimeAdapterHTTPProxy      RuntimeAdapterRef = "http-proxy@1"
	RuntimeAdapterCaidoGraphQL   RuntimeAdapterRef = "caido-graphql@1"
	RuntimeAdapterInteractshHTTP RuntimeAdapterRef = "interactsh-http@1"

	RuntimeCredentialOTLPHeaders     RuntimeCredentialKind = "otlp-headers@1"
	RuntimeCredentialProxyBasic      RuntimeCredentialKind = "http-proxy-basic@1"
	RuntimeCredentialProxyBearer     RuntimeCredentialKind = "http-proxy-bearer@1"
	RuntimeCredentialCaidoBearer     RuntimeCredentialKind = "caido-bearer@1"
	RuntimeCredentialInteractshToken RuntimeCredentialKind = "interactsh-token@1"
	RuntimeCredentialOriginBasic     RuntimeCredentialKind = "http-origin-basic@1"
	RuntimeCredentialOriginBearer    RuntimeCredentialKind = "http-origin-bearer@1"
)

type RuntimeAdapterRef string

func (r RuntimeAdapterRef) Validate() error {
	switch r {
	case RuntimeAdapterOTLPHTTP, RuntimeAdapterHTTPProxy, RuntimeAdapterCaidoGraphQL, RuntimeAdapterInteractshHTTP:
		return nil
	default:
		return contracts.Invalidf("unknown RuntimeAdapter ref")
	}
}

type RuntimeCredentialKind string

func (k RuntimeCredentialKind) Validate() error {
	switch k {
	case RuntimeCredentialOTLPHeaders, RuntimeCredentialProxyBasic, RuntimeCredentialProxyBearer,
		RuntimeCredentialCaidoBearer, RuntimeCredentialInteractshToken, RuntimeCredentialOriginBasic, RuntimeCredentialOriginBearer:
		return nil
	default:
		return contracts.Invalidf("unknown Runtime credential kind")
	}
}

// ValidateRuntimeAdapterRefs requires a non-null, bounded, sorted and unique
// list of known RuntimeAdapter refs.
func ValidateRuntimeAdapterRefs(values []RuntimeAdapterRef) error {
	if values == nil || len(values) > 64 {
		return contracts.Invalidf("RuntimeAdapter refs must be a non-null bounded array")
	}
	previous := ""
	for _, value := range values {
		if err := value.Validate(); err != nil {
			return err
		}
		if string(value) <= previous {
			return contracts.Invalidf("RuntimeAdapter refs must be sorted and unique")
		}
		previous = string(value)
	}
	return nil
}
