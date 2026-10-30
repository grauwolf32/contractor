package engagementcontract

import (
	"net"
	"net/netip"
	"net/url"
	"strconv"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
)

// HTTPAssignment grants bounded, model-selected HTTP within the pinned scope.
type HTTPAssignment struct {
	Limits                *HTTPLimits                       `json:"limits,omitempty"`
	Adapter               runtimesettings.RuntimeAdapterRef `json:"adapter" strictjson:"required"`
	Routes                []Route                           `json:"routes" strictjson:"required"`
	Exclusions            []Route                           `json:"exclusions" strictjson:"required"`
	Identities            []Identity                        `json:"identities" strictjson:"required"`
	ProtectedDestinations []ProtectedDestination            `json:"protectedDestinations" strictjson:"required"`
}

// ProtectedDestination denies trusted infrastructure by hostname and resolved
// address. The Runtime also resolves the hostname in its own network.
type ProtectedDestination struct {
	Hostname  string   `json:"hostname" strictjson:"required"`
	Addresses []string `json:"addresses" strictjson:"required"`
	Port      int      `json:"port" strictjson:"required"`
}

func (p HTTPAssignment) Validate() error {
	if p.Limits != nil && p.Limits.Validate() != nil {
		return contracts.Invalidf("controlled HTTP limits are invalid")
	}
	if p.Adapter != runtimesettings.RuntimeAdapterHTTPTarget || len(p.Routes) < 1 || len(p.Routes) > 128 || p.Exclusions == nil || len(p.Exclusions) > 128 || len(p.Identities) < 1 || len(p.Identities) > 8 || len(p.ProtectedDestinations) < 1 || len(p.ProtectedDestinations) > 256 {
		return contracts.Invalidf("controlled HTTP assignment is invalid")
	}
	seen := map[string]bool{}
	credentials := map[string]bool{}
	for _, identity := range p.Identities {
		if identity.Validate() != nil || seen[identity.ID] {
			return contracts.Invalidf("engagement assigned identity is invalid")
		}
		seen[identity.ID] = true
		if identity.Credential != nil {
			if credentials[identity.Credential.CredentialID] {
				return contracts.Invalidf("engagement credential selection is ambiguous")
			}
			credentials[identity.Credential.CredentialID] = true
		}
	}
	if !seen["anonymous"] {
		return contracts.Invalidf("engagement anonymous identity is required")
	}
	for _, routes := range [][]Route{p.Routes, p.Exclusions} {
		for _, route := range routes {
			if route.Validate() != nil {
				return contracts.Invalidf("controlled HTTP route is invalid")
			}
		}
	}
	for _, destination := range p.ProtectedDestinations {
		if destination.Validate() != nil {
			return contracts.Invalidf("engagement protected destination is invalid")
		}
	}
	return nil
}

func (p ProtectedDestination) Validate() error {
	if p.Port < 1 || p.Port > 65535 || !uniqueStrings(p.Addresses, 1, 32) {
		return contracts.Invalidf("engagement protected destination is invalid")
	}
	candidate := "http://" + net.JoinHostPort(p.Hostname, strconv.Itoa(p.Port)) + "/"
	canonical, err := CanonicalURL(candidate)
	if err != nil {
		return contracts.Invalidf("engagement protected hostname is invalid")
	}
	parsed, err := url.Parse(canonical)
	if err != nil || parsed.Hostname() != p.Hostname {
		return contracts.Invalidf("engagement protected hostname is not canonical")
	}
	for _, raw := range p.Addresses {
		ip, err := netip.ParseAddr(raw)
		if err != nil || ip.String() != raw || ip.Is4In6() {
			return contracts.Invalidf("engagement protected address is invalid")
		}
	}
	return nil
}
