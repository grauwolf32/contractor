// Package engagementcontract owns the closed, non-secret engagement contract shared by
// Run/Audit admission and the private allocation projection. It performs no I/O.
package engagementcontract

import (
	"fmt"
	"strings"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

const (
	Schema           = "contractor.engagement.v1"
	MediaType        = "application/vnd.contractor.engagement+json"
	MaxDocumentBytes = 4 * 1024 * 1024
	MaxIdentities    = 8
)

type Document struct {
	Schema     string      `json:"schema" strictjson:"required"`
	Egress     Egress      `json:"egress" strictjson:"required"`
	Identities []Identity  `json:"identities" strictjson:"required"`
	Limits     *HTTPLimits `json:"limits,omitempty"`
}
type Egress struct {
	Routes     []Route `json:"routes" strictjson:"required"`
	Exclusions []Route `json:"exclusions" strictjson:"required"`
}
type Route struct {
	Origin        string   `json:"origin" strictjson:"required"`
	PathPrefixes  []string `json:"pathPrefixes" strictjson:"required"`
	Methods       []string `json:"methods" strictjson:"required"`
	AddressRanges []string `json:"addressRanges" strictjson:"required"`
}
type Identity struct {
	ID         string                                `json:"id" strictjson:"required"`
	Kind       string                                `json:"kind" strictjson:"required"`
	Origins    []string                              `json:"origins" strictjson:"required"`
	Role       *string                               `json:"role,omitempty"`
	Tenant     *string                               `json:"tenant,omitempty"`
	Credential *runtimesettings.RuntimeCredentialRef `json:"credential,omitempty"`
}

// HTTPLimits are optional ceilings enforced independently by each allocation.
// Absence means the corresponding engagement limit is not configured.
type HTTPLimits struct {
	TargetRequests     *uint32 `json:"targetRequests,omitempty"`
	RequestsPerSecond  *uint32 `json:"requestsPerSecond,omitempty"`
	ConcurrentRequests *uint32 `json:"concurrentRequests,omitempty"`
	RequestBodyBytes   *uint32 `json:"requestBodyBytes,omitempty"`
	ResponseBodyBytes  *uint32 `json:"responseBodyBytes,omitempty"`
}

func (l HTTPLimits) Validate() error {
	for _, value := range []*uint32{l.TargetRequests, l.RequestsPerSecond, l.ConcurrentRequests, l.RequestBodyBytes, l.ResponseBodyBytes} {
		if value != nil && *value == 0 {
			return invalid()
		}
	}
	return nil
}

func invalid() error { return fmt.Errorf("engagement is invalid") }

// Decode rejects unknown/duplicate members, omitted arrays, null optionals and
// invalid Unicode before semantic validation; errors never contain input data.
func Decode(data []byte) (Document, error) {
	var value Document
	if len(data) > MaxDocumentBytes || strictjson.DecodeNonNull(data, &value) != nil {
		return Document{}, invalid()
	}
	if err := value.Validate(); err != nil {
		return Document{}, invalid()
	}
	return value, nil
}
func (d Document) Validate() error {
	if d.Schema != Schema || (d.Limits != nil && d.Limits.Validate() != nil) {
		return invalid()
	}
	if d.Egress.Routes == nil || len(d.Egress.Routes) < 1 || len(d.Egress.Routes) > 128 || d.Egress.Exclusions == nil || len(d.Egress.Exclusions) > 128 || d.Identities == nil || len(d.Identities) < 1 || len(d.Identities) > MaxIdentities {
		return invalid()
	}
	origins := map[string]bool{}
	for _, route := range d.Egress.Routes {
		if route.Validate() != nil {
			return invalid()
		}
		origins[route.Origin] = true
	}
	if len(origins) > 16 {
		return invalid()
	}
	for _, route := range d.Egress.Exclusions {
		if route.Validate() != nil || !origins[route.Origin] {
			return invalid()
		}
	}
	seen := map[string]bool{}
	credentials := map[string]bool{}
	anonymous := 0
	for _, identity := range d.Identities {
		if identity.Validate() != nil || seen[identity.ID] {
			return invalid()
		}
		seen[identity.ID] = true
		if identity.Credential != nil {
			if credentials[identity.Credential.CredentialID] {
				return invalid()
			}
			credentials[identity.Credential.CredentialID] = true
		}
		if identity.Kind == "anonymous" {
			anonymous++
		}
		for _, origin := range identity.Origins {
			if !origins[origin] {
				return invalid()
			}
		}
	}
	if anonymous != 1 {
		return invalid()
	}
	encoded, err := strictjson.Canonical(d)
	if err != nil || len(encoded) > MaxDocumentBytes {
		return invalid()
	}
	return nil
}
func (i Identity) Validate() error {
	if !validID(i.ID) || !uniqueStrings(i.Origins, 1, 16) {
		return invalid()
	}
	for _, origin := range i.Origins {
		if !validOrigin(origin) {
			return invalid()
		}
	}
	if i.Role != nil && !text(*i.Role, 256) || i.Tenant != nil && !text(*i.Tenant, 256) {
		return invalid()
	}
	if i.Kind == "anonymous" {
		if i.ID != "anonymous" || i.Credential != nil {
			return invalid()
		}
		return nil
	}
	if i.ID == "anonymous" || i.Credential == nil || i.Credential.Validate() != nil {
		return invalid()
	}
	var kind runtimesettings.RuntimeCredentialKind
	switch i.Kind {
	case "basic":
		kind = runtimesettings.RuntimeCredentialTargetBasic
	case "bearer":
		kind = runtimesettings.RuntimeCredentialTargetBearer
	case "cookies":
		kind = runtimesettings.RuntimeCredentialTargetCookies
	default:
		return invalid()
	}
	if i.Credential.Kind != kind {
		return invalid()
	}
	return nil
}
func validID(s string) bool {
	if len(s) < 1 || len(s) > 64 || s[0] < 'a' || s[0] > 'z' {
		return false
	}
	for _, c := range s {
		if !(c >= 'a' && c <= 'z' || c >= '0' && c <= '9' || c == '-') {
			return false
		}
	}
	return true
}
func text(s string, max int) bool {
	if len(s) == 0 || len(s) > max || !utf8.ValidString(s) || strings.TrimSpace(s) == "" {
		return false
	}
	for _, c := range s {
		if c < 32 || c == 127 {
			return false
		}
	}
	return true
}
func uniqueStrings(values []string, min, max int) bool {
	if values == nil || len(values) < min || len(values) > max {
		return false
	}
	seen := map[string]bool{}
	for _, v := range values {
		if seen[v] {
			return false
		}
		seen[v] = true
	}
	return true
}
