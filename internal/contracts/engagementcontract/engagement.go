// Package engagementcontract owns the closed, non-secret engagement contract shared by
// Audit admission and the private allocation projection. It performs no I/O.
package engagementcontract

import (
	"fmt"
	"slices"
	"strings"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

const (
	Schema           = "contractor.engagement.v1"
	MediaType        = "application/vnd.contractor.engagement+json"
	Capability       = "engagement@1"
	MaxDocumentBytes = 4 * 1024 * 1024
)

type Document struct {
	Schema     string     `json:"schema" strictjson:"required"`
	AccessMode string     `json:"accessMode" strictjson:"required"`
	Target     Target     `json:"target" strictjson:"required"`
	Scope      Scope      `json:"scope" strictjson:"required"`
	Identities []Identity `json:"identities" strictjson:"required"`
	Inputs     []Input    `json:"inputs" strictjson:"required"`
	Budgets    Budgets    `json:"budgets" strictjson:"required"`
	Policy     Policy     `json:"policy" strictjson:"required"`
}
type Target struct {
	Origins    []string    `json:"origins" strictjson:"required"`
	Deployment *Deployment `json:"deployment,omitempty"`
}
type Deployment struct {
	Marker string `json:"marker" strictjson:"required"`
}
type Scope struct {
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
type Input struct {
	ID            string                `json:"id" strictjson:"required"`
	Role          string                `json:"role" strictjson:"required"`
	Artifact      contracts.ArtifactRef `json:"artifact" strictjson:"required"`
	ContentDigest string                `json:"contentDigest" strictjson:"required"`
}
type Policy struct {
	Version              string   `json:"version" strictjson:"required"`
	ActiveChecks         string   `json:"activeChecks" strictjson:"required"`
	RequiredCapabilities []string `json:"requiredCapabilities" strictjson:"required"`
	OptionalCapabilities []string `json:"optionalCapabilities" strictjson:"required"`
}
type Budgets struct {
	HTTPLimits
	ActiveSeconds uint32 `json:"activeSeconds" strictjson:"required"`
	Rounds        uint32 `json:"rounds" strictjson:"required"`
	SubmittedRuns uint32 `json:"submittedRuns" strictjson:"required"`
	EvidenceBytes uint32 `json:"evidenceBytes" strictjson:"required"`
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

func (l HTTPLimits) Clone() HTTPLimits {
	result := l
	for _, pair := range []struct {
		source *uint32
		target **uint32
	}{
		{l.TargetRequests, &result.TargetRequests}, {l.RequestsPerSecond, &result.RequestsPerSecond},
		{l.ConcurrentRequests, &result.ConcurrentRequests}, {l.RequestBodyBytes, &result.RequestBodyBytes},
		{l.ResponseBodyBytes, &result.ResponseBodyBytes},
	} {
		if pair.source != nil {
			value := *pair.source
			*pair.target = &value
		}
	}
	return result
}

func invalid() error { return fmt.Errorf("engagement engagement is invalid") }

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
func (p Policy) Digest() (string, error) {
	if p.Validate() != nil {
		return "", invalid()
	}
	return contentdigest.CanonicalJSON(p)
}
func (b Budgets) Validate() error {
	if err := b.HTTPLimits.Validate(); err != nil {
		return err
	}
	for _, pair := range [][2]uint32{{b.ActiveSeconds, 86400}, {b.Rounds, 8}, {b.SubmittedRuns, 512}, {b.EvidenceBytes, 67108864}} {
		if pair[0] == 0 || pair[0] > pair[1] {
			return invalid()
		}
	}
	return nil
}
func (p Policy) Validate() error {
	if p.Version != "1" || p.ActiveChecks != "automatic" || !uniqueStrings(p.RequiredCapabilities, 0, 128) || !uniqueStrings(p.OptionalCapabilities, 0, 128) {
		return invalid()
	}
	for _, v := range append(slices.Clone(p.RequiredCapabilities), p.OptionalCapabilities...) {
		if len(v) > 256 || !contracts.ValidSelector(v) {
			return invalid()
		}
	}
	for _, v := range p.RequiredCapabilities {
		if slices.Contains(p.OptionalCapabilities, v) {
			return invalid()
		}
	}
	return nil
}
func (d Document) Validate() error {
	if d.Schema != Schema || (d.AccessMode != "blackbox" && d.AccessMode != "source-assisted") || d.Policy.Validate() != nil || d.Budgets.Validate() != nil {
		return invalid()
	}
	if !uniqueStrings(d.Target.Origins, 1, 16) || d.Scope.Routes == nil || len(d.Scope.Routes) < 1 || len(d.Scope.Routes) > 128 || d.Scope.Exclusions == nil || len(d.Scope.Exclusions) > 128 || d.Identities == nil || len(d.Identities) < 1 || len(d.Identities) > 8 || d.Inputs == nil || len(d.Inputs) > 128 {
		return invalid()
	}
	if d.Target.Deployment != nil && !text(d.Target.Deployment.Marker, 256) {
		return invalid()
	}
	for _, origin := range d.Target.Origins {
		if !validOrigin(origin) {
			return invalid()
		}
	}
	for _, route := range append(slices.Clone(d.Scope.Routes), d.Scope.Exclusions...) {
		if route.Validate() != nil || !slices.Contains(d.Target.Origins, route.Origin) {
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
			if !slices.Contains(d.Target.Origins, origin) {
				return invalid()
			}
		}
	}
	if anonymous != 1 {
		return invalid()
	}
	inputs := map[string]bool{}
	sources := 0
	for _, input := range d.Inputs {
		if !validID(input.ID) || inputs[input.ID] || input.Artifact.ValidateExact() != nil || !contentdigest.Valid(input.ContentDigest) || !slices.Contains([]string{"source", "openapi", "request-examples", "documentation"}, input.Role) {
			return invalid()
		}
		inputs[input.ID] = true
		if input.Role == "source" {
			sources++
		}
	}
	if (d.AccessMode == "source-assisted" && sources == 0) || (d.AccessMode == "blackbox" && sources > 0) {
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
