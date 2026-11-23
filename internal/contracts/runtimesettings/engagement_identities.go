package runtimesettings

import (
	"regexp"
	"strings"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/contracts"
)

// HTTPIdentityMaterial is private allocation material for one exact, held
// target credential. It never belongs in the engagement or public provenance.
type HTTPIdentityMaterial struct {
	Credential RuntimeCredentialRef              `json:"credential" strictjson:"required"`
	Fields     map[string]contracts.SecretString `json:"fields" strictjson:"required"`
}

var targetIdentityPattern = regexp.MustCompile(`^[a-z][a-z0-9-]{0,63}$`)

func (m HTTPIdentityMaterial) Validate() error {
	if m.Credential.Kind == RuntimeCredentialTargetAPIKeys {
		if m.Credential.Validate() != nil {
			return contracts.Invalidf("engagement identity material is invalid")
		}
		return ValidateAPIKeyFields(m.Fields)
	}
	if m.Credential.Validate() != nil || len(m.Fields) < 1 || len(m.Fields) > 128 {
		return contracts.Invalidf("engagement identity material is invalid")
	}
	total := 0
	for name, field := range m.Fields {
		value := field.Reveal()
		if m.Credential.Kind == RuntimeCredentialTargetCookies {
			if !contracts.ValidCookieName(name) || !contracts.ValidCookieValue(value) || len(value) > 8192 {
				return contracts.Invalidf("engagement cookie material is invalid")
			}
			total += len(name)
		} else if !validTargetMaterialValue(m.Credential.Kind, value) {
			return contracts.Invalidf("engagement identity material is invalid")
		}
		total += len(value)
	}
	if total > 16*1024 {
		return contracts.Invalidf("engagement identity material exceeds its byte bound")
	}
	switch m.Credential.Kind {
	case RuntimeCredentialTargetBasic:
		username := m.Fields["username"].Reveal()
		if len(m.Fields) != 2 || username == "" || len(username) > 256 || strings.Contains(username, ":") || m.Fields["password"].Reveal() == "" {
			return contracts.Invalidf("engagement basic material is invalid")
		}
	case RuntimeCredentialTargetBearer:
		if len(m.Fields) != 1 || m.Fields["token"].Reveal() == "" {
			return contracts.Invalidf("engagement bearer material is invalid")
		}
	case RuntimeCredentialTargetCookies:
	default:
		return contracts.Invalidf("engagement credential kind is invalid")
	}
	return nil
}

func validTargetMaterialValue(kind RuntimeCredentialKind, value string) bool {
	if len(value) < 1 || len(value) > 8192 {
		return false
	}
	if kind == RuntimeCredentialTargetBearer {
		return contracts.ValidCredentialFieldValue(value)
	}
	if !utf8.ValidString(value) {
		return false
	}
	for _, raw := range []byte(value) {
		if raw < 0x20 || raw == 0x7f {
			return false
		}
	}
	return true
}

// ValidateAPIKeyFields is shared by encrypted credential creation and private
// allocation validation. Prefixes identify placement; every value stays private.
func ValidateAPIKeyFields(fields map[string]contracts.SecretString) error {
	invalid := func() error { return contracts.Invalidf("engagement API key material is invalid") }
	if len(fields) < 1 || len(fields) > 128 {
		return invalid()
	}
	total := 0
	headers := map[string]bool{}
	for qualified, secret := range fields {
		location, name, found := strings.Cut(qualified, ":")
		value := secret.Reveal()
		if !found || len(name) < 1 || len(name) > 256 || len(value) < 1 || len(value) > 8192 || !utf8.ValidString(name) || !utf8.ValidString(value) {
			return invalid()
		}
		total += len(qualified) + len(value)
		if total > 16*1024 {
			return invalid()
		}
		switch location {
		case "header":
			lower := strings.ToLower(name)
			_, forbidden := forbiddenRuntimeHeaderNames[lower]
			if !headerNamePattern.MatchString(name) || forbidden || lower == "cookie" || lower == "set-cookie" || headers[lower] || !contracts.ValidCredentialFieldValue(value) {
				return invalid()
			}
			headers[lower] = true
		case "cookie":
			if !contracts.ValidCookieName(name) || !contracts.ValidCookieValue(value) {
				return invalid()
			}
		case "query":
			for _, text := range []string{name, value} {
				for _, c := range text {
					if c < 32 || c == 127 {
						return invalid()
					}
				}
			}
		default:
			return invalid()
		}
	}
	return nil
}
