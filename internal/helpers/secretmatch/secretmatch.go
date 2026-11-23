// Package secretmatch is the Server's one definition of how known secret
// values are found in text (OBS-R31, LIM-OBS-3). It matches the Runtime's
// contractor_runtime.redaction: text equal to a secret matches whole; only a
// secret of at least MinSubstringBytes UTF-8 bytes matches inside other text,
// and those are replaced longest first, so a short credential such as a proxy
// username never garbles ordinary words and a short secret that prefixes a
// longer one cannot expose the longer one's remainder.
package secretmatch

import (
	"slices"
	"strings"
)

// MinSubstringBytes is LIM-OBS-3: a secret this long is specific enough to be
// matched anywhere in text; a shorter one only as a complete string.
const MinSubstringBytes = 16

// Long returns the non-empty secrets of at least MinSubstringBytes bytes,
// unique, longest first and then in byte order.
func Long(secrets []string) []string {
	result := make([]string, 0, len(secrets))
	for _, secret := range secrets {
		if len(secret) >= MinSubstringBytes && !slices.Contains(result, secret) {
			result = append(result, secret)
		}
	}
	slices.SortFunc(result, func(a, b string) int {
		if len(a) != len(b) {
			return len(b) - len(a)
		}
		return strings.Compare(a, b)
	})
	return result
}

// Contains reports whether text equals a non-empty secret or contains a long
// one.
func Contains(text string, secrets []string) bool {
	for _, secret := range secrets {
		if secret == "" {
			continue
		}
		if secret == text || len(secret) >= MinSubstringBytes && strings.Contains(text, secret) {
			return true
		}
	}
	return false
}

// Redact returns marker when text equals a non-empty secret, and otherwise
// text with every long secret replaced by marker, longest first.
func Redact(text string, secrets []string, marker string) string {
	for _, secret := range secrets {
		if secret != "" && secret == text {
			return marker
		}
	}
	for _, secret := range Long(secrets) {
		text = strings.ReplaceAll(text, secret, marker)
	}
	return text
}
