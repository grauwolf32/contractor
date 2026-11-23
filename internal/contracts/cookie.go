package contracts

import "regexp"

var cookieNamePattern = regexp.MustCompile("^[!#$%&'*+.^_`|~0-9A-Za-z-]{1,256}$")

// ValidCookieName validates a supplied cookie name without its value.
func ValidCookieName(value string) bool { return cookieNamePattern.MatchString(value) }

// ValidCookieValue accepts the RFC 6265 cookie-octet subset, including empty values.
func ValidCookieValue(value string) bool {
	for _, c := range []byte(value) {
		if c < 0x21 || c > 0x7e || c == '"' || c == ',' || c == ';' || c == '\\' {
			return false
		}
	}
	return true
}
