package scan

import (
	"net"
	"strings"
	"unicode/utf8"

	"golang.org/x/net/idna"
)

var scanIDNA = idna.New(idna.ValidateForRegistration(), idna.BidiRule())

// ValidHost accepts the shared scanner DNS/IP host grammar without resolution.
func ValidHost(host string) bool {
	if !utf8.ValidString(host) || strings.Contains(host, "%") {
		return false
	}
	if net.ParseIP(host) != nil {
		return true
	}
	name := strings.TrimSuffix(strings.ToLower(host), ".")
	if !isASCII(name) {
		var err error
		name, err = scanIDNA.ToASCII(name)
		if err != nil {
			return false
		}
	}
	if len(name) == 0 || len(name) > 253 {
		return false
	}
	for _, label := range strings.Split(name, ".") {
		if len(label) == 0 || len(label) > 63 || label[0] == '-' {
			return false
		}
		for _, char := range label {
			if !(char >= 'a' && char <= 'z' || char >= '0' && char <= '9' || char == '-' || char == '_') {
				return false
			}
		}
		if strings.HasPrefix(label, "xn--") {
			if _, err := scanIDNA.ToUnicode(label); err != nil {
				return false
			}
		}
	}
	return true
}

func isASCII(value string) bool {
	for _, c := range value {
		if c > 127 {
			return false
		}
	}
	return true
}
