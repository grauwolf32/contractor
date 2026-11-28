package contracts

import (
	"net"
	"net/url"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

const maxRuntimeEndpointBytes = 2048

// ParseRuntimeEndpoint owns the bounded HTTP endpoint grammar shared by Gateway
// publication, RuntimeConfig and allocations. Callers apply their path/origin
// rules to the returned URL. It performs no network I/O and never echoes input.
func ParseRuntimeEndpoint(field, value string) (*url.URL, error) {
	if len(value) == 0 || len(value) > maxRuntimeEndpointBytes || !utf8.ValidString(value) || value != strings.TrimSpace(value) {
		return nil, Invalidf("%s is invalid", field)
	}
	parsed, err := url.Parse(value)
	if err != nil || parsed.Host == "" || parsed.Hostname() == "" || parsed.User != nil ||
		strings.Contains(value, "#") || parsed.RawQuery != "" || parsed.ForceQuery ||
		(parsed.Scheme != "http" && parsed.Scheme != "https") {
		return nil, Invalidf("%s must be an absolute HTTP(S) URL without userinfo, query, or fragment", field)
	}
	host := parsed.Hostname()
	if strings.Contains(host, "%") || strings.ContainsAny(host, "<>\\^|`{}") ||
		strings.IndexFunc(host, unicode.IsSpace) >= 0 || strings.HasSuffix(parsed.Host, ":") {
		return nil, Invalidf("%s host is invalid", field)
	}
	if port := parsed.Port(); port != "" {
		portNumber, err := strconv.ParseUint(port, 10, 16)
		if err != nil || portNumber == 0 {
			return nil, Invalidf("%s port is invalid", field)
		}
	}
	if net.ParseIP(host) == nil {
		if strings.Contains(host, ":") {
			return nil, Invalidf("%s host is invalid", field)
		}
		labels := strings.Split(strings.TrimSuffix(host, "."), ".")
		last := labels[len(labels)-1]
		if last == "" || allDecimalDigits(last) {
			return nil, Invalidf("%s host is invalid", field)
		}
	}
	return parsed, nil
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

// ParsePrivateRPCOrigin applies the private gRPC origin restriction to the
// common endpoint grammar. Paths cannot disappear during authority derivation.
func ParsePrivateRPCOrigin(field, value string) (*url.URL, error) {
	parsed, err := ParseRuntimeEndpoint(field, value)
	if err != nil {
		return nil, err
	}
	if parsed.Scheme != "https" || (parsed.Path != "" && parsed.Path != "/") {
		return nil, Invalidf("%s must be an HTTPS origin", field)
	}
	return parsed, nil
}
