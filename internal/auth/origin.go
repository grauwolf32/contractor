package auth

import (
	"fmt"
	"net"
	"net/url"
	"sort"
	"strings"
)

const maximumBrowserOrigins = 16

type OriginPolicy struct {
	allowed map[string]struct{}
	values  []string
}

func NewOriginPolicy(origins []string, allowInsecureDevelopment bool) (OriginPolicy, error) {
	if len(origins) == 0 || len(origins) > maximumBrowserOrigins {
		return OriginPolicy{}, fmt.Errorf("%w: one through %d browser origins are required", ErrInvalidOrigin, maximumBrowserOrigins)
	}
	result := OriginPolicy{allowed: make(map[string]struct{}, len(origins))}
	for _, source := range origins {
		if err := validateOrigin(source, allowInsecureDevelopment); err != nil {
			return OriginPolicy{}, err
		}
		if _, duplicate := result.allowed[source]; duplicate {
			return OriginPolicy{}, fmt.Errorf("%w: duplicate browser origin", ErrInvalidOrigin)
		}
		result.allowed[source] = struct{}{}
		result.values = append(result.values, source)
	}
	sort.Strings(result.values)
	return result, nil
}

func (p OriginPolicy) Allows(origin string) bool {
	_, ok := p.allowed[origin]
	return ok
}

func (p OriginPolicy) Values() []string { return append([]string(nil), p.values...) }

func validateOrigin(source string, allowInsecureDevelopment bool) error {
	if source == "" || len(source) > 2048 || source == "*" || source == "null" || strings.TrimSpace(source) != source {
		return ErrInvalidOrigin
	}
	parsed, err := url.Parse(source)
	if err != nil || parsed.Host == "" || parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" ||
		parsed.Path != "" || parsed.RawPath != "" || parsed.Opaque != "" || parsed.Scheme != strings.ToLower(parsed.Scheme) ||
		parsed.Host != strings.ToLower(parsed.Host) || parsed.String() != source {
		return fmt.Errorf("%w: browser origin must be one canonical origin", ErrInvalidOrigin)
	}
	if parsed.Scheme == "https" {
		return nil
	}
	if parsed.Scheme != "http" || !allowInsecureDevelopment || !developmentHostname(parsed.Hostname()) {
		return fmt.Errorf("%w: browser origin must use HTTPS", ErrInvalidOrigin)
	}
	return nil
}

func developmentHostname(host string) bool {
	if strings.EqualFold(host, "localhost") {
		return true
	}
	address := net.ParseIP(host)
	if address == nil || strings.Contains(host, ":") {
		return false
	}
	if address.IsLoopback() {
		return true
	}
	ipv4 := address.To4()
	return ipv4 != nil && (ipv4[0] == 10 ||
		ipv4[0] == 172 && ipv4[1] >= 16 && ipv4[1] <= 31 ||
		ipv4[0] == 192 && ipv4[1] == 168)
}
