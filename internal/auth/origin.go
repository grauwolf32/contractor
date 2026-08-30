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

func NewOriginPolicy(origins []string, allowInsecureLoopback bool) (OriginPolicy, error) {
	if len(origins) == 0 || len(origins) > maximumBrowserOrigins {
		return OriginPolicy{}, fmt.Errorf("%w: one through %d browser origins are required", ErrInvalidOrigin, maximumBrowserOrigins)
	}
	result := OriginPolicy{allowed: make(map[string]struct{}, len(origins))}
	for _, source := range origins {
		if err := validateOrigin(source, allowInsecureLoopback); err != nil {
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

func validateOrigin(source string, allowInsecureLoopback bool) error {
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
	if parsed.Scheme != "http" || !allowInsecureLoopback || !loopbackHostname(parsed.Hostname()) {
		return fmt.Errorf("%w: browser origin must use HTTPS", ErrInvalidOrigin)
	}
	return nil
}

func loopbackHostname(host string) bool {
	if strings.EqualFold(host, "localhost") {
		return true
	}
	address := net.ParseIP(host)
	return address != nil && address.IsLoopback()
}
