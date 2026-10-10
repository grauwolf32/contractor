package runtimesettings

// Validators shared by the Runtime settings and their provenance: Runtime
// endpoints, CA bundles and sorted label bindings.

import (
	"net"
	"net/url"
	"strconv"
	"strings"
	"unicode"

	"github.com/grauwolf32/contractor/internal/cabundle"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func validateRuntimeEndpoint(field, value string) error {
	if len(value) == 0 || len([]byte(value)) > 2048 || value != strings.TrimSpace(value) {
		return contracts.Invalidf("%s is invalid", field)
	}
	parsed, err := url.Parse(value)
	if err != nil || parsed.Host == "" || parsed.Hostname() == "" || parsed.User != nil ||
		strings.Contains(value, "#") || parsed.RawQuery != "" || parsed.ForceQuery ||
		(parsed.Scheme != "http" && parsed.Scheme != "https") {
		return contracts.Invalidf("%s must be an absolute HTTP(S) URL without userinfo, query, or fragment", field)
	}
	host := parsed.Hostname()
	if strings.Contains(host, "%") || strings.ContainsAny(host, "<>\\^|`{}") ||
		strings.IndexFunc(host, unicode.IsSpace) >= 0 || strings.HasSuffix(parsed.Host, ":") {
		return contracts.Invalidf("%s host is invalid", field)
	}
	if port := parsed.Port(); port != "" {
		portNumber, err := strconv.ParseUint(port, 10, 16)
		if err != nil || portNumber == 0 {
			return contracts.Invalidf("%s port is invalid", field)
		}
	}
	if net.ParseIP(host) == nil {
		if strings.Contains(host, ":") {
			return contracts.Invalidf("%s host is invalid", field)
		}
		labels := strings.Split(strings.TrimSuffix(host, "."), ".")
		last := labels[len(labels)-1]
		if last == "" || allDecimalDigits(last) {
			return contracts.Invalidf("%s host is invalid", field)
		}
	}
	return nil
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

func validateCABundle(owner, value string) error {
	if err := cabundle.Validate(value); err != nil {
		return contracts.Invalidf("%s CA bundle is invalid", owner)
	}
	return nil
}

func validateProvenanceBindings(field string, values []RuntimeLabelBindingProvenance) error {
	if values == nil || len(values) > 32 {
		return contracts.Invalidf("provenance %s must be a non-null bounded array", field)
	}
	previous := ""
	for _, value := range values {
		if len(value.Label) == 0 || len(value.Label) > 63 || value.Label == "default" ||
			!contracts.ValidIdentifier(value.Label) || value.Label <= previous || value.BindingRevision == 0 {
			return contracts.Invalidf("provenance %s contains invalid or unsorted bindings", field)
		}
		if err := value.Config.Validate(); err != nil {
			return err
		}
		previous = value.Label
	}
	return nil
}
