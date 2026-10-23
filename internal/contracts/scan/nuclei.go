package scan

import (
	"encoding/json"
	"net/url"
	"regexp"
	"strings"

	"github.com/grauwolf32/contractor/internal/artifactpolicy"
	"github.com/grauwolf32/contractor/internal/contracts"
)

const (
	NucleiRulepackMediaType    = "application/vnd.contractor.nuclei-rulepack+zip"
	MaxNucleiRulepackZIPBytes  = 16 * 1024 * 1024
	MaxNucleiRulepackFileBytes = 1024 * 1024
)

var nucleiFilter = regexp.MustCompile(`^[A-Za-z0-9_*][A-Za-z0-9_.*-]*(,[A-Za-z0-9_*][A-Za-z0-9_.*-]*)*$`)

// ValidateNucleiArguments validates resolved tool inputs at ordinary Run admission.
// It owns the scanner contract independently of a Workflow name; scan planners
// validate their generated inputs separately.
func ValidateNucleiArguments(arguments map[string]any) error {
	for name := range arguments {
		switch name {
		case "url", "template_ids", "tags", "severity", "rate_limit", "timeout_seconds", "templates_digest", "rulepack_ref":
		default:
			return contracts.Invalidf("unknown Nuclei argument")
		}
	}
	target, ok := arguments["url"].(string)
	if !ok {
		return contracts.Invalidf("Nuclei URL is required")
	}
	request := PreparedHTTPRequest{Method: "GET", URL: target, Headers: []HTTPRequestHeader{}}
	if request.Validate() != nil || !(strings.HasPrefix(target, "http://") || strings.HasPrefix(target, "https://")) {
		return contracts.Invalidf("invalid Nuclei URL")
	}
	u, err := url.Parse(target)
	if err != nil || !ValidHost(u.Hostname()) || strings.Contains(u.Host, "%") || (strings.HasPrefix(u.Host, "[") && !strings.Contains(u.Hostname(), ":")) {
		return contracts.Invalidf("invalid Nuclei host")
	}
	for _, name := range []string{"template_ids", "tags", "severity"} {
		if value, exists := arguments[name]; exists {
			text, ok := value.(string)
			if !ok || len(text) > 2048 || (text != "" && !nucleiFilter.MatchString(text)) {
				return contracts.Invalidf("invalid Nuclei filter")
			}
			if name == "severity" && text != "" {
				for _, severity := range strings.Split(text, ",") {
					switch severity {
					case "info", "low", "medium", "high", "critical", "unknown":
					default:
						return contracts.Invalidf("invalid Nuclei severity")
					}
				}
			}
		}
	}
	for _, pair := range []struct {
		name string
		max  int64
	}{{"rate_limit", 1000}, {"timeout_seconds", 3600}} {
		if value, exists := arguments[pair.name]; exists {
			data, err := json.Marshal(value)
			if err != nil {
				return contracts.Invalidf("invalid Nuclei integer")
			}
			var number int64
			if json.Unmarshal(data, &number) != nil || number < 1 || number > pair.max {
				return contracts.Invalidf("invalid Nuclei integer")
			}
		}
	}
	if value, exists := arguments["templates_digest"]; exists {
		digest, ok := value.(string)
		if !ok || (digest != "" && contracts.ValidateDigest("templates_digest", digest) != nil) {
			return contracts.Invalidf("invalid Nuclei package digest")
		}
	}
	if value, exists := arguments["rulepack_ref"]; exists {
		encoded, err := json.Marshal(value)
		if err != nil {
			return contracts.Invalidf("invalid Nuclei rulepack ref")
		}
		var ref contracts.ArtifactRef
		if _, err := contracts.DecodeStrictObject(encoded, &ref); err != nil || ref.ValidateExact() != nil || artifactpolicy.IsModelHiddenBinding(ref.Namespace, ref.Name) {
			return contracts.Invalidf("invalid Nuclei rulepack ref")
		}
	}

	return nil
}
