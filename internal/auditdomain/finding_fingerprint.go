package auditdomain

import (
	"encoding/json"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/strictjson"
)

const (
	FindingFingerprintSchema          = "contractor.finding.fingerprint-projection.v1"
	MaximumFingerprintReportBytes     = 8 * 1024 * 1024
	MaximumFingerprintProjectionBytes = 16 * 1024
	MaximumFingerprintResults         = 10000
)

var fingerprintID = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:/-]{0,159}$`)
var fingerprintHex = regexp.MustCompile(`^[0-9a-f]{64}$`)
var fingerprintCommit = regexp.MustCompile(`^[0-9a-f]{40}$`)
var fingerprintLocator = regexp.MustCompile(`^/runs/[0-9]+/results/[0-9]+$`)

// FindingFingerprint is immutable derived evidence, separate from the model
// tool result and from analyst triage. Only qualified keys can group findings.
type FindingFingerprint struct {
	Schema        string                   `json:"schema"`
	Availability  string                   `json:"availability"`
	Reason        string                   `json:"reason,omitempty"`
	Algorithm     string                   `json:"algorithm,omitempty"`
	Value         string                   `json:"value,omitempty"`
	Adapter       string                   `json:"adapter,omitempty"`
	RuleID        string                   `json:"ruleId,omitempty"`
	Repository    string                   `json:"repository,omitempty"`
	Commit        string                   `json:"commit,omitempty"`
	SourceDigest  string                   `json:"sourceDigest,omitempty"`
	Report        *contracts.ExactArtifact `json:"report,omitempty"`
	Source        *contracts.ExactArtifact `json:"source,omitempty"`
	RecordLocator string                   `json:"recordLocator,omitempty"`
}

// UnavailableFindingFingerprint preserves an observation without guessing a key.
func UnavailableFindingFingerprint(reason string) FindingFingerprint {
	return FindingFingerprint{Schema: FindingFingerprintSchema, Availability: "unavailable", Reason: reason}
}

// EncodeFindingFingerprint validates every persisted projection on write.
func EncodeFindingFingerprint(value FindingFingerprint) ([]byte, error) {
	if err := validateFindingFingerprint(value); err != nil {
		return nil, err
	}
	encoded, err := strictjson.Canonical(value)
	if err != nil || len(encoded) > MaximumFingerprintProjectionBytes {
		return nil, invalid(CodeLimitExceeded, "fingerprint")
	}
	return encoded, nil
}

// DecodeFindingFingerprint rejects malformed stored projections on every read.
func DecodeFindingFingerprint(data []byte) (FindingFingerprint, error) {
	var value FindingFingerprint
	if len(data) > MaximumFingerprintProjectionBytes {
		return value, invalid(CodeLimitExceeded, "fingerprint")
	}
	var members map[string]json.RawMessage
	if err := strictjson.Decode(data, &members); err != nil {
		return value, err
	}
	for _, member := range members {
		if strings.TrimSpace(string(member)) == "null" {
			return value, invalid(CodeInvalid, "fingerprint")
		}
	}
	if err := strictjson.Decode(data, &value); err != nil {
		return value, err
	}
	if (value.Availability == "unavailable" && len(members) != 3) || (value.Availability == "available" && len(members) != 12) {
		return value, invalid(CodeInvalid, "fingerprint")
	}
	return value, validateFindingFingerprint(value)
}

func validateFindingFingerprint(value FindingFingerprint) error {
	if value.Schema != FindingFingerprintSchema {
		return invalid(CodeInvalid, "fingerprint.schema")
	}
	if value.Availability == "unavailable" {
		allowed := map[string]bool{"not_recorded": true, "not_provided": true, "input_limit": true, "producer_key_unavailable": true, "producer_report_invalid": true, "producer_record_ambiguous": true, "asset_unavailable": true, "discarded": true}
		if !allowed[value.Reason] || value.Algorithm != "" || value.Value != "" || value.Adapter != "" || value.RuleID != "" || value.Repository != "" || value.Commit != "" || value.SourceDigest != "" || value.Report != nil || value.Source != nil || value.RecordLocator != "" {
			return invalid(CodeInvalid, "fingerprint.unavailable")
		}
		return nil
	}
	if value.Availability != "available" || value.Reason != "" ||
		value.Algorithm != "contractor/producer-issue/v1" || value.Adapter != "pinned-sarif-native/v1" ||
		!fingerprintHex.MatchString(value.Value) || !fingerprintID.MatchString(value.RuleID) ||
		!validFingerprintRepository(value.Repository) || !fingerprintCommit.MatchString(value.Commit) ||
		!contentdigest.Valid(value.SourceDigest) || value.Report == nil || value.Source == nil ||
		value.Report.Validate() != nil || value.Source.Validate() != nil ||
		!value.Report.HasMetadata() || !value.Source.HasMetadata() ||
		value.Source.Digest != value.SourceDigest || value.Report.MediaType != "application/sarif+json" ||
		value.Report.SizeBytes > MaximumFingerprintReportBytes || !fingerprintLocator.MatchString(value.RecordLocator) {
		return invalid(CodeInvalid, "fingerprint")
	}
	return nil
}

func validFingerprintRepository(repository string) bool {
	parsed, err := url.Parse(repository)
	if err == nil && parsed.User != nil {
		if _, password := parsed.User.Password(); password {
			return false
		}
	}
	return err == nil && len(repository) <= 4096 && parsed.Host != "" &&
		(parsed.Scheme == "https" || parsed.Scheme == "ssh") && parsed.RawQuery == "" &&
		!parsed.ForceQuery && parsed.Fragment == "" && (parsed.Scheme != "https" || parsed.User == nil)
}

// FingerprintPinnedSARIF derives only snapshot-qualified full native keys from
// an owner-pinned report. The caller establishes pinned-input/Git authority;
// neither report prose nor model-authored descriptors supply that authority.
func FingerprintPinnedSARIF(data []byte, repository, commit string, source, report contracts.ExactArtifact, locations []FindingLocation) FindingFingerprint {
	unavailable := UnavailableFindingFingerprint
	if len(data) > MaximumFingerprintReportBytes {
		return unavailable("input_limit")
	}
	var document map[string]json.RawMessage
	if strictjson.Decode(data, &document) != nil || stringField(document, "version") != "2.1.0" {
		return unavailable("producer_report_invalid")
	}
	var runs []map[string]json.RawMessage
	if strictjson.Decode(document["runs"], &runs) != nil {
		return unavailable("producer_report_invalid")
	}
	if len(runs) > 32 {
		return unavailable("input_limit")
	}
	var found *FindingFingerprint
	examined := 0
	for ri, run := range runs {
		var provenance []struct {
			RepositoryURI string `json:"repositoryUri"`
			RevisionID    string `json:"revisionId"`
		}
		// Standard reports may carry additional provenance members. Read only
		// exact fields through their raw maps, with duplicate rejection above.
		var rawProvenance []map[string]json.RawMessage
		if strictjson.Decode(run["versionControlProvenance"], &rawProvenance) != nil {
			continue
		}
		for _, item := range rawProvenance {
			provenance = append(provenance, struct {
				RepositoryURI string `json:"repositoryUri"`
				RevisionID    string `json:"revisionId"`
			}{stringField(item, "repositoryUri"), stringField(item, "revisionId")})
		}
		if len(provenance) != 1 || provenance[0].RepositoryURI != repository || provenance[0].RevisionID != commit {
			continue
		}
		var tool map[string]json.RawMessage
		var driver map[string]json.RawMessage
		if strictjson.Decode(run["tool"], &tool) != nil || strictjson.Decode(tool["driver"], &driver) != nil {
			continue
		}
		name := stringField(driver, "name")
		version := stringField(driver, "semanticVersion")
		if version == "" {
			version = stringField(driver, "version")
		}
		if name == "" || version == "" || len(name) > 160 || len(version) > 160 || strings.ContainsAny(name+version, "\x00\r\n") {
			continue
		}
		var results []map[string]json.RawMessage
		if strictjson.Decode(run["results"], &results) != nil {
			return unavailable("producer_report_invalid")
		}
		examined += len(results)
		if examined > MaximumFingerprintResults {
			return unavailable("input_limit")
		}
		for index, result := range results {
			rule := stringField(result, "ruleId")
			if !fingerprintID.MatchString(rule) || !sarifLocationMatches(result["locations"], locations) {
				continue
			}
			var fingerprints map[string]string
			if strictjson.Decode(result["fingerprints"], &fingerprints) != nil || len(fingerprints) == 0 || len(fingerprints) > 16 {
				continue
			}
			valid := true
			for key, value := range fingerprints {
				if !fingerprintID.MatchString(key) || value == "" || len(value) > 1024 || !utf8.ValidString(value) || strings.ContainsAny(value, "\x00\r\n") {
					valid = false
				}
			}
			if !valid {
				continue
			}
			native := []any{name, version, fingerprints}
			canonical, err := strictjson.Canonical(native)
			if err != nil || len(canonical) > 4096 {
				continue
			}
			digest, err := contentdigest.CanonicalJSON([]any{"contractor/producer-issue/v1", []any{"git", repository}, "contractor/pinned-sarif", "snapshot/v1", rule, []any{"snapshot", []any{commit, source.Digest}}, native})
			if err != nil {
				return unavailable("producer_report_invalid")
			}
			value := FindingFingerprint{Schema: FindingFingerprintSchema, Availability: "available", Algorithm: "contractor/producer-issue/v1", Value: strings.TrimPrefix(digest, "sha256:"), Adapter: "pinned-sarif-native/v1", RuleID: rule, Repository: repository, Commit: commit, SourceDigest: source.Digest, Report: &report, Source: &source, RecordLocator: "/runs/" + strconv.Itoa(ri) + "/results/" + strconv.Itoa(index)}
			if validateFindingFingerprint(value) != nil {
				return unavailable("producer_report_invalid")
			}
			if found != nil {
				return unavailable("producer_record_ambiguous")
			}
			found = &value
		}
	}
	if found == nil {
		return unavailable("producer_key_unavailable")
	}
	return *found
}

func stringField(value map[string]json.RawMessage, key string) string {
	var result string
	_ = strictjson.Decode(value[key], &result)
	return result
}

func sarifLocationMatches(data json.RawMessage, locations []FindingLocation) bool {
	var values []map[string]json.RawMessage
	if strictjson.Decode(data, &values) != nil || len(values) == 0 || len(values) > 256 {
		return false
	}
	var physical, artifact, region map[string]json.RawMessage
	if strictjson.Decode(values[0]["physicalLocation"], &physical) != nil || strictjson.Decode(physical["artifactLocation"], &artifact) != nil || strictjson.Decode(physical["region"], &region) != nil || stringField(artifact, "uriBaseId") != "" {
		return false
	}
	uri, err := url.Parse(stringField(artifact, "uri"))
	if err != nil || uri.IsAbs() || uri.Host != "" || uri.RawQuery != "" || uri.Fragment != "" {
		return false
	}
	file := uri.Path
	if validateFindingSourceLocation(FindingLocation{File: file}) != nil {
		return false
	}
	var line int64
	if strictjson.Decode(region["startLine"], &line) != nil || line < 1 {
		return false
	}
	for _, location := range locations {
		if location.File == file && location.Line != nil && *location.Line == line {
			return true
		}
		if location.File == file && location.Range != nil && location.Range.StartLine == line {
			var end int64
			if strictjson.Decode(region["endLine"], &end) == nil && location.Range.EndLine == end {
				return true
			}
		}
	}
	return false
}
