package config

import (
	"fmt"
	"math"
	"net/url"
	"regexp"
	"strings"
)

var (
	idPattern        = regexp.MustCompile(`^[a-z][a-z0-9_-]*$`)
	versionPattern   = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._+-]*$`)
	mediaTypePattern = regexp.MustCompile(`^[a-z0-9!#$&^_.+-]+/[a-z0-9!#$&^_.+-]+$`)
)

const maxJSONSafeInteger = 1<<53 - 1

// ParseSelector validates and separates an exact <id>@<version> selector.
func ParseSelector(raw string) (Selector, error) {
	if strings.Count(raw, "@") != 1 {
		return Selector{}, fmt.Errorf("selector %q must use exact <id>@<version> syntax", raw)
	}
	id, version, _ := strings.Cut(raw, "@")
	if !idPattern.MatchString(id) {
		return Selector{}, fmt.Errorf("selector %q has invalid id", raw)
	}
	if !versionPattern.MatchString(version) {
		return Selector{}, fmt.Errorf("selector %q has invalid version", raw)
	}
	return Selector{ID: id, Version: version}, nil
}

func validateMetadata(metadata *metadataSource) (Selector, error) {
	if metadata == nil {
		return Selector{}, fmt.Errorf("metadata is required")
	}
	if !idPattern.MatchString(metadata.Name) {
		return Selector{}, fmt.Errorf("metadata.name %q does not match %s", metadata.Name, idPattern)
	}
	if !versionPattern.MatchString(metadata.Version) {
		return Selector{}, fmt.Errorf("metadata.version %q does not match %s", metadata.Version, versionPattern)
	}
	return Selector{ID: metadata.Name, Version: metadata.Version}, nil
}

func validateEnvelope(apiVersion, kind, expectedKind string, metadata *metadataSource) (Selector, error) {
	if apiVersion != "contractor/v1alpha1" {
		return Selector{}, fmt.Errorf("apiVersion must be %q", "contractor/v1alpha1")
	}
	if kind != expectedKind {
		return Selector{}, fmt.Errorf("kind %q does not match %s subtree", kind, expectedKind)
	}
	return validateMetadata(metadata)
}

func validateIdentifier(field, value string) error {
	if strings.TrimSpace(value) == "" {
		return fmt.Errorf("%s must not be empty or whitespace-only", field)
	}
	return nil
}

func validateInstructionRef(raw string) (string, error) {
	if raw == "" {
		return "", fmt.Errorf("instruction ref is required")
	}
	if strings.Contains(raw, `\`) {
		return "", fmt.Errorf("instruction ref %q contains a backslash", raw)
	}
	parsed, err := url.Parse(raw)
	if err != nil {
		return "", fmt.Errorf("parse instruction ref %q: %w", raw, err)
	}
	if parsed.IsAbs() || parsed.Scheme != "" || strings.HasPrefix(raw, "/") {
		return "", fmt.Errorf("instruction ref %q must be relative and have no URI scheme", raw)
	}
	segments := strings.Split(raw, "/")
	for _, segment := range segments {
		if segment == "" || segment == "." || segment == ".." {
			return "", fmt.Errorf("instruction ref %q contains an invalid path segment", raw)
		}
	}
	if len(segments) < 2 || segments[0] != "instructions" {
		return "", fmt.Errorf("instruction ref %q must be below instructions/", raw)
	}
	return strings.Join(segments, "/"), nil
}

func validateMediaTypes(field string, values []string) ([]string, error) {
	if len(values) == 0 {
		return nil, fmt.Errorf("%s.mediaTypes must be a non-empty list", field)
	}
	seen := make(map[string]struct{}, len(values))
	result := append([]string(nil), values...)
	for _, value := range result {
		if value != "*/*" && !mediaTypePattern.MatchString(value) {
			return nil, fmt.Errorf("%s.mediaTypes contains non-canonical media type %q", field, value)
		}
		if _, exists := seen[value]; exists {
			return nil, fmt.Errorf("%s.mediaTypes contains duplicate %q", field, value)
		}
		seen[value] = struct{}{}
	}
	if _, any := seen["*/*"]; any && len(seen) != 1 {
		return nil, fmt.Errorf("%s.mediaTypes cannot combine */* with specific values", field)
	}
	return result, nil
}

func mediaTypesIntersect(left, right []string) bool {
	if len(left) == 1 && left[0] == "*/*" || len(right) == 1 && right[0] == "*/*" {
		return true
	}
	rightSet := make(map[string]struct{}, len(right))
	for _, item := range right {
		rightSet[item] = struct{}{}
	}
	for _, item := range left {
		if _, ok := rightSet[item]; ok {
			return true
		}
	}
	return false
}

func validateModelPolicySpec(spec *modelPolicySpecSource) error {
	if spec == nil {
		return fmt.Errorf("spec is required")
	}
	if strings.TrimSpace(spec.Model) == "" {
		return fmt.Errorf("spec.model must not be empty or whitespace-only")
	}
	if spec.MaxOutputTokens <= 0 {
		return fmt.Errorf("spec.maxOutputTokens must be positive")
	}
	if spec.MaxOutputTokens > maxJSONSafeInteger {
		return fmt.Errorf("spec.maxOutputTokens exceeds the I-JSON safe integer range")
	}
	if spec.Temperature != nil && (math.IsNaN(*spec.Temperature) || math.IsInf(*spec.Temperature, 0) || *spec.Temperature < 0) {
		return fmt.Errorf("spec.temperature must be finite and non-negative")
	}
	return nil
}
