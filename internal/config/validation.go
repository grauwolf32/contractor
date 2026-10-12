package config

import (
	"fmt"
	"math"
	"net/url"
	"strings"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
)

const maxJSONSafeInteger = 1<<53 - 1

func validateMetadata(metadata *metadataSource) (resolved.Selector, error) {
	if metadata == nil {
		return resolved.Selector{}, fmt.Errorf("metadata is required")
	}
	if !contracts.ValidConfigID(metadata.Name) {
		return resolved.Selector{}, fmt.Errorf("metadata.name %q must match [a-z][a-z0-9_-]* and have at most %d characters",
			metadata.Name, contracts.MaxConfigIDLength)
	}
	if !contracts.ValidConfigVersion(metadata.Version) {
		return resolved.Selector{}, fmt.Errorf("metadata.version %q must match [A-Za-z0-9][A-Za-z0-9._+-]* and have at most %d characters",
			metadata.Version, contracts.MaxConfigVersionLength)
	}
	return resolved.Selector{ID: metadata.Name, Version: metadata.Version}, nil
}

func validateEnvelope(apiVersion, kind, expectedKind string, metadata *metadataSource) (resolved.Selector, error) {
	if apiVersion != "contractor/v1alpha1" {
		return resolved.Selector{}, fmt.Errorf("apiVersion must be %q", "contractor/v1alpha1")
	}
	if kind != expectedKind {
		return resolved.Selector{}, fmt.Errorf("kind %q does not match %s subtree", kind, expectedKind)
	}
	return validateMetadata(metadata)
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

func validateModelPolicySpec(spec *modelPolicySpecSource) error {
	if spec == nil {
		return fmt.Errorf("spec is required")
	}
	if strings.TrimSpace(spec.Model) == "" {
		return fmt.Errorf("spec.model must not be empty or whitespace-only")
	}
	if spec.ContextWindowTokens != nil &&
		(*spec.ContextWindowTokens <= 0 || *spec.ContextWindowTokens > contracts.MaxModelContextTokens) {
		return fmt.Errorf("spec.contextWindowTokens must be between 1 and %d when present", contracts.MaxModelContextTokens)
	}
	if spec.MaxOutputTokens != nil && *spec.MaxOutputTokens <= 0 {
		return fmt.Errorf("spec.maxOutputTokens must be positive when present")
	}
	if spec.MaxOutputTokens != nil && *spec.MaxOutputTokens > maxJSONSafeInteger {
		return fmt.Errorf("spec.maxOutputTokens exceeds the I-JSON safe integer range")
	}
	if spec.MaxModelCalls != nil && (*spec.MaxModelCalls <= 0 || *spec.MaxModelCalls > contracts.MaxWorkerModelCalls) {
		return fmt.Errorf("spec.maxModelCalls must be between 1 and %d when present", contracts.MaxWorkerModelCalls)
	}
	if spec.MaxToolCalls != nil && (*spec.MaxToolCalls <= 0 || *spec.MaxToolCalls > contracts.MaxWorkerToolCalls) {
		return fmt.Errorf("spec.maxToolCalls must be between 1 and %d when present", contracts.MaxWorkerToolCalls)
	}
	if spec.MaxWorkerCalls != nil && (*spec.MaxWorkerCalls <= 0 || *spec.MaxWorkerCalls > contracts.MaxPlannerWorkerCalls) {
		return fmt.Errorf("spec.maxWorkerCalls must be between 1 and %d when present", contracts.MaxPlannerWorkerCalls)
	}
	if spec.MaxTotalTokens != nil && (*spec.MaxTotalTokens <= 0 || *spec.MaxTotalTokens > contracts.MaxWorkerTotalTokens) {
		return fmt.Errorf("spec.maxTotalTokens must be between 1 and %d when present", contracts.MaxWorkerTotalTokens)
	}
	if spec.ContextWindowTokens != nil && spec.MaxOutputTokens != nil &&
		*spec.MaxOutputTokens >= *spec.ContextWindowTokens {
		return fmt.Errorf("spec.maxOutputTokens must be below contextWindowTokens")
	}
	if spec.Temperature != nil && (math.IsNaN(*spec.Temperature) || math.IsInf(*spec.Temperature, 0) || *spec.Temperature < 0) {
		return fmt.Errorf("spec.temperature must be finite and non-negative")
	}
	return nil
}
