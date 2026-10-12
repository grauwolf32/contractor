package resolved

import (
	"fmt"
	"strings"

	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func validateExecutionConfigRef(ref ExecutionConfigRef) error {
	if _, err := ParseSelector(ref.ConfigID + "@" + ref.Version); err != nil {
		return err
	}
	if !contentdigest.Valid(ref.Digest) {
		return fmt.Errorf("digest %q must use sha256:<64 lowercase hex>", ref.Digest)
	}
	return nil
}

// The checks below are shared by the catalog loader, which applies them to
// authored manifests, and by the rules here and in auditprofile, which apply
// them again to persisted snapshots. field names the checked value in the
// error.

// ValidateIdentifier rejects an empty or whitespace-only value.
func ValidateIdentifier(field, value string) error {
	if strings.TrimSpace(value) == "" {
		return fmt.Errorf("%s must not be empty or whitespace-only", field)
	}
	return nil
}

// ValidateWorkflowName applies the shared name grammar of contracts.ValidName
// to a Stage or logical Agent name, wherever a Workflow, ExecutionConfig or
// persisted snapshot declares or references one. The public API and
// Operations show both as ConfigId values (WFL-R84), so the catalog accepts
// no name they could not show.
func ValidateWorkflowName(field, value string) error {
	if !contracts.ValidName(value) {
		return fmt.Errorf(
			"%s %q must be 1 to %d ASCII letters, digits, dots, underscores or hyphens, starting with a letter or digit",
			field, value, contracts.MaxNameLength,
		)
	}
	return nil
}

// validateStageAgentNames checks every logical Agent name of one resolved
// Stage; field names the Stage in the error.
func validateStageAgentNames(field string, agents map[string]ResolvedAgentBinding) error {
	for logicalName := range agents {
		if err := ValidateWorkflowName(field+" logical Agent name", logicalName); err != nil {
			return err
		}
	}
	return nil
}

// ValidateMediaTypes checks one slot's accepted media types: a non-empty list
// of distinct canonical media types, or the single wildcard */*. It returns a
// copy the caller owns.
func ValidateMediaTypes(field string, values []string) ([]string, error) {
	if len(values) == 0 {
		return nil, fmt.Errorf("%s.mediaTypes must be a non-empty list", field)
	}
	seen := make(map[string]struct{}, len(values))
	result := append([]string(nil), values...)
	for _, value := range result {
		if value != "*/*" && !contracts.ValidMediaType(value) {
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

// MediaTypesIntersect reports whether two slots accept a common media type;
// the wildcard */* accepts every type.
func MediaTypesIntersect(left, right []string) bool {
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

// ValidateMapKey checks a slot, parameter or mapping name: non-empty and
// without a slash.
func ValidateMapKey(field, value string) error {
	if err := ValidateIdentifier(field, value); err != nil {
		return err
	}
	if strings.Contains(value, "/") {
		return fmt.Errorf("%s %q must not contain slash", field, value)
	}
	return nil
}

// ValidateArtifactComponent checks an artifact namespace or name against the
// ArtifactStore name grammar.
func ValidateArtifactComponent(field, value string) error {
	if err := contracts.ValidateArtifactName(value); err != nil {
		return fmt.Errorf("%s: %w", field, err)
	}
	return nil
}
