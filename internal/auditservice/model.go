package auditservice

import (
	"encoding/json"
	"fmt"
	"slices"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/auditbaseline"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/runlabels"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

func encodeDraftSelection(value DraftSelection) (json.RawMessage, error) {
	if err := validateDraftSelection(value); err != nil {
		return nil, err
	}
	return json.Marshal(value)
}

func DecodeDraftSelection(data []byte) (DraftSelection, error) {
	var result DraftSelection
	if err := strictjson.Decode(data, &result); err != nil || validateDraftSelection(result) != nil {
		return DraftSelection{}, fmt.Errorf("%w: stored Audit input selection is invalid", ErrInvalid)
	}
	return result, nil
}

func EncodeBaseline(value BaselineSnapshot) (json.RawMessage, error) {
	if err := ValidateBaseline(value); err != nil {
		return nil, err
	}
	return json.Marshal(value)
}

func DecodeBaseline(data []byte) (BaselineSnapshot, error) {
	var fields map[string]json.RawMessage
	if err := strictjson.Decode(data, &fields); err != nil {
		return BaselineSnapshot{}, fmt.Errorf("%w: stored Audit baseline is invalid", ErrInvalid)
	}
	if raw, exists := fields["engagement"]; exists {
		var engagement auditbaseline.EngagementSnapshot
		if string(raw) == "null" || strictjson.DecodeNonNull(raw, &engagement) != nil {
			return BaselineSnapshot{}, fmt.Errorf("%w: stored Audit engagement is invalid", ErrInvalid)
		}
	}

	var result BaselineSnapshot
	if err := strictjson.Decode(data, &result); err != nil || ValidateBaseline(result) != nil {
		return BaselineSnapshot{}, fmt.Errorf("%w: stored Audit baseline is invalid", ErrInvalid)
	}
	// The decoder owns this graph. Keep the empty-collection normalization
	// formerly supplied by the copy without duplicating populated collections.
	if result.RuntimeConfig.Labels == nil {
		result.RuntimeConfig.Labels = []runtimeconfig.PinnedLabel{}
	}
	if result.RuntimeConfig.LLMCredentialIDs == nil {
		result.RuntimeConfig.LLMCredentialIDs = []string{}
	}
	if result.RuntimeConfig.RuntimeCredentialIDs == nil {
		result.RuntimeConfig.RuntimeCredentialIDs = []string{}
	}

	return result, nil
}

func validateDraftSelection(value DraftSelection) error {
	if err := validateAuditLabels(value.Labels, value.RequiredPlacementLabels); err != nil {
		return err
	}
	if value.RequiredPlacementLabels == nil || contracts.ValidatePlacementLabels(value.RequiredPlacementLabels) != nil {
		return fmt.Errorf("%w: Audit placement requirements are invalid", ErrInvalid)
	}
	if value.Schema != DraftSelectionSchema || len(value.Inputs) == 0 {
		return fmt.Errorf("%w: Audit input selection is invalid", ErrInvalid)
	}
	for name, input := range value.Inputs {
		if !validComponent(name) || validateExactArtifact(input, true) != nil {
			return fmt.Errorf("%w: Audit input selection is invalid", ErrInvalid)
		}
	}
	if _, err := normalizeScope(value.Scope); err != nil {
		return err
	}
	return nil
}

func ValidateBaseline(value BaselineSnapshot) error {
	if value.InitialAssertions != nil && auditdomain.ValidateInitialAssertions(*value.InitialAssertions) != nil {
		return fmt.Errorf("%w: Audit assertion selection is invalid", ErrInvalid)
	}
	if err := validateEngagementBaseline(value); err != nil {
		return err
	}
	if err := validateAuditLabels(value.Labels, value.RequiredPlacementLabels); err != nil {
		return err
	}
	if value.RequiredPlacementLabels == nil || contracts.ValidatePlacementLabels(value.RequiredPlacementLabels) != nil {
		return fmt.Errorf("%w: Audit baseline placement requirements are invalid", ErrInvalid)
	}
	if value.Schema != BaselineSchema || !value.ActiveChecks.Valid() || len(value.Inputs) == 0 ||
		value.RuntimeLabels == nil || value.Skills == nil || value.LLMCredentialIDs == nil ||
		value.RuntimeCredentialIDs == nil || value.Standards == nil {
		return fmt.Errorf("%w: Audit baseline shape is invalid", ErrInvalid)
	}
	for name, input := range value.Inputs {
		if !validComponent(name) || validateExactArtifact(input, true) != nil {
			return fmt.Errorf("%w: Audit baseline input is invalid", ErrInvalid)
		}
	}
	if _, err := normalizeScope(value.Scope); err != nil {
		return err
	}
	if normalized, err := normalizeLabels(value.RuntimeLabels); err != nil || !slices.Equal(normalized, value.RuntimeLabels) {
		return fmt.Errorf("%w: Audit baseline Runtime labels are invalid", ErrInvalid)
	}
	if err := value.RuntimeConfig.Validate(); err != nil || !slices.Equal(value.RuntimeLabels, value.RuntimeConfig.ExplicitLabels()) {
		return fmt.Errorf("%w: Audit baseline RuntimeConfig is invalid", ErrInvalid)
	}
	previous := ""
	for _, skill := range value.Skills {
		if skill.Name <= previous || skill.Source == nil || skill.Initialized() || skill.Validate() != nil {
			return fmt.Errorf("%w: Audit baseline Skill snapshot is invalid", ErrInvalid)
		}
		previous = skill.Name
	}
	if !validSortedIDs(value.LLMCredentialIDs) || !validSortedIDs(value.RuntimeCredentialIDs) {
		return fmt.Errorf("%w: Audit baseline credential snapshot is invalid", ErrInvalid)
	}
	previousStandard := ""
	for _, standard := range value.Standards {
		key := standard.Reference.Scheme + "\x00" + standard.Reference.Version
		if key <= previousStandard || auditstandards.ValidatePinnedPackage(standard) != nil {
			return fmt.Errorf("%w: Audit baseline standard snapshot is invalid", ErrInvalid)
		}
		previousStandard = key
	}
	if value.ProjectHTTPTarget != nil {
		if err := value.ProjectHTTPTarget.Validate(); err != nil {
			return fmt.Errorf("%w: Audit baseline HTTP target is invalid", ErrInvalid)
		}
	}

	return nil
}

func normalizeScope(value Scope) (Scope, error) {
	for _, candidate := range []string{value.Objective, value.Target, value.AuthorizationScope} {
		if !utf8.ValidString(candidate) || strings.ContainsRune(candidate, 0) || len([]byte(candidate)) > maximumScopeValue {
			return Scope{}, fmt.Errorf("%w: Audit scope is invalid", ErrInvalid)
		}
	}
	return value, nil
}

func normalizeLabels(value []string) ([]string, error) {
	return runtimeconfig.NormalizeLabels(value)
}

func validateExactArtifact(value auditstore.ExactArtifact, metadata bool) error {
	if value.Ref.ValidateExact() != nil || !contentdigest.Valid(value.Digest) {
		return ErrInvalid
	}
	if metadata && (value.MediaType == "" || value.SizeBytes < 0) {
		return ErrInvalid
	}
	return nil
}

func validComponent(value string) bool {
	return value != "" && len([]byte(value)) <= 128 && utf8.ValidString(value) &&
		!strings.Contains(value, "/") && !strings.ContainsRune(value, 0)
}

func validSortedIDs(values []string) bool {
	previous := ""
	for _, value := range values {
		if !validComponent(value) || value <= previous {
			return false
		}
		previous = value
	}
	return true
}

func mergeIDs(sets ...[]string) []string {
	values := make(map[string]struct{})
	for _, set := range sets {
		for _, value := range set {
			values[value] = struct{}{}
		}
	}
	result := make([]string, 0, len(values))
	for value := range values {
		result = append(result, value)
	}
	sort.Strings(result)
	return result
}

func validateAuditLabels(labels runlabels.RunMetadataLabels, required []string) error {
	if labels == nil {
		return fmt.Errorf("%w: Audit labels must be an object", ErrInvalid)
	}
	if _, err := runlabels.NormalizeAuditLabels(labels); err != nil {
		return fmt.Errorf("%w: invalid Audit labels: %v", ErrInvalid, err)
	}
	derived, err := labels.PlacementRequirements()
	if err != nil || !slices.Equal(derived, required) {
		return fmt.Errorf("%w: Audit placement requirements differ from labels", ErrInvalid)
	}
	return nil
}
