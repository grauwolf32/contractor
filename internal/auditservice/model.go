package auditservice

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
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
	if err := decodeStrict(data, &result); err != nil || validateDraftSelection(result) != nil {
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
	var result BaselineSnapshot
	if err := decodeStrict(data, &result); err != nil || ValidateBaseline(result) != nil {
		return BaselineSnapshot{}, fmt.Errorf("%w: stored Audit baseline is invalid", ErrInvalid)
	}
	return cloneBaseline(result), nil
}

func validateDraftSelection(value DraftSelection) error {
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
	if value.Schema != BaselineSchema || len(value.Inputs) == 0 ||
		value.RuntimeLabels == nil || value.Skills == nil || value.LLMCredentialIDs == nil ||
		value.RuntimeCredentialIDs == nil || value.Inventory.ExecutionManifest.Items == nil {
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
	if normalized, err := normalizeLabels(value.RuntimeLabels); err != nil || !equalStrings(normalized, value.RuntimeLabels) {
		return fmt.Errorf("%w: Audit baseline Runtime labels are invalid", ErrInvalid)
	}
	if err := value.RuntimeConfig.Validate(); err != nil || !equalStrings(value.RuntimeLabels, value.RuntimeConfig.ExplicitLabels()) {
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
	if validateExactArtifact(value.Inventory.Worklist, true) != nil ||
		!validDigest(value.Inventory.SourceContentDigest) ||
		!validDigest(value.Inventory.CanonicalInventoryDigest) ||
		value.Inventory.Gaps == nil || !sort.StringsAreSorted(value.Inventory.Gaps) ||
		auditdomain.ValidateDispatchExecutionManifest(value.Inventory.ExecutionManifest) != nil {
		return fmt.Errorf("%w: Audit baseline inventory is invalid", ErrInvalid)
	}
	if selection := value.Inventory.StandardSelection; selection != nil {
		if selection.Scope == "" || !utf8.ValidString(selection.Scope) || len([]byte(selection.Scope)) > 512 ||
			strings.TrimSpace(selection.Scope) != selection.Scope || strings.ContainsRune(selection.Scope, 0) ||
			len(selection.Levels) == 0 || len(selection.Levels) > config.MaxAuditStandardLevels ||
			len(selection.EntryIDs) == 0 || len(selection.EntryIDs) > config.MaxAuditStandardEntries ||
			!strictSortedAuditValues(selection.Levels) || !strictSortedAuditValues(selection.EntryIDs) {
			return fmt.Errorf("%w: Audit baseline standard selection is invalid", ErrInvalid)
		}
	}
	return nil
}

func strictSortedAuditValues(values []string) bool {
	previous := ""
	for _, value := range values {
		if !utf8.ValidString(value) || value == "" || value <= previous ||
			len([]byte(value)) > 128 || strings.TrimSpace(value) != value || strings.ContainsRune(value, 0) {
			return false
		}
		previous = value
	}
	return true
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
	return runtimeconfig.NormalizeRunLabels(value)
}

func decodeStrict(data []byte, target any) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return err
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		return errors.New("JSON value has trailing data")
	}
	return nil
}

func validateExactArtifact(value auditstore.ExactArtifact, metadata bool) error {
	if value.Ref.ValidateExact() != nil || !validDigest(value.Digest) {
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

func validDigest(value string) bool {
	if len(value) != len("sha256:")+sha256.Size*2 || !strings.HasPrefix(value, "sha256:") {
		return false
	}
	_, err := hex.DecodeString(strings.TrimPrefix(value, "sha256:"))
	return err == nil
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

func digestBytes(value []byte) string {
	digest := sha256.Sum256(value)
	return "sha256:" + hex.EncodeToString(digest[:])
}

func deterministicID(prefix string, values ...string) string {
	digest := sha256.New()
	_, _ = digest.Write([]byte("contractor.audit.identity.v1\x00" + prefix))
	for _, value := range values {
		_, _ = digest.Write([]byte{'\x00'})
		_, _ = digest.Write([]byte(value))
	}
	return prefix + "-" + hex.EncodeToString(digest.Sum(nil))
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

func equalStrings(left, right []string) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
}

func cloneExactInputs(source map[string]auditstore.ExactArtifact) map[string]auditstore.ExactArtifact {
	result := make(map[string]auditstore.ExactArtifact, len(source))
	for name, value := range source {
		if value.Ref.Revision != nil {
			revision := *value.Ref.Revision
			value.Ref.Revision = &revision
		}
		result[name] = value
	}
	return result
}

func cloneBaseline(source BaselineSnapshot) BaselineSnapshot {
	result := source
	result.Inputs = cloneExactInputs(source.Inputs)
	result.RuntimeLabels = append([]string(nil), source.RuntimeLabels...)
	result.RuntimeConfig = source.RuntimeConfig.Clone()
	result.Skills = append([]contracts.RunSkillSnapshot(nil), source.Skills...)
	for index := range result.Skills {
		if result.Skills[index].Source != nil {
			ref := *result.Skills[index].Source
			if ref.Revision != nil {
				revision := *ref.Revision
				ref.Revision = &revision
			}
			result.Skills[index].Source = &ref
		}
	}
	result.LLMCredentialIDs = append([]string(nil), source.LLMCredentialIDs...)
	result.RuntimeCredentialIDs = append([]string(nil), source.RuntimeCredentialIDs...)
	result.Standards = append([]auditstandards.PinnedPackage(nil), source.Standards...)
	for index := range result.Standards {
		cloneExactPackageRef(&result.Standards[index].Catalog)
		cloneExactPackageRef(&result.Standards[index].Retained)
	}
	if source.ProjectHTTPTarget != nil {
		target := *source.ProjectHTTPTarget
		if source.ProjectHTTPTarget.Credential != nil {
			credential := *source.ProjectHTTPTarget.Credential
			target.Credential = &credential
		}
		result.ProjectHTTPTarget = &target
	}
	result.Inventory.Gaps = append([]string(nil), source.Inventory.Gaps...)
	result.Inventory.StandardSelection = cloneAuditStandardSelection(source.Inventory.StandardSelection)
	result.Inventory.ExecutionManifest.Items = append(
		[]auditdomain.ExecutionItem(nil), source.Inventory.ExecutionManifest.Items...,
	)
	for index := range result.Inventory.ExecutionManifest.Items {
		item := &result.Inventory.ExecutionManifest.Items[index]
		if item.TaskRef != nil {
			item.TaskRef = exactRefPointer(*item.TaskRef)
		}
		item.Inputs = append([]auditdomain.ExactInput(nil), item.Inputs...)
		for inputIndex := range item.Inputs {
			if item.Inputs[inputIndex].Ref.Revision != nil {
				revision := *item.Inputs[inputIndex].Ref.Revision
				item.Inputs[inputIndex].Ref.Revision = &revision
			}
		}
	}
	return result
}

func cloneAuditStandardSelection(
	source *config.AuditStandardSelection,
) *config.AuditStandardSelection {
	if source == nil {
		return nil
	}
	result := *source
	result.Levels = append([]string(nil), source.Levels...)
	result.EntryIDs = append([]string(nil), source.EntryIDs...)
	return &result
}

func cloneExactPackageRef(value *auditstandards.ExactPackage) {
	if value != nil && value.Artifact.Revision != nil {
		revision := *value.Artifact.Revision
		value.Artifact.Revision = &revision
	}
}
