package auditstandards

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/url"
	"regexp"
	"sort"
	"strings"
	"unicode/utf8"
)

var (
	schemePattern  = regexp.MustCompile(`^[a-z][a-z0-9.-]{0,63}$`)
	versionPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$`)
	entryIDPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9:._/-]{0,127}$`)
	rolePattern    = regexp.MustCompile(`^[a-z][a-z0-9_-]{0,63}$`)
)

var supportedLicenses = map[string]map[DisclosurePolicy]bool{
	"CC-BY-4.0":              {DisclosureMetadata: true, DisclosureIdentifiers: true, DisclosureFull: true},
	"CC-BY-SA-4.0":           {DisclosureMetadata: true, DisclosureIdentifiers: true, DisclosureFull: true},
	"Apache-2.0":             {DisclosureMetadata: true, DisclosureIdentifiers: true, DisclosureFull: true},
	"MIT":                    {DisclosureMetadata: true, DisclosureIdentifiers: true, DisclosureFull: true},
	"LicenseRef-Proprietary": {DisclosureMetadata: true},
}

var allowedMethods = map[string]bool{
	"source-analysis": true, "configuration-review": true,
	"documentation-review": true, "active-test": true, "manual-review": true,
}

var allowedAssessments = map[string]bool{
	"satisfied": true, "violated": true, "inconclusive": true,
	"not-tested": true, "blocked": true, "not-applicable": true,
	"supported": true, "refuted": true,
}

var allowedEvidenceKinds = map[string]bool{
	"artifact": true, "observation": true, "tool-result": true,
	"manual-attestation": true, "runtime-metric": true,
}

func DecodeDocument(data []byte) (Document, error) {
	if len(data) == 0 || len(data) > MaximumManifestBytes || !utf8.Valid(data) || bytes.IndexByte(data, 0) >= 0 {
		return Document{}, validationError(CodeLimitExceeded, ManifestPath)
	}
	if err := rejectDuplicateJSONKeys(data); err != nil {
		return Document{}, validationError(CodeManifestInvalid, ManifestPath)
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var document Document
	if err := decoder.Decode(&document); err != nil {
		return Document{}, validationError(CodeManifestInvalid, ManifestPath)
	}
	if err := requireJSONEOF(decoder); err != nil {
		return Document{}, validationError(CodeManifestInvalid, ManifestPath)
	}
	normalizeDocument(&document)
	if err := validateDocument(document); err != nil {
		return Document{}, err
	}
	return document, nil
}

func normalizeDocument(document *Document) {
	if document.EvidenceContracts == nil {
		document.EvidenceContracts = []EvidenceContract{}
	}
	if document.Entries == nil {
		document.Entries = []Entry{}
	}
	if document.Mappings == nil {
		document.Mappings = []Mapping{}
	}
	for index := range document.EvidenceContracts {
		contract := &document.EvidenceContracts[index]
		if contract.Assessments == nil {
			contract.Assessments = []string{}
		}
		if contract.EvidenceKinds == nil {
			contract.EvidenceKinds = []string{}
		}
		sort.Strings(contract.Assessments)
		sort.Strings(contract.EvidenceKinds)
	}
	for index := range document.Entries {
		entry := &document.Entries[index]
		if entry.AllowedMethods == nil {
			entry.AllowedMethods = []string{}
		}
		sort.Strings(entry.AllowedMethods)
	}
	for index := range document.Mappings {
		mapping := &document.Mappings[index]
		if mapping.EntryIDs == nil {
			mapping.EntryIDs = []string{}
		}
		sort.Strings(mapping.EntryIDs)
	}
	sort.Slice(document.EvidenceContracts, func(i, j int) bool {
		left, right := document.EvidenceContracts[i], document.EvidenceContracts[j]
		if left.ID == right.ID {
			return left.Version < right.Version
		}
		return left.ID < right.ID
	})
	sort.Slice(document.Entries, func(i, j int) bool { return document.Entries[i].ID < document.Entries[j].ID })
	sort.Slice(document.Mappings, func(i, j int) bool { return document.Mappings[i].Key < document.Mappings[j].Key })
}

func validateDocument(document Document) error {
	if document.Schema != Schema {
		return validationError(CodeManifestInvalid, ManifestPath)
	}
	metadata := document.Standard
	if !schemePattern.MatchString(metadata.Scheme) || !versionPattern.MatchString(metadata.Version) {
		return validationError(CodeIdentityMismatch, ManifestPath)
	}
	if !validText(metadata.Title, 512, true) || !validText(metadata.Description, 4096, true) ||
		!validText(metadata.Source.Name, 512, true) || !validText(metadata.Source.Revision, 512, false) ||
		!validHTTPSURL(metadata.Source.URL) {
		return validationError(CodeManifestInvalid, ManifestPath)
	}
	if !validText(metadata.License.Attribution, 4096, true) || !validHTTPSURL(metadata.License.URL) {
		return validationError(CodeLicenseInvalid, ManifestPath)
	}
	policies, supported := supportedLicenses[metadata.License.ID]
	if !supported || !policies[metadata.License.Disclosure] {
		return validationError(CodeLicenseInvalid, ManifestPath)
	}
	if len(document.EvidenceContracts) == 0 || len(document.EvidenceContracts) > MaximumEvidenceContracts ||
		len(document.Entries) == 0 || len(document.Entries) > MaximumEntries ||
		len(document.Mappings) > MaximumMappings {
		return validationError(CodeLimitExceeded, ManifestPath)
	}

	contracts := make(map[string]EvidenceContract, len(document.EvidenceContracts))
	for _, contract := range document.EvidenceContracts {
		if !entryIDPattern.MatchString(contract.ID) || !versionPattern.MatchString(contract.Version) ||
			contract.MinimumEvidence < 0 || contract.MaximumEvidence < contract.MinimumEvidence ||
			contract.MaximumEvidence > 64 || len(contract.Assessments) == 0 || len(contract.Assessments) > 16 ||
			len(contract.EvidenceKinds) > 16 {
			return validationError(CodeManifestInvalid, ManifestPath)
		}
		if contract.HumanReview != "never" && contract.HumanReview != "on-inconclusive" && contract.HumanReview != "required" {
			return validationError(CodeManifestInvalid, ManifestPath)
		}
		if !validUniqueEnum(contract.Assessments, allowedAssessments) ||
			!validUniqueEnum(contract.EvidenceKinds, allowedEvidenceKinds) {
			return validationError(CodeManifestInvalid, ManifestPath)
		}
		key := contractKey(contract.Reference())
		if _, duplicate := contracts[key]; duplicate {
			return validationError(CodeManifestInvalid, ManifestPath)
		}
		contracts[key] = contract
	}

	entries := make(map[string]Entry, len(document.Entries))
	for _, entry := range document.Entries {
		if !entryIDPattern.MatchString(entry.ID) || (entry.Kind != "risk" && entry.Kind != "requirement") ||
			!validText(entry.Title, 1024, true) || !validText(entry.Statement, MaximumEntryStatementBytes, true) ||
			!validText(entry.Level, 128, false) || len(entry.AllowedMethods) == 0 || len(entry.AllowedMethods) > 16 ||
			!validUniqueEnum(entry.AllowedMethods, allowedMethods) {
			return validationError(CodeManifestInvalid, ManifestPath)
		}
		if entry.Applicability.Mode != "always" && entry.Applicability.Mode != "profile-rule" &&
			entry.Applicability.Mode != "human-review" {
			return validationError(CodeManifestInvalid, ManifestPath)
		}
		if entry.Applicability.Mode == "profile-rule" {
			if !validText(entry.Applicability.Rule, 4096, true) {
				return validationError(CodeManifestInvalid, ManifestPath)
			}
		} else if entry.Applicability.Rule != "" {
			return validationError(CodeManifestInvalid, ManifestPath)
		}
		if _, exists := contracts[contractKey(entry.EvidenceContract)]; !exists {
			return validationError(CodeDanglingMapping, ManifestPath)
		}
		if _, duplicate := entries[entry.ID]; duplicate {
			return validationError(CodeManifestInvalid, ManifestPath)
		}
		entries[entry.ID] = entry
	}

	mappings := make(map[string]struct{}, len(document.Mappings))
	for _, mapping := range document.Mappings {
		if !entryIDPattern.MatchString(mapping.Key) || !rolePattern.MatchString(mapping.WorkflowRole) ||
			!allowedMethods[mapping.Method] || len(mapping.EntryIDs) == 0 || len(mapping.EntryIDs) > 64 ||
			!validText(mapping.Title, 1024, true) ||
			!validText(mapping.Objective, MaximumMappingObjectiveBytes, true) {
			return validationError(CodeManifestInvalid, ManifestPath)
		}
		if _, duplicate := mappings[mapping.Key]; duplicate {
			return validationError(CodeManifestInvalid, ManifestPath)
		}
		mappings[mapping.Key] = struct{}{}
		contract, exists := contracts[contractKey(mapping.EvidenceContract)]
		if !exists {
			return validationError(CodeDanglingMapping, ManifestPath)
		}
		_ = contract
		seenEntries := make(map[string]struct{}, len(mapping.EntryIDs))
		for _, entryID := range mapping.EntryIDs {
			entry, exists := entries[entryID]
			if !exists || entry.EvidenceContract != mapping.EvidenceContract || !contains(entry.AllowedMethods, mapping.Method) {
				return validationError(CodeDanglingMapping, ManifestPath)
			}
			if _, duplicate := seenEntries[entryID]; duplicate {
				return validationError(CodeManifestInvalid, ManifestPath)
			}
			seenEntries[entryID] = struct{}{}
		}
	}
	return nil
}

func contractKey(ref EvidenceContractRef) string { return ref.ID + "\x00" + ref.Version }

func validText(value string, maximum int, required bool) bool {
	if !utf8.ValidString(value) || strings.ContainsRune(value, 0) || len([]byte(value)) > maximum || value != strings.TrimSpace(value) {
		return false
	}
	return !required || value != ""
}

func validHTTPSURL(raw string) bool {
	if !validText(raw, 2048, true) {
		return false
	}
	parsed, err := url.Parse(raw)
	return err == nil && parsed.Scheme == "https" && parsed.Host != "" && parsed.User == nil
}

func validUniqueEnum(values []string, allowed map[string]bool) bool {
	seen := make(map[string]struct{}, len(values))
	for _, value := range values {
		if !allowed[value] {
			return false
		}
		if _, duplicate := seen[value]; duplicate {
			return false
		}
		seen[value] = struct{}{}
	}
	return true
}

func contains(values []string, candidate string) bool {
	index := sort.SearchStrings(values, candidate)
	return index < len(values) && values[index] == candidate
}

func requireJSONEOF(decoder *json.Decoder) error {
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		if err == nil {
			return errors.New("trailing JSON value")
		}
		return err
	}
	return nil
}

func rejectDuplicateJSONKeys(data []byte) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	var visit func() error
	visit = func() error {
		token, err := decoder.Token()
		if err != nil {
			return err
		}
		delimiter, compound := token.(json.Delim)
		if !compound {
			return nil
		}
		switch delimiter {
		case '{':
			seen := map[string]struct{}{}
			for decoder.More() {
				keyToken, err := decoder.Token()
				if err != nil {
					return err
				}
				key, ok := keyToken.(string)
				if !ok {
					return errors.New("object key is not a string")
				}
				if _, duplicate := seen[key]; duplicate {
					return fmt.Errorf("duplicate JSON key %q", key)
				}
				seen[key] = struct{}{}
				if err := visit(); err != nil {
					return err
				}
			}
			closing, err := decoder.Token()
			if err != nil || closing != json.Delim('}') {
				return errors.New("invalid object closing token")
			}
		case '[':
			for decoder.More() {
				if err := visit(); err != nil {
					return err
				}
			}
			closing, err := decoder.Token()
			if err != nil || closing != json.Delim(']') {
				return errors.New("invalid array closing token")
			}
		default:
			return errors.New("unexpected JSON delimiter")
		}
		return nil
	}
	if err := visit(); err != nil {
		return err
	}
	if decoder.More() {
		return errors.New("trailing JSON value")
	}
	return nil
}

func ValidatePinnedPackage(value PinnedPackage) error {
	if err := validateReference(value.Reference); err != nil || !validText(value.Title, 512, true) ||
		!validText(value.Source.Name, 512, true) || !validText(value.Source.Revision, 512, false) ||
		!validHTTPSURL(value.Source.URL) || !validText(value.License.Attribution, 4096, true) ||
		!validHTTPSURL(value.License.URL) {
		return validationError(CodeManifestInvalid, ManifestPath)
	}
	policies, supported := supportedLicenses[value.License.ID]
	if !supported || !policies[value.License.Disclosure] {
		return validationError(CodeLicenseInvalid, ManifestPath)
	}
	if err := validateExactPackage(value.Catalog); err != nil {
		return err
	}
	if err := validateExactPackage(value.Retained); err != nil {
		return err
	}
	if value.Catalog.Artifact.Namespace != CatalogNamespace ||
		!strings.HasPrefix(value.Retained.Artifact.Namespace, "audit-") ||
		value.Catalog.Digest != value.Retained.Digest || value.Catalog.SizeBytes != value.Retained.SizeBytes {
		return validationError(CodeIdentityMismatch, ManifestPath)
	}
	return nil
}

func validateExactPackage(value ExactPackage) error {
	if value.Artifact.ValidateExact() != nil || value.MediaType != MediaType ||
		value.SizeBytes <= 0 || value.SizeBytes > MaximumPackageBytes ||
		len(value.Digest) != len("sha256:")+64 || !strings.HasPrefix(value.Digest, "sha256:") {
		return validationError(CodeManifestInvalid, ManifestPath)
	}
	if _, err := hex.DecodeString(strings.TrimPrefix(value.Digest, "sha256:")); err != nil {
		return validationError(CodeManifestInvalid, ManifestPath)
	}
	return nil
}
