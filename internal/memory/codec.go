// Package memory defines the language-neutral stored Memory note contract.
// It deliberately contains no tool, scheduling, HTTP or persistence policy.
package memory

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/ucarion/jcs"
)

const (
	SchemaVersion            = "contractor.memory-note/v1"
	MediaType                = "application/vnd.contractor.memory-note+json"
	ArtifactNamePrefix       = "memory."
	MaximumArtifactNameBytes = 128
	MaximumPayloadBytes      = 32 * 1024
	MaximumNameBytes         = MaximumArtifactNameBytes - len(ArtifactNamePrefix)
	MaximumDescription       = 512
	MaximumTags              = 3
	MaximumTagBytes          = 64
	MaximumExactOrdinal      = uint64(1<<53 - 1)
)

var (
	namePattern = regexp.MustCompile(`^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$`)
	tagPattern  = regexp.MustCompile(`^[a-z][a-z0-9_-]*$`)
)

type ErrorReason string

const (
	ReasonName            ErrorReason = "name"
	ReasonContent         ErrorReason = "content"
	ReasonDescription     ErrorReason = "description"
	ReasonTags            ErrorReason = "tags"
	ReasonOrdinal         ErrorReason = "ordinal"
	ReasonPayloadTooLarge ErrorReason = "payload_too_large"
	ReasonMalformed       ErrorReason = "malformed"
	ReasonSchema          ErrorReason = "schema"
	ReasonBindingMismatch ErrorReason = "binding_mismatch"
	ReasonNonCanonical    ErrorReason = "noncanonical"
)

// CodecError intentionally retains only a closed reason. Caller-provided note
// text can therefore never enter an error, log or metric through this type.
type CodecError struct {
	Reason ErrorReason
}

func (e *CodecError) Error() string { return "invalid memory note (" + string(e.Reason) + ")" }

func ErrorReasonOf(err error) (ErrorReason, bool) {
	var codecError *CodecError
	if !errors.As(err, &codecError) {
		return "", false
	}
	return codecError.Reason, true
}

func invalid(reason ErrorReason) error { return &CodecError{Reason: reason} }

// StoredNote is the exact timestamp-free Artifact payload. Tags may be
// supplied in arbitrary order to Encode; the returned bytes always contain a
// sorted unique array.
type StoredNote struct {
	SchemaVersion string   `json:"schemaVersion"`
	Name          string   `json:"name"`
	Content       string   `json:"content"`
	Description   string   `json:"description"`
	Tags          []string `json:"tags"`
	Ordinal       uint64   `json:"ordinal"`
}

// Note and Preview are model-facing projections. Storage identity and
// revision are structurally unrepresentable.
type Note struct {
	Name        string    `json:"name"`
	Content     string    `json:"content"`
	Description string    `json:"description"`
	Tags        []string  `json:"tags"`
	Ordinal     uint64    `json:"ordinal"`
	CreatedAt   time.Time `json:"created_at"`
	UpdatedAt   time.Time `json:"updated_at"`
}

type Preview struct {
	Name        string    `json:"name"`
	Description string    `json:"description"`
	Tags        []string  `json:"tags"`
	Ordinal     uint64    `json:"ordinal"`
	CreatedAt   time.Time `json:"created_at"`
	UpdatedAt   time.Time `json:"updated_at"`
}

func FullProjection(note StoredNote, createdAt, updatedAt time.Time) Note {
	return Note{
		Name: note.Name, Content: note.Content, Description: note.Description,
		Tags: append([]string(nil), note.Tags...), Ordinal: note.Ordinal,
		CreatedAt: createdAt.UTC(), UpdatedAt: updatedAt.UTC(),
	}
}

func PreviewProjection(note StoredNote, createdAt, updatedAt time.Time) Preview {
	return Preview{
		Name: note.Name, Description: note.Description,
		Tags: append([]string(nil), note.Tags...), Ordinal: note.Ordinal,
		CreatedAt: createdAt.UTC(), UpdatedAt: updatedAt.UTC(),
	}
}

func ArtifactName(name string) (string, error) {
	if err := validateName(name); err != nil {
		return "", err
	}
	return ArtifactNamePrefix + name, nil
}

func NameFromArtifact(artifactName string) (string, error) {
	if !strings.HasPrefix(artifactName, ArtifactNamePrefix) {
		return "", invalid(ReasonName)
	}
	name := strings.TrimPrefix(artifactName, ArtifactNamePrefix)
	if err := validateName(name); err != nil {
		return "", err
	}
	return name, nil
}

// Normalize validates author input and returns a detached canonical logical
// value. It is the only path that deduplicates/reorders tags.
func Normalize(note StoredNote) (StoredNote, error) {
	if note.SchemaVersion != SchemaVersion {
		return StoredNote{}, invalid(ReasonSchema)
	}
	if err := validateName(note.Name); err != nil {
		return StoredNote{}, err
	}
	if err := validateContent(note.Content); err != nil {
		return StoredNote{}, invalid(ReasonContent)
	}
	if !utf8.ValidString(note.Description) || len([]byte(note.Description)) > MaximumDescription {
		return StoredNote{}, invalid(ReasonDescription)
	}
	if note.Ordinal > MaximumExactOrdinal {
		return StoredNote{}, invalid(ReasonOrdinal)
	}
	tags, err := normalizeTags(note.Tags)
	if err != nil {
		return StoredNote{}, err
	}
	return StoredNote{
		SchemaVersion: SchemaVersion,
		Name:          note.Name,
		Content:       note.Content,
		Description:   note.Description,
		Tags:          tags,
		Ordinal:       note.Ordinal,
	}, nil
}

// Encode returns RFC 8785 canonical JSON and enforces the complete 32-KiB
// payload bound after tag normalization.
func Encode(note StoredNote) ([]byte, error) {
	normalized, err := Normalize(note)
	if err != nil {
		return nil, err
	}
	encoded, err := canonicalBytes(normalized)
	if err != nil {
		return nil, invalid(ReasonMalformed)
	}
	if len(encoded) > MaximumPayloadBytes {
		return nil, invalid(ReasonPayloadTooLarge)
	}
	return encoded, nil
}

// Decode accepts only the exact canonical stored representation. In
// particular, it does not silently repair tag order or duplicate tags.
func Decode(artifactName string, payload []byte) (StoredNote, error) {
	if len(payload) > MaximumPayloadBytes {
		return StoredNote{}, invalid(ReasonPayloadTooLarge)
	}
	if !utf8.Valid(payload) || rejectDuplicateRootKeys(payload) != nil {
		return StoredNote{}, invalid(ReasonMalformed)
	}
	fields, ok := storedFields(payload)
	if !ok {
		return StoredNote{}, invalid(ReasonSchema)
	}
	var note StoredNote
	var validString bool
	if note.SchemaVersion, validString = decodeString(fields["schemaVersion"]); !validString {
		return StoredNote{}, invalid(ReasonMalformed)
	}
	if note.Name, validString = decodeString(fields["name"]); !validString {
		return StoredNote{}, invalid(ReasonMalformed)
	}
	if note.Content, validString = decodeString(fields["content"]); !validString {
		return StoredNote{}, invalid(ReasonMalformed)
	}
	if note.Description, validString = decodeString(fields["description"]); !validString {
		return StoredNote{}, invalid(ReasonMalformed)
	}
	if bytes.Equal(fields["tags"], []byte("null")) ||
		json.Unmarshal(fields["tags"], &note.Tags) != nil {
		return StoredNote{}, invalid(ReasonMalformed)
	}
	ordinal, ordinalReason := decodeOrdinal(fields["ordinal"])
	if ordinalReason != "" {
		return StoredNote{}, invalid(ordinalReason)
	}
	note.Ordinal = ordinal
	if note.SchemaVersion != SchemaVersion {
		return StoredNote{}, invalid(ReasonSchema)
	}
	if err := validateName(note.Name); err != nil {
		return StoredNote{}, err
	}
	if err := validateContent(note.Content); err != nil {
		return StoredNote{}, invalid(ReasonContent)
	}
	if !utf8.ValidString(note.Description) || len([]byte(note.Description)) > MaximumDescription {
		return StoredNote{}, invalid(ReasonDescription)
	}
	if note.Ordinal > MaximumExactOrdinal {
		return StoredNote{}, invalid(ReasonOrdinal)
	}
	normalizedTags, err := normalizeTags(note.Tags)
	if err != nil || note.Tags == nil || !equalStrings(note.Tags, normalizedTags) {
		return StoredNote{}, invalid(ReasonTags)
	}
	bindingName, err := NameFromArtifact(artifactName)
	if err != nil {
		return StoredNote{}, err
	}
	if bindingName != note.Name {
		return StoredNote{}, invalid(ReasonBindingMismatch)
	}
	canonical, err := canonicalBytes(note)
	if err != nil {
		return StoredNote{}, invalid(ReasonMalformed)
	}
	if !bytes.Equal(payload, canonical) {
		return StoredNote{}, invalid(ReasonNonCanonical)
	}
	return note, nil
}

func validateName(name string) error {
	if !utf8.ValidString(name) || len(name) == 0 || len([]byte(name)) > MaximumNameBytes || !namePattern.MatchString(name) {
		return invalid(ReasonName)
	}
	return nil
}

func validateContent(content string) error {
	if content == "" || !utf8.ValidString(content) {
		return invalid(ReasonContent)
	}
	return nil
}

func normalizeTags(tags []string) ([]string, error) {
	unique := make(map[string]struct{}, len(tags))
	for _, tag := range tags {
		if !utf8.ValidString(tag) || len([]byte(tag)) > MaximumTagBytes || !tagPattern.MatchString(tag) {
			return nil, invalid(ReasonTags)
		}
		unique[tag] = struct{}{}
	}
	if len(unique) > MaximumTags {
		return nil, invalid(ReasonTags)
	}
	result := make([]string, 0, len(unique))
	for tag := range unique {
		result = append(result, tag)
	}
	sort.Strings(result)
	return result, nil
}

func canonicalBytes(note StoredNote) ([]byte, error) {
	// The float64 conversion is exact because Normalize/Decode enforce the JCS
	// exact-integer range before this helper is reached.
	tags := make([]any, len(note.Tags))
	for index, tag := range note.Tags {
		tags[index] = tag
	}
	formatted, err := jcs.Format(map[string]any{
		"schemaVersion": note.SchemaVersion,
		"name":          note.Name,
		"content":       note.Content,
		"description":   note.Description,
		"tags":          tags,
		"ordinal":       float64(note.Ordinal),
	})
	if err != nil {
		return nil, err
	}
	return []byte(formatted), nil
}

func rejectDuplicateRootKeys(payload []byte) error {
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.UseNumber()
	opening, err := decoder.Token()
	if err != nil || opening != json.Delim('{') {
		return errors.New("invalid root")
	}
	seen := make(map[string]struct{}, 6)
	for decoder.More() {
		token, err := decoder.Token()
		if err != nil {
			return err
		}
		key, ok := token.(string)
		if !ok {
			return errors.New("invalid key")
		}
		if _, exists := seen[key]; exists {
			return errors.New("duplicate key")
		}
		seen[key] = struct{}{}
		var value json.RawMessage
		if err := decoder.Decode(&value); err != nil {
			return err
		}
	}
	closing, err := decoder.Token()
	if err != nil || closing != json.Delim('}') {
		return errors.New("invalid root")
	}
	return requireEOF(decoder)
}

func storedFields(payload []byte) (map[string]json.RawMessage, bool) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(payload, &fields); err != nil || len(fields) != 6 {
		return nil, false
	}
	for _, name := range []string{"schemaVersion", "name", "content", "description", "tags", "ordinal"} {
		if _, ok := fields[name]; !ok {
			return nil, false
		}
	}
	return fields, true
}

func decodeString(raw json.RawMessage) (string, bool) {
	var value any
	if err := json.Unmarshal(raw, &value); err != nil {
		return "", false
	}
	result, ok := value.(string)
	return result, ok
}

func decodeOrdinal(raw json.RawMessage) (uint64, ErrorReason) {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var value json.Number
	if err := decoder.Decode(&value); err != nil || requireEOF(decoder) != nil {
		return 0, ReasonMalformed
	}
	text := value.String()
	if strings.HasPrefix(text, "-") {
		return 0, ReasonOrdinal
	}
	if strings.ContainsAny(text, ".eE") {
		return 0, ReasonMalformed
	}
	ordinal, err := strconv.ParseUint(text, 10, 64)
	if err != nil {
		return 0, ReasonOrdinal
	}
	return ordinal, ""
}

func requireEOF(decoder *json.Decoder) error {
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		if err != nil {
			return err
		}
		return errors.New("multiple JSON values")
	}
	return nil
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
