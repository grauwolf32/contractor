package memory

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"
)

type fixtureDocument struct {
	SchemaVersion string `json:"schemaVersion"`
	Valid         []struct {
		ID        string     `json:"id"`
		Input     StoredNote `json:"input"`
		Canonical string     `json:"canonical"`
	} `json:"valid"`
	GeneratedValid []struct {
		ID    string `json:"id"`
		Input struct {
			SchemaVersion     string          `json:"schemaVersion"`
			Name              string          `json:"name"`
			NameRepeat        *fixtureRepeat  `json:"nameRepeat"`
			Content           string          `json:"content"`
			ContentRepeat     *fixtureRepeat  `json:"contentRepeat"`
			Description       string          `json:"description"`
			DescriptionRepeat *fixtureRepeat  `json:"descriptionRepeat"`
			Tags              []string        `json:"tags"`
			TagRepeats        []fixtureRepeat `json:"tagRepeats"`
			Ordinal           uint64          `json:"ordinal"`
		} `json:"input"`
		CanonicalBytes  int    `json:"canonicalBytes"`
		CanonicalSHA256 string `json:"canonicalSha256"`
	} `json:"generatedValid"`
	InvalidInput []struct {
		ID       string      `json:"id"`
		Mutation string      `json:"mutation"`
		Reason   ErrorReason `json:"reason"`
	} `json:"invalidInput"`
	InvalidStored []struct {
		ID          string      `json:"id"`
		BindingName string      `json:"bindingName"`
		Payload     string      `json:"payload"`
		Reason      ErrorReason `json:"reason"`
	} `json:"invalidStored"`
}

type fixtureRepeat struct {
	Value string `json:"value"`
	Count int    `json:"count"`
}

func TestSharedCodecFixtures(t *testing.T) {
	fixtures := loadFixtures(t)
	if fixtures.SchemaVersion != "contractor.memory-codec-fixtures/v1" {
		t.Fatalf("fixture schema = %q", fixtures.SchemaVersion)
	}
	for _, fixture := range fixtures.Valid {
		t.Run("valid/"+fixture.ID, func(t *testing.T) {
			encoded, err := Encode(fixture.Input)
			if err != nil {
				t.Fatal(err)
			}
			if string(encoded) != fixture.Canonical {
				t.Fatalf("canonical mismatch\n got: %s\nwant: %s", encoded, fixture.Canonical)
			}
			artifactName, err := ArtifactName(fixture.Input.Name)
			if err != nil {
				t.Fatal(err)
			}
			decoded, err := Decode(artifactName, encoded)
			if err != nil {
				t.Fatal(err)
			}
			normalized, _ := Normalize(fixture.Input)
			if !reflect.DeepEqual(decoded, normalized) {
				t.Fatalf("decoded = %+v, want %+v", decoded, normalized)
			}
		})
	}
	for _, fixture := range fixtures.GeneratedValid {
		t.Run("valid/"+fixture.ID, func(t *testing.T) {
			name := fixture.Input.Name
			if fixture.Input.NameRepeat != nil {
				name = strings.Repeat(fixture.Input.NameRepeat.Value, fixture.Input.NameRepeat.Count)
			}
			content := fixture.Input.Content
			if fixture.Input.ContentRepeat != nil {
				content = strings.Repeat(fixture.Input.ContentRepeat.Value, fixture.Input.ContentRepeat.Count)
			}
			description := fixture.Input.Description
			if fixture.Input.DescriptionRepeat != nil {
				description = strings.Repeat(fixture.Input.DescriptionRepeat.Value, fixture.Input.DescriptionRepeat.Count)
			}
			tags := append([]string(nil), fixture.Input.Tags...)
			for _, repeat := range fixture.Input.TagRepeats {
				tags = append(tags, strings.Repeat(repeat.Value, repeat.Count))
			}
			note := StoredNote{
				SchemaVersion: fixture.Input.SchemaVersion,
				Name:          name,
				Content:       content,
				Description:   description,
				Tags:          tags,
				Ordinal:       fixture.Input.Ordinal,
			}
			encoded, err := Encode(note)
			if err != nil {
				t.Fatal(err)
			}
			digest := sha256.Sum256(encoded)
			if len(encoded) != fixture.CanonicalBytes || hex.EncodeToString(digest[:]) != fixture.CanonicalSHA256 {
				t.Fatalf("generated canonical = bytes %d sha256 %x", len(encoded), digest)
			}
			if _, err := Decode(ArtifactNamePrefix+note.Name, encoded); err != nil {
				t.Fatal(err)
			}
		})
	}
	for _, fixture := range fixtures.InvalidInput {
		t.Run("invalid-input/"+fixture.ID, func(t *testing.T) {
			note, canaries := mutatedNote(fixture.Mutation)
			_, err := Encode(note)
			assertReason(t, err, fixture.Reason, canaries...)
		})
	}
	for _, fixture := range fixtures.InvalidStored {
		t.Run("invalid-stored/"+fixture.ID, func(t *testing.T) {
			_, err := Decode(fixture.BindingName, []byte(fixture.Payload))
			assertReason(t, err, fixture.Reason, "stored_note", "other_note")
		})
	}
}

func TestModelProjectionsNormalizeServerTimestampsToUTC(t *testing.T) {
	location := time.FixedZone("configured-local", 3*60*60)
	created := time.Date(2026, 9, 1, 10, 11, 12, 0, location)
	updated := created.Add(time.Second)
	note := StoredNote{SchemaVersion: SchemaVersion, Name: "note", Content: "body", Tags: []string{}}
	full := FullProjection(note, created, updated)
	preview := PreviewProjection(note, created, updated)
	if full.CreatedAt.Location() != time.UTC || full.UpdatedAt.Location() != time.UTC ||
		preview.CreatedAt.Location() != time.UTC || preview.UpdatedAt.Location() != time.UTC ||
		full.CreatedAt.Format(time.RFC3339) != "2026-09-01T07:11:12Z" ||
		preview.UpdatedAt.Format(time.RFC3339) != "2026-09-01T07:11:13Z" {
		t.Fatalf("UTC projections = full:%+v preview:%+v", full, preview)
	}
}

func TestCodecRejectsOversizedAndInvalidUTF8StoredPayloads(t *testing.T) {
	_, err := Decode("memory.note", []byte(strings.Repeat("x", MaximumPayloadBytes+1)))
	assertReason(t, err, ReasonPayloadTooLarge)
	_, err = Decode("memory.note", []byte{0xff})
	assertReason(t, err, ReasonMalformed)
}

func TestArtifactNameMappingAndModelProjections(t *testing.T) {
	if name, err := ArtifactName("repo_overview"); err != nil || name != "memory.repo_overview" {
		t.Fatalf("ArtifactName = (%q, %v)", name, err)
	}
	if name, err := NameFromArtifact("memory.repo_overview"); err != nil || name != "repo_overview" {
		t.Fatalf("NameFromArtifact = (%q, %v)", name, err)
	}
	if _, err := NameFromArtifact("report.repo_overview"); err == nil {
		t.Fatal("non-memory binding accepted")
	}
	maximumName := "a" + strings.Repeat("b", MaximumNameBytes-1)
	maximumArtifactName, err := ArtifactName(maximumName)
	if err != nil || len(maximumArtifactName) != MaximumArtifactNameBytes {
		t.Fatalf("maximum ArtifactName = (%d bytes, %v)", len(maximumArtifactName), err)
	}
	if _, err := ArtifactName(maximumName + "b"); err == nil {
		t.Fatal("Memory name exceeding the prefixed Artifact limit was accepted")
	}
	note, err := Normalize(baseNote())
	if err != nil {
		t.Fatal(err)
	}
	created := time.Date(2026, 9, 1, 10, 0, 0, 0, time.UTC)
	updated := created.Add(time.Second)
	fullJSON, _ := json.Marshal(FullProjection(note, created, updated))
	previewJSON, _ := json.Marshal(PreviewProjection(note, created, updated))
	for _, forbidden := range []string{"artifact", "revision", "namespace"} {
		if strings.Contains(string(fullJSON), forbidden) || strings.Contains(string(previewJSON), forbidden) {
			t.Fatalf("storage identity leaked in projections: %s / %s", fullJSON, previewJSON)
		}
	}
	if strings.Contains(string(previewJSON), `"content"`) || !strings.Contains(string(fullJSON), `"content"`) {
		t.Fatalf("full/preview content projection = %s / %s", fullJSON, previewJSON)
	}
}

func loadFixtures(t *testing.T) fixtureDocument {
	t.Helper()
	data, err := os.ReadFile("../../testdata/memory/cases.json")
	if err != nil {
		t.Fatal(err)
	}
	var result fixtureDocument
	decoder := json.NewDecoder(strings.NewReader(string(data)))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&result); err != nil {
		t.Fatal(err)
	}
	return result
}

func baseNote() StoredNote {
	return StoredNote{
		SchemaVersion: SchemaVersion,
		Name:          "safe_note",
		Content:       "recognizable-content-canary",
		Description:   "recognizable-description-canary",
		Tags:          []string{"safe-tag"},
		Ordinal:       1,
	}
}

func mutatedNote(mutation string) (StoredNote, []string) {
	note := baseNote()
	switch mutation {
	case "uppercase_name":
		note.Name = "Uppercase"
	case "long_name":
		note.Name = strings.Repeat("a", MaximumNameBytes+1)
	case "invalid_name_utf8":
		note.Name = string([]byte{0xff})
	case "empty_content":
		note.Content = ""
	case "invalid_content_utf8":
		note.Content = string([]byte{0xff})
	case "long_description_utf8":
		note.Description = strings.Repeat("é", MaximumDescription/2+1)
	case "invalid_description_utf8":
		note.Description = string([]byte{0xff})
	case "too_many_unique_tags":
		note.Tags = []string{"a", "b", "c", "d"}
	case "invalid_tag":
		note.Tags = []string{"Uppercase"}
	case "long_tag":
		note.Tags = []string{strings.Repeat("a", MaximumTagBytes+1)}
	case "ordinal_above_exact_range":
		note.Ordinal = MaximumExactOrdinal + 1
	case "payload_above_limit":
		note.Content = strings.Repeat("x", 32651)
	case "wrong_schema":
		note.SchemaVersion = "contractor.memory-note/v0"
	default:
		panic("unknown fixture mutation")
	}
	return note, []string{note.Content, note.Description, strings.Join(note.Tags, ",")}
}

func assertReason(t *testing.T, err error, expected ErrorReason, canaries ...string) {
	t.Helper()
	reason, ok := ErrorReasonOf(err)
	if !ok || reason != expected {
		t.Fatalf("error = %v, reason = %q, want %q", err, reason, expected)
	}
	message := err.Error()
	for _, canary := range canaries {
		if len(canary) >= 8 && strings.Contains(message, canary) {
			t.Fatalf("codec error leaked supplied value: %q", message)
		}
	}
}
