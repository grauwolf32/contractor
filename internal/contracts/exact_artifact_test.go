package contracts

import (
	"encoding/json"
	"errors"
	"testing"
)

func TestExactArtifactVerifyIsTheOneContentCheck(t *testing.T) {
	t.Parallel()
	revision := "rev-1"
	ref := ArtifactRef{Namespace: "audit-test", Name: "task", Revision: &revision}
	found := ExactContent(ref, "application/zip", []byte("package"))
	if found.SizeBytes != 7 || !found.HasMetadata() || found.Validate() != nil {
		t.Fatalf("exact content = %+v", found)
	}
	digestOnly := ExactArtifact{Ref: ref, Digest: found.Digest}
	for name, expected := range map[string]ExactArtifact{"complete": found, "digest-only": digestOnly} {
		if err := expected.Verify(found); err != nil {
			t.Fatalf("%s descriptor rejected its content: %v", name, err)
		}
	}

	otherRevision := "rev-2"
	for name, candidate := range map[string]ExactArtifact{
		"revision":   {Ref: ArtifactRef{Namespace: "audit-test", Name: "task", Revision: &otherRevision}, Digest: found.Digest, MediaType: found.MediaType, SizeBytes: found.SizeBytes},
		"name":       {Ref: ArtifactRef{Namespace: "audit-test", Name: "other", Revision: &revision}, Digest: found.Digest, MediaType: found.MediaType, SizeBytes: found.SizeBytes},
		"digest":     ExactContent(ref, "application/zip", []byte("changed")),
		"media type": {Ref: ref, Digest: found.Digest, MediaType: "application/json", SizeBytes: found.SizeBytes},
		"size":       {Ref: ref, Digest: found.Digest, MediaType: found.MediaType, SizeBytes: found.SizeBytes + 1},
	} {
		if err := found.Verify(candidate); !errors.Is(err, ErrExactArtifactMismatch) {
			t.Fatalf("%s mismatch = %v", name, err)
		}
	}
	// A digest-only descriptor pins the bytes but cannot pin a media type.
	if err := digestOnly.Verify(ExactArtifact{Ref: ref, Digest: found.Digest, MediaType: "application/json", SizeBytes: 7}); err != nil {
		t.Fatalf("digest-only descriptor checked metadata it does not record: %v", err)
	}

	for name, invalid := range map[string]ExactArtifact{
		"versionless ref":       {Ref: ArtifactRef{Namespace: "audit-test", Name: "task"}, Digest: found.Digest},
		"digest":                {Ref: ref, Digest: "sha256:ABC"},
		"size without metadata": {Ref: ref, Digest: found.Digest, SizeBytes: 7},
		"media type":            {Ref: ref, Digest: found.Digest, MediaType: "Application/Zip"},
		"negative size":         {Ref: ref, Digest: found.Digest, MediaType: "application/zip", SizeBytes: -1},
	} {
		if err := invalid.Validate(); !errors.Is(err, ErrValidation) {
			t.Fatalf("%s descriptor validated: %v", name, err)
		}
		if err := invalid.Verify(found); !errors.Is(err, ErrValidation) {
			t.Fatalf("%s descriptor verified: %v", name, err)
		}
	}
}

func TestExactArtifactJSONCarriesMetadataTogether(t *testing.T) {
	t.Parallel()
	revision := "rev-1"
	ref := ArtifactRef{Namespace: "audit-test", Name: "empty", Revision: &revision}
	empty := ExactContent(ref, "text/plain", nil)
	for name, tc := range map[string]struct {
		value ExactArtifact
		want  string
	}{
		"empty content keeps its size": {empty, `{"ref":{"namespace":"audit-test","name":"empty","revision":"rev-1"},"digest":"` + empty.Digest + `","mediaType":"text/plain","sizeBytes":0}`},
		"digest-only carries neither":  {ExactArtifact{Ref: ref, Digest: empty.Digest}, `{"ref":{"namespace":"audit-test","name":"empty","revision":"rev-1"},"digest":"` + empty.Digest + `"}`},
	} {
		encoded, err := json.Marshal(tc.value)
		if err != nil || string(encoded) != tc.want {
			t.Fatalf("%s = (%s, %v), want %s", name, encoded, err, tc.want)
		}
		var decoded ExactArtifact
		if err := json.Unmarshal(encoded, &decoded); err != nil || decoded.Verify(tc.value) != nil {
			t.Fatalf("%s round trip = (%+v, %v)", name, decoded, err)
		}
	}
	if _, err := json.Marshal(ExactArtifact{Ref: ref, Digest: empty.Digest, SizeBytes: 3}); err == nil {
		t.Fatal("a size without a media type encoded")
	}
}
