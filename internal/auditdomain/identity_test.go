package auditdomain

import "testing"

// TestIdentityEncodingsArePinned guards persisted identifiers and digests:
// these values are stored in PostgreSQL rows and artifact names, so any change
// to the encoding is a data migration, not a refactor.
func TestIdentityEncodingsArePinned(t *testing.T) {
	cases := []struct {
		name, got, want string
	}{
		{"digest", DigestBytes([]byte("hello")), "sha256:2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"},
		{"empty digest", DigestBytes(nil), "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
		{"id with values", DeterministicID("round", "audit-1", "x"), "round-1a9f229057d8da43e7ac29b10be7c9f6c3e421a57bbe113977585dd016d17c33"},
		{"id without values", DeterministicID("item"), "item-d4d6e7604c05fdd1eec5cc825463052bf59c58c8c2bdd4663c832b277720536c"},
		{"id with empty values", DeterministicID("round", "", ""), "round-77862fca6b7eb6abdc3fcdc94bbf8d5a4ecf599bddd31ca112643ec697bf780e"},
		{"artifact namespace", ArtifactNamespace("a1"), "audit-ec197504e4716ea21f7caad402e49f18f69878a966b394880b2212795e1cf986"},
	}
	for _, tc := range cases {
		if tc.got != tc.want {
			t.Errorf("%s = %s, want %s", tc.name, tc.got, tc.want)
		}
	}
}
