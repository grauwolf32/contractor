package credentials

import (
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

// TestSharedCredentialBoundsCountUTF8Bytes checks the Basic credential
// username and password of every Basic kind (LIM-RCF-23) and the Git SSH key
// (LIM-GIT-3) with each shared character at and one byte over the declared
// bound. The key's run precedes a valid PEM block, which the parser finds
// after any leading text.
func TestSharedCredentialBoundsCountUTF8Bytes(t *testing.T) {
	t.Parallel()
	characters, fields := contractstest.ReadUTF8ByteBoundCases(t)
	basic := map[string]func(string, string) (RuntimeCredentialMaterial, error){
		"http-proxy":  NewHTTPProxyBasicCredential,
		"http-origin": NewHTTPOriginBasicCredential,
		"target":      NewTargetBasicCredential,
	}
	checked := map[string]bool{}
	for _, field := range fields {
		switch field.Schema {
		case "BasicCredentialUsername", "BasicCredentialPassword":
			bound := MaximumBasicUsernameBytes
			if field.Schema == "BasicCredentialPassword" {
				bound = MaximumRuntimeSecretBytes
			}
			if field.Rule != "LIM-RCF-23" || field.MaxBytes != bound {
				t.Fatalf("%s: shared bound %d (%s), Go bound %d", field.Schema, field.MaxBytes, field.Rule, bound)
			}
			for kind, create := range basic {
				for _, value := range field.Values(characters) {
					username, password := value.Value, "password"
					if field.Schema == "BasicCredentialPassword" {
						username, password = "user", value.Value
					}
					if _, err := create(username, password); (err == nil) != value.Within {
						t.Errorf("%s %s of %d bytes of %q: %v, within %v", kind, field.Schema, len(value.Value), value.Character, err, value.Within)
					}
				}
			}
		case "GitPrivateKey":
			if field.Rule != "LIM-GIT-3" || field.MaxBytes != MaximumGitKeyBytes {
				t.Fatalf("%s: shared bound %d (%s), Go bound %d", field.Schema, field.MaxBytes, field.Rule, MaximumGitKeyBytes)
			}
			for _, value := range field.EmbeddedValues(characters, "", "\n"+string(testGitPrivateKey(t))) {
				if _, err := parseGitKey([]byte(value.Value)); (err == nil) != value.Within {
					t.Errorf("Git key of %d bytes of %q: %v, within %v", len(value.Value), value.Character, err, value.Within)
				}
			}
		default:
			continue
		}
		checked[field.Schema] = true
	}
	if len(checked) != 3 {
		t.Fatalf("shared credential byte bounds checked: %v", checked)
	}
}

// TestSharedOTLPHeaderNamesAreASCII pins the ASCII header-name grammar that the
// public schema declares (LIM-RCF-20): a name with any shared non-ASCII
// character is rejected, so maxLength counts its bytes.
func TestSharedOTLPHeaderNamesAreASCII(t *testing.T) {
	t.Parallel()
	characters, _ := contractstest.ReadUTF8ByteBoundCases(t)
	for _, character := range characters {
		_, err := NewOTLPHeadersCredential(map[string]string{"x-" + character: "value"})
		if ascii := len(character) == 1; (err == nil) != ascii {
			t.Errorf("OTLP header name with %q: %v", character, err)
		}
	}
}
