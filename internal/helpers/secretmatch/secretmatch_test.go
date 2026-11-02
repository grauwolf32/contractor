package secretmatch

import (
	"slices"
	"testing"
)

// The rule matches the Runtime's redaction.py: complete strings for every
// secret, substrings only for long ones, longest first (OBS-R31, LIM-OBS-3).
func TestRedactMatchesCompleteShortAndLongestFirstLongSecrets(t *testing.T) {
	t.Parallel()
	username := "admin"
	password := "admin-0123456789-password-xyz" // starts with the username
	token := "tok-0123456789abc"                // 17 bytes
	secrets := []string{username, password, token, ""}
	for _, test := range []struct{ text, want string }{
		{"admin", "[REDACTED]"},
		{"administrator login failed", "administrator login failed"},
		{"path/main.go", "path/main.go"},
		{"auth " + password + " end", "auth [REDACTED] end"},
		{password, "[REDACTED]"},
		{"x" + token + token, "x[REDACTED][REDACTED]"},
		{"", ""},
	} {
		if got := Redact(test.text, secrets, "[REDACTED]"); got != test.want {
			t.Errorf("Redact(case %q) = %q, want %q", test.want, got, test.want)
		}
	}
	if Contains("administrator", secrets) || !Contains("admin", secrets) || !Contains("a"+token, secrets) {
		t.Fatal("Contains does not follow the complete-or-long rule")
	}
	// A multibyte secret is measured in UTF-8 bytes: ten two-byte runes and two
	// ASCII bytes are long enough to match anywhere.
	multibyte := "секрет!!" + "ключ"
	if got := Redact("x"+multibyte+"y", []string{multibyte}, "#"); got != "x#y" {
		t.Fatalf("multibyte secret was not matched as a substring: %q", got)
	}
	if got := Long([]string{"b-0123456789abcdef", "a-0123456789abcdefgh", "short", "b-0123456789abcdef"}); !slices.Equal(got,
		[]string{"a-0123456789abcdefgh", "b-0123456789abcdef"}) {
		t.Fatalf("Long = %q", got)
	}
}
