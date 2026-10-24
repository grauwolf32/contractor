package credentials

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"
)

func TestTargetCredentialMaterialRoundTripsThroughCanonicalDecoding(t *testing.T) {
	fields := map[string]string{"username": "alice@example.test", "password": "pässwörd with spaces"}
	cases := []struct {
		name  string
		build func() (RuntimeCredentialMaterial, error)
		kind  RuntimeCredentialKind
	}{
		{"basic", func() (RuntimeCredentialMaterial, error) { return NewTargetBasicCredential("alice", "s3cret") }, RuntimeCredentialTargetBasic},
		{"bearer", func() (RuntimeCredentialMaterial, error) { return NewTargetBearerCredential("tok \t en") }, RuntimeCredentialTargetBearer},
		{"login", func() (RuntimeCredentialMaterial, error) { return NewTargetLoginCredential(fields) }, RuntimeCredentialTargetLogin},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			material, err := tc.build()
			if err != nil {
				t.Fatalf("build: %v", err)
			}
			defer material.Destroy()
			if material.Kind() != tc.kind {
				t.Fatalf("kind = %q", material.Kind())
			}
			err = material.WithPlaintext(func(kind RuntimeCredentialKind, canonical []byte) error {
				decoded, err := runtimeCredentialMaterialFromCanonical(kind, append([]byte(nil), canonical...))
				if err != nil {
					return err
				}
				decoded.Destroy()
				return nil
			})
			if err != nil {
				t.Fatalf("canonical decode: %v", err)
			}
		})
	}
}

func TestTargetLoginCredentialRejectsOutOfBoundsFields(t *testing.T) {
	tooMany := map[string]string{}
	for i := 0; i <= MaximumTargetLoginFields; i++ {
		tooMany["f"+strings.Repeat("x", i)] = "v"
	}
	tooLarge := map[string]string{}
	for i := 0; i < 3; i++ {
		tooLarge["f"+strings.Repeat("x", i)] = strings.Repeat("v", 6000)
	}
	for name, fields := range map[string]map[string]string{
		"empty":          {},
		"too many":       tooMany,
		"too large":      tooLarge,
		"uppercase name": {"Password": "v"},
		"empty value":    {"password": ""},
		"control value":  {"password": "a\nb"},
		"invalid utf8":   {"password": "\xff"},
	} {
		if _, err := NewTargetLoginCredential(fields); !errors.Is(err, ErrRuntimeCredentialInvalid) {
			t.Fatalf("%s: err = %v", name, err)
		}
	}
	if _, err := NewTargetBasicCredential("a:b", "p"); !errors.Is(err, ErrRuntimeCredentialInvalid) {
		t.Fatalf("basic username with colon: err = %v", err)
	}
	if _, err := NewTargetBearerCredential(" leading"); !errors.Is(err, ErrRuntimeCredentialInvalid) {
		t.Fatalf("bearer leading space: err = %v", err)
	}
}

func TestTargetLoginMaterialIsWipedAfterCallbackFailureAndNeverFormatted(t *testing.T) {
	const canary = "canary-login-secret-41c9"
	material, err := NewTargetLoginCredential(map[string]string{"password": canary})
	if err != nil {
		t.Fatal(err)
	}
	var seen []byte
	failure := errors.New("consumer failed")
	err = material.WithPlaintext(func(_ RuntimeCredentialKind, canonical []byte) error {
		seen = canonical
		return failure
	})
	if !errors.Is(err, failure) {
		t.Fatalf("callback error = %v", err)
	}
	for _, text := range []string{fmt.Sprintf("%v %+v %#v %s", material, material, material, material)} {
		if strings.Contains(text, canary) {
			t.Fatalf("formatted material exposes the secret: %s", text)
		}
	}
	if encoded, err := json.Marshal(material); err == nil || strings.Contains(string(encoded), canary) {
		t.Fatalf("material encoded: %s, %v", encoded, err)
	}
	material.Destroy()
	if strings.Contains(string(seen), canary) {
		t.Fatal("Destroy left the plaintext in the material buffer")
	}
}
