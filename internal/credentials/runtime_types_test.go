package credentials

import (
	"bytes"
	"crypto/aes"
	"crypto/cipher"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"
)

func TestRuntimeCredentialMaterialsAreCanonicalBoundedAndRedacted(t *testing.T) {
	t.Parallel()
	secret := "Bearer runtime-secret-never-log"
	headers, err := NewOTLPHeadersCredential(map[string]string{
		"X-Tenant": "tenant-a", "Authorization": secret,
	})
	if err != nil {
		t.Fatal(err)
	}
	basic, err := NewHTTPProxyBasicCredential("proxy-user", "proxy-password")
	if err != nil {
		t.Fatal(err)
	}
	bearer, err := NewHTTPProxyBearerCredential("proxy-bearer")
	if err != nil {
		t.Fatal(err)
	}
	for _, material := range []*RuntimeCredentialMaterial{&headers, &basic, &bearer} {
		for _, formatted := range []string{
			fmt.Sprint(material), fmt.Sprintf("%v", material), fmt.Sprintf("%+v", material), fmt.Sprintf("%#v", material),
		} {
			if !strings.Contains(formatted, "REDACTED") || strings.Contains(formatted, secret) || strings.Contains(formatted, "proxy-password") {
				t.Fatalf("unsafe Runtime credential formatting: %q", formatted)
			}
		}
		if encoded, err := json.Marshal(material); err == nil || bytes.Contains(encoded, []byte(secret)) {
			t.Fatalf("Runtime credential JSON = (%s, %v)", encoded, err)
		}
		if len(material.canonical) == 0 || len(material.canonical) > MaximumRuntimePlaintextBytes {
			t.Fatalf("canonical plaintext size = %d", len(material.canonical))
		}
	}
	if got := string(headers.canonical); got != `{"headers":{"authorization":"Bearer runtime-secret-never-log","x-tenant":"tenant-a"}}` {
		t.Fatal("canonical OTLP header normalization differs")
	}

	aliased := headers.canonical
	headers.Destroy()
	if headers.kind != "" || headers.canonical != nil || !bytes.Equal(aliased, make([]byte, len(aliased))) {
		t.Fatal("Destroy did not wipe Runtime credential plaintext")
	}
	if err := headers.WithPlaintext(func(RuntimeCredentialKind, []byte) error { return nil }); !errors.Is(err, ErrRuntimeCredentialInvalid) {
		t.Fatalf("use-after-destroy error = %v", err)
	}
}

func TestRuntimeCredentialConstructorsRejectUnsafeInputs(t *testing.T) {
	t.Parallel()
	headerTests := map[string]map[string]string{
		"empty":              {},
		"host":               {"Host": "collector"},
		"hop by hop":         {"Connection": "close"},
		"proxy auth":         {"Proxy-Authorization": "secret"},
		"routing":            {"X-Forwarded-For": "127.0.0.1"},
		"bad name":           {"bad header": "value"},
		"case duplicate":     {"Authorization": "one", "authorization": "two"},
		"CRLF":               {"Authorization": "one\r\ntwo"},
		"oversized value":    {"Authorization": strings.Repeat("x", MaximumOTLPHeaderValueBytes+1)},
		"invalid value byte": {"Authorization": "one\x00two"},
	}
	for name, headers := range headerTests {
		if _, err := NewOTLPHeadersCredential(headers); !errors.Is(err, ErrRuntimeCredentialInvalid) {
			t.Fatalf("%s header error = %v", name, err)
		}
	}
	if _, err := NewHTTPProxyBasicCredential("user:name", "password"); !errors.Is(err, ErrRuntimeCredentialInvalid) {
		t.Fatalf("colon username error = %v", err)
	}
	if _, err := NewHTTPProxyBasicCredential("", "password"); !errors.Is(err, ErrRuntimeCredentialInvalid) {
		t.Fatalf("empty username error = %v", err)
	}
	if _, err := NewHTTPProxyBasicCredential(strings.Repeat("u", 257), "password"); !errors.Is(err, ErrRuntimeCredentialInvalid) {
		t.Fatalf("oversized username error = %v", err)
	}
	if _, err := NewHTTPProxyBasicCredential("user", "line\npassword"); !errors.Is(err, ErrRuntimeCredentialInvalid) {
		t.Fatalf("CRLF password error = %v", err)
	}
	if _, err := NewHTTPProxyBasicCredential("user", ""); !errors.Is(err, ErrRuntimeCredentialInvalid) {
		t.Fatalf("empty password error = %v", err)
	}
	if _, err := NewHTTPProxyBearerCredential(""); !errors.Is(err, ErrRuntimeCredentialInvalid) {
		t.Fatalf("empty bearer error = %v", err)
	}
	if _, err := NewHTTPProxyBearerCredential(strings.Repeat("x", MaximumRuntimeSecretBytes+1)); !errors.Is(err, ErrRuntimeCredentialInvalid) {
		t.Fatalf("oversized bearer error = %v", err)
	}
	if _, err := NewHTTPProxyBearerCredential(string([]byte{0xff})); !errors.Is(err, ErrRuntimeCredentialInvalid) {
		t.Fatalf("invalid UTF-8 bearer error = %v", err)
	}
	manyValues := make(map[string]string)
	for index := 0; index < 5; index++ {
		manyValues[fmt.Sprintf("x-secret-%d", index)] = strings.Repeat("x", MaximumOTLPHeaderValueBytes)
	}
	if _, err := NewOTLPHeadersCredential(manyValues); !errors.Is(err, ErrRuntimeCredentialInvalid) {
		t.Fatalf("total OTLP header bound error = %v", err)
	}
	tooManyHeaders := make(map[string]string)
	for index := 0; index <= MaximumOTLPHeaders; index++ {
		tooManyHeaders[fmt.Sprintf("x-header-%d", index)] = "value"
	}
	if _, err := NewOTLPHeadersCredential(tooManyHeaders); !errors.Is(err, ErrRuntimeCredentialInvalid) {
		t.Fatalf("OTLP header count bound error = %v", err)
	}
}

func TestRuntimeCredentialCipherUsesSeparateAADAndSecretMAC(t *testing.T) {
	t.Parallel()
	key := bytes.Repeat([]byte{0x63}, 32)
	cipher, err := newTokenCipher(key, bytes.NewReader(bytes.Repeat([]byte{0x17}, 48)))
	if err != nil {
		t.Fatal(err)
	}
	material, _ := NewHTTPProxyBearerCredential("exact-bearer-secret")
	envelope, err := cipher.SealRuntimeCredential("caido-auth", material)
	if err != nil {
		t.Fatal(err)
	}
	opened, err := cipher.OpenRuntimeCredential("caido-auth", RuntimeCredentialProxyBearer, envelope)
	if err != nil {
		t.Fatal(err)
	}
	defer opened.Destroy()
	if err := opened.WithPlaintext(func(kind RuntimeCredentialKind, plaintext []byte) error {
		if kind != RuntimeCredentialProxyBearer || string(plaintext) != string(material.canonical) {
			t.Fatal("opened Runtime credential differs from exact input")
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := cipher.OpenRuntimeCredential("other-auth", RuntimeCredentialProxyBearer, envelope); !errors.Is(err, ErrCrypto) {
		t.Fatalf("wrong identity error = %v", err)
	}
	if _, err := cipher.OpenRuntimeCredential("caido-auth", RuntimeCredentialProxyBasic, envelope); !errors.Is(err, ErrCrypto) {
		t.Fatalf("wrong kind error = %v", err)
	}
	corrupt := cloneEnvelope(envelope)
	corrupt.Ciphertext[len(corrupt.Ciphertext)-1] ^= 0xff
	if _, err := cipher.OpenRuntimeCredential("caido-auth", RuntimeCredentialProxyBearer, corrupt); !errors.Is(err, ErrCrypto) {
		t.Fatalf("corrupt ciphertext error = %v", err)
	}

	firstMAC, err := cipher.RuntimeCredentialRequestMAC("caido-auth", material)
	if err != nil {
		t.Fatal(err)
	}
	secondMAC, _ := cipher.RuntimeCredentialRequestMAC("caido-auth", material)
	changed, _ := NewHTTPProxyBearerCredential("changed-bearer-secret")
	changedMAC, _ := cipher.RuntimeCredentialRequestMAC("caido-auth", changed)
	otherCipher, _ := NewTokenCipher(bytes.Repeat([]byte{0x64}, 32))
	otherMAC, _ := otherCipher.RuntimeCredentialRequestMAC("caido-auth", material)
	if !bytes.Equal(firstMAC, secondMAC) || bytes.Equal(firstMAC, changedMAC) || bytes.Equal(firstMAC, otherMAC) || len(firstMAC) != 32 {
		t.Fatal("Runtime credential request MAC is not stable and domain-keyed")
	}
	for _, formatted := range []string{fmt.Sprint(cipher), fmt.Sprintf("%#v", cipher)} {
		if !strings.Contains(formatted, "REDACTED") || strings.Contains(formatted, fmt.Sprintf("%x", key)) {
			t.Fatalf("unsafe credential cipher formatting: %q", formatted)
		}
	}
}

func TestRuntimeMACDerivationDoesNotChangeLegacyLiteLLMCiphertext(t *testing.T) {
	t.Parallel()
	key := bytes.Repeat([]byte{0x44}, 32)
	nonce := bytes.Repeat([]byte{0x23}, 12)
	cipherUnderTest, err := newTokenCipher(key, bytes.NewReader(nonce))
	if err != nil {
		t.Fatal(err)
	}
	token, _ := NewToken("legacy-litellm-token")
	gateway := testPinnedGatewayRef("1", strings.Repeat("4", 64))
	envelope, err := cipherUnderTest.Seal("legacy-worker", gateway, token)
	if err != nil {
		t.Fatal(err)
	}
	block, _ := aes.NewCipher(key)
	legacyAEAD, _ := cipher.NewGCM(block)
	legacyAAD := []byte(`{"schemaVersion":"contractor.credentials/v1","credentialId":"legacy-worker","llmGateway":{"gatewayId":"local-litellm","version":"1","digest":"sha256:4444444444444444444444444444444444444444444444444444444444444444"}}`)
	want := legacyAEAD.Seal(nil, nonce, []byte(token.value), legacyAAD)
	if !bytes.Equal(envelope.Nonce, nonce) || !bytes.Equal(envelope.Ciphertext, want) || envelope.SchemaVersion != CredentialSchemaVersion {
		t.Fatal("legacy LiteLLM ciphertext contract changed after Runtime MAC derivation")
	}
}
