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

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
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
	caido, err := NewCaidoBearerCredential("caido-bearer")
	if err != nil {
		t.Fatal(err)
	}
	originBasic, err := NewHTTPOriginBasicCredential("origin-user", "origin-password")
	if err != nil {
		t.Fatal(err)
	}
	originBearer, err := NewHTTPOriginBearerCredential("origin-bearer")
	if err != nil {
		t.Fatal(err)
	}
	for _, material := range []*RuntimeCredentialMaterial{&headers, &basic, &bearer, &caido, &originBasic, &originBearer} {
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
	if _, err := NewCaidoBearerCredential(""); !errors.Is(err, ErrRuntimeCredentialInvalid) {
		t.Fatalf("empty Caido bearer error = %v", err)
	}
	if _, err := NewCaidoBearerCredential("line\ntoken"); !errors.Is(err, ErrRuntimeCredentialInvalid) {
		t.Fatalf("control-byte Caido bearer error = %v", err)
	}
	if _, err := NewHTTPOriginBasicCredential("user:name", "password"); !errors.Is(err, ErrRuntimeCredentialInvalid) {
		t.Fatalf("origin colon username error = %v", err)
	}
	if _, err := NewHTTPOriginBearerCredential(""); !errors.Is(err, ErrRuntimeCredentialInvalid) {
		t.Fatalf("empty origin bearer error = %v", err)
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
	tokenCipher, err := newTokenCipher(key, bytes.NewReader(bytes.Repeat([]byte{0x17}, 48)))
	if err != nil {
		t.Fatal(err)
	}
	material, _ := NewHTTPProxyBearerCredential("exact-bearer-secret")
	envelope, err := tokenCipher.SealRuntimeCredential("caido-auth", material)
	if err != nil {
		t.Fatal(err)
	}
	opened, err := tokenCipher.OpenRuntimeCredential("caido-auth", RuntimeCredentialProxyBearer, envelope)
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
	if _, err := tokenCipher.OpenRuntimeCredential("other-auth", RuntimeCredentialProxyBearer, envelope); !errors.Is(err, ErrCrypto) {
		t.Fatalf("wrong identity error = %v", err)
	}
	if _, err := tokenCipher.OpenRuntimeCredential("caido-auth", RuntimeCredentialProxyBasic, envelope); !errors.Is(err, ErrCrypto) {
		t.Fatalf("wrong kind error = %v", err)
	}
	corrupt := cloneEnvelope(envelope)
	corrupt.Ciphertext[len(corrupt.Ciphertext)-1] ^= 0xff
	if _, err := tokenCipher.OpenRuntimeCredential("caido-auth", RuntimeCredentialProxyBearer, corrupt); !errors.Is(err, ErrCrypto) {
		t.Fatalf("corrupt ciphertext error = %v", err)
	}

	firstMAC, err := tokenCipher.RuntimeCredentialRequestMAC("caido-auth", material)
	if err != nil {
		t.Fatal(err)
	}
	secondMAC, _ := tokenCipher.RuntimeCredentialRequestMAC("caido-auth", material)
	changed, _ := NewHTTPProxyBearerCredential("changed-bearer-secret")
	changedMAC, _ := tokenCipher.RuntimeCredentialRequestMAC("caido-auth", changed)
	otherCipher, _ := NewTokenCipher(bytes.Repeat([]byte{0x64}, 32))
	otherMAC, _ := otherCipher.RuntimeCredentialRequestMAC("caido-auth", material)
	if !bytes.Equal(firstMAC, secondMAC) || bytes.Equal(firstMAC, changedMAC) || bytes.Equal(firstMAC, otherMAC) || len(firstMAC) != 32 {
		t.Fatal("Runtime credential request MAC is not stable and domain-keyed")
	}
	for _, formatted := range []string{fmt.Sprint(tokenCipher), fmt.Sprintf("%#v", tokenCipher)} {
		if !strings.Contains(formatted, "REDACTED") || strings.Contains(formatted, fmt.Sprintf("%x", key)) {
			t.Fatalf("unsafe credential cipher formatting: %q", formatted)
		}
	}
}

func TestCaidoBearerCredentialRoundTripsOnlyUnderExactKindAndIdentity(t *testing.T) {
	t.Parallel()
	tokenCipher, err := NewTokenCipher(bytes.Repeat([]byte{0x39}, 32))
	if err != nil {
		t.Fatal(err)
	}
	material, _ := NewCaidoBearerCredential("caido-exact-secret")
	envelope, err := tokenCipher.SealRuntimeCredential("caido-auth", material)
	if err != nil {
		t.Fatal(err)
	}
	opened, err := tokenCipher.OpenRuntimeCredential("caido-auth", RuntimeCredentialCaidoBearer, envelope)
	if err != nil {
		t.Fatal(err)
	}
	defer opened.Destroy()
	if err := opened.WithPlaintext(func(kind RuntimeCredentialKind, plaintext []byte) error {
		if kind != RuntimeCredentialCaidoBearer || string(plaintext) != `{"token":"caido-exact-secret"}` {
			return errors.New("Caido material differs")
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := tokenCipher.OpenRuntimeCredential("caido-auth", RuntimeCredentialProxyBearer, envelope); !errors.Is(err, ErrCrypto) {
		t.Fatalf("cross-kind Caido open error = %v", err)
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

func TestInteractshTokenCredentialRoundTripsOnlyUnderExactKindAndIdentity(t *testing.T) {
	t.Parallel()
	tokenCipher, err := NewTokenCipher(bytes.Repeat([]byte{0x39}, 32))
	if err != nil {
		t.Fatal(err)
	}
	material, _ := NewInteractshTokenCredential("interactsh-exact-secret")
	envelope, err := tokenCipher.SealRuntimeCredential("interactsh-auth", material)
	if err != nil {
		t.Fatal(err)
	}
	opened, err := tokenCipher.OpenRuntimeCredential("interactsh-auth", RuntimeCredentialInteractshToken, envelope)
	if err != nil {
		t.Fatal(err)
	}
	defer opened.Destroy()
	if err := opened.WithPlaintext(func(kind RuntimeCredentialKind, plaintext []byte) error {
		if kind != RuntimeCredentialInteractshToken || string(plaintext) != `{"token":"interactsh-exact-secret"}` {
			return errors.New("Interactsh material differs")
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := tokenCipher.OpenRuntimeCredential("interactsh-auth", RuntimeCredentialProxyBearer, envelope); !errors.Is(err, ErrCrypto) {
		t.Fatalf("cross-kind Interactsh open error = %v", err)
	}
}

// TestSharedCredentialFieldValueCases holds the Runtime credential
// constructors, the decoding of stored material, LLM credential tokens and
// development tokens to api/testdata/v1alpha1/credential-field-value-cases.json.
// A stored row written before the rule fails to open (FAIL-RCF-3) instead of
// reaching an allocation.
func TestSharedCredentialFieldValueCases(t *testing.T) {
	t.Parallel()
	cases := contractstest.ReadCredentialFieldValueCases(t)
	tokenCipher, err := NewTokenCipher(bytes.Repeat([]byte{0x5a}, 32))
	if err != nil {
		t.Fatal(err)
	}
	constructors := map[RuntimeCredentialKind]func(string) (RuntimeCredentialMaterial, error){
		RuntimeCredentialOriginBearer:    NewHTTPOriginBearerCredential,
		RuntimeCredentialProxyBearer:     NewHTTPProxyBearerCredential,
		RuntimeCredentialCaidoBearer:     NewCaidoBearerCredential,
		RuntimeCredentialInteractshToken: NewInteractshTokenCredential,
		RuntimeCredentialOTLPHeaders: func(value string) (RuntimeCredentialMaterial, error) {
			return NewOTLPHeadersCredential(map[string]string{"authorization": value})
		},
	}
	covered := map[RuntimeCredentialKind]bool{}
	llmCredential := false
	for _, field := range cases.Fields {
		switch {
		case field.Credential != nil:
			kind := RuntimeCredentialKind(field.Credential.Kind)
			construct, ok := constructors[kind]
			if !ok {
				t.Fatalf("%s: no constructor for kind %q", field.Name, kind)
			}
			covered[kind] = true
			for _, test := range field.Cases(cases.Grammar) {
				material, err := construct(test.Value)
				if (err == nil) != test.Valid || err != nil && !errors.Is(err, ErrRuntimeCredentialInvalid) {
					t.Errorf("%s %s: constructor error = %v, want valid %t", field.Name, test.Name, err, test.Valid)
				}
				material.Destroy()
				// Seal the material as an earlier Server could have stored it,
				// without the constructor, and open it as allocation preparation does.
				document := map[string]any{}
				contractstest.SetJSONPointer(t, document, field.Credential.Member, test.Value)
				canonical, err := json.Marshal(document)
				if err != nil {
					t.Fatal(err)
				}
				envelope, err := tokenCipher.SealRuntimeCredential("stored", RuntimeCredentialMaterial{kind: kind, canonical: canonical})
				if err != nil {
					t.Fatal(err)
				}
				opened, err := tokenCipher.OpenRuntimeCredential("stored", kind, envelope)
				if (err == nil) != test.Valid || err != nil && !errors.Is(err, ErrCrypto) {
					t.Errorf("%s %s: stored material open error = %v, want valid %t", field.Name, test.Name, err, test.Valid)
				}
				opened.Destroy()
			}
		case field.LLMCredential:
			llmCredential = true
			for _, test := range field.Cases(cases.Grammar) {
				if _, err := NewToken(test.Value); (err == nil) != test.Valid {
					t.Errorf("%s %s: NewToken error = %v, want valid %t", field.Name, test.Name, err, test.Valid)
				}
			}
			// A development token has no byte bound of its own.
			for _, test := range cases.Grammar {
				_, err := NewStaticProvider([]StaticEntry{{
					Metadata: resolved.CredentialMetadata{
						Ref:        llmgateway.LLMCredentialRef{CredentialID: "development-worker"},
						LLMGateway: testPinnedGatewayRef("1", strings.Repeat("5", 64)), Unrestricted: true,
					},
					Token: contracts.NewSecretString(test.Value),
				}})
				if (err == nil) != test.Valid {
					t.Errorf("%s %s: development token error = %v, want valid %t", field.Name, test.Name, err, test.Valid)
				}
			}
		}
	}
	for kind := range constructors {
		if !covered[kind] {
			t.Errorf("the shared cases do not cover Runtime credential kind %q", kind)
		}
	}
	if !llmCredential {
		t.Error("the shared cases do not cover the LLM credential token")
	}
}
