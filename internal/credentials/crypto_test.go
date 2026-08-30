package credentials

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestTokenCipherRoundTripUsesFreshNonceAndBoundAAD(t *testing.T) {
	t.Parallel()
	key := bytes.Repeat([]byte{0x2a}, 32)
	cipher, err := NewTokenCipher(key)
	if err != nil {
		t.Fatal(err)
	}
	wantFingerprint := sha256.Sum256(key)
	if want := "sha256:" + hex.EncodeToString(wantFingerprint[:]); cipher.KeyID() != want {
		t.Fatalf("key ID = %q, want %q", cipher.KeyID(), want)
	}
	token, err := NewToken("sk-exact-token\nbytes")
	if err != nil {
		t.Fatal(err)
	}
	gateway := testPinnedGatewayRef("1", strings.Repeat("1", 64))
	first, err := cipher.Seal("worker-primary", gateway, token)
	if err != nil {
		t.Fatal(err)
	}
	second, err := cipher.Seal("worker-primary", gateway, token)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Equal(first.Nonce, second.Nonce) || bytes.Equal(first.Ciphertext, second.Ciphertext) {
		t.Fatal("two seals reused nonce or ciphertext")
	}
	opened, err := cipher.Open("worker-primary", gateway, first)
	if err != nil {
		t.Fatal(err)
	}
	if opened.value != token.value {
		t.Fatalf("opened token differs from exact input")
	}

	tests := map[string]func() (string, contracts.LLMGatewayConfigRef, EncryptedEnvelope, *TokenCipher){
		"credential identity": func() (string, contracts.LLMGatewayConfigRef, EncryptedEnvelope, *TokenCipher) {
			return "worker-secondary", gateway, cloneEnvelope(first), cipher
		},
		"Gateway identity": func() (string, contracts.LLMGatewayConfigRef, EncryptedEnvelope, *TokenCipher) {
			return "worker-primary", testPinnedGatewayRef("2", strings.Repeat("2", 64)), cloneEnvelope(first), cipher
		},
		"nonce": func() (string, contracts.LLMGatewayConfigRef, EncryptedEnvelope, *TokenCipher) {
			envelope := cloneEnvelope(first)
			envelope.Nonce[0] ^= 0xff
			return "worker-primary", gateway, envelope, cipher
		},
		"ciphertext": func() (string, contracts.LLMGatewayConfigRef, EncryptedEnvelope, *TokenCipher) {
			envelope := cloneEnvelope(first)
			envelope.Ciphertext[len(envelope.Ciphertext)-1] ^= 0xff
			return "worker-primary", gateway, envelope, cipher
		},
		"key": func() (string, contracts.LLMGatewayConfigRef, EncryptedEnvelope, *TokenCipher) {
			other, otherErr := NewTokenCipher(bytes.Repeat([]byte{0x17}, 32))
			if otherErr != nil {
				t.Fatal(otherErr)
			}
			return "worker-primary", gateway, cloneEnvelope(first), other
		},
	}
	for name, mutate := range tests {
		name, mutate := name, mutate
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			credentialID, currentGateway, envelope, currentCipher := mutate()
			if _, err := currentCipher.Open(credentialID, currentGateway, envelope); !errors.Is(err, ErrCrypto) {
				t.Fatalf("Open error = %v, want ErrCrypto", err)
			}
		})
	}
}

func TestTokenCipherRejectsInvalidInputsAndRandomFailure(t *testing.T) {
	t.Parallel()
	if _, err := NewTokenCipher([]byte("short")); !errors.Is(err, ErrKeyUnavailable) {
		t.Fatalf("short key error = %v", err)
	}
	cipher, err := newTokenCipher(bytes.Repeat([]byte{7}, 32), failingReader{})
	if err != nil {
		t.Fatal(err)
	}
	token, _ := NewToken("sk-token")
	if _, err := cipher.Seal("worker", testPinnedGatewayRef("1", strings.Repeat("1", 64)), token); !errors.Is(err, ErrCrypto) {
		t.Fatalf("random failure error = %v", err)
	}
	if _, err := cipher.Seal("INVALID", testPinnedGatewayRef("1", strings.Repeat("1", 64)), token); !errors.Is(err, ErrInvalid) {
		t.Fatalf("invalid identity error = %v", err)
	}
	if _, err := (*TokenCipher)(nil).Open("worker", testPinnedGatewayRef("1", strings.Repeat("1", 64)), EncryptedEnvelope{}); !errors.Is(err, ErrKeyUnavailable) {
		t.Fatalf("nil cipher error = %v", err)
	}
}

func TestTokenIsBoundedAndRedacted(t *testing.T) {
	t.Parallel()
	secret := "sk-never-print-this"
	token, err := NewToken(secret)
	if err != nil {
		t.Fatal(err)
	}
	for _, formatted := range []string{
		fmt.Sprint(token), fmt.Sprintf("%s", token), fmt.Sprintf("%v", token), fmt.Sprintf("%#v", token),
	} {
		if strings.Contains(formatted, secret) || !strings.Contains(formatted, "REDACTED") {
			t.Fatalf("unsafe token formatting: %q", formatted)
		}
	}
	if encoded, err := json.Marshal(token); err == nil || bytes.Contains(encoded, []byte(secret)) {
		t.Fatalf("token JSON = %q, %v", encoded, err)
	}
	for name, value := range map[string]string{
		"empty":         "",
		"oversized":     strings.Repeat("x", MaximumTokenBytes+1),
		"invalid UTF-8": string([]byte{0xff}),
	} {
		if _, err := NewToken(value); !errors.Is(err, ErrInvalid) {
			t.Fatalf("%s token error = %v", name, err)
		}
	}
}

func TestEffectiveGatewayPolicyRequiresCanonicalBoundedValues(t *testing.T) {
	t.Parallel()
	budget := 4.5
	two := 2
	valid := EffectiveGatewayPolicy{
		ModelPolicies: []contracts.ModelPolicyRef{
			{PolicyID: "planner", Version: "1", Digest: "sha256:" + strings.Repeat("1", 64)},
			{PolicyID: "worker", Version: "1", Digest: "sha256:" + strings.Repeat("2", 64)},
		},
		Models: []string{"model-a", "model-b"}, MaxBudget: &budget, BudgetDuration: "30d",
		TPMLimit: &two, RPMLimit: &two, MaxParallelRequests: &two,
	}
	if err := validateEffectiveGatewayPolicy(valid); err != nil {
		t.Fatalf("valid policy: %v", err)
	}
	tests := map[string]func(EffectiveGatewayPolicy) EffectiveGatewayPolicy{
		"unsorted refs": func(value EffectiveGatewayPolicy) EffectiveGatewayPolicy {
			value.ModelPolicies[0], value.ModelPolicies[1] = value.ModelPolicies[1], value.ModelPolicies[0]
			return value
		},
		"unsorted models": func(value EffectiveGatewayPolicy) EffectiveGatewayPolicy {
			value.Models[0], value.Models[1] = value.Models[1], value.Models[0]
			return value
		},
		"invalid duration": func(value EffectiveGatewayPolicy) EffectiveGatewayPolicy {
			value.BudgetDuration = "monthly"
			return value
		},
		"duration without budget": func(value EffectiveGatewayPolicy) EffectiveGatewayPolicy {
			value.MaxBudget = nil
			return value
		},
	}
	for name, mutate := range tests {
		value := clonePolicy(valid)
		if err := validateEffectiveGatewayPolicy(mutate(value)); !errors.Is(err, ErrInvalid) {
			t.Fatalf("%s error = %v", name, err)
		}
	}
}

func TestGatewayPolicyValidatesSetWithoutRequiringInputOrder(t *testing.T) {
	t.Parallel()
	budget := 1.25
	policy := GatewayPolicy{
		ModelPolicies: []contracts.ModelPolicyRef{
			{PolicyID: "worker", Version: "1", Digest: "sha256:" + strings.Repeat("2", 64)},
			{PolicyID: "planner", Version: "1", Digest: "sha256:" + strings.Repeat("1", 64)},
		},
		MaxBudget: &budget, BudgetDuration: "1mo",
	}
	if err := policy.Validate(); err != nil {
		t.Fatalf("valid unordered request policy: %v", err)
	}
	policy.ModelPolicies[1] = policy.ModelPolicies[0]
	if err := policy.Validate(); !errors.Is(err, ErrInvalid) {
		t.Fatalf("duplicate request policy error = %v", err)
	}
}

type failingReader struct{}

func (failingReader) Read([]byte) (int, error) { return 0, errors.New("random source failed") }

func cloneEnvelope(value EncryptedEnvelope) EncryptedEnvelope {
	value.Nonce = append([]byte(nil), value.Nonce...)
	value.Ciphertext = append([]byte(nil), value.Ciphertext...)
	return value
}

func clonePolicy(value EffectiveGatewayPolicy) EffectiveGatewayPolicy {
	value.ModelPolicies = append([]contracts.ModelPolicyRef(nil), value.ModelPolicies...)
	value.Models = append([]string(nil), value.Models...)
	return value
}

func testPinnedGatewayRef(version, digest string) contracts.LLMGatewayConfigRef {
	return contracts.LLMGatewayConfigRef{
		GatewayID: "local-litellm", Version: version, Digest: "sha256:" + digest,
	}
}
