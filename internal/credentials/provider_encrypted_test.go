package credentials

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestEncryptedProviderReturnsSafeMetadataAndDecryptsExactBinding(t *testing.T) {
	t.Parallel()
	secret := "sk-provider-secret"
	record, cipher := sealedTestRecord(t, "managed-worker", secret)
	provider, err := NewEncryptedProvider(&credentialRecordReader{record: record}, cipher)
	if err != nil {
		t.Fatal(err)
	}
	metadata, err := provider.LookupLLMCredential(t.Context(), record.CredentialID)
	if err != nil {
		t.Fatal(err)
	}
	if metadata.Ref.CredentialID != record.CredentialID || metadata.LLMGateway != record.LLMGateway {
		t.Fatalf("metadata = %+v", metadata)
	}
	resolved, err := provider.ResolveLLMCredential(
		t.Context(), contracts.LLMCredentialRef{CredentialID: record.CredentialID}, record.LLMGateway,
	)
	if err != nil || resolved.Reveal() != secret {
		t.Fatalf("resolved credential = (%s, %v)", resolved, err)
	}
	wrongGateway := record.LLMGateway
	wrongGateway.Digest = "sha256:" + strings.Repeat("9", 64)
	if _, err := provider.ResolveLLMCredential(
		t.Context(), contracts.LLMCredentialRef{CredentialID: record.CredentialID}, wrongGateway,
	); !errors.Is(err, ErrNotFound) {
		t.Fatalf("wrong Gateway error = %v", err)
	}
	if _, err := provider.ResolveLLMCredential(
		t.Context(), contracts.LLMCredentialRef{CredentialID: "missing"}, record.LLMGateway,
	); !errors.Is(err, ErrNotFound) {
		t.Fatalf("missing credential error = %v", err)
	}
}

func TestEncryptedProviderFailsSafelyForStorageCryptoAndKeyErrors(t *testing.T) {
	t.Parallel()
	secret := "sk-do-not-leak"
	record, cipher := sealedTestRecord(t, "managed-worker", secret)
	tests := map[string]struct {
		reader RecordReader
		cipher *TokenCipher
		want   error
	}{
		"storage": {
			reader: &credentialRecordReader{err: errors.New("database included " + secret)},
			cipher: cipher, want: nil,
		},
		"corrupt envelope": {
			reader: &credentialRecordReader{record: corruptTestRecord(record)},
			cipher: cipher, want: ErrCrypto,
		},
		"missing key": {reader: &credentialRecordReader{record: record}, want: ErrKeyUnavailable},
	}
	for name, test := range tests {
		name, test := name, test
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			provider, err := NewEncryptedProvider(test.reader, test.cipher)
			if err != nil {
				t.Fatal(err)
			}
			_, err = provider.ResolveLLMCredential(
				t.Context(), contracts.LLMCredentialRef{CredentialID: record.CredentialID}, record.LLMGateway,
			)
			if err == nil || (test.want != nil && !errors.Is(err, test.want)) {
				t.Fatalf("ResolveLLMCredential error = %v, want %v", err, test.want)
			}
			if strings.Contains(err.Error(), secret) {
				t.Fatalf("error leaked credential: %v", err)
			}
		})
	}
}

func TestCompositeProviderDetectsIdentityCollisionsBeforeResolution(t *testing.T) {
	t.Parallel()
	gateway := testPinnedGatewayRef("1", strings.Repeat("1", 64))
	entry := func(token string) StaticEntry {
		return StaticEntry{
			Metadata: workflowconfig.CredentialMetadata{
				Ref: contracts.LLMCredentialRef{CredentialID: "shared"}, LLMGateway: gateway,
			},
			Token: contracts.NewSecretString(token),
		}
	}
	first, _ := NewStaticProvider([]StaticEntry{entry("first-secret")})
	second, _ := NewStaticProvider([]StaticEntry{entry("second-secret")})
	composite, err := NewCompositeProvider(first, second)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := composite.LookupLLMCredential(t.Context(), "shared"); !errors.Is(err, ErrConflict) {
		t.Fatalf("duplicate lookup error = %v", err)
	}
	if _, err := composite.ResolveLLMCredential(
		t.Context(), contracts.LLMCredentialRef{CredentialID: "shared"}, gateway,
	); !errors.Is(err, ErrConflict) {
		t.Fatalf("duplicate resolution error = %v", err)
	} else if strings.Contains(err.Error(), "secret") {
		t.Fatalf("duplicate error leaked credential: %v", err)
	}
	if _, err := NewCompositeProvider(nil); err == nil {
		t.Fatal("empty composite provider succeeded")
	}
}

func TestCredentialRecordJSONExcludesInternalAndSecretFields(t *testing.T) {
	t.Parallel()
	record, _ := sealedTestRecord(t, "managed-worker", "sk-json-secret")
	encoded, err := json.Marshal(record)
	if err != nil {
		t.Fatal(err)
	}
	for _, forbidden := range [][]byte{
		[]byte("sk-json-secret"), record.Envelope.Nonce, record.Envelope.Ciphertext,
		[]byte(record.RemoteKeyID), []byte(record.Envelope.KeyID),
	} {
		if len(forbidden) > 0 && bytes.Contains(encoded, forbidden) {
			t.Fatalf("public record JSON contains internal value: %s", encoded)
		}
	}
	if !bytes.Contains(encoded, []byte(`"credentialId":"managed-worker"`)) ||
		!bytes.Contains(encoded, []byte(`"effectivePolicy"`)) {
		t.Fatalf("record JSON lacks safe metadata: %s", encoded)
	}
	if envelope, err := json.Marshal(record.Envelope); err != nil || string(envelope) != "{}" {
		t.Fatalf("envelope JSON = %s, %v", envelope, err)
	}
}

type credentialRecordReader struct {
	record Record
	err    error
}

func (r *credentialRecordReader) GetCredential(_ context.Context, id string) (Record, error) {
	if r.err != nil {
		return Record{}, r.err
	}
	if r.record.CredentialID != id {
		return Record{}, ErrNotFound
	}
	return r.record, nil
}

func sealedTestRecord(t *testing.T, credentialID, secret string) (Record, *TokenCipher) {
	t.Helper()
	cipher, err := NewTokenCipher(bytes.Repeat([]byte{0x31}, 32))
	if err != nil {
		t.Fatal(err)
	}
	token, err := NewToken(secret)
	if err != nil {
		t.Fatal(err)
	}
	gateway := testPinnedGatewayRef("1", strings.Repeat("1", 64))
	envelope, err := cipher.Seal(credentialID, gateway, token)
	if err != nil {
		t.Fatal(err)
	}
	return Record{
		CredentialID: credentialID, LLMGateway: gateway, RemoteKeyID: strings.Repeat("a", 64),
		Label: "Managed worker", EffectivePolicy: EffectiveGatewayPolicy{
			ModelPolicies: []contracts.ModelPolicyRef{{
				PolicyID: "worker", Version: "1", Digest: "sha256:" + strings.Repeat("2", 64),
			}},
			Models: []string{"qwen/model"},
		},
		Envelope: envelope, CreatedAt: time.Date(2026, 8, 30, 12, 0, 0, 0, time.UTC),
	}, cipher
}

func corruptTestRecord(record Record) Record {
	record.Envelope = cloneEnvelope(record.Envelope)
	record.Envelope.Ciphertext[0] ^= 0xff
	return record
}
