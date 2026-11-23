package credentials

import (
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

// Every credentials secret holder redacts itself for every fmt verb, flag and
// nesting, for log/slog and for encoding/json (SEC-R88).
func TestCredentialSecretHoldersRedactForEveryVerbAndEncoder(t *testing.T) {
	t.Parallel()
	token, err := NewToken("hunter2-token")
	if err != nil {
		t.Fatal(err)
	}
	contractstest.AssertRedacted(t, token, "hunter2-token", "[REDACTED]")

	material, err := NewHTTPProxyBearerCredential("hunter2-material")
	if err != nil {
		t.Fatal(err)
	}
	contractstest.AssertRedacted(t, material, "hunter2-material", "[REDACTED]")

	key := []byte("0123456789abcdef0123456789abcdef")
	cipher, err := NewTokenCipher(key)
	if err != nil {
		t.Fatal(err)
	}
	contractstest.AssertRedacted(t, cipher, string(key), "credentials.TokenCipher([REDACTED])")
	contractstest.AssertRedacted(t, *cipher, string(cipher.runtimeMACKey[:]), "credentials.TokenCipher([REDACTED])")

	creation := RuntimeCredentialCreation{
		IdempotencyKeyDigest: "hunter2-digest", RequestMAC: []byte("hunter2-mac"), CreatedAt: time.Unix(0, 0),
	}
	contractstest.AssertRedacted(t, creation, "hunter2-mac", "RuntimeCredentialCreation([REDACTED])")
	contractstest.AssertRedacted(t, creation, "hunter2-digest", "")

	record := RuntimeCredentialRecord{
		Metadata: RuntimeCredentialMetadata{CredentialID: "proxy", Kind: RuntimeCredentialProxyBearer},
		Envelope: EncryptedEnvelope{Ciphertext: []byte("hunter2-ciphertext")},
	}
	contractstest.AssertRedacted(t, record, "hunter2-ciphertext",
		`RuntimeCredentialRecord{CredentialID:"proxy" Kind:"http-proxy-bearer@1" [ENCRYPTED]}`)

	provider, err := NewStaticProvider(nil)
	if err != nil {
		t.Fatal(err)
	}
	contractstest.AssertRedacted(t, provider, "hunter2", "credentials.StaticProvider([REDACTED])")
	encrypted := &EncryptedProvider{cipher: cipher}
	contractstest.AssertRedacted(t, encrypted, string(cipher.runtimeMACKey[:]), "credentials.EncryptedProvider([REDACTED])")
}
