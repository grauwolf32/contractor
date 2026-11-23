package credentials

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/pem"
	"errors"
	"os"
	"testing"
)

func testGitPrivateKey(t *testing.T) []byte {
	t.Helper()
	_, key, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	data, err := x509.MarshalPKCS8PrivateKey(key)
	if err != nil {
		t.Fatal(err)
	}
	return pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: data})
}
func TestGitKeyEnvelopeAuthenticatesOwnerPurposeAndGeneration(t *testing.T) {
	c, _ := NewTokenCipher(bytes.Repeat([]byte{1}, 32))
	key := testGitPrivateKey(t)
	e, err := c.sealGitKey("owner", "generation", key)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := c.openGitKey("owner", "generation", e)
	if err != nil || !bytes.Equal(decoded, key) {
		t.Fatal("round trip failed")
	}
	for _, identity := range [][2]string{{"foreign", "generation"}, {"owner", "other"}} {
		if _, err := c.openGitKey(identity[0], identity[1], e); !errors.Is(err, ErrCrypto) {
			t.Fatal("identity swap accepted")
		}
	}
	e.SchemaVersion = "runtime-credential@1"
	if _, err := c.openGitKey("owner", "generation", e); !errors.Is(err, ErrCrypto) {
		t.Fatal("purpose swap accepted")
	}
	for _, key := range [][]byte{nil, []byte("secret-canary-invalid"), bytes.Repeat([]byte{1}, MaximumGitKeyBytes+1)} {
		if _, err := parseGitKey(key); !errors.Is(err, ErrGitKeyInvalid) {
			t.Fatal("invalid key accepted")
		}
	}
	rsaKey, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := parseGitKey(pem.EncodeToMemory(&pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(rsaKey)})); err != nil {
		t.Fatal(err)
	}
}
func TestGitKeyPostgresOwnerIsolationReplacementAndStartupVerification(t *testing.T) {
	database := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if database == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx := t.Context()
	pool := isolatedCredentialPool(t, ctx, database)
	c, _ := NewTokenCipher(bytes.Repeat([]byte{1}, 32))
	keys := NewGitKeys(pool, c)
	key := testGitPrivateKey(t)
	first, err := keys.Replace(ctx, "owner", key)
	if err != nil || !first.Configured {
		t.Fatal(err)
	}
	signer, err := keys.Signer(ctx, "owner")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := keys.Signer(ctx, "foreign"); !errors.Is(err, ErrGitKeyMissing) {
		t.Fatal("foreign owner obtained key")
	}
	if err := keys.Delete(ctx, "foreign"); err != nil {
		t.Fatal(err)
	}
	if _, err := keys.Replace(ctx, "owner", []byte("invalid")); !errors.Is(err, ErrGitKeyInvalid) {
		t.Fatal(err)
	}
	still, _ := keys.Metadata(ctx, "owner")
	if still.Fingerprint != first.Fingerprint {
		t.Fatal("invalid replacement changed key")
	}
	var ciphertext []byte
	if err := pool.QueryRow(ctx, `SELECT ciphertext FROM git_ssh_keys`).Scan(&ciphertext); err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(ciphertext, key) {
		t.Fatal("plaintext stored")
	}
	if err := NewGitKeys(pool, c).Verify(ctx); err != nil {
		t.Fatal(err)
	}
	bad, _ := NewTokenCipher(bytes.Repeat([]byte{2}, 32))
	if err := NewGitKeys(pool, bad).Verify(ctx); !errors.Is(err, ErrCrypto) {
		t.Fatal("wrong master key accepted")
	}
	count, _ := keys.Count(ctx)
	if _, err := RequireTokenCipher("", count); !errors.Is(err, ErrKeyUnavailable) {
		t.Fatal("missing master key accepted")
	}
	second, err := keys.Replace(ctx, "owner", testGitPrivateKey(t))
	if err != nil || second.Fingerprint == first.Fingerprint {
		t.Fatal("replacement failed")
	}
	if err := keys.Delete(ctx, "owner"); err != nil {
		t.Fatal(err)
	}
	if _, err := signer.Sign(rand.Reader, []byte("captured generation")); err != nil {
		t.Fatal(err)
	}
	if _, err := keys.Signer(ctx, "owner"); !errors.Is(err, ErrGitKeyMissing) {
		t.Fatal("removed key still configured")
	}
	if err := keys.Verify(ctx); err != nil {
		t.Fatal(err)
	}
}
