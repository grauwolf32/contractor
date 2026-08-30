package credentials

import (
	"crypto/aes"
	"crypto/cipher"
	cryptorand "crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"

	"github.com/grauwolf32/contractor/internal/contracts"
)

type TokenCipher struct {
	aead   cipher.AEAD
	keyID  string
	random io.Reader
}

func NewTokenCipher(key []byte) (*TokenCipher, error) {
	return newTokenCipher(key, cryptorand.Reader)
}

func newTokenCipher(key []byte, random io.Reader) (*TokenCipher, error) {
	if len(key) != 32 || random == nil {
		return nil, ErrKeyUnavailable
	}
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, ErrKeyUnavailable
	}
	aead, err := cipher.NewGCM(block)
	if err != nil || aead.NonceSize() != 12 {
		return nil, ErrKeyUnavailable
	}
	fingerprint := sha256.Sum256(key)
	return &TokenCipher{
		aead: aead, keyID: "sha256:" + hex.EncodeToString(fingerprint[:]), random: random,
	}, nil
}

func (c *TokenCipher) KeyID() string {
	if c == nil {
		return ""
	}
	return c.keyID
}

func (c *TokenCipher) Seal(
	credentialID string,
	gateway contracts.LLMGatewayConfigRef,
	token Token,
) (EncryptedEnvelope, error) {
	if c == nil || c.aead == nil {
		return EncryptedEnvelope{}, ErrKeyUnavailable
	}
	if err := validateCipherIdentity(credentialID, gateway); err != nil {
		return EncryptedEnvelope{}, err
	}
	if len(token.value) == 0 || len(token.value) > MaximumTokenBytes {
		return EncryptedEnvelope{}, fmt.Errorf("%w: token is outside the bounded envelope", ErrInvalid)
	}
	aad, err := credentialAAD(credentialID, gateway)
	if err != nil {
		return EncryptedEnvelope{}, ErrCrypto
	}
	nonce := make([]byte, c.aead.NonceSize())
	if _, err := io.ReadFull(c.random, nonce); err != nil {
		return EncryptedEnvelope{}, ErrCrypto
	}
	plaintext := []byte(token.value)
	ciphertext := c.aead.Seal(nil, nonce, plaintext, aad)
	for index := range plaintext {
		plaintext[index] = 0
	}
	return EncryptedEnvelope{
		SchemaVersion: CredentialSchemaVersion,
		KeyID:         c.keyID,
		Nonce:         append([]byte(nil), nonce...),
		Ciphertext:    append([]byte(nil), ciphertext...),
	}, nil
}

func (c *TokenCipher) Open(
	credentialID string,
	gateway contracts.LLMGatewayConfigRef,
	envelope EncryptedEnvelope,
) (Token, error) {
	if c == nil || c.aead == nil {
		return Token{}, ErrKeyUnavailable
	}
	if err := validateCipherIdentity(credentialID, gateway); err != nil {
		return Token{}, err
	}
	if envelope.SchemaVersion != CredentialSchemaVersion || envelope.KeyID != c.keyID ||
		len(envelope.Nonce) != c.aead.NonceSize() || len(envelope.Ciphertext) < c.aead.Overhead()+1 ||
		len(envelope.Ciphertext) > MaximumTokenBytes+c.aead.Overhead() {
		return Token{}, ErrCrypto
	}
	aad, err := credentialAAD(credentialID, gateway)
	if err != nil {
		return Token{}, ErrCrypto
	}
	plaintext, err := c.aead.Open(nil, envelope.Nonce, envelope.Ciphertext, aad)
	if err != nil {
		return Token{}, ErrCrypto
	}
	token, err := NewToken(string(plaintext))
	for index := range plaintext {
		plaintext[index] = 0
	}
	if err != nil {
		return Token{}, ErrCrypto
	}
	return token, nil
}

func validateCipherIdentity(credentialID string, gateway contracts.LLMGatewayConfigRef) error {
	if err := (contracts.LLMCredentialRef{CredentialID: credentialID}).Validate(); err != nil {
		return fmt.Errorf("%w: credential identity is invalid", ErrInvalid)
	}
	if err := gateway.ValidateRef(); err != nil {
		return fmt.Errorf("%w: Gateway identity is invalid", ErrInvalid)
	}
	return nil
}

func credentialAAD(credentialID string, gateway contracts.LLMGatewayConfigRef) ([]byte, error) {
	return json.Marshal(struct {
		SchemaVersion string                        `json:"schemaVersion"`
		CredentialID  string                        `json:"credentialId"`
		LLMGateway    contracts.LLMGatewayConfigRef `json:"llmGateway"`
	}{
		SchemaVersion: CredentialSchemaVersion,
		CredentialID:  credentialID,
		LLMGateway:    gateway,
	})
}
