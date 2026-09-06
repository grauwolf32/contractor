package credentials

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"time"

	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
	"golang.org/x/crypto/ssh"
)

const MaximumGitKeyBytes = 32 << 10
const gitKeySchema = "git-ssh-key@1"

var ErrGitKeyInvalid = errors.New("Git SSH key must be a supported unencrypted private key of at most 32 KiB")
var ErrGitKeyMissing = errors.New("Git SSH key is not configured")

type GitKeyMetadata struct {
	Configured  bool       `json:"configured"`
	Fingerprint string     `json:"fingerprint,omitempty"`
	KeyType     string     `json:"keyType,omitempty"`
	UpdatedAt   *time.Time `json:"updatedAt,omitempty"`
}

// GitKeys never exposes stored private bytes. Signer captures a generation for
// one admitted import and remains valid if the owner later replaces the key.
type GitKeys struct {
	db     postgres.DBTX
	cipher *TokenCipher
}

func NewGitKeys(db postgres.DBTX, cipher *TokenCipher) *GitKeys {
	return &GitKeys{db: db, cipher: cipher}
}
func (s *GitKeys) Count(ctx context.Context) (int64, error) {
	var count int64
	err := s.db.QueryRow(ctx, `SELECT count(*) FROM git_ssh_keys`).Scan(&count)
	return count, err
}
func validGitOwner(owner string) bool {
	return len(owner) > 0 && len(owner) <= 256 && strings.TrimSpace(owner) == owner
}
func (s *GitKeys) Metadata(ctx context.Context, owner string) (GitKeyMetadata, error) {
	if !validGitOwner(owner) {
		return GitKeyMetadata{}, ErrGitKeyInvalid
	}
	var m GitKeyMetadata
	err := s.db.QueryRow(ctx, `SELECT fingerprint,key_type,updated_at FROM git_ssh_keys WHERE owner_id=$1`, owner).Scan(&m.Fingerprint, &m.KeyType, &m.UpdatedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return GitKeyMetadata{}, nil
	}
	m.Configured = err == nil
	return m, err
}
func (s *GitKeys) Replace(ctx context.Context, owner string, key []byte) (GitKeyMetadata, error) {
	if !validGitOwner(owner) {
		return GitKeyMetadata{}, ErrGitKeyInvalid
	}
	signer, err := parseGitKey(key)
	if err != nil {
		return GitKeyMetadata{}, err
	}
	var id [16]byte
	if _, err := rand.Read(id[:]); err != nil {
		return GitKeyMetadata{}, ErrCrypto
	}
	generation := hex.EncodeToString(id[:])
	envelope, err := s.cipher.sealGitKey(owner, generation, key)
	if err != nil {
		return GitKeyMetadata{}, err
	}
	m := GitKeyMetadata{Configured: true, Fingerprint: ssh.FingerprintSHA256(signer.PublicKey()), KeyType: signer.PublicKey().Type()}
	err = s.db.QueryRow(ctx, `INSERT INTO git_ssh_keys(owner_id,generation,encryption_schema_version,key_id,nonce,ciphertext,fingerprint,key_type)
 VALUES($1,$2,$3,$4,$5,$6,$7,$8) ON CONFLICT(owner_id) DO UPDATE SET
 generation=EXCLUDED.generation,encryption_schema_version=EXCLUDED.encryption_schema_version,key_id=EXCLUDED.key_id,
 nonce=EXCLUDED.nonce,ciphertext=EXCLUDED.ciphertext,fingerprint=EXCLUDED.fingerprint,key_type=EXCLUDED.key_type,updated_at=clock_timestamp()
 RETURNING updated_at`, owner, generation, envelope.SchemaVersion, envelope.KeyID, envelope.Nonce, envelope.Ciphertext, m.Fingerprint, m.KeyType).Scan(&m.UpdatedAt)
	return m, err
}
func (s *GitKeys) Delete(ctx context.Context, owner string) error {
	if !validGitOwner(owner) {
		return ErrGitKeyInvalid
	}
	_, err := s.db.Exec(ctx, `DELETE FROM git_ssh_keys WHERE owner_id=$1`, owner)
	return err
}
func (s *GitKeys) Signer(ctx context.Context, owner string) (ssh.Signer, error) {
	if !validGitOwner(owner) {
		return nil, ErrGitKeyInvalid
	}
	var generation, fingerprint, keyType string
	var envelope EncryptedEnvelope
	err := s.db.QueryRow(ctx, `SELECT generation,encryption_schema_version,key_id,nonce,ciphertext,fingerprint,key_type FROM git_ssh_keys WHERE owner_id=$1`, owner).Scan(&generation, &envelope.SchemaVersion, &envelope.KeyID, &envelope.Nonce, &envelope.Ciphertext, &fingerprint, &keyType)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, ErrGitKeyMissing
	}
	if err != nil {
		return nil, err
	}
	key, err := s.cipher.openGitKey(owner, generation, envelope)
	if err != nil {
		return nil, err
	}
	defer wipeBytes(key)
	signer, err := parseGitKey(key)
	if err != nil || ssh.FingerprintSHA256(signer.PublicKey()) != fingerprint || signer.PublicKey().Type() != keyType {
		return nil, ErrCrypto
	}
	return signer, nil
}

// Verify decrypts rows in bounded owner batches, not merely their key IDs.
func (s *GitKeys) Verify(ctx context.Context) error {
	after := ""
	for {
		rows, err := s.db.Query(ctx, `SELECT owner_id FROM git_ssh_keys WHERE owner_id>$1 ORDER BY owner_id LIMIT 100`, after)
		if err != nil {
			return err
		}
		var owners []string
		for rows.Next() {
			var owner string
			if err := rows.Scan(&owner); err != nil {
				rows.Close()
				return err
			}
			owners = append(owners, owner)
		}
		err = rows.Err()
		rows.Close()
		if err != nil {
			return err
		}
		if len(owners) == 0 {
			return nil
		}
		for _, owner := range owners {
			if _, err := s.Signer(ctx, owner); err != nil && !errors.Is(err, ErrGitKeyMissing) {
				return err
			}
		}
		after = owners[len(owners)-1]
	}
}
func parseGitKey(key []byte) (ssh.Signer, error) {
	if len(key) == 0 || len(key) > MaximumGitKeyBytes {
		return nil, ErrGitKeyInvalid
	}
	signer, err := ssh.ParsePrivateKey(key)
	if err != nil {
		return nil, ErrGitKeyInvalid
	}
	return signer, nil
}
func gitKeyAAD(owner, generation string) []byte {
	value, _ := json.Marshal(struct{ Schema, Owner, Generation string }{gitKeySchema, owner, generation})
	return value
}
func (c *TokenCipher) sealGitKey(owner, generation string, key []byte) (EncryptedEnvelope, error) {
	if c == nil || c.aead == nil {
		return EncryptedEnvelope{}, ErrKeyUnavailable
	}
	nonce := make([]byte, c.aead.NonceSize())
	if _, err := io.ReadFull(c.random, nonce); err != nil {
		return EncryptedEnvelope{}, ErrCrypto
	}
	return EncryptedEnvelope{SchemaVersion: gitKeySchema, KeyID: c.keyID, Nonce: nonce, Ciphertext: c.aead.Seal(nil, nonce, key, gitKeyAAD(owner, generation))}, nil
}
func (c *TokenCipher) openGitKey(owner, generation string, e EncryptedEnvelope) ([]byte, error) {
	if c == nil || c.aead == nil {
		return nil, ErrKeyUnavailable
	}
	if e.SchemaVersion != gitKeySchema || e.KeyID != c.keyID || len(e.Nonce) != c.aead.NonceSize() || len(e.Ciphertext) < 17 || len(e.Ciphertext) > MaximumGitKeyBytes+16 {
		return nil, ErrCrypto
	}
	key, err := c.aead.Open(nil, e.Nonce, e.Ciphertext, gitKeyAAD(owner, generation))
	if err != nil {
		return nil, ErrCrypto
	}
	return key, nil
}
