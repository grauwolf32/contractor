package credentials

import (
	"bytes"
	"encoding/base64"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"golang.org/x/sys/unix"
)

func TestLoadTokenCipherAcceptsStrictOwnerOnlyBase64File(t *testing.T) {
	t.Parallel()
	key := bytes.Repeat([]byte{0x41}, 32)
	for _, suffix := range []string{"", "\n"} {
		suffix := suffix
		t.Run(map[bool]string{true: "trailing-newline", false: "no-newline"}[suffix != ""], func(t *testing.T) {
			t.Parallel()
			path := writeMasterKeyFile(t, base64.StdEncoding.EncodeToString(key)+suffix, 0o600)
			cipher, err := LoadTokenCipher(path)
			if err != nil {
				t.Fatal(err)
			}
			want, _ := NewTokenCipher(key)
			if cipher.KeyID() != want.KeyID() {
				t.Fatalf("key ID = %q, want %q", cipher.KeyID(), want.KeyID())
			}
		})
	}
}

func TestLoadTokenCipherRejectsUnsafeOrMalformedFiles(t *testing.T) {
	t.Parallel()
	valid := base64.StdEncoding.EncodeToString(bytes.Repeat([]byte{0x42}, 32))
	tests := map[string]func(*testing.T) string{
		"relative": func(t *testing.T) string { return "relative-key" },
		"missing":  func(t *testing.T) string { return filepath.Join(t.TempDir(), "missing") },
		"group readable": func(t *testing.T) string {
			return writeMasterKeyFile(t, valid, 0o640)
		},
		"world readable": func(t *testing.T) string {
			return writeMasterKeyFile(t, valid, 0o604)
		},
		"wrong length": func(t *testing.T) string {
			return writeMasterKeyFile(t, base64.StdEncoding.EncodeToString([]byte("short")), 0o600)
		},
		"malformed base64": func(t *testing.T) string {
			return writeMasterKeyFile(t, strings.Repeat("!", 44), 0o600)
		},
		"CRLF": func(t *testing.T) string { return writeMasterKeyFile(t, valid+"\r\n", 0o600) },
		"two newlines": func(t *testing.T) string {
			return writeMasterKeyFile(t, valid+"\n\n", 0o600)
		},
		"symlink": func(t *testing.T) string {
			target := writeMasterKeyFile(t, valid, 0o600)
			link := filepath.Join(t.TempDir(), "key-link")
			if err := os.Symlink(target, link); err != nil {
				t.Fatal(err)
			}
			return link
		},
		"non-regular": func(t *testing.T) string {
			path := filepath.Join(t.TempDir(), "key-fifo")
			if err := unix.Mkfifo(path, 0o600); err != nil {
				t.Fatal(err)
			}
			return path
		},
	}
	for name, makePath := range tests {
		name, makePath := name, makePath
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			path := makePath(t)
			if _, err := LoadTokenCipher(path); !errors.Is(err, ErrKeyUnavailable) {
				t.Fatalf("LoadTokenCipher error = %v", err)
			} else if strings.Contains(err.Error(), valid) {
				t.Fatalf("error contains key material: %v", err)
			}
		})
	}
}

func TestRequireTokenCipherEnforcesActiveRowStartupRule(t *testing.T) {
	t.Parallel()
	if cipher, err := RequireTokenCipher("", 0); err != nil || cipher != nil {
		t.Fatalf("empty store result = (%v, %v)", cipher, err)
	}
	if _, err := RequireTokenCipher("", 1); !errors.Is(err, ErrKeyUnavailable) {
		t.Fatalf("active store without key error = %v", err)
	}
	if _, err := RequireTokenCipher("not-absolute", 0); !errors.Is(err, ErrKeyUnavailable) {
		t.Fatalf("configured invalid key error = %v", err)
	}
	if _, err := RequireTokenCipher("", -1); !errors.Is(err, ErrKeyUnavailable) {
		t.Fatalf("negative active-row count error = %v", err)
	}
}

func writeMasterKeyFile(t *testing.T, contents string, mode os.FileMode) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "credential-master-key")
	if err := os.WriteFile(path, []byte(contents), mode); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(path, mode); err != nil {
		t.Fatal(err)
	}
	return path
}
