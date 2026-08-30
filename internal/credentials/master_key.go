package credentials

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"golang.org/x/sys/unix"
)

const maximumEncodedMasterKeyBytes = 45

func LoadTokenCipher(path string) (*TokenCipher, error) {
	key, err := loadMasterKey(path)
	if err != nil {
		return nil, err
	}
	result, err := NewTokenCipher(key)
	for index := range key {
		key[index] = 0
	}
	if err != nil {
		return nil, ErrKeyUnavailable
	}
	return result, nil
}

func loadMasterKey(path string) ([]byte, error) {
	if !filepath.IsAbs(path) || filepath.Clean(path) != path {
		return nil, fmt.Errorf("%w: master-key file path must be clean and absolute", ErrKeyUnavailable)
	}
	info, err := os.Lstat(path)
	if err != nil || info.Mode()&os.ModeSymlink != 0 || !info.Mode().IsRegular() || info.Mode().Perm()&0o077 != 0 {
		return nil, fmt.Errorf("%w: master-key file must be regular, non-symlinked, and owner-only", ErrKeyUnavailable)
	}
	fd, err := unix.Open(path, unix.O_RDONLY|unix.O_NOFOLLOW|unix.O_CLOEXEC, 0)
	if err != nil {
		return nil, ErrKeyUnavailable
	}
	handle := os.NewFile(uintptr(fd), "credential-master-key")
	defer handle.Close()
	var stat unix.Stat_t
	if err := unix.Fstat(fd, &stat); err != nil || stat.Mode&unix.S_IFMT != unix.S_IFREG || stat.Mode&0o077 != 0 ||
		stat.Size < 1 || stat.Size > maximumEncodedMasterKeyBytes {
		return nil, ErrKeyUnavailable
	}
	encoded, err := io.ReadAll(io.LimitReader(handle, maximumEncodedMasterKeyBytes+1))
	if err != nil || len(encoded) == 0 || len(encoded) > maximumEncodedMasterKeyBytes {
		return nil, ErrKeyUnavailable
	}
	defer wipeBytes(encoded)
	if encoded[len(encoded)-1] == '\n' {
		encoded = encoded[:len(encoded)-1]
	}
	if len(encoded) == 0 || bytes.IndexByte(encoded, '\r') >= 0 || bytes.IndexByte(encoded, '\n') >= 0 {
		return nil, ErrKeyUnavailable
	}
	decoded := make([]byte, base64.StdEncoding.DecodedLen(len(encoded)))
	decodedLength, err := base64.StdEncoding.Strict().Decode(decoded, encoded)
	if err != nil || decodedLength != 32 {
		wipeBytes(decoded)
		return nil, ErrKeyUnavailable
	}
	return decoded[:decodedLength], nil
}

func wipeBytes(value []byte) {
	for index := range value {
		value[index] = 0
	}
}

func RequireTokenCipher(path string, activeCredentials int64) (*TokenCipher, error) {
	if activeCredentials < 0 {
		return nil, ErrKeyUnavailable
	}
	if strings.TrimSpace(path) == "" {
		if activeCredentials != 0 {
			return nil, fmt.Errorf("%w: encrypted credential rows require a master key", ErrKeyUnavailable)
		}
		return nil, nil
	}
	cipher, err := LoadTokenCipher(path)
	if err != nil {
		return nil, err
	}
	return cipher, nil
}
