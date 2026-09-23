package credentials

import (
	"bytes"
	"encoding/base64"
	"errors"
	"fmt"
	"strings"

	"github.com/grauwolf32/contractor/internal/securefile"
)

const maximumEncodedMasterKeyBytes = 45

func LoadTokenCipher(path string) (*TokenCipher, error) {
	key, err := loadMasterKey(path)
	if err != nil {
		return nil, err
	}
	result, err := NewTokenCipher(key)
	clear(key)
	if err != nil {
		return nil, ErrKeyUnavailable
	}
	return result, nil
}

func loadMasterKey(path string) ([]byte, error) {
	encoded, err := securefile.Read(path, maximumEncodedMasterKeyBytes)
	switch {
	case err == nil:
	case errors.Is(err, securefile.ErrPath):
		return nil, fmt.Errorf("%w: master-key file path must be clean and absolute", ErrKeyUnavailable)
	case errors.Is(err, securefile.ErrUnsafePath):
		return nil, fmt.Errorf("%w: master-key file must be regular, non-symlinked, and owner-only", ErrKeyUnavailable)
	default:
		return nil, ErrKeyUnavailable
	}
	defer clear(encoded)
	if encoded[len(encoded)-1] == '\n' {
		encoded = encoded[:len(encoded)-1]
	}
	if len(encoded) == 0 || bytes.IndexByte(encoded, '\r') >= 0 || bytes.IndexByte(encoded, '\n') >= 0 {
		return nil, ErrKeyUnavailable
	}
	decoded := make([]byte, base64.StdEncoding.DecodedLen(len(encoded)))
	decodedLength, err := base64.StdEncoding.Strict().Decode(decoded, encoded)
	if err != nil || decodedLength != 32 {
		clear(decoded)
		return nil, ErrKeyUnavailable
	}
	return decoded[:decodedLength], nil
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
