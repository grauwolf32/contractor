package auth

import (
	"crypto/rand"
	"crypto/subtle"
	"encoding/base64"
	"fmt"
	"strings"
	"unicode/utf8"

	"golang.org/x/crypto/argon2"
)

const (
	ArgonMemoryKiB       uint32 = 64 * 1024
	ArgonIterations      uint32 = 3
	ArgonParallelism     uint8  = 1
	ArgonSaltBytes              = 16
	ArgonOutputBytes            = 32
	MinimumPasswordBytes        = 12
	MaximumPasswordBytes        = 1024
)

type passwordHash struct {
	salt   [ArgonSaltBytes]byte
	output [ArgonOutputBytes]byte
}

func ValidatePassword(password []byte) error {
	if len(password) < MinimumPasswordBytes || len(password) > MaximumPasswordBytes || !utf8.Valid(password) {
		return fmt.Errorf("%w: password must contain %d through %d UTF-8 bytes", ErrInvalidCredentials, MinimumPasswordBytes, MaximumPasswordBytes)
	}
	return nil
}

func HashPassword(password []byte) (string, error) {
	if err := ValidatePassword(password); err != nil {
		return "", err
	}
	var salt [ArgonSaltBytes]byte
	if _, err := rand.Read(salt[:]); err != nil {
		return "", fmt.Errorf("generate Argon2id salt: %w", err)
	}
	output := argon2.IDKey(password, salt[:], ArgonIterations, ArgonMemoryKiB, ArgonParallelism, ArgonOutputBytes)
	defer wipe(output)
	return fmt.Sprintf(
		"$argon2id$v=19$m=%d,t=%d,p=%d$%s$%s",
		ArgonMemoryKiB,
		ArgonIterations,
		ArgonParallelism,
		base64.RawStdEncoding.EncodeToString(salt[:]),
		base64.RawStdEncoding.EncodeToString(output),
	), nil
}

func parsePasswordHash(source string) (passwordHash, error) {
	if len(source) < 64 || len(source) > 256 {
		return passwordHash{}, ErrInvalidBootstrap
	}
	parts := strings.Split(source, "$")
	if len(parts) != 6 || parts[0] != "" || parts[1] != "argon2id" || parts[2] != "v=19" ||
		parts[3] != "m=65536,t=3,p=1" {
		return passwordHash{}, ErrInvalidBootstrap
	}
	salt, err := base64.RawStdEncoding.Strict().DecodeString(parts[4])
	if err != nil || len(salt) != ArgonSaltBytes || base64.RawStdEncoding.EncodeToString(salt) != parts[4] {
		wipe(salt)
		return passwordHash{}, ErrInvalidBootstrap
	}
	defer wipe(salt)
	output, err := base64.RawStdEncoding.Strict().DecodeString(parts[5])
	if err != nil || len(output) != ArgonOutputBytes || base64.RawStdEncoding.EncodeToString(output) != parts[5] {
		wipe(output)
		return passwordHash{}, ErrInvalidBootstrap
	}
	defer wipe(output)
	var result passwordHash
	copy(result.salt[:], salt)
	copy(result.output[:], output)
	return result, nil
}

func (h passwordHash) verify(password []byte) bool {
	output := argon2.IDKey(password, h.salt[:], ArgonIterations, ArgonMemoryKiB, ArgonParallelism, ArgonOutputBytes)
	defer wipe(output)
	return subtle.ConstantTimeCompare(output, h.output[:]) == 1
}

func wipe(value []byte) {
	for index := range value {
		value[index] = 0
	}
}
