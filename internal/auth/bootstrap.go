package auth

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"go.yaml.in/yaml/v4"
	"golang.org/x/sys/unix"
)

const maximumBootstrapBytes = 8 * 1024

type Bootstrap struct {
	Principal Principal
	hash      passwordHash
	valid     bool
}

func (b Bootstrap) String() string {
	return fmt.Sprintf("auth.Bootstrap{UserID:%q Username:%q PasswordHash:[REDACTED]}", b.Principal.UserID, b.Principal.Username)
}

func (b Bootstrap) GoString() string { return b.String() }

type bootstrapDocument struct {
	User bootstrapUser `yaml:"user"`
}

type bootstrapUser struct {
	UserID       string `yaml:"userId"`
	Username     string `yaml:"username"`
	PasswordHash string `yaml:"passwordHash"`
}

func NewBootstrap(userID, username, encodedHash string) (Bootstrap, error) {
	principal, err := NewPrincipal(userID, username)
	if err != nil {
		return Bootstrap{}, err
	}
	hash, err := parsePasswordHash(encodedHash)
	if err != nil {
		return Bootstrap{}, ErrInvalidBootstrap
	}
	return Bootstrap{Principal: principal, hash: hash, valid: true}, nil
}

func LoadBootstrap(path string) (Bootstrap, error) {
	data, err := readBootstrap(path)
	if err != nil {
		return Bootstrap{}, err
	}
	defer wipe(data)
	var documents []bootstrapDocument
	if err := yaml.Load(
		data,
		&documents,
		yaml.WithAllDocuments(),
		yaml.WithKnownFields(),
		yaml.WithUniqueKeys(),
	); err != nil || len(documents) != 1 {
		return Bootstrap{}, fmt.Errorf("%w: decode strict local-auth YAML", ErrInvalidBootstrap)
	}
	return NewBootstrap(
		documents[0].User.UserID,
		documents[0].User.Username,
		documents[0].User.PasswordHash,
	)
}

func BootstrapYAML(userID, username, encodedHash string) ([]byte, error) {
	if _, err := NewBootstrap(userID, username, encodedHash); err != nil {
		return nil, err
	}
	document := bootstrapDocument{User: bootstrapUser{
		UserID: userID, Username: username, PasswordHash: encodedHash,
	}}
	encoded, err := yaml.Marshal(document)
	if err != nil {
		return nil, fmt.Errorf("encode local-auth bootstrap: %w", err)
	}
	return encoded, nil
}

func readBootstrap(path string) ([]byte, error) {
	if strings.TrimSpace(path) == "" || !filepath.IsAbs(path) || filepath.Clean(path) != path {
		return nil, fmt.Errorf("%w: local-auth path must be clean and absolute", ErrInvalidBootstrap)
	}
	info, err := os.Lstat(path)
	if err != nil || info.Mode()&os.ModeSymlink != 0 || !info.Mode().IsRegular() || info.Mode().Perm()&0o077 != 0 {
		return nil, fmt.Errorf("%w: local-auth file must be regular, non-symlinked, and owner-only", ErrInvalidBootstrap)
	}
	fd, err := unix.Open(path, unix.O_RDONLY|unix.O_NOFOLLOW|unix.O_CLOEXEC, 0)
	if err != nil {
		return nil, ErrInvalidBootstrap
	}
	handle := os.NewFile(uintptr(fd), "local-auth-bootstrap")
	defer handle.Close()
	var stat unix.Stat_t
	if err := unix.Fstat(fd, &stat); err != nil || stat.Mode&unix.S_IFMT != unix.S_IFREG ||
		os.FileMode(stat.Mode).Perm()&0o077 != 0 || stat.Size < 1 || stat.Size > maximumBootstrapBytes {
		return nil, ErrInvalidBootstrap
	}
	data, err := io.ReadAll(io.LimitReader(handle, maximumBootstrapBytes+1))
	if err != nil || len(data) == 0 || len(data) > maximumBootstrapBytes {
		wipe(data)
		return nil, ErrInvalidBootstrap
	}
	return data, nil
}
