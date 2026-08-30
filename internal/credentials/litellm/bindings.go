// Package litellm implements Contractor's digest-pinned LiteLLM virtual-key
// management boundary. It never exposes or serializes Gateway admin keys.
package litellm

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/credentials"
	"go.yaml.in/yaml/v4"
	"golang.org/x/sys/unix"
)

const (
	maximumBindingsFileBytes = 1 << 20
	maximumAdminBindings     = 256
)

type GatewayLookup interface {
	LLMGateway(string) (contracts.ResolvedLLMGatewayConfig, error)
}

type adminKey struct{ value string }

func (k adminKey) reveal() string   { return k.value }
func (k adminKey) String() string   { return "[REDACTED]" }
func (k adminKey) GoString() string { return "litellm.adminKey([REDACTED])" }

type exactBinding struct {
	managementURL string
	key           adminKey
}

// AdminBindings is an immutable in-memory map loaded once during Server
// startup. The exact Gateway ref is resolved while loading so a forged
// same-ref request cannot redirect an admin key to another origin.
type AdminBindings struct {
	bindings map[contracts.LLMGatewayConfigRef]exactBinding
}

func (b *AdminBindings) Len() int {
	if b == nil {
		return 0
	}
	return len(b.bindings)
}

func (b *AdminBindings) String() string   { return "litellm.AdminBindings([REDACTED])" }
func (b *AdminBindings) GoString() string { return "litellm.AdminBindings([REDACTED])" }

type bindingDocument struct {
	Bindings []bindingSource `yaml:"bindings"`
}

type bindingSource struct {
	LLMGateway   gatewayRefSource `yaml:"llmGateway"`
	AdminKeyFile string           `yaml:"adminKeyFile"`
}

type gatewayRefSource struct {
	GatewayID string `yaml:"gatewayId"`
	Version   string `yaml:"version"`
	Digest    string `yaml:"digest"`
}

// LoadAdminBindings loads a non-secret bootstrap document and every referenced
// owner-only admin-key file. An empty path deliberately configures no managed
// Gateways; it is not an implicit default binding.
func LoadAdminBindings(path string, gateways GatewayLookup) (*AdminBindings, error) {
	result := &AdminBindings{bindings: make(map[contracts.LLMGatewayConfigRef]exactBinding)}
	if path == "" {
		return result, nil
	}
	if gateways == nil {
		return nil, fmt.Errorf("%w: Gateway binding resolver is unavailable", credentials.ErrManagerUnavailable)
	}
	documentBytes, err := readSecureFile(path, maximumBindingsFileBytes, 0o022, "admin bindings")
	if err != nil {
		return nil, err
	}
	defer wipe(documentBytes)
	var documents []bindingDocument
	if err := yaml.Load(
		documentBytes,
		&documents,
		yaml.WithAllDocuments(),
		yaml.WithKnownFields(),
		yaml.WithUniqueKeys(),
	); err != nil {
		return nil, fmt.Errorf("%w: decode strict Gateway admin bindings", credentials.ErrManagerUnavailable)
	}
	if len(documents) != 1 || len(documents[0].Bindings) == 0 ||
		len(documents[0].Bindings) > maximumAdminBindings {
		return nil, fmt.Errorf("%w: Gateway admin bindings must contain 1 through %d entries", credentials.ErrManagerUnavailable, maximumAdminBindings)
	}
	for _, source := range documents[0].Bindings {
		ref := contracts.LLMGatewayConfigRef{
			GatewayID: source.LLMGateway.GatewayID,
			Version:   source.LLMGateway.Version,
			Digest:    source.LLMGateway.Digest,
		}
		if err := ref.ValidateRef(); err != nil {
			return nil, fmt.Errorf("%w: Gateway admin binding has an invalid exact ref", credentials.ErrManagerUnavailable)
		}
		if _, duplicate := result.bindings[ref]; duplicate {
			return nil, fmt.Errorf("%w: duplicate exact Gateway admin binding", credentials.ErrManagerUnavailable)
		}
		gateway, err := gateways.LLMGateway(ref.GatewayID + "@" + ref.Version)
		if err != nil || gateway.Ref != ref || gateway.Validate() != nil || gateway.CredentialManager == nil ||
			gateway.CredentialManager.Implementation != contracts.LiteLLMVirtualKeysManager {
			return nil, fmt.Errorf("%w: Gateway admin binding does not resolve to its exact managed Gateway", credentials.ErrManagerUnavailable)
		}
		key, err := loadAdminKey(source.AdminKeyFile)
		if err != nil {
			return nil, err
		}
		result.bindings[ref] = exactBinding{
			managementURL: gateway.CredentialManager.ManagementURL,
			key:           key,
		}
	}
	return result, nil
}

func (b *AdminBindings) bindingFor(gateway contracts.ResolvedLLMGatewayConfig) (exactBinding, error) {
	if b == nil || gateway.Validate() != nil || gateway.CredentialManager == nil ||
		gateway.CredentialManager.Implementation != contracts.LiteLLMVirtualKeysManager {
		return exactBinding{}, credentials.ErrManagerUnavailable
	}
	binding, exists := b.bindings[gateway.Ref]
	if !exists || binding.managementURL != gateway.CredentialManager.ManagementURL || binding.key.reveal() == "" {
		return exactBinding{}, credentials.ErrManagerUnavailable
	}
	return binding, nil
}

func loadAdminKey(path string) (adminKey, error) {
	data, err := readSecureFile(path, credentials.MaximumTokenBytes+1, 0o077, "Gateway admin key")
	if err != nil {
		return adminKey{}, err
	}
	defer wipe(data)
	if len(data) > 0 && data[len(data)-1] == '\n' {
		data = data[:len(data)-1]
	}
	if len(data) < 3 || len(data) > credentials.MaximumTokenBytes || !utf8.Valid(data) ||
		!bytes.HasPrefix(data, []byte("sk-")) || bytes.IndexByte(data, '\r') >= 0 ||
		bytes.IndexByte(data, '\n') >= 0 {
		return adminKey{}, fmt.Errorf("%w: Gateway admin key is malformed", credentials.ErrManagerUnavailable)
	}
	return adminKey{value: string(data)}, nil
}

func readSecureFile(path string, maximumBytes int64, forbiddenPermissions os.FileMode, kind string) ([]byte, error) {
	if strings.TrimSpace(path) == "" || !filepath.IsAbs(path) || filepath.Clean(path) != path {
		return nil, fmt.Errorf("%w: %s file path must be clean and absolute", credentials.ErrManagerUnavailable, kind)
	}
	info, err := os.Lstat(path)
	if err != nil || info.Mode()&os.ModeSymlink != 0 || !info.Mode().IsRegular() ||
		info.Mode().Perm()&forbiddenPermissions != 0 {
		return nil, fmt.Errorf("%w: %s file permissions or type are unsafe", credentials.ErrManagerUnavailable, kind)
	}
	fd, err := unix.Open(path, unix.O_RDONLY|unix.O_NOFOLLOW|unix.O_CLOEXEC, 0)
	if err != nil {
		return nil, fmt.Errorf("%w: open %s file", credentials.ErrManagerUnavailable, kind)
	}
	handle := os.NewFile(uintptr(fd), kind)
	defer handle.Close()
	var stat unix.Stat_t
	if err := unix.Fstat(fd, &stat); err != nil || stat.Mode&unix.S_IFMT != unix.S_IFREG ||
		os.FileMode(stat.Mode).Perm()&forbiddenPermissions != 0 || stat.Size < 1 || stat.Size > maximumBytes {
		return nil, fmt.Errorf("%w: %s file is unsafe or outside its size bound", credentials.ErrManagerUnavailable, kind)
	}
	data, err := io.ReadAll(io.LimitReader(handle, maximumBytes+1))
	if err != nil || len(data) == 0 || int64(len(data)) > maximumBytes {
		wipe(data)
		return nil, fmt.Errorf("%w: read bounded %s file", credentials.ErrManagerUnavailable, kind)
	}
	return data, nil
}

func wipe(value []byte) {
	for index := range value {
		value[index] = 0
	}
}
