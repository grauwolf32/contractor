// Package litellm implements Contractor's digest-pinned LiteLLM virtual-key
// management boundary. It never exposes or serializes Gateway admin keys.
package litellm

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/helpers/securefile"
	"go.yaml.in/yaml/v4"
)

const (
	maximumBindingsFileBytes = 1 << 20
	maximumAdminBindings     = 256
)

type GatewayLookup interface {
	LLMGateway(string) (llmgateway.ResolvedLLMGatewayConfig, error)
}

// adminKey formats and logs only as a redaction marker, for every fmt verb
// and flag, and refuses JSON encoding (SEC-R88).
type adminKey struct{ value string }

func (k adminKey) reveal() string   { return k.value }
func (k adminKey) String() string   { return "[REDACTED]" }
func (k adminKey) GoString() string { return "litellm.adminKey([REDACTED])" }
func (k adminKey) Format(state fmt.State, verb rune) {
	if verb == 'v' && state.Flag('#') {
		_, _ = io.WriteString(state, k.GoString())
		return
	}
	_, _ = io.WriteString(state, k.String())
}
func (k adminKey) LogValue() slog.Value { return slog.StringValue(k.String()) }
func (k adminKey) MarshalJSON() ([]byte, error) {
	return nil, errors.New("Gateway admin key cannot be serialized")
}

// exactBinding redacts itself as a whole (SEC-R88): fmt cannot call the
// admin key's own redaction through the unexported key field.
type exactBinding struct {
	managementURL string
	key           adminKey
}

func (b exactBinding) String() string {
	return fmt.Sprintf("litellm.exactBinding{managementURL:%q key:[REDACTED]}", b.managementURL)
}
func (b exactBinding) GoString() string               { return b.String() }
func (b exactBinding) Format(state fmt.State, _ rune) { _, _ = io.WriteString(state, b.String()) }
func (b exactBinding) LogValue() slog.Value           { return slog.StringValue(b.String()) }
func (b exactBinding) MarshalJSON() ([]byte, error)   { return json.Marshal(b.String()) }

// AdminBindings is an immutable in-memory map loaded once during Server
// startup. The exact Gateway ref is resolved while loading so a forged
// same-ref request cannot redirect an admin key to another origin.
type AdminBindings struct {
	bindings map[llmgateway.LLMGatewayConfigRef]exactBinding
}

func (b *AdminBindings) Len() int {
	if b == nil {
		return 0
	}
	return len(b.bindings)
}

// AdminBindings redacts itself as a whole for every fmt verb and flag, for
// log/slog and for JSON (SEC-R88); the value receivers make a copied value
// redact too.
func (b AdminBindings) String() string                 { return "litellm.AdminBindings([REDACTED])" }
func (b AdminBindings) GoString() string               { return b.String() }
func (b AdminBindings) Format(state fmt.State, _ rune) { _, _ = io.WriteString(state, b.String()) }
func (b AdminBindings) LogValue() slog.Value           { return slog.StringValue(b.String()) }
func (b AdminBindings) MarshalJSON() ([]byte, error)   { return json.Marshal(b.String()) }

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
	result := &AdminBindings{bindings: make(map[llmgateway.LLMGatewayConfigRef]exactBinding)}
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
	defer clear(documentBytes)
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
		ref := llmgateway.LLMGatewayConfigRef{
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
			gateway.CredentialManager.Implementation != llmgateway.LiteLLMVirtualKeysManager {
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

func (b *AdminBindings) bindingFor(gateway llmgateway.ResolvedLLMGatewayConfig) (exactBinding, error) {
	if b == nil || gateway.Validate() != nil || gateway.CredentialManager == nil ||
		gateway.CredentialManager.Implementation != llmgateway.LiteLLMVirtualKeysManager {
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
	defer clear(data)
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
	data, err := securefile.ReadRestricted(path, maximumBytes, forbiddenPermissions)
	switch {
	case err == nil:
		return data, nil
	case errors.Is(err, securefile.ErrPath):
		return nil, fmt.Errorf("%w: %s file path must be clean and absolute", credentials.ErrManagerUnavailable, kind)
	case errors.Is(err, securefile.ErrUnsafePath):
		return nil, fmt.Errorf("%w: %s file permissions or type are unsafe", credentials.ErrManagerUnavailable, kind)
	case errors.Is(err, securefile.ErrOpen):
		return nil, fmt.Errorf("%w: open %s file", credentials.ErrManagerUnavailable, kind)
	case errors.Is(err, securefile.ErrUnsafeHandle):
		return nil, fmt.Errorf("%w: %s file is unsafe or outside its size bound", credentials.ErrManagerUnavailable, kind)
	default:
		return nil, fmt.Errorf("%w: read bounded %s file", credentials.ErrManagerUnavailable, kind)
	}
}
