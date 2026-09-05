package credentials

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"time"
	"unicode/utf8"
)

const (
	RuntimeCredentialSchemaVersion = "contractor.runtime-credentials/v1"
	MaximumRuntimePlaintextBytes   = 32 * 1024
	MaximumRuntimeSecretBytes      = 8 * 1024
	MaximumOTLPHeaderValueBytes    = 4 * 1024
	MaximumOTLPHeaderValuesBytes   = 16 * 1024
	MaximumOTLPHeaders             = 32
)

type RuntimeCredentialKind string

const (
	RuntimeCredentialOTLPHeaders  RuntimeCredentialKind = "otlp-headers@1"
	RuntimeCredentialProxyBasic   RuntimeCredentialKind = "http-proxy-basic@1"
	RuntimeCredentialProxyBearer  RuntimeCredentialKind = "http-proxy-bearer@1"
	RuntimeCredentialCaidoBearer  RuntimeCredentialKind = "caido-bearer@1"
	RuntimeCredentialOriginBasic  RuntimeCredentialKind = "http-origin-basic@1"
	RuntimeCredentialOriginBearer RuntimeCredentialKind = "http-origin-bearer@1"
)

var (
	ErrRuntimeCredentialInvalid  = errors.New("invalid Runtime adapter credential")
	ErrRuntimeCredentialConflict = errors.New("Runtime adapter credential conflict")
	ErrRuntimeCredentialNotFound = errors.New("Runtime adapter credential not found")
	ErrRuntimeCredentialInUse    = errors.New("Runtime adapter credential is in use")

	headerNamePattern    = regexp.MustCompile("^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
	forbiddenOTLPHeaders = map[string]struct{}{
		"connection": {}, "content-length": {}, "forwarded": {}, "host": {},
		"keep-alive": {}, "proxy-authenticate": {}, "proxy-authorization": {},
		"te": {}, "trailer": {}, "transfer-encoding": {}, "upgrade": {},
		"via": {}, "x-real-ip": {},
	}
)

type RuntimeCredentialMaterial struct {
	kind      RuntimeCredentialKind
	canonical []byte
}

func NewOTLPHeadersCredential(headers map[string]string) (RuntimeCredentialMaterial, error) {
	if len(headers) < 1 || len(headers) > MaximumOTLPHeaders {
		return RuntimeCredentialMaterial{}, runtimeInvalid("OTLP header count is invalid")
	}
	normalized := make(map[string]string, len(headers))
	totalValueBytes := 0
	for name, value := range headers {
		lowerName := strings.ToLower(name)
		if len(name) == 0 || len(name) > 64 || !isASCII(name) || !headerNamePattern.MatchString(name) ||
			forbiddenOTLPHeader(lowerName) {
			return RuntimeCredentialMaterial{}, runtimeInvalid("OTLP header name is invalid")
		}
		if _, exists := normalized[lowerName]; exists {
			return RuntimeCredentialMaterial{}, runtimeInvalid("OTLP headers contain a case-insensitive duplicate")
		}
		if len(value) < 1 || len(value) > MaximumOTLPHeaderValueBytes || !utf8.ValidString(value) || !validHeaderValue(value) {
			return RuntimeCredentialMaterial{}, runtimeInvalid("OTLP header value is invalid")
		}
		totalValueBytes += len(value)
		if totalValueBytes > MaximumOTLPHeaderValuesBytes {
			return RuntimeCredentialMaterial{}, runtimeInvalid("OTLP header values exceed their total bound")
		}
		normalized[lowerName] = value
	}
	return newRuntimeCredentialMaterial(RuntimeCredentialOTLPHeaders, struct {
		Headers map[string]string `json:"headers"`
	}{Headers: normalized})
}

func NewHTTPProxyBasicCredential(username, password string) (RuntimeCredentialMaterial, error) {
	if !validBoundedSecret(username, 256) || !validBoundedSecret(password, MaximumRuntimeSecretBytes) || strings.Contains(username, ":") {
		return RuntimeCredentialMaterial{}, runtimeInvalid("HTTP proxy basic credential is invalid")
	}
	return newRuntimeCredentialMaterial(RuntimeCredentialProxyBasic, struct {
		Password string `json:"password"`
		Username string `json:"username"`
	}{Password: password, Username: username})
}

func NewHTTPProxyBearerCredential(token string) (RuntimeCredentialMaterial, error) {
	if !validBoundedSecret(token, MaximumRuntimeSecretBytes) {
		return RuntimeCredentialMaterial{}, runtimeInvalid("HTTP proxy bearer credential is invalid")
	}
	return newRuntimeCredentialMaterial(RuntimeCredentialProxyBearer, struct {
		Token string `json:"token"`
	}{Token: token})
}

func NewCaidoBearerCredential(token string) (RuntimeCredentialMaterial, error) {
	if !validBoundedSecret(token, MaximumRuntimeSecretBytes) {
		return RuntimeCredentialMaterial{}, runtimeInvalid("Caido bearer credential is invalid")
	}
	return newRuntimeCredentialMaterial(RuntimeCredentialCaidoBearer, struct {
		Token string `json:"token"`
	}{Token: token})
}

func NewHTTPOriginBasicCredential(username, password string) (RuntimeCredentialMaterial, error) {
	if !validBoundedSecret(username, 256) || !validBoundedSecret(password, MaximumRuntimeSecretBytes) || strings.Contains(username, ":") {
		return RuntimeCredentialMaterial{}, runtimeInvalid("HTTP origin basic credential is invalid")
	}
	return newRuntimeCredentialMaterial(RuntimeCredentialOriginBasic, struct {
		Password string `json:"password"`
		Username string `json:"username"`
	}{Password: password, Username: username})
}

func NewHTTPOriginBearerCredential(token string) (RuntimeCredentialMaterial, error) {
	if !validBoundedSecret(token, MaximumRuntimeSecretBytes) {
		return RuntimeCredentialMaterial{}, runtimeInvalid("HTTP origin bearer credential is invalid")
	}
	return newRuntimeCredentialMaterial(RuntimeCredentialOriginBearer, struct {
		Token string `json:"token"`
	}{Token: token})
}

func (m RuntimeCredentialMaterial) Kind() RuntimeCredentialKind { return m.kind }

func (m RuntimeCredentialMaterial) String() string { return "[REDACTED]" }

func (m RuntimeCredentialMaterial) GoString() string {
	return "credentials.RuntimeCredentialMaterial([REDACTED])"
}

func (m RuntimeCredentialMaterial) MarshalJSON() ([]byte, error) {
	return nil, errors.New("Runtime credential material cannot be serialized")
}

// WithPlaintext exposes the canonical secret only for the synchronous
// consumer callback. The slice must not be retained or modified.
func (m *RuntimeCredentialMaterial) WithPlaintext(fn func(RuntimeCredentialKind, []byte) error) error {
	if m == nil || fn == nil || !validRuntimeCredentialKind(m.kind) || len(m.canonical) == 0 {
		return ErrRuntimeCredentialInvalid
	}
	return fn(m.kind, m.canonical)
}

func (m *RuntimeCredentialMaterial) Destroy() {
	if m == nil {
		return
	}
	wipeBytes(m.canonical)
	m.canonical = nil
	m.kind = ""
}

type RuntimeCredentialMetadata struct {
	CredentialID string                `json:"credentialId"`
	Kind         RuntimeCredentialKind `json:"kind"`
	CreatedBy    string                `json:"createdBy"`
	CreatedAt    time.Time             `json:"createdAt"`
}

type RuntimeCredentialRecord struct {
	Metadata RuntimeCredentialMetadata `json:"metadata"`
	Envelope EncryptedEnvelope         `json:"-"`
}

func (r RuntimeCredentialRecord) String() string {
	return fmt.Sprintf("RuntimeCredentialRecord{CredentialID:%q Kind:%q [ENCRYPTED]}", r.Metadata.CredentialID, r.Metadata.Kind)
}

func (r RuntimeCredentialRecord) GoString() string { return r.String() }

type RuntimeCredentialCreation struct {
	IdempotencyKeyDigest string                `json:"-"`
	RequestMAC           []byte                `json:"-"`
	CredentialID         string                `json:"-"`
	Kind                 RuntimeCredentialKind `json:"-"`
	ActorID              string                `json:"-"`
	CreatedAt            time.Time             `json:"-"`
}

func (c RuntimeCredentialCreation) String() string   { return "RuntimeCredentialCreation([REDACTED])" }
func (c RuntimeCredentialCreation) GoString() string { return c.String() }
func (c RuntimeCredentialCreation) MarshalJSON() ([]byte, error) {
	return nil, errors.New("Runtime credential creation replay cannot be serialized")
}

type RuntimeCredentialTombstone struct {
	CredentialID string
	ActorID      string
	DeletedAt    time.Time
}

type RuntimeCredentialCreateRequest struct {
	CredentialID   string
	Material       RuntimeCredentialMaterial
	IdempotencyKey string
	ActorID        string
}

type RuntimeCredentialCreateResult struct {
	Credential RuntimeCredentialMetadata
	Replayed   bool
}

type RuntimeCredentialDeleteResult struct {
	Replayed bool
}

type RuntimeCredentialUsage struct {
	BindingLabels []string
	ProjectIDs    []string
	RunIDs        []string
	AuditIDs      []string
	AllocationIDs []string
}

func (u RuntimeCredentialUsage) Empty() bool {
	return len(u.BindingLabels) == 0 && len(u.ProjectIDs) == 0 && len(u.RunIDs) == 0 &&
		len(u.AuditIDs) == 0 && len(u.AllocationIDs) == 0
}

type RuntimeCredentialInUseError struct{ Usage RuntimeCredentialUsage }

func (e *RuntimeCredentialInUseError) Error() string { return ErrRuntimeCredentialInUse.Error() }
func (e *RuntimeCredentialInUseError) Unwrap() error { return ErrRuntimeCredentialInUse }

type RuntimeCredentialUsageChecker interface {
	InspectRuntimeCredentialUsage(context.Context, string, int) (RuntimeCredentialUsage, error)
}

func newRuntimeCredentialMaterial(kind RuntimeCredentialKind, value any) (RuntimeCredentialMaterial, error) {
	canonical, err := json.Marshal(value)
	if err != nil || len(canonical) == 0 || len(canonical) > MaximumRuntimePlaintextBytes {
		return RuntimeCredentialMaterial{}, runtimeInvalid("Runtime credential plaintext is invalid")
	}
	return RuntimeCredentialMaterial{kind: kind, canonical: canonical}, nil
}

func runtimeCredentialMaterialFromCanonical(kind RuntimeCredentialKind, canonical []byte) (RuntimeCredentialMaterial, error) {
	if len(canonical) == 0 || len(canonical) > MaximumRuntimePlaintextBytes {
		return RuntimeCredentialMaterial{}, ErrCrypto
	}
	var material RuntimeCredentialMaterial
	switch kind {
	case RuntimeCredentialOTLPHeaders:
		var source struct {
			Headers map[string]string `json:"headers"`
		}
		if decodeStrictJSON(canonical, &source) != nil {
			return RuntimeCredentialMaterial{}, ErrCrypto
		}
		material, _ = NewOTLPHeadersCredential(source.Headers)
	case RuntimeCredentialProxyBasic:
		var source struct {
			Password string `json:"password"`
			Username string `json:"username"`
		}
		if decodeStrictJSON(canonical, &source) != nil {
			return RuntimeCredentialMaterial{}, ErrCrypto
		}
		material, _ = NewHTTPProxyBasicCredential(source.Username, source.Password)
	case RuntimeCredentialProxyBearer:
		var source struct {
			Token string `json:"token"`
		}
		if decodeStrictJSON(canonical, &source) != nil {
			return RuntimeCredentialMaterial{}, ErrCrypto
		}
		material, _ = NewHTTPProxyBearerCredential(source.Token)
	case RuntimeCredentialCaidoBearer:
		var source struct {
			Token string `json:"token"`
		}
		if decodeStrictJSON(canonical, &source) != nil {
			return RuntimeCredentialMaterial{}, ErrCrypto
		}
		material, _ = NewCaidoBearerCredential(source.Token)
	case RuntimeCredentialOriginBasic:
		var source struct {
			Password string `json:"password"`
			Username string `json:"username"`
		}
		if decodeStrictJSON(canonical, &source) != nil {
			return RuntimeCredentialMaterial{}, ErrCrypto
		}
		material, _ = NewHTTPOriginBasicCredential(source.Username, source.Password)
	case RuntimeCredentialOriginBearer:
		var source struct {
			Token string `json:"token"`
		}
		if decodeStrictJSON(canonical, &source) != nil {
			return RuntimeCredentialMaterial{}, ErrCrypto
		}
		material, _ = NewHTTPOriginBearerCredential(source.Token)
	default:
		return RuntimeCredentialMaterial{}, ErrCrypto
	}
	if len(material.canonical) == 0 || !bytes.Equal(material.canonical, canonical) {
		material.Destroy()
		return RuntimeCredentialMaterial{}, ErrCrypto
	}
	return material, nil
}

func validRuntimeCredentialKind(kind RuntimeCredentialKind) bool {
	return kind == RuntimeCredentialOTLPHeaders || kind == RuntimeCredentialProxyBasic ||
		kind == RuntimeCredentialProxyBearer || kind == RuntimeCredentialCaidoBearer ||
		kind == RuntimeCredentialOriginBasic || kind == RuntimeCredentialOriginBearer
}

func validBoundedSecret(value string, maximum int) bool {
	if len(value) < 1 || len(value) > maximum || !utf8.ValidString(value) {
		return false
	}
	for _, raw := range []byte(value) {
		if raw < 0x20 || raw == 0x7f {
			return false
		}
	}
	return true
}

func validHeaderValue(value string) bool {
	for _, raw := range []byte(value) {
		if raw < 0x20 && raw != '\t' || raw == 0x7f {
			return false
		}
	}
	return true
}

func forbiddenOTLPHeader(name string) bool {
	if _, exists := forbiddenOTLPHeaders[name]; exists {
		return true
	}
	return strings.HasPrefix(name, "x-forwarded-") || strings.HasPrefix(name, "proxy-")
}

func isASCII(value string) bool {
	for _, raw := range []byte(value) {
		if raw > 0x7f {
			return false
		}
	}
	return true
}

func runtimeInvalid(message string) error {
	return fmt.Errorf("%w: %s", ErrRuntimeCredentialInvalid, message)
}

func normalizeRuntimeUsage(value RuntimeCredentialUsage, maximum int) RuntimeCredentialUsage {
	normalize := func(items []string) []string {
		result := append([]string(nil), items...)
		sort.Strings(result)
		if len(result) > maximum {
			result = result[:maximum]
		}
		return result
	}
	value.BindingLabels = normalize(value.BindingLabels)
	value.ProjectIDs = normalize(value.ProjectIDs)
	value.RunIDs = normalize(value.RunIDs)
	value.AuditIDs = normalize(value.AuditIDs)
	value.AllocationIDs = normalize(value.AllocationIDs)
	return value
}
