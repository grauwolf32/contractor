package credentials

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"regexp"
	"sort"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/apierror"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/credentialerrors"
	"github.com/grauwolf32/contractor/internal/strictjson"
)

const (
	RuntimeCredentialSchemaVersion = "contractor.runtime-credentials/v1"
	MaximumRuntimePlaintextBytes   = 32 * 1024
	MaximumRuntimeSecretBytes      = 8 * 1024
	// MaximumBasicUsernameBytes bounds a Basic credential username
	// (LIM-RCF-23); its password is bounded by MaximumRuntimeSecretBytes.
	MaximumBasicUsernameBytes    = 256
	MaximumOTLPHeaderValueBytes  = 4 * 1024
	MaximumOTLPHeaderValuesBytes = 16 * 1024
	MaximumOTLPHeaders           = 32
	// Supplied cookies fit the existing encrypted plaintext envelope.
	MaximumTargetCookies     = 128
	MaximumTargetCookieBytes = 16 * 1024
)

// RuntimeCredentialKind is a stored Runtime credential kind. Its values are
// runtimesettings.RuntimeCredentialKind's, the one Server definition.
type RuntimeCredentialKind string

const (
	RuntimeCredentialOTLPHeaders     = RuntimeCredentialKind(runtimesettings.RuntimeCredentialOTLPHeaders)
	RuntimeCredentialProxyBasic      = RuntimeCredentialKind(runtimesettings.RuntimeCredentialProxyBasic)
	RuntimeCredentialProxyBearer     = RuntimeCredentialKind(runtimesettings.RuntimeCredentialProxyBearer)
	RuntimeCredentialCaidoBearer     = RuntimeCredentialKind(runtimesettings.RuntimeCredentialCaidoBearer)
	RuntimeCredentialInteractshToken = RuntimeCredentialKind(runtimesettings.RuntimeCredentialInteractshToken)
	RuntimeCredentialOriginBasic     = RuntimeCredentialKind(runtimesettings.RuntimeCredentialOriginBasic)
	RuntimeCredentialOriginBearer    = RuntimeCredentialKind(runtimesettings.RuntimeCredentialOriginBearer)
	RuntimeCredentialTargetBasic     = RuntimeCredentialKind(runtimesettings.RuntimeCredentialTargetBasic)
	RuntimeCredentialTargetBearer    = RuntimeCredentialKind(runtimesettings.RuntimeCredentialTargetBearer)
	RuntimeCredentialTargetCookies   = RuntimeCredentialKind(runtimesettings.RuntimeCredentialTargetCookies)
)

var (
	ErrRuntimeCredentialInvalid  = credentialerrors.RuntimeInvalid
	ErrRuntimeCredentialConflict = apierror.New(apierror.RuntimeConfigConflict, "Runtime adapter credential conflict",
		apierror.WithMessage("Runtime credential identity or idempotency key conflicts"))
	ErrRuntimeCredentialNotFound = credentialerrors.RuntimeNotFound
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
		if !validFieldValue(value, MaximumOTLPHeaderValueBytes) {
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
	if !validBoundedSecret(username, MaximumBasicUsernameBytes) || !validBoundedSecret(password, MaximumRuntimeSecretBytes) || strings.Contains(username, ":") {
		return RuntimeCredentialMaterial{}, runtimeInvalid("HTTP proxy basic credential is invalid")
	}
	return newRuntimeCredentialMaterial(RuntimeCredentialProxyBasic, struct {
		Password string `json:"password"`
		Username string `json:"username"`
	}{Password: password, Username: username})
}

func NewHTTPProxyBearerCredential(token string) (RuntimeCredentialMaterial, error) {
	if !validFieldValue(token, MaximumRuntimeSecretBytes) {
		return RuntimeCredentialMaterial{}, runtimeInvalid("HTTP proxy bearer credential is invalid")
	}
	return newRuntimeCredentialMaterial(RuntimeCredentialProxyBearer, struct {
		Token string `json:"token"`
	}{Token: token})
}

func NewCaidoBearerCredential(token string) (RuntimeCredentialMaterial, error) {
	if !validFieldValue(token, MaximumRuntimeSecretBytes) {
		return RuntimeCredentialMaterial{}, runtimeInvalid("Caido bearer credential is invalid")
	}
	return newRuntimeCredentialMaterial(RuntimeCredentialCaidoBearer, struct {
		Token string `json:"token"`
	}{Token: token})
}

func NewInteractshTokenCredential(token string) (RuntimeCredentialMaterial, error) {
	if !validFieldValue(token, MaximumRuntimeSecretBytes) {
		return RuntimeCredentialMaterial{}, runtimeInvalid("Interactsh bearer credential is invalid")
	}
	return newRuntimeCredentialMaterial(RuntimeCredentialInteractshToken, struct {
		Token string `json:"token"`
	}{Token: token})
}

func NewHTTPOriginBasicCredential(username, password string) (RuntimeCredentialMaterial, error) {
	if !validBoundedSecret(username, MaximumBasicUsernameBytes) || !validBoundedSecret(password, MaximumRuntimeSecretBytes) || strings.Contains(username, ":") {
		return RuntimeCredentialMaterial{}, runtimeInvalid("HTTP origin basic credential is invalid")
	}
	return newRuntimeCredentialMaterial(RuntimeCredentialOriginBasic, struct {
		Password string `json:"password"`
		Username string `json:"username"`
	}{Password: password, Username: username})
}

func NewHTTPOriginBearerCredential(token string) (RuntimeCredentialMaterial, error) {
	if !validFieldValue(token, MaximumRuntimeSecretBytes) {
		return RuntimeCredentialMaterial{}, runtimeInvalid("HTTP origin bearer credential is invalid")
	}
	return newRuntimeCredentialMaterial(RuntimeCredentialOriginBearer, struct {
		Token string `json:"token"`
	}{Token: token})
}

// NewTargetBasicCredential is a pentest target identity's Basic pair. It
// keeps the HTTP origin Basic bounds (RCF-R114).
func NewTargetBasicCredential(username, password string) (RuntimeCredentialMaterial, error) {
	if !validBoundedSecret(username, MaximumBasicUsernameBytes) || !validBoundedSecret(password, MaximumRuntimeSecretBytes) || strings.Contains(username, ":") {
		return RuntimeCredentialMaterial{}, runtimeInvalid("Target basic credential is invalid")
	}
	return newRuntimeCredentialMaterial(RuntimeCredentialTargetBasic, struct {
		Password string `json:"password"`
		Username string `json:"username"`
	}{Password: password, Username: username})
}

// NewTargetBearerCredential is a pentest target identity's bearer token. It
// keeps the bearer token bound and field-value grammar (RCF-R114).
func NewTargetBearerCredential(token string) (RuntimeCredentialMaterial, error) {
	if !validFieldValue(token, MaximumRuntimeSecretBytes) {
		return RuntimeCredentialMaterial{}, runtimeInvalid("Target bearer credential is invalid")
	}
	return newRuntimeCredentialMaterial(RuntimeCredentialTargetBearer, struct {
		Token string `json:"token"`
	}{Token: token})
}

// NewTargetCookiesCredential stores ready user-supplied session cookies.
// Names are HTTP tokens; values cannot contain cookie separators or framing.
func NewTargetCookiesCredential(cookies map[string]string) (RuntimeCredentialMaterial, error) {
	if len(cookies) < 1 || len(cookies) > MaximumTargetCookies {
		return RuntimeCredentialMaterial{}, runtimeInvalid("Target cookie count is invalid")
	}
	totalValueBytes := 0
	for name, value := range cookies {
		if !contracts.ValidCookieName(name) {
			return RuntimeCredentialMaterial{}, runtimeInvalid("Target cookie name is invalid")
		}
		if !contracts.ValidCookieValue(value) || len(value) > MaximumRuntimeSecretBytes {
			return RuntimeCredentialMaterial{}, runtimeInvalid("Target cookie value is invalid")
		}
		totalValueBytes += len(name) + len(value)
		if totalValueBytes > MaximumTargetCookieBytes {
			return RuntimeCredentialMaterial{}, runtimeInvalid("Target cookie values exceed their total bound")
		}
	}
	return newRuntimeCredentialMaterial(RuntimeCredentialTargetCookies, struct {
		Cookies map[string]string `json:"cookies"`
	}{Cookies: cookies})
}

func (m RuntimeCredentialMaterial) Kind() RuntimeCredentialKind { return m.kind }

// RuntimeCredentialMaterial formats and logs only as a redaction marker, for
// every fmt verb and flag, and refuses JSON encoding (SEC-R88).
func (m RuntimeCredentialMaterial) String() string { return "[REDACTED]" }

func (m RuntimeCredentialMaterial) GoString() string {
	return "credentials.RuntimeCredentialMaterial([REDACTED])"
}

func (m RuntimeCredentialMaterial) Format(state fmt.State, verb rune) {
	formatRedacted(state, verb, m.String(), m.GoString())
}

func (m RuntimeCredentialMaterial) LogValue() slog.Value { return slog.StringValue(m.String()) }

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
	clear(m.canonical)
	m.canonical = nil
	m.kind = ""
}

type RuntimeCredentialMetadata struct {
	CredentialID string                `json:"credentialId"`
	Kind         RuntimeCredentialKind `json:"kind"`
	CreatedBy    string                `json:"createdBy"`
	CreatedAt    time.Time             `json:"createdAt"`
}

// RuntimeCredentialUser is the authenticated principal on whose behalf
// owner-scoped configuration, such as a Project HTTP target, references a
// Runtime credential. Runtime credentials are global rows; only this check
// keeps one owner from pointing its target at another owner's secret.
type RuntimeCredentialUser struct {
	UserID string
	// Operations is the Operations capability. It administers every Runtime
	// credential and may therefore reference any of them.
	Operations bool
}

// MayUse reports whether the principal may reference the credential: it
// created the credential, or it holds the Operations capability.
func (u RuntimeCredentialUser) MayUse(metadata RuntimeCredentialMetadata) bool {
	return u.Operations || u.UserID != "" && metadata.CreatedBy == u.UserID
}

// RuntimeCredentialRecord formats and logs as its identity and kind with the
// envelope elided, for every fmt verb and flag, and encodes only its metadata
// (SEC-R88).
type RuntimeCredentialRecord struct {
	Metadata RuntimeCredentialMetadata `json:"metadata"`
	Envelope EncryptedEnvelope         `json:"-"`
}

func (r RuntimeCredentialRecord) String() string {
	return fmt.Sprintf("RuntimeCredentialRecord{CredentialID:%q Kind:%q [ENCRYPTED]}", r.Metadata.CredentialID, r.Metadata.Kind)
}

func (r RuntimeCredentialRecord) GoString() string { return r.String() }

func (r RuntimeCredentialRecord) Format(state fmt.State, verb rune) {
	formatRedacted(state, verb, r.String(), r.GoString())
}

func (r RuntimeCredentialRecord) LogValue() slog.Value { return slog.StringValue(r.String()) }

func (r RuntimeCredentialRecord) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Metadata RuntimeCredentialMetadata `json:"metadata"`
	}{Metadata: r.Metadata})
}

// RuntimeCredentialCreation formats and logs only as a redaction marker, for
// every fmt verb and flag, and refuses JSON encoding (SEC-R88): its request
// MAC and idempotency digest identify the secret request.
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
func (c RuntimeCredentialCreation) Format(state fmt.State, verb rune) {
	formatRedacted(state, verb, c.String(), c.GoString())
}
func (c RuntimeCredentialCreation) LogValue() slog.Value { return slog.StringValue(c.String()) }
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

// ErrorKind answers the error with runtime_credential_in_use; its usage is
// the details.
func (e *RuntimeCredentialInUseError) ErrorKind() apierror.Kind {
	return apierror.RuntimeCredentialInUse
}

// RuntimeCredentialUsageChecker reports the RuntimeConfig bindings, Projects,
// Runs and allocations that reference a Runtime credential: every field of
// RuntimeCredentialUsage except AuditIDs, which RuntimeCredentialAuditHoldLookup
// reports.
type RuntimeCredentialUsageChecker interface {
	InspectRuntimeCredentialUsage(context.Context, string, int) (RuntimeCredentialUsage, error)
}

// RuntimeCredentialAuditHoldLookup reports, up to a limit, the Audits whose
// dispatch hold keeps a Runtime credential for future child Runs (AUD-R16);
// auditstore, which owns the Audits, implements it.
type RuntimeCredentialAuditHoldLookup interface {
	ListHeldAuditIDsByRuntimeCredential(context.Context, string, int) ([]string, error)
}

func newRuntimeCredentialMaterial(kind RuntimeCredentialKind, value any) (RuntimeCredentialMaterial, error) {
	canonical, err := json.Marshal(value)
	if err != nil || len(canonical) == 0 || len(canonical) > MaximumRuntimePlaintextBytes {
		return RuntimeCredentialMaterial{}, runtimeInvalid("Runtime credential plaintext is invalid")
	}
	return RuntimeCredentialMaterial{kind: kind, canonical: canonical}, nil
}

// DecodeRuntimeCredentialMaterial strictly decodes the creation material of
// kind, the JSON object a public create request carries, and builds it with
// the kind's constructor. Duplicate keys, unknown or case-aliased members,
// lone surrogate escapes, trailing data, an unknown kind and any value outside
// the kind's rules are invalid; the error names no input. encoding/json
// would turn invalid UTF-8 and a lone surrogate escape into U+FFFD, so both
// are refused before decoding.
func DecodeRuntimeCredentialMaterial(kind RuntimeCredentialKind, data []byte) (RuntimeCredentialMaterial, error) {
	if len(data) == 0 || len(data) > MaximumRuntimePlaintextBytes || !utf8.Valid(data) ||
		!strictjson.ValidUnicodeEscapes(data) {
		return RuntimeCredentialMaterial{}, runtimeInvalid("Runtime credential material is invalid")
	}
	material, err := decodeRuntimeCredentialMaterial(kind, data)
	if err != nil {
		return RuntimeCredentialMaterial{}, runtimeInvalid("Runtime credential material is invalid")
	}
	return material, nil
}

func decodeRuntimeCredentialMaterial(kind RuntimeCredentialKind, data []byte) (RuntimeCredentialMaterial, error) {
	switch kind {
	case RuntimeCredentialOTLPHeaders:
		var source struct {
			Headers map[string]string `json:"headers"`
		}
		if err := strictjson.Decode(data, &source); err != nil {
			return RuntimeCredentialMaterial{}, err
		}
		return NewOTLPHeadersCredential(source.Headers)
	case RuntimeCredentialProxyBasic, RuntimeCredentialOriginBasic, RuntimeCredentialTargetBasic:
		var source struct {
			Password string `json:"password"`
			Username string `json:"username"`
		}
		if err := strictjson.Decode(data, &source); err != nil {
			return RuntimeCredentialMaterial{}, err
		}
		switch kind {
		case RuntimeCredentialProxyBasic:
			return NewHTTPProxyBasicCredential(source.Username, source.Password)
		case RuntimeCredentialOriginBasic:
			return NewHTTPOriginBasicCredential(source.Username, source.Password)
		default:
			return NewTargetBasicCredential(source.Username, source.Password)
		}
	case RuntimeCredentialProxyBearer, RuntimeCredentialCaidoBearer, RuntimeCredentialInteractshToken,
		RuntimeCredentialOriginBearer, RuntimeCredentialTargetBearer:
		var source struct {
			Token string `json:"token"`
		}
		if err := strictjson.Decode(data, &source); err != nil {
			return RuntimeCredentialMaterial{}, err
		}
		switch kind {
		case RuntimeCredentialProxyBearer:
			return NewHTTPProxyBearerCredential(source.Token)
		case RuntimeCredentialCaidoBearer:
			return NewCaidoBearerCredential(source.Token)
		case RuntimeCredentialInteractshToken:
			return NewInteractshTokenCredential(source.Token)
		case RuntimeCredentialOriginBearer:
			return NewHTTPOriginBearerCredential(source.Token)
		default:
			return NewTargetBearerCredential(source.Token)
		}
	case RuntimeCredentialTargetCookies:
		var source struct {
			Cookies map[string]string `json:"cookies"`
		}
		if err := strictjson.Decode(data, &source); err != nil {
			return RuntimeCredentialMaterial{}, err
		}
		material, err := NewTargetCookiesCredential(source.Cookies)
		clear(source.Cookies)
		return material, err
	default:
		return RuntimeCredentialMaterial{}, runtimeInvalid("Runtime credential kind is invalid")
	}
}

// runtimeCredentialMaterialFromCanonical decodes decrypted plaintext: it must
// be valid material of kind and exactly the canonical JSON its constructor
// produces, or the record is refused as a decryption failure (FAIL-RCF-3).
func runtimeCredentialMaterialFromCanonical(kind RuntimeCredentialKind, canonical []byte) (RuntimeCredentialMaterial, error) {
	if len(canonical) == 0 || len(canonical) > MaximumRuntimePlaintextBytes {
		return RuntimeCredentialMaterial{}, ErrCrypto
	}
	material, err := decodeRuntimeCredentialMaterial(kind, canonical)
	if err != nil {
		return RuntimeCredentialMaterial{}, ErrCrypto
	}
	if len(material.canonical) == 0 || !bytes.Equal(material.canonical, canonical) {
		material.Destroy()
		return RuntimeCredentialMaterial{}, ErrCrypto
	}
	return material, nil
}

func validRuntimeCredentialKind(kind RuntimeCredentialKind) bool {
	return runtimesettings.RuntimeCredentialKind(kind).Validate() == nil
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

// validFieldValue reports whether a bearer token or OTLP header value is
// within its byte bound and is sent unchanged as an HTTP field value. Basic
// credentials are base64-encoded before sending and keep validBoundedSecret.
func validFieldValue(value string, maximum int) bool {
	return len(value) <= maximum && contracts.ValidCredentialFieldValue(value)
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
