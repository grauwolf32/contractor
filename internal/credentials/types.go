package credentials

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"regexp"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/contracts"
)

const (
	CredentialSchemaVersion = "contractor.credentials/v1"
	MaximumTokenBytes       = 16 * 1024
	MaximumRemoteKeyIDBytes = 512
	MaximumCredentialLabel  = 256
)

var (
	ErrInvalid            = errors.New("invalid LLM credential")
	ErrConflict           = errors.New("LLM credential conflict")
	ErrCrypto             = errors.New("LLM credential cryptographic failure")
	ErrKeyUnavailable     = errors.New("LLM credential master key is unavailable")
	ErrManagerUnavailable = errors.New("Gateway credential manager is unavailable")
	budgetDurationPattern = regexp.MustCompile(`^[1-9][0-9]*(s|m|h|d|mo)$`)
)

// Token is a bounded secret returned by a Gateway credential manager. It is
// deliberately not JSON serializable and formats only as a redaction marker.
// The package converts it to contracts.SecretString only at the final execution
// resolver boundary.
type Token struct{ value string }

func NewToken(value string) (Token, error) {
	if len(value) == 0 || len(value) > MaximumTokenBytes || !utf8.ValidString(value) {
		return Token{}, fmt.Errorf("%w: generated token must be 1 through %d UTF-8 bytes", ErrInvalid, MaximumTokenBytes)
	}
	return Token{value: value}, nil
}

func (t Token) String() string   { return "[REDACTED]" }
func (t Token) GoString() string { return "credentials.Token([REDACTED])" }
func (t Token) MarshalJSON() ([]byte, error) {
	return nil, errors.New("credential token cannot be serialized")
}

type GatewayPolicy struct {
	ModelPolicies       []contracts.ModelPolicyRef `json:"modelPolicies"`
	MaxBudget           *float64                   `json:"maxBudget,omitempty"`
	BudgetDuration      string                     `json:"budgetDuration,omitempty"`
	TPMLimit            *int                       `json:"tpmLimit,omitempty"`
	RPMLimit            *int                       `json:"rpmLimit,omitempty"`
	MaxParallelRequests *int                       `json:"maxParallelRequests,omitempty"`
}

func (p GatewayPolicy) Validate() error {
	if len(p.ModelPolicies) == 0 || len(p.ModelPolicies) > 128 {
		return fmt.Errorf("%w: Gateway policy has invalid ModelPolicy selections", ErrInvalid)
	}
	seen := make(map[contracts.ModelPolicyRef]struct{}, len(p.ModelPolicies))
	for _, ref := range p.ModelPolicies {
		if err := validateModelPolicyRef(ref); err != nil {
			return err
		}
		if _, exists := seen[ref]; exists {
			return fmt.Errorf("%w: Gateway policy repeats a ModelPolicy ref", ErrInvalid)
		}
		seen[ref] = struct{}{}
	}
	return validateGatewayLimits(
		p.MaxBudget, p.BudgetDuration, p.TPMLimit, p.RPMLimit, p.MaxParallelRequests,
	)
}

type EffectiveGatewayPolicy struct {
	ModelPolicies       []contracts.ModelPolicyRef `json:"modelPolicies"`
	Models              []string                   `json:"models"`
	MaxBudget           *float64                   `json:"maxBudget,omitempty"`
	BudgetDuration      string                     `json:"budgetDuration,omitempty"`
	TPMLimit            *int                       `json:"tpmLimit,omitempty"`
	RPMLimit            *int                       `json:"rpmLimit,omitempty"`
	MaxParallelRequests *int                       `json:"maxParallelRequests,omitempty"`
}

func (p EffectiveGatewayPolicy) Validate() error { return validateEffectiveGatewayPolicy(p) }

// EncryptedEnvelope never participates in a public encoding. Nonce and
// ciphertext are safe at rest but remain internal to avoid confusing them
// with credential metadata.
type EncryptedEnvelope struct {
	SchemaVersion string `json:"-"`
	KeyID         string `json:"-"`
	Nonce         []byte `json:"-"`
	Ciphertext    []byte `json:"-"`
}

type Record struct {
	CredentialID    string                        `json:"credentialId"`
	LLMGateway      contracts.LLMGatewayConfigRef `json:"llmGateway"`
	RemoteKeyID     string                        `json:"-"`
	Label           string                        `json:"label,omitempty"`
	EffectivePolicy EffectiveGatewayPolicy        `json:"effectivePolicy"`
	Envelope        EncryptedEnvelope             `json:"-"`
	CreatedAt       time.Time                     `json:"createdAt"`
}

type Tombstone struct {
	CredentialID string    `json:"credentialId"`
	ActorID      string    `json:"actorId"`
	DeletedAt    time.Time `json:"deletedAt"`
}

type OperationKind string

const (
	OperationCreate OperationKind = "create"
	OperationDelete OperationKind = "delete"
)

type OperationPhase string

const (
	OperationPrepared  OperationPhase = "prepared"
	OperationCompleted OperationPhase = "completed"
)

type Operation struct {
	OperationID    string
	IdempotencyKey string
	RequestHash    string
	CredentialID   string
	Kind           OperationKind
	Phase          OperationPhase
	Request        json.RawMessage
	CreatedAt      time.Time
	UpdatedAt      time.Time
}

type ManagerCreateRequest struct {
	OperationID  string
	CredentialID string
	LLMGateway   contracts.ResolvedLLMGatewayConfig
	Label        string
	Policy       GatewayPolicy
}

type ManagerDeleteRequest struct {
	OperationID  string
	CredentialID string
	LLMGateway   contracts.ResolvedLLMGatewayConfig
	RemoteKeyID  string
}

type GeneratedCredential struct {
	Token           Token
	RemoteKeyID     string
	EffectivePolicy EffectiveGatewayPolicy
}

func (g GeneratedCredential) Validate() error {
	if len(g.Token.value) == 0 || len(g.Token.value) > MaximumTokenBytes || !utf8.ValidString(g.Token.value) {
		return fmt.Errorf("%w: manager returned an invalid token", ErrInvalid)
	}
	if strings.TrimSpace(g.RemoteKeyID) == "" || len(g.RemoteKeyID) > MaximumRemoteKeyIDBytes ||
		!utf8.ValidString(g.RemoteKeyID) {
		return fmt.Errorf("%w: manager returned an invalid remote key ID", ErrInvalid)
	}
	return validateEffectiveGatewayPolicy(g.EffectivePolicy)
}

// GatewayCredentialManager is the secret-bearing outbound boundary. Concrete
// LiteLLM behavior is introduced by V4-004; lifecycle orchestration is V4-003A.
type GatewayCredentialManager interface {
	Create(context.Context, ManagerCreateRequest) (GeneratedCredential, error)
	Delete(context.Context, ManagerDeleteRequest) error
	RecoverCreate(context.Context, ManagerCreateRequest) error
}

func validateEffectiveGatewayPolicy(policy EffectiveGatewayPolicy) error {
	if len(policy.ModelPolicies) == 0 || len(policy.ModelPolicies) > 128 ||
		len(policy.Models) == 0 || len(policy.Models) > 128 {
		return fmt.Errorf("%w: effective Gateway policy has invalid model selections", ErrInvalid)
	}
	seenRefs := make(map[contracts.ModelPolicyRef]struct{}, len(policy.ModelPolicies))
	previousRef := ""
	for index, ref := range policy.ModelPolicies {
		if err := validateModelPolicyRef(ref); err != nil {
			return err
		}
		if _, exists := seenRefs[ref]; exists {
			return fmt.Errorf("%w: effective Gateway policy repeats a ModelPolicy ref", ErrInvalid)
		}
		seenRefs[ref] = struct{}{}
		currentRef := ref.PolicyID + "\x00" + ref.Version + "\x00" + ref.Digest
		if index > 0 && currentRef <= previousRef {
			return fmt.Errorf("%w: effective Gateway ModelPolicy refs are not canonical", ErrInvalid)
		}
		previousRef = currentRef
	}
	seenModels := make(map[string]struct{}, len(policy.Models))
	previousModel := ""
	for index, model := range policy.Models {
		if strings.TrimSpace(model) == "" || !utf8.ValidString(model) || utf8.RuneCountInString(model) > 256 {
			return fmt.Errorf("%w: effective Gateway policy has an invalid model", ErrInvalid)
		}
		if _, exists := seenModels[model]; exists {
			return fmt.Errorf("%w: effective Gateway policy repeats a model", ErrInvalid)
		}
		seenModels[model] = struct{}{}
		if index > 0 && model <= previousModel {
			return fmt.Errorf("%w: effective Gateway models are not canonical", ErrInvalid)
		}
		previousModel = model
	}
	return validateGatewayLimits(
		policy.MaxBudget, policy.BudgetDuration,
		policy.TPMLimit, policy.RPMLimit, policy.MaxParallelRequests,
	)
}

func validateModelPolicyRef(ref contracts.ModelPolicyRef) error {
	if err := (contracts.ResolvedModelPolicy{Ref: ref, Model: "validation-only"}).Validate(); err != nil {
		return fmt.Errorf("%w: Gateway policy has an invalid ModelPolicy ref", ErrInvalid)
	}
	return nil
}

func validateGatewayLimits(
	maxBudget *float64,
	budgetDuration string,
	tpmLimit, rpmLimit, maxParallelRequests *int,
) error {
	if maxBudget != nil && (*maxBudget <= 0 || math.IsNaN(*maxBudget) || math.IsInf(*maxBudget, 0)) {
		return fmt.Errorf("%w: effective Gateway max budget is invalid", ErrInvalid)
	}
	if budgetDuration != "" && maxBudget == nil {
		return fmt.Errorf("%w: effective Gateway budget duration requires max budget", ErrInvalid)
	}
	if budgetDuration != "" && (len(budgetDuration) > 32 ||
		!budgetDurationPattern.MatchString(budgetDuration)) {
		return fmt.Errorf("%w: effective Gateway budget duration is invalid", ErrInvalid)
	}
	for _, value := range []*int{tpmLimit, rpmLimit, maxParallelRequests} {
		if value != nil && (*value <= 0 || int64(*value) > math.MaxInt32) {
			return fmt.Errorf("%w: effective Gateway integer limit is invalid", ErrInvalid)
		}
	}
	return nil
}
