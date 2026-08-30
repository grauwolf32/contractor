package litellm

import (
	"encoding/json"

	"github.com/grauwolf32/contractor/internal/contracts"
)

type keyMetadata struct {
	CredentialID   string `json:"contractorCredentialId"`
	GatewayID      string `json:"contractorGatewayId"`
	GatewayVersion string `json:"contractorGatewayVersion"`
	GatewayDigest  string `json:"contractorGatewayDigest"`
	OperationID    string `json:"contractorOperationId"`
	Label          string `json:"contractorLabel,omitempty"`
}

type generateKeyRequest struct {
	KeyAlias            string      `json:"key_alias"`
	KeyType             string      `json:"key_type"`
	Models              []string    `json:"models"`
	Metadata            keyMetadata `json:"metadata"`
	MaxBudget           *float64    `json:"max_budget,omitempty"`
	BudgetDuration      string      `json:"budget_duration,omitempty"`
	TPMLimit            *int        `json:"tpm_limit,omitempty"`
	RPMLimit            *int        `json:"rpm_limit,omitempty"`
	MaxParallelRequests *int        `json:"max_parallel_requests,omitempty"`

	// ModelPolicies are Contractor provenance and never leave the process.
	ModelPolicies []contracts.ModelPolicyRef `json:"-"`
}

// generateKeyResponse names every top-level field emitted by the pinned
// LiteLLM 1.85.0 GenerateKeyResponse. Irrelevant provider fields remain raw,
// are never stored, and exist here only so DisallowUnknownFields can detect an
// unreviewed compatibility change.
type generateKeyResponse struct {
	Key                       string          `json:"key"`
	Token                     string          `json:"token"`
	TokenID                   string          `json:"token_id"`
	KeyAlias                  string          `json:"key_alias"`
	Models                    []string        `json:"models"`
	Metadata                  keyMetadata     `json:"metadata"`
	MaxBudget                 *float64        `json:"max_budget"`
	BudgetDuration            *string         `json:"budget_duration"`
	TPMLimit                  *int            `json:"tpm_limit"`
	RPMLimit                  *int            `json:"rpm_limit"`
	MaxParallelRequests       *int            `json:"max_parallel_requests"`
	AllowedRoutes             []string        `json:"allowed_routes"`
	Blocked                   *bool           `json:"blocked"`
	AccessGroupIDs            json.RawMessage `json:"access_group_ids"`
	AgentID                   json.RawMessage `json:"agent_id"`
	Aliases                   json.RawMessage `json:"aliases"`
	AllowedCacheControls      json.RawMessage `json:"allowed_cache_controls"`
	AllowedPassthroughRoutes  json.RawMessage `json:"allowed_passthrough_routes"`
	AllowedVectorStoreIndexes json.RawMessage `json:"allowed_vector_store_indexes"`
	BudgetID                  json.RawMessage `json:"budget_id"`
	BudgetLimits              json.RawMessage `json:"budget_limits"`
	Config                    json.RawMessage `json:"config"`
	CreatedAt                 json.RawMessage `json:"created_at"`
	CreatedBy                 json.RawMessage `json:"created_by"`
	Duration                  json.RawMessage `json:"duration"`
	EnforcedParams            json.RawMessage `json:"enforced_params"`
	Expires                   json.RawMessage `json:"expires"`
	Guardrails                json.RawMessage `json:"guardrails"`
	KeyName                   json.RawMessage `json:"key_name"`
	LiteLLMBudgetTable        json.RawMessage `json:"litellm_budget_table"`
	ModelMaxBudget            json.RawMessage `json:"model_max_budget"`
	ModelRPMLimit             json.RawMessage `json:"model_rpm_limit"`
	ModelTPMLimit             json.RawMessage `json:"model_tpm_limit"`
	ObjectPermission          json.RawMessage `json:"object_permission"`
	OrganizationID            json.RawMessage `json:"organization_id"`
	Permissions               json.RawMessage `json:"permissions"`
	Policies                  json.RawMessage `json:"policies"`
	ProjectID                 json.RawMessage `json:"project_id"`
	Prompts                   json.RawMessage `json:"prompts"`
	RouterSettings            json.RawMessage `json:"router_settings"`
	RPMLimitType              json.RawMessage `json:"rpm_limit_type"`
	Spend                     json.RawMessage `json:"spend"`
	Tags                      json.RawMessage `json:"tags"`
	TeamID                    json.RawMessage `json:"team_id"`
	TPMLimitType              json.RawMessage `json:"tpm_limit_type"`
	UpdatedAt                 json.RawMessage `json:"updated_at"`
	UpdatedBy                 json.RawMessage `json:"updated_by"`
	UserID                    json.RawMessage `json:"user_id"`
}

type deleteKeyRequest struct {
	Keys       []string `json:"keys,omitempty"`
	KeyAliases []string `json:"key_aliases,omitempty"`
}

type deleteKeyResponse struct {
	DeletedKeys []string `json:"deleted_keys"`
}

type liteLLMErrorResponse struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Param   string `json:"param"`
		Code    string `json:"code"`
	} `json:"error"`
}
