package public

// Versioned configuration and managed credentials: the catalog and
// publisher ports, and the bodies the configuration routes speak.

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/credentials"
)

// ConfigurationCatalog is the atomically swappable read view consumed by the
// public API. Both a bootstrap Snapshot and the managed configuration Manager
// implement it.
type ConfigurationCatalog interface {
	AgentTemplateWorkflowBindings(string) (config.AgentTemplateWorkflowBindings, error)
	AgentInstructions(string) (config.AgentInstructions, error)
	Workflow(string) (config.ResolvedWorkflow, error)
	Workflows() []config.ResolvedWorkflow
	ResolveRunWorkflow(
		context.Context,
		string,
		config.ExecutionConfigPatch,
		config.CredentialLookup,
	) (config.ResolvedWorkflow, error)
	Configurations(config.ConfigurationKind) ([]config.ConfigurationResource, error)
	Configuration(config.ConfigurationKind, string) (config.ConfigurationResource, error)
}

type ConfigurationPublisher interface {
	Publish(context.Context, config.PublicationRequest) (config.PublicationResult, error)
}

type ManagedCredentialLifecycle interface {
	ListCredentials(context.Context, string, int) ([]credentials.Record, error)
	GetCredential(context.Context, string) (credentials.Record, error)
	Create(context.Context, credentials.CreateRequest) (credentials.CreateResult, error)
	Delete(context.Context, credentials.DeleteRequest) (credentials.DeleteResult, error)
	WithRunCreation(context.Context, func() error) error
}

type configurationPageResponse struct {
	Items []config.ConfigurationResource `json:"items"`
	Page  pageInfoResponse               `json:"page"`
}

type publishConfigurationRequest struct {
	Name        string                         `json:"name"`
	Version     string                         `json:"version"`
	ModelPolicy *config.ModelPolicyPublication `json:"modelPolicy,omitempty"`
	LLMGateway  *config.LLMGatewayPublication  `json:"llmGateway,omitempty"`
}

type createCredentialRequest struct {
	CredentialID  string                        `json:"credentialId"`
	LLMGateway    contracts.LLMGatewayConfigRef `json:"llmGateway"`
	Label         *string                       `json:"label,omitempty"`
	GatewayPolicy credentials.GatewayPolicy     `json:"gatewayPolicy"`
}

func (r *createCredentialRequest) UnmarshalJSON(data []byte) error {
	type wire createCredentialRequest
	var decoded wire
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&decoded); err != nil {
		return err
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return err
	}
	if raw, present := fields["label"]; present && bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return errors.New("credential label cannot be null")
	}
	if raw, present := fields["gatewayPolicy"]; present && !bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		var policyFields map[string]json.RawMessage
		if err := json.Unmarshal(raw, &policyFields); err != nil {
			return err
		}
		for _, name := range []string{
			"maxBudget", "budgetDuration", "tpmLimit", "rpmLimit", "maxParallelRequests",
		} {
			if value, exists := policyFields[name]; exists && bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
				return errors.New("Gateway policy optional fields cannot be null")
			}
		}
		if _, exists := policyFields["budgetDuration"]; exists && decoded.GatewayPolicy.BudgetDuration == "" {
			return errors.New("Gateway policy budget duration cannot be empty")
		}
	}
	*r = createCredentialRequest(decoded)
	return nil
}

type credentialPageResponse struct {
	Items []credentials.Record `json:"items"`
	Page  pageInfoResponse     `json:"page"`
}
