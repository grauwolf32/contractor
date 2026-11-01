package public

import (
	"encoding/json"
	"errors"
	"time"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

type workflowPageResponse struct {
	Items []workflowSummaryResponse `json:"items"`
	Page  pageInfoResponse          `json:"page"`
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
	CredentialID  string                         `json:"credentialId"`
	LLMGateway    llmgateway.LLMGatewayConfigRef `json:"llmGateway"`
	Label         *string                        `json:"label,omitempty"`
	GatewayPolicy credentials.GatewayPolicy      `json:"gatewayPolicy"`
}

func (r *createCredentialRequest) UnmarshalJSON(data []byte) error {
	type wire createCredentialRequest
	var decoded wire
	fields, err := decodeStrictWithPresence(data, &decoded)
	if err != nil {
		return err
	}
	if fields.null("label") {
		return errors.New("credential label cannot be null")
	}
	if raw, present := fields["gatewayPolicy"]; present && !isJSONNull(raw) {
		var policyFields jsonMembers
		if err := json.Unmarshal(raw, &policyFields); err != nil {
			return err
		}
		for _, name := range []string{
			"maxBudget", "budgetDuration", "tpmLimit", "rpmLimit", "maxParallelRequests",
		} {
			if policyFields.null(name) {
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

type runtimeConfigResourceResponse struct {
	Ref       runtimeconfig.Ref `json:"ref"`
	Document  json.RawMessage   `json:"document"`
	BuiltIn   bool              `json:"builtIn"`
	CreatedBy string            `json:"createdBy"`
	CreatedAt time.Time         `json:"createdAt"`
}

type runtimeConfigPageResponse struct {
	Items []runtimeConfigResourceResponse `json:"items"`
	Page  pageInfoResponse                `json:"page"`
}

type runtimeLabelPageResponse struct {
	Items []runtimeLabelResponse `json:"items"`
	Page  pageInfoResponse       `json:"page"`
}

type runtimeLabelResponse struct {
	Label     string            `json:"label"`
	Config    runtimeconfig.Ref `json:"config"`
	Revision  string            `json:"revision"`
	CreatedBy string            `json:"createdBy"`
	CreatedAt time.Time         `json:"createdAt"`
	UpdatedBy string            `json:"updatedBy"`
	UpdatedAt time.Time         `json:"updatedAt"`
}

type runtimeLabelMutationRequest struct {
	Config runtimeconfig.Ref `json:"config"`
}

type createRuntimeCredentialRequest struct {
	CredentialID string
	Material     credentials.RuntimeCredentialMaterial
}

func (r *createRuntimeCredentialRequest) UnmarshalJSON(data []byte) error {
	var envelope struct {
		CredentialID string                            `json:"credentialId"`
		Kind         credentials.RuntimeCredentialKind `json:"kind"`
		Material     json.RawMessage                   `json:"material"`
	}
	if err := strictjson.Decode(data, &envelope); err != nil {
		return err
	}
	if len(envelope.Material) == 0 || isJSONNull(envelope.Material) {
		return errors.New("Runtime credential material is required")
	}
	material, err := credentials.DecodeRuntimeCredentialMaterial(envelope.Kind, envelope.Material)
	if err != nil {
		return err
	}
	r.CredentialID = envelope.CredentialID
	r.Material = material
	return nil
}

type runtimeCredentialPageResponse struct {
	Items []credentials.RuntimeCredentialMetadata `json:"items"`
	Page  pageInfoResponse                        `json:"page"`
}

type runtimeAgentPrincipalResponse struct {
	RuntimeAgentID          string                                         `json:"runtimeAgentId"`
	Labels                  []string                                       `json:"labels"`
	Revision                string                                         `json:"revision"`
	Availability            controlplane.RuntimeAgentPrincipalAvailability `json:"availability"`
	RequiredRuntimeAdapters []string                                       `json:"requiredRuntimeAdapters"`
	MissingRuntimeAdapters  []string                                       `json:"missingRuntimeAdapters"`
	Live                    *controlplane.RuntimeAgentObservation          `json:"live,omitempty"`
	CreatedBy               string                                         `json:"createdBy"`
	CreatedAt               time.Time                                      `json:"createdAt"`
	UpdatedBy               string                                         `json:"updatedBy"`
	UpdatedAt               time.Time                                      `json:"updatedAt"`
}

type runtimeAgentPrincipalPageResponse struct {
	Items []runtimeAgentPrincipalResponse `json:"items"`
	Page  pageInfoResponse                `json:"page"`
}

type runtimeAgentLabelsMutationRequest struct {
	Labels []string `json:"labels"`
}

func (r *runtimeAgentLabelsMutationRequest) UnmarshalJSON(data []byte) error {
	var envelope struct {
		Labels json.RawMessage `json:"labels"`
	}
	if err := strictjson.Decode(data, &envelope); err != nil {
		return err
	}
	if len(envelope.Labels) == 0 || isJSONNull(envelope.Labels) {
		return errors.New("Runtime Agent labels must be an array")
	}
	if err := json.Unmarshal(envelope.Labels, &r.Labels); err != nil || r.Labels == nil {
		return errors.New("Runtime Agent labels must be an array")
	}
	return nil
}
