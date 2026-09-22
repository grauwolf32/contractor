package public

// Runtime configuration: RuntimeConfigs, their label bindings, Runtime
// credentials and Runtime Agent principals.

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"time"

	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

type RuntimeConfigManagement interface {
	Publish(context.Context, []byte, string, string) (runtimeconfig.PublishResult, error)
	ListVersions(context.Context, string, string, int) ([]runtimeconfig.Version, error)
	GetVersion(context.Context, string, string) (runtimeconfig.Version, error)
	ListBindings(context.Context, string, int) ([]runtimeconfig.Binding, error)
	GetBinding(context.Context, string) (runtimeconfig.Binding, error)
	CreateBinding(context.Context, string, runtimeconfig.Ref, string, string, time.Time) (runtimeconfig.BindingMutationResult, error)
	Rebind(context.Context, string, uint64, runtimeconfig.Ref, string, string, time.Time) (runtimeconfig.BindingMutationResult, error)
	DeleteBinding(context.Context, string, uint64, string, string, time.Time) (runtimeconfig.BindingMutationResult, error)
}

type RuntimeCredentialManagement interface {
	List(context.Context, string, int) ([]credentials.RuntimeCredentialMetadata, error)
	Get(context.Context, string) (credentials.RuntimeCredentialMetadata, error)
	Create(context.Context, credentials.RuntimeCredentialCreateRequest) (credentials.RuntimeCredentialCreateResult, error)
	Delete(context.Context, string, string) (credentials.RuntimeCredentialDeleteResult, error)
	ValidateRuntimeCredential(context.Context, string, ...string) error
	WithCredentialReferences(context.Context, func() error) error
}

type RuntimeAgentPrincipalManagement interface {
	List(context.Context, string, int) ([]controlplane.RuntimeAgentPrincipalProjection, error)
	Get(context.Context, string) (controlplane.RuntimeAgentPrincipalProjection, error)
	ReplaceLabels(context.Context, string, uint64, []string, string, string, time.Time) (controlplane.RuntimeAgentPrincipalProjection, bool, error)
	Delete(context.Context, string, uint64, string, string, time.Time) (bool, error)
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
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&envelope); err != nil {
		return err
	}
	if len(envelope.Material) == 0 || bytes.Equal(bytes.TrimSpace(envelope.Material), []byte("null")) {
		return errors.New("Runtime credential material is required")
	}
	var material credentials.RuntimeCredentialMaterial
	var err error
	switch envelope.Kind {
	case credentials.RuntimeCredentialOTLPHeaders:
		var value struct {
			Headers map[string]string `json:"headers"`
		}
		if err = decodeStrictPublicJSON(envelope.Material, &value); err == nil {
			material, err = credentials.NewOTLPHeadersCredential(value.Headers)
		}
	case credentials.RuntimeCredentialProxyBasic:
		var value struct {
			Username string `json:"username"`
			Password string `json:"password"`
		}
		if err = decodeStrictPublicJSON(envelope.Material, &value); err == nil {
			material, err = credentials.NewHTTPProxyBasicCredential(value.Username, value.Password)
		}
	case credentials.RuntimeCredentialProxyBearer:
		var value struct {
			Token string `json:"token"`
		}
		if err = decodeStrictPublicJSON(envelope.Material, &value); err == nil {
			material, err = credentials.NewHTTPProxyBearerCredential(value.Token)
		}
	case credentials.RuntimeCredentialCaidoBearer:
		var value struct {
			Token string `json:"token"`
		}
		if err = decodeStrictPublicJSON(envelope.Material, &value); err == nil {
			material, err = credentials.NewCaidoBearerCredential(value.Token)
		}
	case credentials.RuntimeCredentialOriginBasic:
		var value struct {
			Username string `json:"username"`
			Password string `json:"password"`
		}
		if err = decodeStrictPublicJSON(envelope.Material, &value); err == nil {
			material, err = credentials.NewHTTPOriginBasicCredential(value.Username, value.Password)
		}
	case credentials.RuntimeCredentialOriginBearer:
		var value struct {
			Token string `json:"token"`
		}
		if err = decodeStrictPublicJSON(envelope.Material, &value); err == nil {
			material, err = credentials.NewHTTPOriginBearerCredential(value.Token)
		}
	default:
		err = credentials.ErrRuntimeCredentialInvalid
	}
	if err != nil {
		material.Destroy()
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
	if err := decodeStrictPublicJSON(data, &envelope); err != nil {
		return err
	}
	if len(envelope.Labels) == 0 || bytes.Equal(bytes.TrimSpace(envelope.Labels), []byte("null")) {
		return errors.New("Runtime Agent labels must be an array")
	}
	if err := json.Unmarshal(envelope.Labels, &r.Labels); err != nil || r.Labels == nil {
		return errors.New("Runtime Agent labels must be an array")
	}
	return nil
}
