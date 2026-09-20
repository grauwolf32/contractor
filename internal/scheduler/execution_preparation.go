package scheduler

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"

	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

// Allocation configuration is materialized only at preparation time. These
// helpers do not acquire claims, persist snapshots or own allocation release.
func (s *Scheduler) workerExecutionSettingsForRun(
	ctx context.Context,
	run runstore.WorkflowRun,
	stage workflowconfig.ResolvedStage,
	reservationSet []controlplane.Reservation,
) (map[string]contracts.WorkerExecutionSettings, error) {
	if len(reservationSet) != len(stage.ExecutionConfig.Agents) {
		return nil, fmt.Errorf("reservation set differs from Worker execution settings")
	}
	reservations := make(map[string]controlplane.Reservation, len(reservationSet))
	for _, reservation := range reservationSet {
		name := reservation.Grant.LogicalAgentName
		if _, duplicate := reservations[name]; duplicate {
			return nil, fmt.Errorf("duplicate Worker reservation %q", name)
		}
		reservations[name] = reservation
	}
	result := make(map[string]contracts.WorkerExecutionSettings, len(stage.ExecutionConfig.Agents))
	for logicalName := range stage.ExecutionConfig.Agents {
		reservation, ok := reservations[logicalName]
		if !ok || reservation.ResolvedRuntimeConfig == nil {
			return nil, fmt.Errorf("Worker %q has no complete Runtime configuration", logicalName)
		}
		resolved := reservation.ResolvedRuntimeConfig.Clone()
		modelFree := stage.Agents[logicalName].Template.IsToolWorker()
		if resolved.Validate() != nil || resolved.ModelFree != modelFree {
			return nil, fmt.Errorf("Worker %q has no complete Runtime configuration", logicalName)
		}
		runtimeSettings, err := s.materializeRuntimeSettings(ctx, resolved)
		if err != nil {
			return nil, err
		}
		if run.ProjectHTTPTarget != nil && agentUsesHTTPRequest(stage.Agents[logicalName].Template) {
			target, targetErr := s.materializeHTTPOriginTarget(ctx, *run.ProjectHTTPTarget)
			if targetErr != nil {
				return nil, targetErr
			}
			runtimeSettings.HTTPOriginTarget = target
		}
		result[logicalName] = contracts.WorkerExecutionSettings{
			ModelPolicy: cloneModelPolicy(resolved.ModelPolicy), RuntimeSettings: runtimeSettings,
			ResolvedRuntimeConfigProvenance: resolved.Provenance,
		}
	}
	return result, nil
}

func agentUsesHTTPRequest(template contracts.ResolvedAgentTemplate) bool {
	for _, selection := range template.Toolsets {
		if selection.Ref.ToolsetID != "http-tools" || selection.Ref.Version != "1" {
			continue
		}
		for _, tool := range selection.Tools {
			if tool == "http_request" {
				return true
			}
		}
	}
	return false
}

func (s *Scheduler) materializeHTTPOriginTarget(
	ctx context.Context,
	reference contracts.HTTPOriginTargetRef,
) (*contracts.HTTPOriginTargetSettings, error) {
	if err := reference.Validate(); err != nil {
		return nil, errors.New("Project HTTP target snapshot is invalid")
	}
	target := &contracts.HTTPOriginTargetSettings{URL: reference.URL}
	if reference.Credential == nil {
		return target, nil
	}
	err := s.useRuntimeCredential(
		ctx, reference.Credential.CredentialID,
		[]contracts.RuntimeCredentialKind{contracts.RuntimeCredentialOriginBasic, contracts.RuntimeCredentialOriginBearer},
		func(kind contracts.RuntimeCredentialKind, plaintext []byte) error {
			switch kind {
			case contracts.RuntimeCredentialOriginBasic:
				var material struct {
					Password string `json:"password"`
					Username string `json:"username"`
				}
				if decodeRuntimeCredential(plaintext, &material) != nil {
					return errors.New("invalid HTTP origin basic credential material")
				}
				target.BasicAuth = &contracts.HTTPProxyBasicAuth{
					Username: contracts.NewSecretString(material.Username),
					Password: contracts.NewSecretString(material.Password),
				}
			case contracts.RuntimeCredentialOriginBearer:
				var material struct {
					Token string `json:"token"`
				}
				if decodeRuntimeCredential(plaintext, &material) != nil {
					return errors.New("invalid HTTP origin bearer credential material")
				}
				token := contracts.NewSecretString(material.Token)
				target.BearerToken = &token
			default:
				return errors.New("invalid HTTP origin credential kind")
			}
			return nil
		},
	)
	if err != nil {
		return nil, errors.New("Project HTTP target credential is unavailable")
	}
	if err := target.Validate(); err != nil {
		return nil, errors.New("Project HTTP target settings are invalid")
	}
	return target, nil
}

func (s *Scheduler) materializeRuntimeSettings(
	ctx context.Context,
	resolved runtimeconfig.ResolvedRuntimeConfig,
) (contracts.RuntimeSettings, error) {
	result := contracts.RuntimeSettings{
		LLMGatewayURL:         resolved.LLMGateway.URL,
		ArtifactAPIURL:        s.options.RuntimeSettings.ArtifactAPIURL,
		RequestTimeoutSeconds: s.options.RuntimeSettings.RequestTimeoutSeconds,
	}
	if resolved.LLMCredential != nil {
		if s.options.Credentials == nil {
			return contracts.RuntimeSettings{}, fmt.Errorf("selected LLM credential is unavailable")
		}
		token, err := s.options.Credentials.ResolveLLMCredential(
			ctx, *resolved.LLMCredential, resolved.LLMGateway.Ref,
		)
		if err != nil || token.Reveal() == "" {
			return contracts.RuntimeSettings{}, fmt.Errorf("selected LLM credential is unavailable")
		}
		result.LLMGatewayToken = &token
	}
	if resolved.WorkerTelemetry != nil {
		telemetry := &contracts.TelemetrySettings{
			Adapter: contracts.RuntimeAdapterOTLPHTTP, Endpoint: resolved.WorkerTelemetry.Endpoint,
			Headers: map[string]contracts.SecretString{}, CaptureContent: resolved.WorkerTelemetry.CaptureContent,
			FlushTimeoutSeconds: resolved.WorkerTelemetry.FlushTimeoutSeconds,
		}
		if resolved.WorkerTelemetry.Export != nil {
			export := *resolved.WorkerTelemetry.Export
			if export.Retry != nil {
				retry := *export.Retry
				export.Retry = &retry
			}
			telemetry.Export = &export
		}
		if credentialID := resolved.WorkerTelemetry.Credential; credentialID != "" {
			if err := s.useRuntimeCredential(
				ctx, credentialID, []contracts.RuntimeCredentialKind{contracts.RuntimeCredentialOTLPHeaders},
				func(kind contracts.RuntimeCredentialKind, plaintext []byte) error {
					var material struct {
						Headers map[string]string `json:"headers"`
					}
					if kind != contracts.RuntimeCredentialOTLPHeaders || decodeRuntimeCredential(plaintext, &material) != nil {
						return errors.New("invalid OTLP credential material")
					}
					for name, value := range material.Headers {
						telemetry.Headers[name] = contracts.NewSecretString(value)
					}
					return nil
				},
			); err != nil {
				return contracts.RuntimeSettings{}, fmt.Errorf("Worker telemetry credential is unavailable")
			}
		}
		result.Telemetry = telemetry
	}
	if resolved.HTTPProxy != nil {
		proxy := &contracts.HTTPProxySettings{
			Adapter: contracts.RuntimeAdapterHTTPProxy, ProxyURL: resolved.HTTPProxy.ProxyURL,
			Targets: make([]contracts.HTTPProxyTarget, len(resolved.HTTPProxy.Targets)),
		}
		for index, target := range resolved.HTTPProxy.Targets {
			proxy.Targets[index] = contracts.HTTPProxyTarget(target)
		}
		if resolved.HTTPProxy.CABundlePEM != "" {
			bundle := resolved.HTTPProxy.CABundlePEM
			proxy.CABundlePEM = &bundle
		}
		if credentialID := resolved.HTTPProxy.Credential; credentialID != "" {
			if err := s.useRuntimeCredential(
				ctx, credentialID,
				[]contracts.RuntimeCredentialKind{
					contracts.RuntimeCredentialProxyBasic, contracts.RuntimeCredentialProxyBearer,
				},
				func(kind contracts.RuntimeCredentialKind, plaintext []byte) error {
					switch kind {
					case contracts.RuntimeCredentialProxyBasic:
						var material struct {
							Password string `json:"password"`
							Username string `json:"username"`
						}
						if decodeRuntimeCredential(plaintext, &material) != nil {
							return errors.New("invalid proxy basic credential material")
						}
						proxy.BasicAuth = &contracts.HTTPProxyBasicAuth{
							Username: contracts.NewSecretString(material.Username),
							Password: contracts.NewSecretString(material.Password),
						}
					case contracts.RuntimeCredentialProxyBearer:
						var material struct {
							Token string `json:"token"`
						}
						if decodeRuntimeCredential(plaintext, &material) != nil {
							return errors.New("invalid proxy bearer credential material")
						}
						token := contracts.NewSecretString(material.Token)
						proxy.BearerToken = &token
					default:
						return errors.New("invalid proxy credential kind")
					}
					return nil
				},
			); err != nil {
				return contracts.RuntimeSettings{}, fmt.Errorf("Worker HTTP proxy credential is unavailable")
			}
		}
		result.HTTPProxy = proxy
	}
	if resolved.Caido != nil {
		timeout := resolved.Caido.RequestTimeoutSeconds
		if timeout == 0 || timeout > result.RequestTimeoutSeconds {
			timeout = result.RequestTimeoutSeconds
		}
		if timeout > contracts.MaxCaidoRequestTimeoutSeconds {
			timeout = contracts.MaxCaidoRequestTimeoutSeconds
		}
		caido := &contracts.CaidoSettings{
			Adapter: contracts.RuntimeAdapterCaidoGraphQL, Endpoint: resolved.Caido.Endpoint,
			RequestTimeoutSeconds: timeout,
		}
		if resolved.Caido.CABundlePEM != "" {
			bundle := resolved.Caido.CABundlePEM
			caido.CABundlePEM = &bundle
		}
		if credentialID := resolved.Caido.Credential; credentialID != "" {
			if err := s.useRuntimeCredential(
				ctx, credentialID, []contracts.RuntimeCredentialKind{contracts.RuntimeCredentialCaidoBearer},
				func(kind contracts.RuntimeCredentialKind, plaintext []byte) error {
					var material struct {
						Token string `json:"token"`
					}
					if kind != contracts.RuntimeCredentialCaidoBearer || decodeRuntimeCredential(plaintext, &material) != nil {
						return errors.New("invalid Caido credential material")
					}
					token := contracts.NewSecretString(material.Token)
					caido.BearerToken = &token
					return nil
				},
			); err != nil {
				return contracts.RuntimeSettings{}, fmt.Errorf("Worker Caido credential is unavailable")
			}
		}
		result.Caido = caido
	}
	if err := result.Validate(); err != nil {
		return contracts.RuntimeSettings{}, fmt.Errorf("materialized Runtime settings are invalid")
	}
	return result, nil
}

func (s *Scheduler) useRuntimeCredential(
	ctx context.Context,
	credentialID string,
	allowed []contracts.RuntimeCredentialKind,
	consumer func(contracts.RuntimeCredentialKind, []byte) error,
) error {
	if s.options.RuntimeCredentials == nil {
		return errors.New("Runtime credential service is unavailable")
	}
	return s.options.RuntimeCredentials.UsePlaintext(ctx, credentialID, allowed, consumer)
}

func decodeRuntimeCredential(data []byte, target any) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return err
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		return errors.New("Runtime credential has trailing data")
	}
	return nil
}

func clearWorkerExecutionSettings(settings map[string]contracts.WorkerExecutionSettings) {
	for name, value := range settings {
		value.RuntimeSettings.LLMGatewayToken = nil
		if value.RuntimeSettings.Telemetry != nil {
			value.RuntimeSettings.Telemetry.Headers = nil
		}
		if value.RuntimeSettings.HTTPProxy != nil {
			value.RuntimeSettings.HTTPProxy.BasicAuth = nil
			value.RuntimeSettings.HTTPProxy.BearerToken = nil
		}
		if value.RuntimeSettings.Caido != nil {
			value.RuntimeSettings.Caido.BearerToken = nil
		}
		if value.RuntimeSettings.HTTPOriginTarget != nil {
			value.RuntimeSettings.HTTPOriginTarget.BasicAuth = nil
			value.RuntimeSettings.HTTPOriginTarget.BearerToken = nil
			value.RuntimeSettings.HTTPOriginTarget = nil
		}
		settings[name] = value
	}
}

func (s *Scheduler) plannerModelAccess(
	ctx context.Context,
	stage workflowconfig.ResolvedStage,
) (*planner.ModelAccess, error) {
	if stage.ExecutionConfig.Planner == nil {
		return nil, nil
	}
	selection := *stage.ExecutionConfig.Planner
	if selection.LLMGateway == nil {
		return nil, fmt.Errorf("Planner has no complete LLM Gateway route")
	}
	token, err := s.resolveCredential(ctx, selection)
	if err != nil {
		return nil, err
	}
	result := &planner.ModelAccess{
		ModelPolicy: cloneModelPolicy(selection.ModelPolicy),
		LLMGateway:  *selection.LLMGateway,
		Token:       token,
	}
	if selection.Credential != nil {
		credential := *selection.Credential
		result.Credential = &credential
	}
	return result, nil
}

func (s *Scheduler) resolveCredential(
	ctx context.Context,
	selection workflowconfig.ResolvedConsumerExecutionConfig,
) (contracts.SecretString, error) {
	if selection.Credential == nil {
		return contracts.NewSecretString(""), nil
	}
	if selection.LLMGateway == nil {
		return contracts.SecretString{}, fmt.Errorf("selected LLM credential has no Gateway")
	}
	if s.options.Credentials == nil {
		return contracts.SecretString{}, fmt.Errorf("selected LLM credential is unavailable")
	}
	token, err := s.options.Credentials.ResolveLLMCredential(
		ctx, *selection.Credential, selection.LLMGateway.Ref,
	)
	if err != nil || token.Reveal() == "" {
		return contracts.SecretString{}, fmt.Errorf("selected LLM credential is unavailable")
	}
	return token, nil
}
