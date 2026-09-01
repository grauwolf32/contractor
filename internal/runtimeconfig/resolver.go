package runtimeconfig

import (
	"errors"
	"fmt"
	"sort"

	"github.com/grauwolf32/contractor/internal/contracts"
)

// ResolutionErrorCode is a stable, content-free failure class. Paths are
// limited to the typed merge units below; neither endpoint nor credential
// material is formatted into an error.
type ResolutionErrorCode string

const (
	ResolutionConflict               ResolutionErrorCode = "runtime_config_conflict"
	ResolutionIncomplete             ResolutionErrorCode = "runtime_config_incomplete"
	ResolutionInvalid                ResolutionErrorCode = "runtime_config_invalid"
	ResolutionGatewayMismatch        ResolutionErrorCode = "runtime_config_gateway_mismatch"
	ResolutionCredentialKindMismatch ResolutionErrorCode = "runtime_credential_kind_mismatch"
	ResolutionModelUnauthorized      ResolutionErrorCode = "runtime_model_unauthorized"
)

type ResolutionError struct {
	Code  ResolutionErrorCode
	Path  string
	cause error
}

func (e *ResolutionError) Error() string {
	return string(e.Code) + " at " + e.Path
}

func (e *ResolutionError) Unwrap() error { return e.cause }

type RuntimeLayer string

const (
	LayerDefault     RuntimeLayer = "default"
	LayerWorkflow    RuntimeLayer = "workflow_execution_config"
	LayerRunLabels   RuntimeLayer = "run_labels"
	LayerRunOverride RuntimeLayer = "run_execution_config"
	LayerEscalation  RuntimeLayer = "escalation_execution_config"
	LayerAgentLabels RuntimeLayer = "agent_labels"
)

// RuntimeFieldOrigin identifies the effective typed layer. Configs is present
// for label-backed layers only and contains exact immutable refs, never bodies.
// More than one ref is retained when equal same-layer values deduplicate.
type RuntimeFieldOrigin struct {
	Layer   RuntimeLayer `json:"layer"`
	Configs []Ref        `json:"configs,omitempty"`
}

type ResolvedRuntimeConfigOrigins struct {
	LLMGateway       *RuntimeFieldOrigin `json:"llmGateway,omitempty"`
	LLMCredential    *RuntimeFieldOrigin `json:"llmCredential,omitempty"`
	WorkerTelemetry  *RuntimeFieldOrigin `json:"workerTelemetry,omitempty"`
	HTTPProxy        *RuntimeFieldOrigin `json:"httpProxy,omitempty"`
	PlannerTelemetry *RuntimeFieldOrigin `json:"plannerTelemetry,omitempty"`
}

func (o ResolvedRuntimeConfigOrigins) Validate() error {
	fields := []struct {
		path   string
		origin *RuntimeFieldOrigin
	}{
		{path: "worker.llmGateway.gateway", origin: o.LLMGateway},
		{path: "worker.llmGateway.credential", origin: o.LLMCredential},
		{path: "worker.telemetry", origin: o.WorkerTelemetry},
		{path: "worker.httpProxy", origin: o.HTTPProxy},
		{path: "planner.telemetry", origin: o.PlannerTelemetry},
	}
	for _, field := range fields {
		origin := field.origin
		if origin == nil {
			continue
		}
		if err := origin.validate(); err != nil {
			return resolutionError(ResolutionInvalid, field.path, ErrInvalid)
		}
	}
	return nil
}

func (o RuntimeFieldOrigin) validate() error {
	switch o.Layer {
	case LayerDefault, LayerRunLabels, LayerAgentLabels:
		if len(o.Configs) == 0 || len(o.Configs) > MaximumRunLabels {
			return ErrInvalid
		}
	case LayerWorkflow, LayerRunOverride, LayerEscalation:
		if len(o.Configs) != 0 {
			return ErrInvalid
		}
	default:
		return ErrInvalid
	}
	previous := ""
	for _, ref := range o.Configs {
		if validateRef(ref) != nil || ref.String() <= previous {
			return ErrInvalid
		}
		previous = ref.String()
	}
	return nil
}

// PinnedRuntimeConfig is one exact binding observation and its immutable body.
// Run values come from WorkflowRun.RuntimeConfig; Agent values are read and
// locked at allocation time by the caller.
type PinnedRuntimeConfig struct {
	Label           string
	BindingRevision uint64
	Config          Ref
	Spec            Spec
}

// WorkerRoutePatch is the executionConfig contribution to the two physical
// Worker route leaves. Gateway null is not a valid operation; Credential clear
// is the one supported tri-state.
type WorkerRoutePatch struct {
	Gateway    Field[contracts.LLMGatewayConfigRef]
	Credential Field[string]
}

// LLMCredentialAuthorization is the safe effective Gateway-key policy needed
// to prove that a candidate route still permits the already selected policy
// and model alias. It contains no token or remote key identifier.
type LLMCredentialAuthorization struct {
	Ref           contracts.LLMCredentialRef
	LLMGateway    contracts.LLMGatewayConfigRef
	ModelPolicies []contracts.ModelPolicyRef
	Models        []string
	Unrestricted  bool
}

type ResolveRuntimeConfigInput struct {
	ModelPolicy        contracts.ResolvedModelPolicy
	Default            PinnedRuntimeConfig
	Workflow           WorkerRoutePatch
	RunLabels          []PinnedRuntimeConfig
	RunOverride        WorkerRoutePatch
	Escalation         WorkerRoutePatch
	AgentLabels        []PinnedRuntimeConfig
	Gateways           map[contracts.LLMGatewayConfigRef]contracts.ResolvedLLMGatewayConfig
	LLMCredentials     map[string]LLMCredentialAuthorization
	RuntimeCredentials map[string]contracts.RuntimeCredentialKind
}

// ResolvedRuntimeConfig is safe non-secret allocation input. RuntimeSettings
// secret material is deliberately resolved only after placement by V8-007.
type ResolvedRuntimeConfig struct {
	ModelPolicy              contracts.ResolvedModelPolicy               `json:"modelPolicy"`
	LLMGateway               contracts.ResolvedLLMGatewayConfig          `json:"llmGateway"`
	LLMCredential            *contracts.LLMCredentialRef                 `json:"llmCredential,omitempty"`
	WorkerTelemetry          *TelemetryConfig                            `json:"workerTelemetry,omitempty"`
	HTTPProxy                *HTTPProxyConfig                            `json:"httpProxy,omitempty"`
	PlannerTelemetry         *TelemetryConfig                            `json:"plannerTelemetry,omitempty"`
	PlannerRuntimeCredential *contracts.RuntimeCredentialRefV2           `json:"plannerRuntimeCredential,omitempty"`
	RequiredRuntimeAdapters  []contracts.RuntimeAdapterRef               `json:"requiredRuntimeAdapters"`
	Origins                  ResolvedRuntimeConfigOrigins                `json:"origins"`
	Provenance               contracts.ResolvedRuntimeConfigProvenanceV2 `json:"provenance"`
}

func (r ResolvedRuntimeConfig) Validate() error {
	if err := r.ModelPolicy.Validate(); err != nil {
		return resolutionError(ResolutionInvalid, "modelPolicy", ErrInvalid)
	}
	if err := r.LLMGateway.Validate(); err != nil || r.LLMGateway.Protocol != contracts.OpenAICompatibleProtocol {
		return resolutionError(ResolutionInvalid, "worker.llmGateway.gateway", ErrInvalid)
	}
	if err := r.Origins.Validate(); err != nil {
		return err
	}
	if err := r.Provenance.Validate(); err != nil {
		return resolutionError(ResolutionInvalid, "provenance", ErrInvalid)
	}
	if r.Provenance.LLMGatewayConfig == nil || *r.Provenance.LLMGatewayConfig != r.LLMGateway.Ref ||
		!sameCredentialRef(r.Provenance.LLMCredential, r.LLMCredential) ||
		!sameRuntimeAdapters(r.Provenance.RuntimeAdapters, r.RequiredRuntimeAdapters) {
		return resolutionError(ResolutionInvalid, "provenance", ErrInvalid)
	}
	return nil
}

type effectiveRuntimeConfig struct {
	gateway          *contracts.LLMGatewayConfigRef
	credential       *contracts.LLMCredentialRef
	workerTelemetry  *TelemetryConfig
	httpProxy        *HTTPProxyConfig
	plannerTelemetry *TelemetryConfig
	origins          ResolvedRuntimeConfigOrigins
}

type layerOrigins struct {
	gateway, credential, workerTelemetry, httpProxy, plannerTelemetry *RuntimeFieldOrigin
}

// ResolveRuntimeConfig applies the normative precedence order without I/O.
// Input maps are immutable catalogs owned by the caller for the duration of
// the call; the result owns every returned slice and pointer.
func ResolveRuntimeConfig(input ResolveRuntimeConfigInput) (ResolvedRuntimeConfig, error) {
	if err := input.ModelPolicy.Validate(); err != nil {
		return ResolvedRuntimeConfig{}, resolutionError(ResolutionInvalid, "modelPolicy", ErrInvalid)
	}
	if err := validatePinned(input.Default, DefaultLabel); err != nil {
		return ResolvedRuntimeConfig{}, err
	}
	runSpec, runOrigins, runPins, err := mergePinnedLayer(input.RunLabels, LayerRunLabels, false)
	if err != nil {
		return ResolvedRuntimeConfig{}, err
	}
	agentSpec, agentOrigins, agentPins, err := mergePinnedLayer(input.AgentLabels, LayerAgentLabels, true)
	if err != nil {
		return ResolvedRuntimeConfig{}, err
	}
	if err := validateRoutePatch(input.Workflow, "workflow.executionConfig"); err != nil {
		return ResolvedRuntimeConfig{}, err
	}
	if err := validateRoutePatch(input.RunOverride, "run.executionConfig"); err != nil {
		return ResolvedRuntimeConfig{}, err
	}
	if err := validateRoutePatch(input.Escalation, "escalation.executionConfig"); err != nil {
		return ResolvedRuntimeConfig{}, err
	}

	var effective effectiveRuntimeConfig
	defaultOrigin := originsForPinned(LayerDefault, []PinnedRuntimeConfig{input.Default})
	applyWorkerSpec(&effective, input.Default.Spec.Worker, defaultOrigin)
	applyPlannerSpec(&effective, input.Default.Spec.Planner, defaultOrigin)
	applyRoutePatch(&effective, input.Workflow, RuntimeFieldOrigin{Layer: LayerWorkflow})
	applyWorkerSpec(&effective, runSpec.Worker, runOrigins)
	applyPlannerSpec(&effective, runSpec.Planner, runOrigins)
	applyRoutePatch(&effective, input.RunOverride, RuntimeFieldOrigin{Layer: LayerRunOverride})
	applyRoutePatch(&effective, input.Escalation, RuntimeFieldOrigin{Layer: LayerEscalation})
	applyWorkerSpec(&effective, agentSpec.Worker, agentOrigins)

	if effective.gateway == nil {
		return ResolvedRuntimeConfig{}, resolutionError(
			ResolutionIncomplete, "worker.llmGateway.gateway", ErrInvalid,
		)
	}
	gateway, ok := input.Gateways[*effective.gateway]
	if !ok || gateway.Ref != *effective.gateway {
		return ResolvedRuntimeConfig{}, resolutionError(
			ResolutionIncomplete, "worker.llmGateway.gateway", ErrInvalid,
		)
	}
	if err := gateway.Validate(); err != nil || gateway.Protocol != contracts.OpenAICompatibleProtocol {
		return ResolvedRuntimeConfig{}, resolutionError(
			ResolutionInvalid, "worker.llmGateway.gateway", ErrInvalid,
		)
	}

	if err := validateLLMRoute(input, gateway.Ref, effective.credential); err != nil {
		return ResolvedRuntimeConfig{}, err
	}
	workerCredentials, plannerCredential, adapters, err := validateAdapterSettings(input, effective)
	if err != nil {
		return ResolvedRuntimeConfig{}, err
	}
	provenance := contracts.ResolvedRuntimeConfigProvenanceV2{
		Default:               bindingProvenance(input.Default),
		RunLabels:             bindingProvenanceList(runPins),
		AgentLabels:           bindingProvenanceList(agentPins),
		RuntimeAdapters:       append([]contracts.RuntimeAdapterRef{}, adapters...),
		LLMGatewayConfig:      cloneGatewayRef(&gateway.Ref),
		LLMCredential:         cloneCredentialRef(effective.credential),
		RuntimeCredentialRefs: workerCredentials,
	}
	if err := provenance.Validate(); err != nil {
		return ResolvedRuntimeConfig{}, resolutionError(ResolutionInvalid, "provenance", ErrInvalid)
	}
	result := ResolvedRuntimeConfig{
		ModelPolicy:              cloneResolvedModelPolicy(input.ModelPolicy),
		LLMGateway:               cloneResolvedGateway(gateway),
		LLMCredential:            cloneCredentialRef(effective.credential),
		WorkerTelemetry:          cloneTelemetry(effective.workerTelemetry),
		HTTPProxy:                cloneHTTPProxy(effective.httpProxy),
		PlannerTelemetry:         cloneTelemetry(effective.plannerTelemetry),
		PlannerRuntimeCredential: plannerCredential,
		RequiredRuntimeAdapters:  append([]contracts.RuntimeAdapterRef{}, adapters...),
		Origins:                  cloneOrigins(effective.origins),
		Provenance:               provenance,
	}
	if err := result.Origins.Validate(); err != nil {
		return ResolvedRuntimeConfig{}, err
	}
	return result, nil
}

func mergePinnedLayer(
	values []PinnedRuntimeConfig, layer RuntimeLayer, agent bool,
) (Spec, layerOrigins, []PinnedRuntimeConfig, error) {
	ordered := append([]PinnedRuntimeConfig{}, values...)
	sort.Slice(ordered, func(i, j int) bool { return ordered[i].Label < ordered[j].Label })
	entries := make([]LayerEntry, 0, len(ordered))
	previous := ""
	for _, value := range ordered {
		if value.Label == DefaultLabel || value.Label == previous {
			return Spec{}, layerOrigins{}, nil, resolutionError(ResolutionInvalid, string(layer), ErrInvalid)
		}
		if err := validatePinned(value, value.Label); err != nil {
			return Spec{}, layerOrigins{}, nil, err
		}
		previous = value.Label
		spec := value.Spec
		if agent {
			if !workerApplicable(spec.Worker) {
				return Spec{}, layerOrigins{}, nil, resolutionError(
					ResolutionInvalid, "agent_labels.worker", ErrInvalid,
				)
			}
			// Agent labels cannot configure the Server-side Planner.
			spec.Planner = PlannerPatch{}
		}
		entries = append(entries, LayerEntry{Label: value.Label, Ref: value.Config, Spec: spec})
	}
	merged, err := MergeSameLayer(entries)
	if err != nil {
		var conflict *MergeConflictError
		path := string(layer)
		if errors.As(err, &conflict) {
			path = conflict.Path
		}
		return Spec{}, layerOrigins{}, nil, resolutionError(ResolutionConflict, path, err)
	}
	return merged, originsForPinned(layer, ordered), ordered, nil
}

func validatePinned(value PinnedRuntimeConfig, expectedLabel string) error {
	if value.Label != expectedLabel || value.BindingRevision == 0 ||
		validateLabel(value.Label) != nil || validateRef(value.Config) != nil {
		return resolutionError(ResolutionInvalid, "label_binding", ErrInvalid)
	}
	return nil
}

func validateRoutePatch(value WorkerRoutePatch, path string) error {
	if value.Gateway.Clear || !value.Gateway.Present && value.Gateway.Value != (contracts.LLMGatewayConfigRef{}) {
		return resolutionError(ResolutionInvalid, path+".llmGateway.gateway", ErrInvalid)
	}
	if value.Gateway.Present && value.Gateway.Value.ValidateRef() != nil {
		return resolutionError(ResolutionInvalid, path+".llmGateway.gateway", ErrInvalid)
	}
	if !value.Credential.Present && (value.Credential.Clear || value.Credential.Value != "") ||
		value.Credential.Present && value.Credential.Clear && value.Credential.Value != "" {
		return resolutionError(ResolutionInvalid, path+".llmGateway.credential", ErrInvalid)
	}
	if value.Credential.Present && !value.Credential.Clear {
		if err := (&contracts.LLMCredentialRef{CredentialID: value.Credential.Value}).Validate(); err != nil {
			return resolutionError(ResolutionInvalid, path+".llmGateway.credential", ErrInvalid)
		}
	}
	return nil
}

func applyRoutePatch(target *effectiveRuntimeConfig, patch WorkerRoutePatch, origin RuntimeFieldOrigin) {
	if patch.Gateway.Present {
		value := patch.Gateway.Value
		target.gateway = &value
		target.origins.LLMGateway = cloneOrigin(&origin)
	}
	if patch.Credential.Present {
		if patch.Credential.Clear {
			target.credential = nil
		} else {
			target.credential = &contracts.LLMCredentialRef{CredentialID: patch.Credential.Value}
		}
		target.origins.LLMCredential = cloneOrigin(&origin)
	}
}

func applyWorkerSpec(target *effectiveRuntimeConfig, patch WorkerPatch, origins layerOrigins) {
	if patch.LLMGateway.Gateway.Present {
		value := patch.LLMGateway.Gateway.Value
		target.gateway = &value
		target.origins.LLMGateway = cloneOrigin(origins.gateway)
	}
	if patch.LLMGateway.Credential.Present {
		if patch.LLMGateway.Credential.Clear {
			target.credential = nil
		} else {
			target.credential = &contracts.LLMCredentialRef{CredentialID: patch.LLMGateway.Credential.Value}
		}
		target.origins.LLMCredential = cloneOrigin(origins.credential)
	}
	if patch.Telemetry.Present {
		if patch.Telemetry.Clear {
			target.workerTelemetry = nil
		} else {
			value := patch.Telemetry.Value
			target.workerTelemetry = &value
		}
		target.origins.WorkerTelemetry = cloneOrigin(origins.workerTelemetry)
	}
	if patch.HTTPProxy.Present {
		if patch.HTTPProxy.Clear {
			target.httpProxy = nil
		} else {
			value := patch.HTTPProxy.Value
			value.Targets = append([]string{}, value.Targets...)
			target.httpProxy = &value
		}
		target.origins.HTTPProxy = cloneOrigin(origins.httpProxy)
	}
}

func applyPlannerSpec(target *effectiveRuntimeConfig, patch PlannerPatch, origins layerOrigins) {
	if !patch.Telemetry.Present {
		return
	}
	if patch.Telemetry.Clear {
		target.plannerTelemetry = nil
	} else {
		value := patch.Telemetry.Value
		target.plannerTelemetry = &value
	}
	target.origins.PlannerTelemetry = cloneOrigin(origins.plannerTelemetry)
}

func validateLLMRoute(
	input ResolveRuntimeConfigInput,
	gateway contracts.LLMGatewayConfigRef,
	credential *contracts.LLMCredentialRef,
) error {
	if credential == nil {
		return nil
	}
	authorization, ok := input.LLMCredentials[credential.CredentialID]
	if !ok || authorization.Ref != *credential {
		return resolutionError(ResolutionIncomplete, "worker.llmGateway.credential", ErrInvalid)
	}
	if authorization.LLMGateway != gateway {
		return resolutionError(
			ResolutionGatewayMismatch, "worker.llmGateway.credential", ErrInvalid,
		)
	}
	if !authorization.Unrestricted &&
		(!containsModelPolicy(authorization.ModelPolicies, input.ModelPolicy.Ref) ||
			!containsString(authorization.Models, input.ModelPolicy.Model)) {
		return resolutionError(
			ResolutionModelUnauthorized, "worker.llmGateway.credential", ErrInvalid,
		)
	}
	return nil
}

func validateAdapterSettings(
	input ResolveRuntimeConfigInput,
	effective effectiveRuntimeConfig,
) ([]contracts.RuntimeCredentialRefV2, *contracts.RuntimeCredentialRefV2, []contracts.RuntimeAdapterRef, error) {
	workerCredentials := make([]contracts.RuntimeCredentialRefV2, 0, 2)
	adapters := make([]contracts.RuntimeAdapterRef, 0, 2)
	if effective.workerTelemetry != nil {
		if err := validateTelemetryConfig(*effective.workerTelemetry, "worker.telemetry"); err != nil {
			return nil, nil, nil, err
		}
		adapters = append(adapters, contracts.RuntimeAdapterOTLPHTTP)
		if effective.workerTelemetry.Credential != "" {
			ref, err := requireRuntimeCredential(
				input.RuntimeCredentials, effective.workerTelemetry.Credential,
				"worker.telemetry.credential", contracts.RuntimeCredentialOTLPHeaders,
			)
			if err != nil {
				return nil, nil, nil, err
			}
			workerCredentials = append(workerCredentials, ref)
		}
	}
	if effective.httpProxy != nil {
		if err := validateHTTPProxyConfig(*effective.httpProxy); err != nil {
			return nil, nil, nil, err
		}
		adapters = append(adapters, contracts.RuntimeAdapterHTTPProxy)
		if effective.httpProxy.Credential != "" {
			ref, err := requireRuntimeCredential(
				input.RuntimeCredentials, effective.httpProxy.Credential,
				"worker.httpProxy.credential",
				contracts.RuntimeCredentialProxyBasic, contracts.RuntimeCredentialProxyBearer,
			)
			if err != nil {
				return nil, nil, nil, err
			}
			workerCredentials = append(workerCredentials, ref)
		}
	}
	var plannerCredential *contracts.RuntimeCredentialRefV2
	if effective.plannerTelemetry != nil {
		if err := validateTelemetryConfig(*effective.plannerTelemetry, "planner.telemetry"); err != nil {
			return nil, nil, nil, err
		}
		if effective.plannerTelemetry.Credential != "" {
			ref, err := requireRuntimeCredential(
				input.RuntimeCredentials, effective.plannerTelemetry.Credential,
				"planner.telemetry.credential", contracts.RuntimeCredentialOTLPHeaders,
			)
			if err != nil {
				return nil, nil, nil, err
			}
			plannerCredential = &ref
		}
	}
	sort.Slice(adapters, func(i, j int) bool { return adapters[i] < adapters[j] })
	sort.Slice(workerCredentials, func(i, j int) bool {
		left := string(workerCredentials[i].Kind) + "\x00" + workerCredentials[i].CredentialID
		right := string(workerCredentials[j].Kind) + "\x00" + workerCredentials[j].CredentialID
		return left < right
	})
	return workerCredentials, plannerCredential, adapters, nil
}

func validateTelemetryConfig(value TelemetryConfig, path string) error {
	if value.Adapter != string(contracts.RuntimeAdapterOTLPHTTP) || value.CaptureContent ||
		value.FlushTimeoutSeconds < 1 || value.FlushTimeoutSeconds > 10 {
		return resolutionError(ResolutionInvalid, path, ErrInvalid)
	}
	if _, err := validateURL(path+".endpoint", value.Endpoint); err != nil {
		return resolutionError(ResolutionInvalid, path, ErrInvalid)
	}
	return nil
}

func validateHTTPProxyConfig(value HTTPProxyConfig) error {
	if value.Adapter != string(contracts.RuntimeAdapterHTTPProxy) {
		return resolutionError(ResolutionInvalid, "worker.httpProxy", ErrInvalid)
	}
	if _, err := validateURL("worker.httpProxy.proxyUrl", value.ProxyURL); err != nil {
		return resolutionError(ResolutionInvalid, "worker.httpProxy", ErrInvalid)
	}
	if value.CABundlePEM != "" && validateCABundle(value.CABundlePEM) != nil {
		return resolutionError(ResolutionInvalid, "worker.httpProxy", ErrInvalid)
	}
	if len(value.Targets) == 0 || len(value.Targets) > 3 {
		return resolutionError(ResolutionInvalid, "worker.httpProxy.targets", ErrInvalid)
	}
	previous := ""
	for _, target := range value.Targets {
		switch contracts.HTTPProxyTarget(target) {
		case contracts.ProxyTargetLLMGateway, contracts.ProxyTargetToolHTTP, contracts.ProxyTargetToolSubprocess:
		default:
			return resolutionError(ResolutionInvalid, "worker.httpProxy.targets", ErrInvalid)
		}
		if target <= previous {
			return resolutionError(ResolutionInvalid, "worker.httpProxy.targets", ErrInvalid)
		}
		previous = target
	}
	return nil
}

func requireRuntimeCredential(
	catalog map[string]contracts.RuntimeCredentialKind,
	credentialID, path string,
	allowed ...contracts.RuntimeCredentialKind,
) (contracts.RuntimeCredentialRefV2, error) {
	kind, ok := catalog[credentialID]
	if !ok {
		return contracts.RuntimeCredentialRefV2{}, resolutionError(ResolutionIncomplete, path, ErrInvalid)
	}
	for _, candidate := range allowed {
		if kind == candidate {
			return contracts.RuntimeCredentialRefV2{CredentialID: credentialID, Kind: kind}, nil
		}
	}
	return contracts.RuntimeCredentialRefV2{}, resolutionError(
		ResolutionCredentialKindMismatch, path, ErrInvalid,
	)
}

func originsForPinned(layer RuntimeLayer, entries []PinnedRuntimeConfig) layerOrigins {
	selectOrigin := func(selected func(Spec) bool) *RuntimeFieldOrigin {
		refs := make([]Ref, 0, len(entries))
		seen := make(map[Ref]struct{}, len(entries))
		for _, entry := range entries {
			if !selected(entry.Spec) {
				continue
			}
			if _, exists := seen[entry.Config]; exists {
				continue
			}
			seen[entry.Config] = struct{}{}
			refs = append(refs, entry.Config)
		}
		if len(refs) == 0 {
			return nil
		}
		sort.Slice(refs, func(i, j int) bool { return refs[i].String() < refs[j].String() })
		return &RuntimeFieldOrigin{Layer: layer, Configs: refs}
	}
	return layerOrigins{
		gateway: selectOrigin(func(spec Spec) bool {
			return spec.Worker.LLMGateway.Gateway.Present
		}),
		credential: selectOrigin(func(spec Spec) bool {
			return spec.Worker.LLMGateway.Credential.Present
		}),
		workerTelemetry: selectOrigin(func(spec Spec) bool {
			return spec.Worker.Telemetry.Present
		}),
		httpProxy: selectOrigin(func(spec Spec) bool {
			return spec.Worker.HTTPProxy.Present
		}),
		plannerTelemetry: selectOrigin(func(spec Spec) bool {
			return spec.Planner.Telemetry.Present
		}),
	}
}

func bindingProvenance(value PinnedRuntimeConfig) contracts.RuntimeLabelBindingProvenanceV2 {
	return contracts.RuntimeLabelBindingProvenanceV2{
		Label: value.Label, BindingRevision: value.BindingRevision,
		Config: contracts.RuntimeConfigRefV2{
			Name: value.Config.Name, Version: value.Config.Version, Digest: value.Config.Digest,
		},
	}
}

func bindingProvenanceList(values []PinnedRuntimeConfig) []contracts.RuntimeLabelBindingProvenanceV2 {
	result := make([]contracts.RuntimeLabelBindingProvenanceV2, len(values))
	for index := range values {
		result[index] = bindingProvenance(values[index])
	}
	return result
}

func workerApplicable(value WorkerPatch) bool {
	return value.LLMGateway.Present || value.Telemetry.Present || value.HTTPProxy.Present
}

func containsModelPolicy(values []contracts.ModelPolicyRef, expected contracts.ModelPolicyRef) bool {
	for _, value := range values {
		if value == expected {
			return true
		}
	}
	return false
}

func containsString(values []string, expected string) bool {
	for _, value := range values {
		if value == expected {
			return true
		}
	}
	return false
}

func resolutionError(code ResolutionErrorCode, path string, cause error) error {
	return &ResolutionError{Code: code, Path: path, cause: cause}
}

func cloneResolvedModelPolicy(value contracts.ResolvedModelPolicy) contracts.ResolvedModelPolicy {
	result := value
	if value.Temperature != nil {
		temperature := *value.Temperature
		result.Temperature = &temperature
	}
	return result
}

func cloneResolvedGateway(value contracts.ResolvedLLMGatewayConfig) contracts.ResolvedLLMGatewayConfig {
	result := value
	if value.CredentialManager != nil {
		manager := *value.CredentialManager
		result.CredentialManager = &manager
	}
	return result
}

func cloneGatewayRef(value *contracts.LLMGatewayConfigRef) *contracts.LLMGatewayConfigRef {
	if value == nil {
		return nil
	}
	result := *value
	return &result
}

func cloneCredentialRef(value *contracts.LLMCredentialRef) *contracts.LLMCredentialRef {
	if value == nil {
		return nil
	}
	result := *value
	return &result
}

func sameCredentialRef(left, right *contracts.LLMCredentialRef) bool {
	return left == nil && right == nil ||
		left != nil && right != nil && *left == *right
}

func sameRuntimeAdapters(left, right []contracts.RuntimeAdapterRef) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
}

func cloneTelemetry(value *TelemetryConfig) *TelemetryConfig {
	if value == nil {
		return nil
	}
	result := *value
	return &result
}

func cloneHTTPProxy(value *HTTPProxyConfig) *HTTPProxyConfig {
	if value == nil {
		return nil
	}
	result := *value
	result.Targets = append([]string{}, value.Targets...)
	return &result
}

func cloneOrigin(value *RuntimeFieldOrigin) *RuntimeFieldOrigin {
	if value == nil {
		return nil
	}
	result := *value
	result.Configs = append([]Ref{}, value.Configs...)
	return &result
}

func cloneOrigins(value ResolvedRuntimeConfigOrigins) ResolvedRuntimeConfigOrigins {
	return ResolvedRuntimeConfigOrigins{
		LLMGateway:       cloneOrigin(value.LLMGateway),
		LLMCredential:    cloneOrigin(value.LLMCredential),
		WorkerTelemetry:  cloneOrigin(value.WorkerTelemetry),
		HTTPProxy:        cloneOrigin(value.HTTPProxy),
		PlannerTelemetry: cloneOrigin(value.PlannerTelemetry),
	}
}

// Clone returns a detached safe resolution snapshot. It is intentionally
// explicit so Registry copies cannot share mutable provenance slices with a
// placement caller.
func (r ResolvedRuntimeConfig) Clone() ResolvedRuntimeConfig {
	result := r
	result.ModelPolicy = cloneResolvedModelPolicy(r.ModelPolicy)
	result.LLMGateway = cloneResolvedGateway(r.LLMGateway)
	result.LLMCredential = cloneCredentialRef(r.LLMCredential)
	result.WorkerTelemetry = cloneTelemetry(r.WorkerTelemetry)
	result.HTTPProxy = cloneHTTPProxy(r.HTTPProxy)
	result.PlannerTelemetry = cloneTelemetry(r.PlannerTelemetry)
	if r.PlannerRuntimeCredential != nil {
		credential := *r.PlannerRuntimeCredential
		result.PlannerRuntimeCredential = &credential
	}
	result.RequiredRuntimeAdapters = append([]contracts.RuntimeAdapterRef{}, r.RequiredRuntimeAdapters...)
	result.Origins = cloneOrigins(r.Origins)
	result.Provenance.RunLabels = append(
		[]contracts.RuntimeLabelBindingProvenanceV2{}, r.Provenance.RunLabels...,
	)
	result.Provenance.AgentLabels = append(
		[]contracts.RuntimeLabelBindingProvenanceV2{}, r.Provenance.AgentLabels...,
	)
	result.Provenance.RuntimeAdapters = append(
		[]contracts.RuntimeAdapterRef{}, r.Provenance.RuntimeAdapters...,
	)
	result.Provenance.LLMGatewayConfig = cloneGatewayRef(r.Provenance.LLMGatewayConfig)
	result.Provenance.LLMCredential = cloneCredentialRef(r.Provenance.LLMCredential)
	result.Provenance.RuntimeCredentialRefs = append(
		[]contracts.RuntimeCredentialRefV2{}, r.Provenance.RuntimeCredentialRefs...,
	)
	return result
}

func (r ResolvedRuntimeConfig) String() string {
	return fmt.Sprintf(
		"ResolvedRuntimeConfig(gateway=%s@%s, adapters=%d)",
		r.LLMGateway.Ref.GatewayID, r.LLMGateway.Ref.Version, len(r.RequiredRuntimeAdapters),
	)
}
