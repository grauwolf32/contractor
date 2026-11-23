package controlplane

import (
	"context"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

func (a *PlacementAllocator) gatewayCatalog(
	selection resolved.ResolvedConsumerExecutionConfig,
	configs []runtimeconfig.PinnedRuntimeConfig,
) (map[llmgateway.LLMGatewayConfigRef]llmgateway.ResolvedLLMGatewayConfig, error) {
	refs := make(map[llmgateway.LLMGatewayConfigRef]struct{})
	if selection.LLMGateway != nil {
		refs[selection.LLMGateway.Ref] = struct{}{}
	}
	for _, pinned := range configs {
		if field := pinned.Spec.Worker.LLMGateway.Gateway; field.Present && !field.Clear {
			refs[field.Value] = struct{}{}
		}
	}
	result := make(map[llmgateway.LLMGatewayConfigRef]llmgateway.ResolvedLLMGatewayConfig, len(refs))
	for ref := range refs {
		gateway, err := a.gateways.LLMGateway(ref.GatewayID + "@" + ref.Version)
		if err != nil || gateway.Ref != ref {
			return nil, runtimeconfig.ErrNotFound
		}
		result[ref] = gateway
	}
	return result, nil
}

// Keep SQL-backed lookup ownership explicit. Optimistic resolution uses pool
// readers; the recheck under the lifecycle barrier uses transaction readers.
type placementCredentialLookups struct {
	llm          resolved.CredentialLookup
	runtime      PlacementRuntimeCredentialLookup
	llmCache     map[string]placementCachedValue[runtimeconfig.LLMCredentialAuthorization]
	runtimeCache map[string]placementCachedValue[runtimesettings.RuntimeCredentialKind]
}

func (lookups placementCredentialLookups) catalogs(
	ctx context.Context,
	selection resolved.ResolvedConsumerExecutionConfig,
	configs []runtimeconfig.PinnedRuntimeConfig,
) (
	map[string]runtimeconfig.LLMCredentialAuthorization,
	map[string]runtimesettings.RuntimeCredentialKind,
	error,
) {
	llmIDs := make(map[string]struct{})
	runtimeIDs := make(map[string]struct{})
	if selection.Credential != nil {
		llmIDs[selection.Credential.CredentialID] = struct{}{}
	}
	for _, pinned := range configs {
		collectSpecCredentialIDs(pinned.Spec, llmIDs, runtimeIDs)
	}
	llmResult := make(map[string]runtimeconfig.LLMCredentialAuthorization, len(llmIDs))
	for id := range llmIDs {
		authorization, err := lookups.llmAuthorization(ctx, id)
		if err != nil {
			return nil, nil, err
		}
		llmResult[id] = authorization
	}
	runtimeResult := make(map[string]runtimesettings.RuntimeCredentialKind, len(runtimeIDs))
	for id := range runtimeIDs {
		kind, err := lookups.runtimeKind(ctx, id)
		if err != nil {
			return nil, nil, err
		}
		runtimeResult[id] = kind
	}
	return llmResult, runtimeResult, nil
}

func (lookups placementCredentialLookups) llmAuthorization(
	ctx context.Context,
	id string,
) (runtimeconfig.LLMCredentialAuthorization, error) {
	if cached, ok := lookups.llmCache[id]; ok {
		return cached.value, cached.err
	}
	metadata, err := lookups.llm.LookupLLMCredential(ctx, id)
	var authorization runtimeconfig.LLMCredentialAuthorization
	if err == nil {
		authorization = runtimeconfig.LLMCredentialAuthorization{
			Ref: metadata.Ref, LLMGateway: metadata.LLMGateway,
			ModelPolicies: append([]contracts.ModelPolicyRef{}, metadata.ModelPolicies...),
			Models:        append([]string{}, metadata.Models...), Unrestricted: metadata.Unrestricted,
		}
	}
	if lookups.llmCache != nil {
		lookups.llmCache[id] = placementCachedValue[runtimeconfig.LLMCredentialAuthorization]{value: authorization, err: err}
	}
	return authorization, err
}

func (lookups placementCredentialLookups) runtimeKind(
	ctx context.Context,
	id string,
) (runtimesettings.RuntimeCredentialKind, error) {
	if cached, ok := lookups.runtimeCache[id]; ok {
		return cached.value, cached.err
	}
	metadata, err := lookups.runtime.Get(ctx, id)
	kind := runtimesettings.RuntimeCredentialKind(metadata.Kind)
	if lookups.runtimeCache != nil {
		lookups.runtimeCache[id] = placementCachedValue[runtimesettings.RuntimeCredentialKind]{value: kind, err: err}
	}
	return kind, err
}

func collectSpecCredentialIDs(
	spec runtimeconfig.Spec,
	llmIDs map[string]struct{},
	runtimeIDs map[string]struct{},
) {
	if field := spec.Worker.LLMGateway.Credential; field.Present && !field.Clear && field.Value != "" {
		llmIDs[field.Value] = struct{}{}
	}
	if field := spec.Worker.Telemetry; field.Present && !field.Clear && field.Value.Credential != "" {
		runtimeIDs[field.Value.Credential] = struct{}{}
	}
	if field := spec.Worker.HTTPProxy; field.Present && !field.Clear && field.Value.Credential != "" {
		runtimeIDs[field.Value.Credential] = struct{}{}
	}
	if field := spec.Worker.Caido; field.Present && !field.Clear && field.Value.Credential != "" {
		runtimeIDs[field.Value.Credential] = struct{}{}
	}
	if field := spec.Worker.Interactsh; field.Present && !field.Clear && field.Value.Credential != "" {
		runtimeIDs[field.Value.Credential] = struct{}{}
	}
	if field := spec.Planner.Telemetry; field.Present && !field.Clear && field.Value.Credential != "" {
		runtimeIDs[field.Value.Credential] = struct{}{}
	}
}
