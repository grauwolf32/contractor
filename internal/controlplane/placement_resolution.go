package controlplane

import (
	"context"
	"errors"
	"strings"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/credentials"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

type placementCachedValue[T any] struct {
	value T
	err   error
}

// placementAttemptCache shares optimistic observations among the passes of
// one placement attempt, that is one ReserveAllContext call: the first pass,
// the retry after a principal refresh and the model Stage hold (CTL-R162).
// The pinning transaction never reads it: selected edges are re-read and
// drift-checked under that transaction (CTL-R88). It dies with the attempt,
// so the next Scheduler poll reads every mutable dependency again.
//
// It holds at most one entry per distinct Agent label, credential ID and
// (binding, ordered Agent-label set) among the location-matching live
// candidates of the attempt, so its size follows the fleet, not the number of
// passes or candidates sharing a configuration.
type placementAttemptCache struct {
	// The pinned default and Run-label versions are immutable references of
	// the Run snapshot, so discardMutable keeps them.
	requestLoaded bool
	defaultConfig runtimeconfig.PinnedRuntimeConfig
	runConfigs    []runtimeconfig.PinnedRuntimeConfig
	requestErr    error

	agentLabels        map[string]placementCachedValue[runtimeconfig.PinnedRuntimeConfig]
	llmCredentials     map[string]placementCachedValue[runtimeconfig.LLMCredentialAuthorization]
	runtimeCredentials map[string]placementCachedValue[runtimesettings.RuntimeCredentialKind]
	resolutions        map[placementResolutionInput]placementCachedValue[runtimeconfig.ResolvedRuntimeConfig]
}

// placementResolutionInput names everything a candidate contributes to
// resolveCandidate: the binding and the principal's exact ordered Agent-label
// set. Labels never contain NUL (runtimeconfig.ValidateLabel).
type placementResolutionInput struct {
	logicalAgentName string
	labels           string
}

func newPlacementAttemptCache() *placementAttemptCache {
	return &placementAttemptCache{
		agentLabels:        make(map[string]placementCachedValue[runtimeconfig.PinnedRuntimeConfig]),
		llmCredentials:     make(map[string]placementCachedValue[runtimeconfig.LLMCredentialAuthorization]),
		runtimeCredentials: make(map[string]placementCachedValue[runtimesettings.RuntimeCredentialKind]),
		resolutions:        make(map[placementResolutionInput]placementCachedValue[runtimeconfig.ResolvedRuntimeConfig]),
	}
}

// discardMutable drops every observation of mutable durable state: Agent-label
// bindings, credential metadata and the resolutions derived from them. A
// changed principal or a drift found by the pinning transaction proves that a
// mutation committed after these reads, and the attempt cannot tell which
// binding or credential mutations accompanied it (CTL-R162).
func (cache *placementAttemptCache) discardMutable() {
	clear(cache.agentLabels)
	clear(cache.llmCredentials)
	clear(cache.runtimeCredentials)
	clear(cache.resolutions)
}

func (a *PlacementAllocator) resolveCandidate(
	ctx context.Context,
	db persistencepostgres.DBTX,
	request ReservationRequest,
	binding BindingRequirement,
	principal AuthenticatedPrincipal,
	lookups placementCredentialLookups,
) (runtimeconfig.ResolvedRuntimeConfig, error) {
	return a.resolveCandidateWithCache(ctx, db, request, binding, principal, lookups, nil)
}

// resolveAttemptCandidate resolves binding for principal's exact ordered
// Agent-label set at most once per attempt (CTL-R162). Another candidate with
// the same set, in this or a later pass, reuses the result, including a
// candidate-local failure.
func (a *PlacementAllocator) resolveAttemptCandidate(
	ctx context.Context,
	request ReservationRequest,
	binding BindingRequirement,
	principal AuthenticatedPrincipal,
	lookups placementCredentialLookups,
	attempt *placementAttemptCache,
) (runtimeconfig.ResolvedRuntimeConfig, error) {
	input := placementResolutionInput{
		logicalAgentName: binding.LogicalAgentName, labels: strings.Join(principal.Labels, "\x00"),
	}
	if cached, ok := attempt.resolutions[input]; ok {
		return cached.value, cached.err
	}
	runtimeConfig, err := a.resolveCandidateWithCache(ctx, a.pool, request, binding, principal, lookups, attempt)
	attempt.resolutions[input] = placementCachedValue[runtimeconfig.ResolvedRuntimeConfig]{value: runtimeConfig, err: err}
	return runtimeConfig, err
}

func (a *PlacementAllocator) resolveCandidateWithCache(
	ctx context.Context,
	db persistencepostgres.DBTX,
	request ReservationRequest,
	binding BindingRequirement,
	principal AuthenticatedPrincipal,
	lookups placementCredentialLookups,
	cache *placementAttemptCache,
) (runtimeconfig.ResolvedRuntimeConfig, error) {
	if request.RuntimeConfig == nil || binding.RuntimeSelection == nil {
		return runtimeconfig.ResolvedRuntimeConfig{}, runtimeconfig.ErrInvalid
	}
	repository := runtimeconfig.NewRepository(db)
	var defaultConfig runtimeconfig.PinnedRuntimeConfig
	var runConfigs []runtimeconfig.PinnedRuntimeConfig
	var err error
	if cache == nil {
		defaultConfig, runConfigs, err = loadRequestPinnedRuntimeConfigs(ctx, repository, request.RuntimeConfig)
	} else {
		defaultConfig, runConfigs, err = cache.requestConfigs(ctx, repository, request.RuntimeConfig)
	}
	if err != nil {
		return runtimeconfig.ResolvedRuntimeConfig{}, err
	}
	agentConfigs := make([]runtimeconfig.PinnedRuntimeConfig, 0, len(principal.Labels))
	for _, label := range principal.Labels {
		var loaded runtimeconfig.PinnedRuntimeConfig
		if cache == nil {
			loaded, err = loadAgentPinnedRuntimeConfig(ctx, repository, label)
		} else {
			loaded, err = cache.agentConfig(ctx, repository, label)
		}
		if err != nil {
			return runtimeconfig.ResolvedRuntimeConfig{}, err
		}
		agentConfigs = append(agentConfigs, loaded)
	}
	workflowPatch, escalationPatch, err := runtimeRoutePatches(*binding.RuntimeSelection)
	if err != nil {
		return runtimeconfig.ResolvedRuntimeConfig{}, err
	}
	if binding.AgentTemplate.IsToolWorker() {
		defaultConfig = runtimeconfig.WithoutWorkerModel(defaultConfig)
		for i := range runConfigs {
			runConfigs[i] = runtimeconfig.WithoutWorkerModel(runConfigs[i])
		}
		for i := range agentConfigs {
			agentConfigs[i] = runtimeconfig.WithoutWorkerModel(agentConfigs[i])
		}
	}
	allConfigs := append([]runtimeconfig.PinnedRuntimeConfig{defaultConfig}, runConfigs...)
	allConfigs = append(allConfigs, agentConfigs...)
	gateways, err := a.gatewayCatalog(*binding.RuntimeSelection, allConfigs)
	if err != nil {
		return runtimeconfig.ResolvedRuntimeConfig{}, err
	}
	llmCredentials, runtimeCredentials, err := lookups.catalogs(ctx, *binding.RuntimeSelection, allConfigs)
	if err != nil {
		return runtimeconfig.ResolvedRuntimeConfig{}, err
	}
	var summarizerModelPolicy *contracts.ResolvedModelPolicy
	if binding.RuntimeSelection.SummarizerModelPolicy != nil {
		policy := binding.RuntimeSelection.SummarizerModelPolicy.Clone()
		summarizerModelPolicy = &policy
	}
	var compactorModelPolicy *contracts.ResolvedModelPolicy
	if binding.ContextManagement.IsCompact() {
		policy := binding.ContextManagement.ModelPolicy.Clone()
		compactorModelPolicy = &policy
	}
	return runtimeconfig.ResolveRuntimeConfig(runtimeconfig.ResolveRuntimeConfigInput{
		ModelFree:   binding.AgentTemplate.IsToolWorker(),
		ModelPolicy: binding.RuntimeSelection.ModelPolicy, SummarizerModelPolicy: summarizerModelPolicy,
		CompactorModelPolicy: compactorModelPolicy,
		Default:              defaultConfig,
		Workflow:             workflowPatch, RunLabels: runConfigs,
		Escalation: escalationPatch, AgentLabels: agentConfigs,
		Gateways: gateways, LLMCredentials: llmCredentials, RuntimeCredentials: runtimeCredentials,
	})
}

func loadRequestPinnedRuntimeConfigs(
	ctx context.Context,
	repository *runtimeconfig.Repository,
	snapshot *runtimeconfig.RunSnapshot,
) (runtimeconfig.PinnedRuntimeConfig, []runtimeconfig.PinnedRuntimeConfig, error) {
	defaultConfig, err := loadPinnedRuntimeConfig(ctx, repository, snapshot.Default)
	if err != nil {
		return runtimeconfig.PinnedRuntimeConfig{}, nil, err
	}
	runConfigs := make([]runtimeconfig.PinnedRuntimeConfig, 0, len(snapshot.Labels))
	for _, pin := range snapshot.Labels {
		loaded, err := loadPinnedRuntimeConfig(ctx, repository, pin)
		if err != nil {
			return runtimeconfig.PinnedRuntimeConfig{}, nil, err
		}
		runConfigs = append(runConfigs, loaded)
	}
	return defaultConfig, runConfigs, nil
}

func (cache *placementAttemptCache) requestConfigs(
	ctx context.Context,
	repository *runtimeconfig.Repository,
	snapshot *runtimeconfig.RunSnapshot,
) (runtimeconfig.PinnedRuntimeConfig, []runtimeconfig.PinnedRuntimeConfig, error) {
	if !cache.requestLoaded {
		cache.requestLoaded = true
		cache.defaultConfig, cache.runConfigs, cache.requestErr = loadRequestPinnedRuntimeConfigs(ctx, repository, snapshot)
	}
	// Model-free projection changes slice entries, so each candidate owns its
	// own shallow slice while the cached pinned specs remain read-only.
	return cache.defaultConfig, append([]runtimeconfig.PinnedRuntimeConfig(nil), cache.runConfigs...), cache.requestErr
}

func loadAgentPinnedRuntimeConfig(
	ctx context.Context,
	repository *runtimeconfig.Repository,
	label string,
) (runtimeconfig.PinnedRuntimeConfig, error) {
	binding, err := repository.GetBinding(ctx, label)
	if err != nil {
		return runtimeconfig.PinnedRuntimeConfig{}, err
	}
	version, err := repository.GetVersionByRef(ctx, binding.Ref)
	if err != nil {
		return runtimeconfig.PinnedRuntimeConfig{}, err
	}
	return runtimeconfig.PinnedRuntimeConfig{
		Label: label, BindingRevision: binding.Revision, Config: binding.Ref, Spec: version.Spec,
	}, nil
}

func (cache *placementAttemptCache) agentConfig(
	ctx context.Context,
	repository *runtimeconfig.Repository,
	label string,
) (runtimeconfig.PinnedRuntimeConfig, error) {
	if cached, ok := cache.agentLabels[label]; ok {
		return cached.value, cached.err
	}
	loaded, err := loadAgentPinnedRuntimeConfig(ctx, repository, label)
	cache.agentLabels[label] = placementCachedValue[runtimeconfig.PinnedRuntimeConfig]{value: loaded, err: err}
	return loaded, err
}

func loadPinnedRuntimeConfig(
	ctx context.Context,
	repository *runtimeconfig.Repository,
	pin runtimeconfig.PinnedLabel,
) (runtimeconfig.PinnedRuntimeConfig, error) {
	version, err := repository.GetVersionByRef(ctx, pin.Config)
	if err != nil {
		return runtimeconfig.PinnedRuntimeConfig{}, err
	}
	return runtimeconfig.PinnedRuntimeConfig{
		Label: pin.Label, BindingRevision: pin.BindingRevision, Config: pin.Config, Spec: version.Spec,
	}, nil
}

func runtimeRoutePatches(
	selection resolved.ResolvedConsumerExecutionConfig,
) (runtimeconfig.WorkerRoutePatch, runtimeconfig.WorkerRoutePatch, error) {
	var workflow, escalation runtimeconfig.WorkerRoutePatch
	patchFor := func(origin string) (*runtimeconfig.WorkerRoutePatch, error) {
		switch {
		case strings.Contains(origin, ".escalate.executionConfig"):
			return &escalation, nil
		case strings.HasPrefix(origin, "workflow.executionConfig"):
			return &workflow, nil
		default:
			return nil, runtimeconfig.ErrInvalid
		}
	}
	if selection.Origins.LLMGateway != "" {
		if selection.LLMGateway == nil {
			return workflow, escalation, runtimeconfig.ErrInvalid
		}
		patch, err := patchFor(selection.Origins.LLMGateway)
		if err != nil {
			return workflow, escalation, err
		}
		patch.Gateway = runtimeconfig.Field[llmgateway.LLMGatewayConfigRef]{
			Present: true, Value: selection.LLMGateway.Ref,
		}
	}
	if selection.Origins.Credential != "" {
		patch, err := patchFor(selection.Origins.Credential)
		if err != nil {
			return workflow, escalation, err
		}
		patch.Credential.Present = true
		if selection.Credential == nil {
			patch.Credential.Clear = true
		} else {
			patch.Credential.Value = selection.Credential.CredentialID
		}
	}
	return workflow, escalation, nil
}

// Candidate-local policy, reference and kind failures remove only that edge.
// Opaque repository/provider errors are deliberately not collapsed into
// capacity: a database or credential backend outage must remain observable to
// Scheduler instead of looking like an idle-capacity shortage forever.
func candidateResolutionUnavailable(err error) bool {
	var resolution *runtimeconfig.ResolutionError
	return errors.As(err, &resolution) ||
		errors.Is(err, runtimeconfig.ErrInvalid) ||
		errors.Is(err, runtimeconfig.ErrNotFound) ||
		errors.Is(err, credentials.ErrNotFound) ||
		errors.Is(err, credentials.ErrRuntimeCredentialNotFound)
}
