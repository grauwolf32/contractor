package runtimeconfig

import (
	"context"
	"errors"
	"fmt"
	"sort"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

const MaximumRunLabels = 32

// PinnedLabel records the exact mutable binding observation that selected one
// immutable RuntimeConfig version. Explicit is false only for the mandatory
// default layer.
type PinnedLabel struct {
	Label           string `json:"label"`
	Explicit        bool   `json:"explicit"`
	BindingRevision uint64 `json:"bindingRevision"`
	Config          Ref    `json:"config"`
}

// RunSnapshot is immutable Run input. Credential identifiers are retained for
// deletion fencing and allocation-time resolution, but secret material never
// crosses this boundary.
type RunSnapshot struct {
	Default              PinnedLabel   `json:"default"`
	Labels               []PinnedLabel `json:"labels"`
	LLMCredentialIDs     []string      `json:"llmCredentialIds"`
	RuntimeCredentialIDs []string      `json:"runtimeCredentialIds"`
}

// NormalizeRunLabels validates the caller-selected set and returns its one
// canonical order. Request order is deliberately not semantic.
func NormalizeRunLabels(source []string) ([]string, error) {
	if len(source) > MaximumRunLabels {
		return nil, invalid("Run selects too many Runtime labels")
	}
	seen := make(map[string]struct{}, len(source))
	result := append([]string{}, source...)
	for _, label := range result {
		if validateLabel(label) != nil || label == DefaultLabel {
			return nil, invalid("Run Runtime label is invalid or reserved")
		}
		if _, duplicate := seen[label]; duplicate {
			return nil, invalid("Run Runtime labels contain a duplicate")
		}
		seen[label] = struct{}{}
	}
	sort.Strings(result)
	return result, nil
}

// PinRunSnapshot must be called with the owning PostgreSQL transaction and a
// credential lookup bound through that same transaction. It locks all
// selected binding rows in lexical order, validates exact immutable versions
// and same-layer conflicts, then returns a compact safe snapshot for the caller
// to insert with the WorkflowRun in the same transaction. Exact bodies remain
// reachable through immutable refs in runtime_config_versions.
func PinRunSnapshot(
	ctx context.Context,
	tx pgx.Tx,
	explicitLabels []string,
	runtimeCredentials RuntimeCredentialValidator,
	llmCredentials TransactionLLMCredentialLookup,
) (RunSnapshot, error) {
	labels, err := NormalizeRunLabels(explicitLabels)
	if err != nil {
		return RunSnapshot{}, err
	}
	allLabels := append([]string{DefaultLabel}, labels...)
	if tx == nil {
		return RunSnapshot{}, invalid("Run RuntimeConfig pinning requires a PostgreSQL transaction")
	}
	repository := NewRepository(tx)
	bindings, err := repository.LockBindings(ctx, allLabels)
	if err != nil {
		return RunSnapshot{}, err
	}
	if len(bindings) != len(allLabels) {
		return RunSnapshot{}, ErrNotFound
	}

	byLabel := make(map[string]Binding, len(bindings))
	for _, binding := range bindings {
		byLabel[binding.Label] = binding
	}
	result := RunSnapshot{Labels: make([]PinnedLabel, 0, len(labels))}
	explicitEntries := make([]LayerEntry, 0, len(labels))
	llmIDs := make(map[string]struct{})
	runtimeIDs := make(map[string]struct{})

	for _, label := range allLabels {
		binding, ok := byLabel[label]
		if !ok {
			return RunSnapshot{}, ErrNotFound
		}
		version, getErr := repository.GetVersionByRef(ctx, binding.Ref)
		if getErr != nil {
			return RunSnapshot{}, getErr
		}
		if validateErr := validateSpecRuntimeCredentials(ctx, version.Spec, runtimeCredentials); validateErr != nil {
			return RunSnapshot{}, validateErr
		}
		if validateErr := validateRunLLMCredential(ctx, version.Spec, llmCredentials); validateErr != nil {
			return RunSnapshot{}, validateErr
		}
		collectCredentialIDs(version.Spec, llmIDs, runtimeIDs)
		pin := PinnedLabel{
			Label: label, Explicit: label != DefaultLabel,
			BindingRevision: binding.Revision, Config: binding.Ref,
		}
		if label == DefaultLabel {
			result.Default = pin
			continue
		}
		result.Labels = append(result.Labels, pin)
		explicitEntries = append(explicitEntries, LayerEntry{
			Label: label, Ref: binding.Ref, Spec: version.Spec,
		})
	}
	if _, err := MergeSameLayer(explicitEntries); err != nil {
		return RunSnapshot{}, err
	}
	result.LLMCredentialIDs = sortedSet(llmIDs)
	result.RuntimeCredentialIDs = sortedSet(runtimeIDs)
	if err := result.Validate(); err != nil {
		return RunSnapshot{}, err
	}
	return result, nil
}

func (s RunSnapshot) Validate() error {
	if s.Default.Label != DefaultLabel || s.Default.Explicit || s.Default.BindingRevision == 0 ||
		validateRef(s.Default.Config) != nil || len(s.Labels) > MaximumRunLabels {
		return invalid("Run RuntimeConfig snapshot default is invalid")
	}
	previous := ""
	for _, pin := range s.Labels {
		if !pin.Explicit || pin.Label == DefaultLabel || pin.Label <= previous ||
			validateLabel(pin.Label) != nil || pin.BindingRevision == 0 || validateRef(pin.Config) != nil {
			return invalid("Run RuntimeConfig label snapshot is invalid")
		}
		previous = pin.Label
	}
	if err := validateCredentialIDSet(s.LLMCredentialIDs, true); err != nil {
		return err
	}
	return validateCredentialIDSet(s.RuntimeCredentialIDs, false)
}

func (s RunSnapshot) ExplicitLabels() []string {
	result := make([]string, len(s.Labels))
	for index := range s.Labels {
		result[index] = s.Labels[index].Label
	}
	return result
}

func (s RunSnapshot) Clone() RunSnapshot {
	result := s
	result.Labels = append([]PinnedLabel{}, s.Labels...)
	result.LLMCredentialIDs = append([]string{}, s.LLMCredentialIDs...)
	result.RuntimeCredentialIDs = append([]string{}, s.RuntimeCredentialIDs...)
	return result
}

func BuiltInRunSnapshot() RunSnapshot {
	return RunSnapshot{
		Default: PinnedLabel{
			Label: DefaultLabel, BindingRevision: 1,
			Config: Ref{Name: BuiltInName, Version: BuiltInVersion, Digest: BuiltInDigest},
		},
		Labels: []PinnedLabel{}, LLMCredentialIDs: []string{}, RuntimeCredentialIDs: []string{},
	}
}

func validateRunLLMCredential(
	ctx context.Context, spec Spec, lookup config.CredentialLookup,
) error {
	patch := spec.Worker.LLMGateway
	if !patch.Present || !patch.Credential.Present || patch.Credential.Clear {
		return nil
	}
	if lookup == nil {
		return invalid("RuntimeConfig LLM credential validator is not configured")
	}
	metadata, err := lookup.LookupLLMCredential(ctx, patch.Credential.Value)
	if err != nil {
		if contextError := ctx.Err(); contextError != nil {
			return contextError
		}
		return persistencepostgres.WrapError(
			invalid("RuntimeConfig references an unavailable LLM credential").Error(), errors.Join(ErrInvalid, err),
		)
	}
	if metadata.Ref.CredentialID != patch.Credential.Value {
		return invalid("RuntimeConfig LLM credential identity does not match")
	}
	if patch.Gateway.Present && metadata.LLMGateway != patch.Gateway.Value {
		return invalid("RuntimeConfig LLM credential is bound to another Gateway")
	}
	return nil
}

func collectCredentialIDs(spec Spec, llm, runtime map[string]struct{}) {
	if patch := spec.Worker.LLMGateway; patch.Present && patch.Credential.Present && !patch.Credential.Clear {
		llm[patch.Credential.Value] = struct{}{}
	}
	for _, credentialID := range []string{
		atomicTelemetryCredential(spec.Worker.Telemetry),
		atomicProxyCredential(spec.Worker.HTTPProxy),
		atomicCaidoCredential(spec.Worker.Caido),
		atomicTelemetryCredential(spec.Planner.Telemetry),
	} {
		if credentialID != "" {
			runtime[credentialID] = struct{}{}
		}
	}
}

func atomicTelemetryCredential(value AtomicPatch[TelemetryConfig]) string {
	if value.Present && !value.Clear {
		return value.Value.Credential
	}
	return ""
}

func atomicProxyCredential(value AtomicPatch[HTTPProxyConfig]) string {
	if value.Present && !value.Clear {
		return value.Value.Credential
	}
	return ""
}

func atomicCaidoCredential(value AtomicPatch[CaidoConfig]) string {
	if value.Present && !value.Clear {
		return value.Value.Credential
	}
	return ""
}

func sortedSet(source map[string]struct{}) []string {
	result := make([]string, 0, len(source))
	for value := range source {
		result = append(result, value)
	}
	sort.Strings(result)
	return result
}

func validateCredentialIDSet(values []string, llm bool) error {
	previous := ""
	for _, value := range values {
		valid := validateID("Runtime credential ID", value, 128) == nil
		if llm {
			valid = (&contracts.LLMCredentialRef{CredentialID: value}).Validate() == nil
		}
		if !valid || value <= previous {
			return invalid("Run RuntimeConfig credential snapshot is invalid")
		}
		previous = value
	}
	return nil
}

func (s RunSnapshot) String() string {
	return fmt.Sprintf("RuntimeConfigSnapshot(default=%s, labels=%d)", s.Default.Config.String(), len(s.Labels))
}
