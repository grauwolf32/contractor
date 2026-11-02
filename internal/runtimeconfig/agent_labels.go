package runtimeconfig

import (
	"context"
	"slices"
	"sort"

	"github.com/grauwolf32/contractor/internal/apierror"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

var ErrAgentLabelNotApplicable = apierror.New(apierror.RuntimeAgentLabelNotApplicable,
	"RuntimeConfig label is not applicable to a Runtime Agent")

// LabelHolder is the label state of one Runtime Agent principal.
type LabelHolder struct {
	RuntimeAgentID string
	Labels         []string
	LabelRevision  uint64
}

// LabelHolders reads the Runtime Agent principals that hold Runtime labels.
// The principal registry (internal/agentprincipal) implements it over its own
// table, so label management keeps every principal's label set valid while it
// rebinds or deletes a label without owning the principals.
type LabelHolders interface {
	// ListLabelHolders returns every principal holding label, ordered by ID.
	ListLabelHolders(ctx context.Context, db persistencepostgres.DBTX, label string) ([]LabelHolder, error)
	// LockLabelHolders locks the given principals FOR UPDATE in ID order
	// inside tx and returns their current state; a missing one is
	// ErrNotFound.
	LockLabelHolders(ctx context.Context, tx pgx.Tx, runtimeAgentIDs []string) ([]LabelHolder, error)
}

// ValidateAgentLabels checks the stored form of a Runtime Agent principal's
// label set: a non-null Runtime label set in canonical order
// (runtimesettings.ValidateCanonicalRuntimeLabels).
func ValidateAgentLabels(labels []string) error {
	if labels == nil || runtimesettings.ValidateCanonicalRuntimeLabels(labels) != nil {
		return invalid("Runtime Agent labels must be a sorted set of at most %d unique valid labels other than default", runtimesettings.MaxRuntimeLabels)
	}
	return nil
}

// ValidateLockedAgentLabels requires each bound RuntimeConfig to carry a Worker
// setting (ErrAgentLabelNotApplicable) and the set to merge in one layer. The
// caller has already locked the bindings; a missing one is ErrUnknownLabel.
func ValidateLockedAgentLabels(ctx context.Context, db persistencepostgres.DBTX, labels []string) error {
	return validateLockedAgentLabelSets(ctx, db, [][]string{labels})
}

// validateLockedAgentLabelSets applies ValidateLockedAgentLabels to each label
// set in order and reports the first set that fails. It reads the bindings of
// every label the sets name with one statement and their versions with
// another, however many sets and labels there are.
func validateLockedAgentLabelSets(ctx context.Context, db persistencepostgres.DBTX, sets [][]string) error {
	repository := NewRepository(db)
	bindings, err := repository.GetBindingsByLabels(ctx, sortedUnion(sets...))
	if err != nil {
		return err
	}
	all := make([]Binding, 0, len(bindings))
	for _, binding := range bindings {
		all = append(all, binding)
	}
	versions, err := repository.versionsOfBindings(ctx, all)
	if err != nil {
		return err
	}
	for _, labels := range sets {
		selected := make([]Binding, len(labels))
		for index, label := range labels {
			selected[index] = bindings[label]
		}
		if err := validateAgentLayer(selected, versions); err != nil {
			return err
		}
	}
	return nil
}

// validateAgentLayer requires every binding's version, read by
// versionsOfBindings, to carry a Worker setting and the bindings to merge in
// one layer.
func validateAgentLayer(bindings []Binding, versions map[Ref]Version) error {
	entries := make([]LayerEntry, 0, len(bindings))
	for _, binding := range bindings {
		worker := versions[binding.Ref].Spec.Worker
		if !worker.LLMGateway.Present && !worker.Telemetry.Present && !worker.HTTPProxy.Present && !worker.Caido.Present && !worker.Interactsh.Present {
			return ErrAgentLabelNotApplicable
		}
		entries = append(entries, LayerEntry{
			Label: binding.Label, Ref: binding.Ref, Spec: Spec{Worker: worker},
		})
	}
	_, err := MergeSameLayer(entries)
	return err
}

func sameLabelHolders(left, right []LabelHolder) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index].RuntimeAgentID != right[index].RuntimeAgentID ||
			left[index].LabelRevision != right[index].LabelRevision ||
			!slices.Equal(left[index].Labels, right[index].Labels) {
			return false
		}
	}
	return true
}

func sortedUnion(groups ...[]string) []string {
	seen := make(map[string]struct{})
	for _, group := range groups {
		for _, value := range group {
			seen[value] = struct{}{}
		}
	}
	result := make([]string, 0, len(seen))
	for value := range seen {
		result = append(result, value)
	}
	sort.Strings(result)
	return result
}
