package runtimeconfig

import (
	"context"
	"errors"
	"slices"
	"sort"

	"github.com/grauwolf32/contractor/internal/apierror"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// maxAgentLabels bounds the Runtime labels one Runtime Agent principal holds.
const maxAgentLabels = 32

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

// ValidateAgentLabels checks the shape of a Runtime Agent principal's label
// set: at most maxAgentLabels valid labels, sorted and unique, never the
// default label.
func ValidateAgentLabels(labels []string) error {
	if labels == nil || len(labels) > maxAgentLabels {
		return invalid("Runtime Agent labels must be a non-null bounded set")
	}
	previous := ""
	for _, label := range labels {
		if ValidateLabel(label) != nil || label == DefaultLabel || label <= previous {
			return invalid("Runtime Agent labels must be sorted, unique, valid, and exclude default")
		}
		previous = label
	}
	return nil
}

// LockAgentLabels share-locks the bindings of an agent label set in db's
// transaction and requires every bound RuntimeConfig to carry a Worker
// setting (ErrAgentLabelNotApplicable) and the set to merge in one layer.
func LockAgentLabels(ctx context.Context, db persistencepostgres.DBTX, labels []string) error {
	if err := ValidateAgentLabels(labels); err != nil {
		return err
	}
	repository := NewRepository(db)
	bindings, err := repository.LockBindings(ctx, labels)
	if err != nil {
		return err
	}
	return validateAgentLayer(ctx, repository, bindings)
}

// ValidateLockedAgentLabels applies the LockAgentLabels rules to bindings the
// caller has already locked; a missing binding is ErrUnknownLabel.
func ValidateLockedAgentLabels(ctx context.Context, db persistencepostgres.DBTX, labels []string) error {
	repository := NewRepository(db)
	bindings := make([]Binding, 0, len(labels))
	for _, label := range labels {
		binding, err := repository.GetBinding(ctx, label)
		if err != nil {
			if errors.Is(err, ErrNotFound) {
				return ErrUnknownLabel
			}
			return err
		}
		bindings = append(bindings, binding)
	}
	return validateAgentLayer(ctx, repository, bindings)
}

func validateAgentLayer(ctx context.Context, repository *Repository, bindings []Binding) error {
	entries := make([]LayerEntry, 0, len(bindings))
	for _, binding := range bindings {
		version, err := repository.GetVersionByRef(ctx, binding.Ref)
		if err != nil {
			return err
		}
		worker := version.Spec.Worker
		if !worker.LLMGateway.Present && !worker.Telemetry.Present && !worker.HTTPProxy.Present && !worker.Caido.Present {
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
