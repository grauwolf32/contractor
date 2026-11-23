package runtimesettings

// Validators shared by the Runtime settings and their provenance: Runtime
// endpoints, CA bundles and sorted label bindings.

import (
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/cabundle"
)

func validateCABundle(owner, value string) error {
	if err := cabundle.Validate(value); err != nil {
		return contracts.Invalidf("%s CA bundle is invalid", owner)
	}
	return nil
}

func validateProvenanceBindings(field string, values []RuntimeLabelBindingProvenance) error {
	if values == nil {
		return contracts.Invalidf("provenance %s must be a non-null array", field)
	}
	labels := make([]string, 0, len(values))
	for _, value := range values {
		labels = append(labels, value.Label)
	}
	if err := ValidateCanonicalRuntimeLabels(labels); err != nil {
		return contracts.Invalidf("provenance %s contains invalid or unsorted bindings", field)
	}
	for _, value := range values {
		if value.BindingRevision == 0 {
			return contracts.Invalidf("provenance %s contains invalid or unsorted bindings", field)
		}
		if err := value.Config.Validate(); err != nil {
			return err
		}
	}
	return nil
}
