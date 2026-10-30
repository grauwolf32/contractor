package runtimesettings

import (
	"slices"

	"github.com/grauwolf32/contractor/internal/contracts"
)

// DefaultRuntimeLabel names the base RuntimeConfig binding. No Run selects it
// and no Runtime Agent holds it (RCF-R118).
const DefaultRuntimeLabel = "default"

// MaxRuntimeLabels bounds the Runtime labels one Run selects and one Runtime
// Agent holds (LIM-RCF-26, RCF-R119).
const MaxRuntimeLabels = 32

// maxRuntimeLabelBytes bounds one Runtime label (LIM-RCF-25).
const maxRuntimeLabelBytes = 63

// ValidRuntimeLabel reports whether label has the Runtime label grammar
// (RCF-R117). The default label has it too.
func ValidRuntimeLabel(label string) bool {
	return len(label) <= maxRuntimeLabelBytes && contracts.ValidIdentifier(label)
}

// CanonicalRuntimeLabels is the one definition of a Runtime label set, which
// a Run selects and a Runtime Agent holds (RCF-R117 to RCF-R120, LIM-RCF-26):
// at most MaxRuntimeLabels labels of the label grammar, none of them the
// default label and none twice. It returns the set in its canonical lexical
// order, because request order has no meaning; nil is the empty set.
// api/testdata/v1alpha1/runtime-label-set-cases.json holds the browser and
// the Runtime to the same verdicts.
func CanonicalRuntimeLabels(labels []string) ([]string, error) {
	if len(labels) > MaxRuntimeLabels {
		return nil, contracts.Invalidf("a Runtime label set has at most %d labels", MaxRuntimeLabels)
	}
	canonical := append([]string{}, labels...)
	slices.Sort(canonical)
	for index, label := range canonical {
		if !ValidRuntimeLabel(label) || label == DefaultRuntimeLabel {
			return nil, contracts.Invalidf("a Runtime label set has an invalid label or the default label")
		}
		if index > 0 && canonical[index-1] == label {
			return nil, contracts.Invalidf("a Runtime label set names a label twice")
		}
	}
	return canonical, nil
}

// ValidateCanonicalRuntimeLabels accepts a valid Runtime label set that is
// already in its canonical order, the form every stored and sent set has.
func ValidateCanonicalRuntimeLabels(labels []string) error {
	canonical, err := CanonicalRuntimeLabels(labels)
	if err != nil {
		return err
	}
	if !slices.Equal(canonical, labels) {
		return contracts.Invalidf("a Runtime label set is not in canonical order")
	}
	return nil
}
