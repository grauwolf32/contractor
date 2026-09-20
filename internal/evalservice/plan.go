package evalservice

import (
	"encoding/json"
	"time"

	"github.com/grauwolf32/contractor/internal/evaldomain"
)

// Preflight is populated by read-only catalog/artifact resolution. Unsupported
// members remain in the matrix. No arbitrary client-supplied capability becomes
// a verified pin merely by appearing in a dataset or a binding.
type Preflight struct {
	Pins         map[string]Pin
	Capabilities []string
	Cases        map[string]Eligibility
	Snapshot     json.RawMessage
}

type Pin struct {
	Value  *string `json:"value"`
	Origin string  `json:"origin"`
}

type Eligibility struct {
	State  string
	Reason *string
}

type Resource struct {
	Path, Kind string
	Document   evaldomain.Frozen `json:"-"`
}

type PlanBundle struct {
	Private   map[string]evaldomain.Frozen   `json:"-"`
	Plan      evaldomain.Frozen              `json:"-"`
	Setup     json.RawMessage                `json:"-"`
	Resources []Resource                     `json:"-"`
	Inputs    map[string]evaldomain.Artifact `json:"-"`
	Cases     map[string]evaldomain.Case     `json:"-"`
}

type documentRef struct {
	Resource string `json:"resource"`
	SHA256   string `json:"sha256"`
}

type blobRef struct {
	Resource  string `json:"resource"`
	SHA256    string `json:"sha256"`
	MediaType string `json:"media_type"`
	SizeBytes int64  `json:"size_bytes"`
}

func jsonBytes(value any) ([]byte, error) {
	b, err := json.Marshal(value)
	if err != nil {
		return nil, evaldomain.Failure("eval_invalid")
	}
	return b, nil
}

func hashJSON(value any) (string, error) {
	b, err := jsonBytes(value)
	if err != nil {
		return "", err
	}
	return evaldomain.Digest(b), nil
}

func observedPin(value string) Pin { return Pin{Value: &value, Origin: "observed"} }

// BuildPlan assembles immutable portable documents without execution effects.
// Each builder owns one document; only the caller persists the complete bundle.
func BuildPlan(portableID string, createdAt time.Time, draft evaldomain.Draft, dataset evaldomain.DatasetInput, preflight map[string]Preflight) (PlanBundle, error) {
	b := planBuilder{
		bundle: PlanBundle{
			Private:   map[string]evaldomain.Frozen{},
			Inputs:    map[string]evaldomain.Artifact{},
			Cases:     map[string]evaldomain.Case{},
			Resources: []Resource{},
		},
		dataset:   dataset,
		preflight: preflight,
	}
	raw, err := jsonBytes(draft)
	if err != nil {
		return b.bundle, err
	}
	// Deep-copy mutable authoring slices before pinning check implementations.
	if err = evaldomain.DecodeInto("Draft", raw, &b.draft); err != nil {
		return b.bundle, err
	}
	raw, err = jsonBytes(dataset)
	if err != nil {
		return b.bundle, err
	}
	if err = evaldomain.Validate("DatasetInput", raw); err != nil {
		return b.bundle, err
	}
	if dataset.DatasetID != draft.Dataset.ID {
		return b.bundle, evaldomain.Failure("eval_member_conflict")
	}
	checks, truth, err := b.buildChecks()
	if err != nil {
		return b.bundle, err
	}
	cases, caseRefs, err := b.buildCases(truth)
	if err != nil {
		return b.bundle, err
	}
	suite, err := b.add("suite.json", portableSuiteSchema, portableSuite{
		SchemaVersion: portableSuiteSchema,
		ID:            draft.Dataset.ID,
		Cases:         caseRefs,
		Scoring:       suiteScoring{Decision: "all_required_pass", Checks: checks},
		Provenance:    suiteProvenance{Sources: []documentRef{}},
	})
	if err != nil {
		return b.bundle, err
	}
	variants, pins, err := b.buildVariants(cases)
	if err != nil {
		return b.bundle, err
	}
	members, order, err := b.buildMembers(portableID, cases, caseRefs, variants)
	if err != nil {
		return b.bundle, err
	}
	comparison, extensions := evaldomain.PortableComparison(draft.Comparison)
	budgets := portableBudgets{
		MaxMembers:             draft.Budgets.MaxMembers,
		MaxInFlight:            draft.Budgets.MaxInFlight,
		WallMS:                 draft.Budgets.WallMS,
		MaxObservedTotalTokens: draft.Budgets.MaxObservedTotalTokens,
	}
	publication := publication{Provider: managedProvider, Connection: managedConnection, Enabled: true}
	experiment, err := b.add("experiment.json", portableExperimentSchema, portableExperiment{
		SchemaVersion: portableExperimentSchema, ID: portableID, Suites: []documentRef{suite}, Variants: variants, Repetitions: draft.Repetitions,
		Order: portableOrder{Kind: draft.Order.Kind, Seed: draft.Order.Seed}, Budgets: budgets, Comparison: comparison, Publication: publication, Extensions: extensions,
	})
	if err != nil {
		return b.bundle, err
	}
	plan := portablePlan{
		SchemaVersion: portablePlanSchema, ExperimentID: portableID, CreatedAt: createdAt.UTC().Format(time.RFC3339Nano), ExperimentRef: experiment,
		Suites: []suiteReference{{ID: draft.Dataset.ID, Ref: suite}}, Variants: variants, Pins: pins, Members: members, ExecutionOrder: order,
		Budgets: budgets, Comparison: comparison, Publication: publication,
	}
	raw, err = jsonBytes(plan)
	if err != nil {
		return b.bundle, err
	}
	b.bundle.Plan, err = evaldomain.Freeze(portablePlanSchema, raw)
	if err != nil {
		return b.bundle, err
	}
	b.bundle.Setup, err = jsonBytes(b.draft.Setup())
	if err == nil {
		err = evaldomain.Validate("ExperimentSetup", b.bundle.Setup)
	}
	return b.bundle, err
}
