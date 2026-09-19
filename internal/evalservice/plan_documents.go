package evalservice

import (
	"encoding/json"

	"github.com/grauwolf32/contractor/internal/evaldomain"
)

const (
	portableCaseSchema       = "playground.case/v2"
	portableSuiteSchema      = "playground.suite/v2"
	portableBindingSchema    = "playground.binding/v1"
	portableExperimentSchema = "playground.experiment/v1"
	portablePlanSchema       = "playground.plan/v1"
	managedProvider          = "contractor@1"
	managedConnection        = "managed"
)

type portableOutput struct {
	MediaTypes []string `json:"media_types"`
	Required   bool     `json:"required"`
}
type caseEvaluation struct {
	GroundTruth map[string]blobRef         `json:"ground_truth"`
	Assertions  map[string]json.RawMessage `json:"assertions"`
}
type datasetProvenance struct {
	ID       string  `json:"id"`
	Revision string  `json:"revision"`
	SHA256   *string `json:"sha256"`
}
type caseProvenance struct {
	Sources  []documentRef       `json:"sources"`
	Datasets []datasetProvenance `json:"datasets"`
}
type portableCase struct {
	SchemaVersion string                    `json:"schema_version"`
	ID            string                    `json:"id"`
	Task          evaldomain.Task           `json:"task"`
	Inputs        map[string]blobRef        `json:"inputs"`
	Requires      []string                  `json:"requires"`
	Outputs       map[string]portableOutput `json:"outputs"`
	Evaluation    caseEvaluation            `json:"evaluation"`
	Provenance    caseProvenance            `json:"provenance"`
}
type portableCheck struct {
	ID                   string            `json:"id"`
	Scorer               string            `json:"scorer"`
	ImplementationSHA256 string            `json:"implementation_sha256"`
	Parameters           map[string]string `json:"parameters"`
	GroundTruthRole      *string           `json:"ground_truth_role"`
	Required             bool              `json:"required"`
	AllowNotApplicable   bool              `json:"allow_not_applicable"`
}
type suiteScoring struct {
	Decision string          `json:"decision"`
	Checks   []portableCheck `json:"checks"`
}
type suiteProvenance struct {
	Sources []documentRef `json:"sources"`
}
type portableSuite struct {
	SchemaVersion string          `json:"schema_version"`
	ID            string          `json:"id"`
	Cases         []documentRef   `json:"cases"`
	Scoring       suiteScoring    `json:"scoring"`
	Provenance    suiteProvenance `json:"provenance"`
}
type portableBudgets struct {
	MaxMembers             int    `json:"max_members"`
	MaxInFlight            int    `json:"max_in_flight"`
	WallMS                 int64  `json:"wall_ms"`
	MaxObservedTotalTokens *int64 `json:"max_observed_total_tokens"`
}
type publication struct {
	Provider   string `json:"provider"`
	Connection string `json:"connection"`
	Enabled    bool   `json:"enabled"`
}
type variantReference struct {
	ID      string      `json:"id"`
	Binding documentRef `json:"binding"`
}
type suiteReference struct {
	ID  string      `json:"id"`
	Ref documentRef `json:"ref"`
}
type planMember struct {
	evaldomain.PublicMember
	Reason *string `json:"reason"`
}
type portableOrder struct {
	Kind string `json:"kind"`
	Seed *int64 `json:"seed"`
}
type portableExperiment struct {
	SchemaVersion string                                  `json:"schema_version"`
	ID            string                                  `json:"id"`
	Suites        []documentRef                           `json:"suites"`
	Variants      []variantReference                      `json:"variants"`
	Repetitions   int                                     `json:"repetitions"`
	Order         portableOrder                           `json:"order"`
	Budgets       portableBudgets                         `json:"budgets"`
	Comparison    evaldomain.PortableComparisonPolicy     `json:"comparison"`
	Publication   publication                             `json:"publication"`
	Extensions    evaldomain.PortableExperimentExtensions `json:"extensions"`
}
type portablePlan struct {
	SchemaVersion  string                              `json:"schema_version"`
	ExperimentID   string                              `json:"experiment_id"`
	CreatedAt      string                              `json:"created_at"`
	ExperimentRef  documentRef                         `json:"experiment_ref"`
	Suites         []suiteReference                    `json:"suites"`
	Variants       []variantReference                  `json:"variants"`
	Pins           map[string]map[string]Pin           `json:"pins"`
	Members        []planMember                        `json:"members"`
	ExecutionOrder []string                            `json:"execution_order"`
	Budgets        portableBudgets                     `json:"budgets"`
	Comparison     evaldomain.PortableComparisonPolicy `json:"comparison"`
	Publication    publication                         `json:"publication"`
}
type bindingSettings struct {
	Variant  evaldomain.Variant `json:"variant"`
	Snapshot json.RawMessage    `json:"snapshot"`
}
type portableBinding struct {
	SchemaVersion string            `json:"schema_version"`
	ID            string            `json:"id"`
	Provider      string            `json:"provider"`
	Capabilities  []string          `json:"capabilities"`
	Connection    string            `json:"connection"`
	Settings      bindingSettings   `json:"settings"`
	InputMapping  map[string]string `json:"input_mapping"`
	OutputMapping map[string]string `json:"output_mapping"`
	Normalizers   []documentRef     `json:"normalizers"`
}
