package auditcontract

import "github.com/grauwolf32/contractor/internal/contracts"

const ProposalsSchema = "contractor.audit.proposals.v1"
const ProposalContextSchema = "contractor.audit.proposal-context.v1"

// Proposals contains producer-local aliases only. The accepting owner assigns
// persisted identities after checking the exact caller-owned context.
type Proposals struct {
	SelectedMaterials map[string]contracts.ArtifactRef `json:"selected_materials,omitempty"`
	Schema            string                           `json:"schema"`
	Hypotheses        map[string]ProposedHypothesis    `json:"hypotheses"`
	Work              []InitialWork                    `json:"work"`
}

type ProposedHypothesis struct {
	Statement string            `json:"statement"`
	Scope     map[string]string `json:"scope"`
	Grounds   string            `json:"grounds"`
	Sources   []string          `json:"sources,omitempty"`
}

// ProposalBinding is frozen in the existing Audit context namespace at Run
// creation. The pinned program resolves both roles; input/output names are free.
type ProposalBinding struct {
	Schema  string                `json:"schema"`
	RunID   string                `json:"run_id"`
	Context contracts.ArtifactRef `json:"context"`
	Result  contracts.ArtifactRef `json:"result"`
}

const ProposalBindingSchema = "contractor.audit.proposal-binding.v1"
const ProposalBindingName = "hypothesis-proposals"

type ProposalContext struct {
	RunID        string              `json:"run_id"`
	InputSources map[string]Material `json:"input_sources"`

	Schema       string                `json:"schema"`
	Program      Program               `json:"program"`
	Materials    map[string]Material   `json:"materials"`
	Observations []ProposalObservation `json:"observations,omitempty"`
}

type ProposalObservation struct {
	TaskID     string      `json:"task_id"`
	Key        string      `json:"key"`
	Gap        string      `json:"gap,omitempty"`
	Hypothesis *Hypothesis `json:"hypothesis,omitempty"`
}
