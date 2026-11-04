package auditcontract

const ProposalsSchema = "contractor.audit.proposals.v1"
const ProposalContextSchema = "contractor.audit.proposal-context.v1"

// Proposals contains producer-local aliases only. The accepting owner assigns
// persisted identities after checking the exact caller-owned context.
type Proposals struct {
	Schema     string                    `json:"schema"`
	Hypotheses map[string]HypothesisSeed `json:"hypotheses"`
	Work       []InitialWork             `json:"work"`
}

type ProposalContext struct {
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
