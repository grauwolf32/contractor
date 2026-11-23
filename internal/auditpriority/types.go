// Package auditpriority validates independent checklist verdicts and computes a
// deterministic selection proposal. It performs no model, Artifact, database or
// Worker calls and grants no authority to admit or execute an Audit Round.
package auditpriority

const (
	MaxCandidates             = 1000
	MinTopN                   = 10
	MaxTopN                   = 1000
	DefaultTopN               = 10
	MaxIdentifierBytes        = 160
	MaxItemVersionBytes       = 160
	MaxContextEvidence        = 100
	MaxVerdictBytes           = 8 << 10
	MaxRationaleBytes         = 2 << 10
	MaxEvidenceIDs            = 16
	MaxMissingContextEntries  = 16
	MaxMissingContextBytes    = 256
	MaxSelectionBytes         = 16 << 20
	PoolSchema                = "contractor.audit.priority-candidates.v1"
	SelectionSchema           = "contractor.audit.priority-selection.v1"
	CodeInvalidPolicy         = "priority_invalid_policy"
	CodeInvalidPool           = "priority_invalid_pool"
	CodeInvalidBinding        = "priority_invalid_binding"
	CodeInvalidVerdict        = "priority_invalid_verdict"
	CodeIncompleteRanking     = "priority_ranking_incomplete"
	CodeInvalidSelection      = "priority_invalid_selection"
	CodeNoRemainingCandidates = "no_remaining_candidates"
	CodeFewerCandidates       = "fewer_candidates_remaining"
	CodeDeferredTopN          = "deferred_top_n"
)

// Error contains a closed diagnostic code, never model or context text.
type Error struct{ Code string }

func (e *Error) Error() string  { return "audit priority: " + e.Code }
func invalid(code string) error { return &Error{Code: code} }

type Priority string

const (
	PriorityCritical Priority = "critical"
	PriorityHigh     Priority = "high"
	PriorityMedium   Priority = "medium"
	PriorityLow      Priority = "low"
)

type Confidence string

const (
	ConfidenceHigh   Confidence = "high"
	ConfidenceMedium Confidence = "medium"
	ConfidenceLow    Confidence = "low"
)

// Policy distinguishes omission from an explicitly invalid zero. This is a
// pure resolution input; profile and public API authoring are separate tasks.
type Policy struct {
	DefaultTopN *int
	MaxTopN     *int
}

type ItemIdentity struct {
	Key     string
	Version string
}

type Candidate struct {
	ID          string `json:"id"`
	ItemKey     string `json:"item_key"`
	ItemVersion string `json:"item_version"`
}

type Pool struct {
	Schema          string      `json:"schema"`
	InventoryDigest string      `json:"inventory_digest"`
	Candidates      []Candidate `json:"candidates"`
}

// CycleBinding binds semantic input, not physical allocation or Stage identity.
// The trusted caller must obtain digests from its exact retained snapshots.
type CycleBinding struct {
	CycleID           string `json:"cycle_id"`
	InventoryDigest   string `json:"inventory_digest"`
	PoolDigest        string `json:"pool_digest"`
	ContextDigest     string `json:"context_digest"`
	PolicyDigest      string `json:"policy_digest"`
	PromptDigest      string `json:"prompt_digest"`
	ModelConfigDigest string `json:"model_config_digest"`
	TopN              int    `json:"top_n"`
}

// BoundVerdict is validated data, not proof of a durable ranking receipt. The
// future caller must verify Run output, journal and ownership before admission.
type BoundVerdict struct {
	Cycle       CycleBinding `json:"cycle"`
	CandidateID string       `json:"candidate_id"`
	Verdict     Verdict      `json:"verdict"`
}

type SelectionRow struct {
	Candidate Candidate `json:"candidate"`
	Verdict   Verdict   `json:"verdict"`
	Rank      int       `json:"rank"`
	Selected  bool      `json:"selected"`
	Code      string    `json:"code"`
}

// Selection is an immutable-by-convention calculation over a complete pool.
// No receipt acceptance, budget reservation, approval or dispatch is implied.
type Selection struct {
	Schema        string         `json:"schema"`
	Cycle         CycleBinding   `json:"cycle"`
	Rows          []SelectionRow `json:"rows"`
	SelectedCount int            `json:"selected_count"`
	DeferredCount int            `json:"deferred_count"`
	Code          string         `json:"code"`
}
