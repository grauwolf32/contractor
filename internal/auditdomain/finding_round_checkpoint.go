package auditdomain

import "github.com/grauwolf32/contractor/internal/helpers/contentdigest"

const FindingRoundBudgetStateSchema = "contractor.audit.finding-round-budget.v1"
const MaximumFindingRoundBudgetStateBytes = 4 << 10

type FindingBudgetDocumentState struct {
	Name    string `json:"name" strictjson:"required"`
	Entries int64  `json:"entries" strictjson:"required"`
	Bytes   int64  `json:"bytes" strictjson:"required"`
	Nodes   int64  `json:"nodes" strictjson:"required"`
	Depth   int    `json:"depth" strictjson:"required"`
}
type FindingRoundBudgetState struct {
	Schema         string                       `json:"schema" strictjson:"required"`
	ShapeDigest    string                       `json:"shape_digest" strictjson:"required"`
	Checks         int                          `json:"checks" strictjson:"required"`
	Full           bool                         `json:"full" strictjson:"required"`
	GeneratedBytes int64                        `json:"generated_bytes" strictjson:"required"`
	Documents      []FindingBudgetDocumentState `json:"documents" strictjson:"required"`
}

func findingBudgetDocuments(round *findingRound) []*boundedDocument {
	return []*boundedDocument{&round.proposals, &round.canonical, &round.coverage, &round.worklist, &round.execution}
}
func (b *FindingRoundBudget) shapeDigest() (string, error) {
	data, err := canonicalJSON(struct {
		Maximum int               `json:"maximum"`
		Shape   FindingRoundShape `json:"shape"`
	}{b.maximumChecks, b.shape})
	return DigestBytes(data), err
}
func (b *FindingRoundBudget) Checkpoint() (FindingRoundBudgetState, error) {
	digest, err := b.shapeDigest()
	if err != nil {
		return FindingRoundBudgetState{}, err
	}
	state := FindingRoundBudgetState{Schema: FindingRoundBudgetStateSchema, ShapeDigest: digest, Checks: b.round.checks, Full: b.full, GeneratedBytes: b.round.generated, Documents: []FindingBudgetDocumentState{}}
	for _, document := range findingBudgetDocuments(&b.round) {
		state.Documents = append(state.Documents, FindingBudgetDocumentState{Name: document.name, Entries: document.entries, Bytes: document.extent.bytes, Nodes: document.extent.nodes, Depth: document.extent.depth})
	}
	return state, validateFindingRoundBudgetState(state)
}
func EncodeFindingRoundBudgetState(value FindingRoundBudgetState) ([]byte, error) {
	if err := validateFindingRoundBudgetState(value); err != nil {
		return nil, err
	}
	return canonicalJSON(value)
}
func DecodeFindingRoundBudgetState(data []byte) (FindingRoundBudgetState, error) {
	if len(data) > MaximumFindingRoundBudgetStateBytes {
		return FindingRoundBudgetState{}, invalid(CodeLimitExceeded, "round_budget_state")
	}
	return decodeDocument(data, validateFindingRoundBudgetState)
}
func validateFindingRoundBudgetState(value FindingRoundBudgetState) error {
	names := []string{"proposal_inventory", "canonical_inventory", "coverage", "worklist", "execution_manifest"}
	if value.Schema != FindingRoundBudgetStateSchema || !contentdigest.Valid(value.ShapeDigest) || value.Checks < 0 || value.Checks > MaximumItems || value.GeneratedBytes < 0 || value.GeneratedBytes > MaximumGeneratedBytes || len(value.Documents) != len(names) {
		return invalid(CodeInvalid, "round_budget_state")
	}
	for i, document := range value.Documents {
		if document.Name != names[i] || document.Entries < 0 || document.Entries > int64(value.Checks) || i > 0 && document.Entries != int64(value.Checks) || document.Bytes < 0 || document.Bytes > MaximumGeneratedBytes || document.Nodes < 0 || document.Nodes > MaximumJSONNodes*MaximumItems || document.Depth < 0 || document.Depth > MaximumJSONDepth || document.Entries == 0 && (document.Bytes != 0 || document.Nodes != 0 || document.Depth != 0) {
			return invalid(CodeInvalid, "round_budget_state.document")
		}
	}
	encoded, err := canonicalJSON(value)
	if err != nil || len(encoded) > MaximumFindingRoundBudgetStateBytes {
		return invalid(CodeLimitExceeded, "round_budget_state")
	}
	return nil
}
func ResumeFindingRoundBudget(maximumChecks int, shape FindingRoundShape, state FindingRoundBudgetState) (*FindingRoundBudget, error) {
	if err := validateFindingRoundBudgetState(state); err != nil {
		return nil, err
	}
	budget, err := NewFindingRoundBudget(maximumChecks, shape)
	if err != nil {
		return nil, err
	}
	digest, err := budget.shapeDigest()
	if err != nil || digest != state.ShapeDigest || state.Checks > budget.maximumChecks {
		return nil, invalid(CodeInvalid, "round_budget_state.shape")
	}
	budget.round.checks = state.Checks
	budget.round.generated = state.GeneratedBytes
	budget.full = state.Full
	for i, document := range findingBudgetDocuments(&budget.round) {
		value := state.Documents[i]
		document.entries = value.Entries
		document.extent = jsonExtent{bytes: value.Bytes, nodes: value.Nodes, depth: value.Depth}
		if !shape.Paged && (document.bytesTotal() > MaximumDocumentBytes || document.shell.nodes+document.extent.nodes > MaximumJSONNodes || max(document.shell.depth, entryArrayDepth+document.extent.depth) > MaximumJSONDepth) {
			return nil, invalid(CodeLimitExceeded, "round_budget_state.document")
		}
	}
	if shape.Paged && findingPagedBudgetBytes(budget.round) > MaximumGeneratedBytes {
		return nil, invalid(CodeLimitExceeded, "round_budget_state.metadata")
	}
	return budget, nil
}

type FindingCheckAdmission struct {
	Admitted      bool
	Unschedulable string
	Inventory     Inventory
}

// AdmitCheck measures the actual bounded source/task/dependency/route tuple.
// Different routes have independent execution-entry sizes; no first-item
// multiplication or selected-source map participates in this durable budget.
func (b *FindingRoundBudget) AdmitCheck(source []byte, options FindingCheckInventoryOptions) (FindingCheckAdmission, error) {
	if b.Full() {
		return FindingCheckAdmission{}, nil
	}
	inventory, err := BuildFindingCheckInventory(source, options)
	if err != nil {
		return FindingCheckAdmission{}, err
	}
	work := inventory.Worklist.Items[0]
	work.Ordinal = MaximumItems - 1
	execution := inventory.ExecutionManifest.Items[0]
	execution.Ordinal = MaximumItems - 1
	execution.Inputs = append([]ExactInput{}, options.Check.ExecutionInputs...)
	ref := inventory.Tasks[0].Document.SourceRef.Clone()
	ref.Namespace = b.shape.TaskNamespace
	ref.Name = placeholderPackageID
	revision := b.shape.TaskRevision
	ref.Revision = &revision
	execution.TaskRef = &ref
	execution.TaskPackageID = placeholderPackageID
	execution.TaskPackageDigest = placeholderDigest
	entry, err := measureJSON(inventory.Tasks[0].Document)
	if err != nil {
		return FindingCheckAdmission{}, err
	}
	workExtent, err := measureJSON(work)
	if err != nil {
		return FindingCheckAdmission{}, err
	}
	executionExtent, err := measureJSON(execution)
	if err != nil {
		return FindingCheckAdmission{}, err
	}
	coverage, err := measureJSON(inventory.Coverage.Rows[0])
	if err != nil {
		return FindingCheckAdmission{}, err
	}
	candidate := measuredCandidate{entry: jsonExtentOf(source), ordinals: []int{0}, worklist: workExtent, execution: executionExtent, checks: []measuredCheck{{canonical: entry, coverage: coverage, tasks: int64(len(inventory.Tasks[0].Package))}}}
	if options.Check.PreparedMetadataBytes < 0 || options.Check.PreparedMetadataBytes > 1<<20 {
		return FindingCheckAdmission{}, invalid(CodeLimitExceeded, "prepared_round.page_metadata")
	}
	if b.shape.Paged && options.Check.PreparedMetadataBytes > 0 {
		if options.Check.PreparedMetadataBytes < executionExtent.bytes {
			return FindingCheckAdmission{}, invalid(CodeInvalid, "prepared_round.page_metadata")
		}
		// Complete DTO bytes supersede legacy whole-document estimates. Each
		// prepared/selection page is independently parsed and bounded by its owner.
		candidate.checks[0].canonical = jsonExtent{}
		candidate.checks[0].coverage = jsonExtent{}
		candidate.worklist = jsonExtent{bytes: options.Check.PreparedMetadataBytes, nodes: workExtent.nodes + executionExtent.nodes + coverage.nodes, depth: max(workExtent.depth, executionExtent.depth, coverage.depth)}
		candidate.execution = jsonExtent{}
	}
	next, limit := b.withFindingCheck(b.round, candidate)
	if limit != "" {
		if _, emptyLimit := b.withFindingCheck(b.empty, candidate); emptyLimit != "" {
			return FindingCheckAdmission{Unschedulable: emptyLimit}, nil
		}
		b.full = true
		return FindingCheckAdmission{}, nil
	}
	b.round = next
	return FindingCheckAdmission{Admitted: true, Inventory: inventory}, nil
}

func findingPagedBudgetBytes(round findingRound) int64 {
	total := round.generated
	for _, document := range findingBudgetDocuments(&round) {
		total += document.bytesTotal()
	}
	return total
}
func (b *FindingRoundBudget) withFindingCheck(round findingRound, candidate measuredCandidate) (findingRound, string) {
	if !b.shape.Paged {
		return b.with(round, candidate, 1)
	}
	next := round
	next.checks++
	extents := []jsonExtent{candidate.entry, candidate.checks[0].canonical, candidate.checks[0].coverage, candidate.worklist, candidate.execution}
	for i, document := range findingBudgetDocuments(&next) {
		document.entries++
		document.extent = document.extent.plus(extents[i])
		if max(document.shell.depth, entryArrayDepth+document.extent.depth) > MaximumJSONDepth {
			return round, document.name + ".depth"
		}
	}
	next.generated += candidate.checks[0].tasks
	if findingPagedBudgetBytes(next) > MaximumGeneratedBytes {
		return round, "prepared_round.metadata_bytes"
	}
	return next, ""
}
