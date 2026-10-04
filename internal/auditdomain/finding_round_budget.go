package auditdomain

import (
	"encoding/json"
	"errors"
	"slices"
	"strconv"
	"strings"
	"sync"

	"github.com/grauwolf32/contractor/internal/contracts"
)

// entryArrayDepth is the nesting depth of the entry array in every document a
// later finding Round produces: each is an object member of the document root.
const entryArrayDepth = 2

var (
	placeholderDigest    = "sha256:" + strings.Repeat("0", 64)
	placeholderPackageID = "task-" + strings.Repeat("0", 64)
	placeholderRoundID   = "worklist-" + strings.Repeat("0", 64)
)

// FindingRoundShape carries the values every item of one later finding Round
// repeats. Artifact revisions are assigned only when the Round is written, so
// Inventory.SourceRef and TaskRevision are provisional values that must be at
// least as long as the revisions the artifact store assigns.
type FindingRoundShape struct {
	Inventory       InventoryOptions
	TaskNamespace   string
	TaskRevision    string
	ExecutionInputs []ExactInput
}

// FindingRoundAdmission reports how much of one candidate proposal a
// FindingRoundBudget admitted.
type FindingRoundAdmission struct {
	// Proposal carries the admitted prefix of the candidate's selected checks
	// when Checks is positive.
	Proposal FindingInventoryProposal
	Checks   int
	// Unschedulable names the limit, such as "proposal_inventory.bytes", that
	// the candidate's first check exceeds even in an otherwise empty Round.
	// Such a proposal can never be verified by a later Round.
	Unschedulable string
}

// FindingRoundBudget admits held finding proposals into one later-Round
// inventory in the caller's order. It measures every admitted check in each
// bounded document the Round derives from it: the proposal inventory, the
// canonical inventory, the coverage, worklist and execution documents of the
// Round package, and the generated item-task packages. An admitted selection
// therefore stays within every item, byte, JSON node and nesting limit.
type FindingRoundBudget struct {
	maximumChecks int
	shape         FindingRoundShape
	overhead      packageOverhead
	empty         findingRound
	round         findingRound
	full          bool
}

// findingRound is the measured extent of one later Round.
type findingRound struct {
	checks    int
	proposals boundedDocument
	canonical boundedDocument
	coverage  boundedDocument
	worklist  boundedDocument
	execution boundedDocument
	generated int64
}

// NewFindingRoundBudget returns an empty budget admitting at most
// maximumChecks checks, never more than MaximumItems.
func NewFindingRoundBudget(maximumChecks int, shape FindingRoundShape) (*FindingRoundBudget, error) {
	taskRef := contracts.ArtifactRef{Namespace: shape.TaskNamespace, Name: placeholderPackageID, Revision: &shape.TaskRevision}
	if shape.Inventory.SourceRef.ValidateExact() != nil || taskRef.ValidateExact() != nil {
		return nil, invalid(CodeReferenceInvalid, "round_shape")
	}
	overhead, err := measuredPackageOverhead()
	if err != nil {
		return nil, err
	}
	empty := findingRound{}
	shells := []struct {
		target *boundedDocument
		name   string
		value  any
	}{
		{&empty.proposals, "proposal_inventory", FindingInventoryDocument{
			Schema: FindingInventorySchema, Proposals: []FindingInventoryProposal{},
		}},
		{&empty.canonical, "canonical_inventory", newFindingInventoryBasis([]map[string]any{})},
		{&empty.coverage, "coverage", CoverageEnvelope{Schema: CoverageSchema, Rows: []CoverageRow{}}},
		{&empty.worklist, "worklist", WorklistManifest{
			Schema: WorklistSchema, Round: shape.Inventory.Round, Items: []WorklistItem{},
		}},
		{&empty.execution, "execution_manifest", ExecutionManifest{
			Schema: ExecutionManifestSchema, Items: []ExecutionItem{},
		}},
	}
	for _, shell := range shells {
		extent, err := measureJSON(shell.value)
		if err != nil {
			return nil, err
		}
		*shell.target = boundedDocument{name: shell.name, shell: extent}
	}
	return &FindingRoundBudget{
		maximumChecks: min(max(maximumChecks, 0), MaximumItems),
		shape:         shape, overhead: overhead, empty: empty, round: empty,
	}, nil
}

// Checks returns the number of admitted checks.
func (b *FindingRoundBudget) Checks() int { return b.round.checks }

// Full reports whether the Round can take no further check: its check limit
// is reached, or a candidate check that fits an empty Round did not fit.
func (b *FindingRoundBudget) Full() bool {
	return b.full || b.round.checks >= b.maximumChecks
}

// Admit adds the longest prefix of candidate's selected checks that keeps the
// Round within its limits. A partial or empty admission of a candidate that
// fits an empty Round makes the budget full.
func (b *FindingRoundBudget) Admit(candidate FindingInventoryProposal) (FindingRoundAdmission, error) {
	if b.Full() {
		return FindingRoundAdmission{}, nil
	}
	measured, err := b.measure(candidate)
	if err != nil {
		return FindingRoundAdmission{}, err
	}
	for count := min(len(measured.checks), b.maximumChecks-b.round.checks); count > 0; count-- {
		next, limit := b.with(b.round, measured, count)
		if limit != "" {
			continue
		}
		b.round = next
		b.full = count < len(measured.checks)
		admitted := candidate
		admitted.SelectedCheckOrdinals = slices.Clone(candidate.SelectedCheckOrdinals[:count])
		return FindingRoundAdmission{Proposal: admitted, Checks: count}, nil
	}
	if _, limit := b.with(b.empty, measured, 1); limit != "" {
		return FindingRoundAdmission{Unschedulable: limit}, nil
	}
	b.full = true
	return FindingRoundAdmission{}, nil
}

// measuredCandidate holds the extents one candidate proposal adds to a Round.
type measuredCandidate struct {
	entry     jsonExtent // the proposal inventory entry with every selected check
	ordinals  []int
	worklist  jsonExtent // one item's worklist entry
	execution jsonExtent // one item's dispatched execution entry
	checks    []measuredCheck
}

// measuredCheck holds the cumulative extents of the candidate's checks up to
// and including this one.
type measuredCheck struct {
	canonical jsonExtent
	coverage  jsonExtent
	tasks     int64
}

func (b *FindingRoundBudget) measure(candidate FindingInventoryProposal) (measuredCandidate, error) {
	ordinals := candidate.SelectedCheckOrdinals
	if len(ordinals) == 0 || len(ordinals) > MaximumCoverageValues {
		return measuredCandidate{}, invalid(CodeInventoryInvalid, "finding_inventory.selected_checks")
	}
	previous := -1
	for _, ordinal := range ordinals {
		if ordinal <= previous || ordinal >= len(candidate.Document.ProposedChecks) {
			return measuredCandidate{}, invalid(CodeInventoryInvalid, "finding_inventory.selected_checks")
		}
		previous = ordinal
	}
	entry, err := measureJSON(candidate)
	if err != nil {
		return measuredCandidate{}, err
	}
	result := measuredCandidate{entry: entry, ordinals: ordinals, checks: make([]measuredCheck, len(ordinals))}
	checks := newFindingChecks(candidate)
	taskRef := contracts.ArtifactRef{
		Namespace: b.shape.TaskNamespace, Name: placeholderPackageID, Revision: &b.shape.TaskRevision,
	}
	var total measuredCheck
	for index, ordinal := range ordinals {
		basisSubject, subject := checks.subject(ordinal)
		if index == 0 {
			// Item keys, task package IDs and digests have fixed lengths, and
			// every check of one proposal shares its subject key, so one
			// worklist and one execution entry measure every check. Item
			// ordinals use the widest possible Round position.
			ordinal := MaximumItems - 1
			if result.worklist, err = measureJSON(newWorklistItem(
				subject, ordinal, placeholderPackageID, b.shape.Inventory,
			)); err != nil {
				return measuredCandidate{}, err
			}
			// Dispatch replaces the inventory inputs with the Workflow inputs
			// and pins each task package revision.
			if result.execution, err = measureJSON(ExecutionItem{
				ItemKey: subject.itemKey, Ordinal: ordinal, SubjectKey: subject.subjectKey,
				TaskPackageID: placeholderPackageID, TaskPackageDigest: placeholderDigest,
				TaskRef: &taskRef, Inputs: b.shape.ExecutionInputs,
			}); err != nil {
				return measuredCandidate{}, err
			}
		}
		canonical, err := measureJSON(basisSubject)
		if err != nil {
			return measuredCandidate{}, err
		}
		coverage, err := measureJSON(newCoverageRow(subject))
		if err != nil {
			return measuredCandidate{}, err
		}
		task, err := json.Marshal(newItemTask(
			subject, JSONMediaType, placeholderDigest, placeholderDigest, b.shape.Inventory,
		))
		if err != nil {
			return measuredCandidate{}, err
		}
		total = measuredCheck{
			canonical: total.canonical.plus(canonical), coverage: total.coverage.plus(coverage),
			tasks: total.tasks + int64(len(task)) + b.overhead.task,
		}
		result.checks[index] = total
	}
	return result, nil
}

// entryWith returns the proposal inventory entry selecting only the first
// count checks. Dropping a trailing ordinal removes its digits, one
// separator and one node.
func (candidate measuredCandidate) entryWith(count int) jsonExtent {
	entry := candidate.entry
	for _, ordinal := range candidate.ordinals[count:] {
		entry.bytes -= int64(len(strconv.Itoa(ordinal))) + 1
		entry.nodes--
	}
	return entry
}

// with returns round after admitting the first count checks of candidate, or
// the name of the first limit the result would exceed.
func (b *FindingRoundBudget) with(round findingRound, candidate measuredCandidate, count int) (findingRound, string) {
	checks := candidate.checks[count-1]
	next := round
	next.checks += count
	var limit string
	for _, step := range []struct {
		target *boundedDocument
		extent jsonExtent
		count  int64
	}{
		{&next.proposals, candidate.entryWith(count), 1},
		{&next.canonical, checks.canonical, int64(count)},
		{&next.coverage, checks.coverage, int64(count)},
		{&next.worklist, candidate.worklist.times(count), int64(count)},
		{&next.execution, candidate.execution.times(count), int64(count)},
	} {
		if *step.target, limit = step.target.with(step.extent, step.count); limit != "" {
			return round, limit
		}
	}
	next.generated += checks.tasks
	if next.generated > MaximumGeneratedBytes {
		return round, "task_packages.bytes"
	}
	archive := b.overhead.round + next.canonical.bytesTotal() + next.coverage.bytesTotal() +
		next.worklist.bytesTotal() + next.execution.bytesTotal()
	if archive > MaximumArchiveBytes {
		return round, "round_package.bytes"
	}
	return next, ""
}

// jsonExtent is the measured size of one JSON value. Bytes count the
// json.Marshal form that canonical encoding parses before formatting; the
// canonical form is never longer, so a fitting extent also fits canonically.
type jsonExtent struct {
	bytes int64
	nodes int64
	depth int
}

func (e jsonExtent) plus(other jsonExtent) jsonExtent {
	return jsonExtent{bytes: e.bytes + other.bytes, nodes: e.nodes + other.nodes, depth: max(e.depth, other.depth)}
}

func (e jsonExtent) times(count int) jsonExtent {
	return jsonExtent{bytes: e.bytes * int64(count), nodes: e.nodes * int64(count), depth: e.depth}
}

// boundedDocument accumulates the entries of the one array of a document
// whose other members are fixed. Entries are separated by single commas.
type boundedDocument struct {
	name    string
	shell   jsonExtent // the document with an empty entry array
	entries int64
	extent  jsonExtent // the entries, without separators
}

func (d boundedDocument) bytesTotal() int64 {
	return d.shell.bytes + d.extent.bytes + max(d.entries-1, 0)
}

// with returns the document after adding count entries of the given total
// extent, or the name of the first limit the result would exceed.
func (d boundedDocument) with(extent jsonExtent, count int64) (boundedDocument, string) {
	next := d
	next.entries += count
	next.extent = next.extent.plus(extent)
	switch {
	case next.bytesTotal() > MaximumDocumentBytes:
		return d, d.name + ".bytes"
	case next.shell.nodes+next.extent.nodes > MaximumJSONNodes:
		return d, d.name + ".nodes"
	case max(next.shell.depth, entryArrayDepth+next.extent.depth) > MaximumJSONDepth:
		return d, d.name + ".depth"
	}
	return next, ""
}

func measureJSON(value any) (jsonExtent, error) {
	encoded, err := json.Marshal(value)
	if err != nil {
		return jsonExtent{}, err
	}
	return jsonExtentOf(encoded), nil
}

// jsonExtentOf measures compact, valid JSON such as json.Marshal output. It
// counts values as readJSONValue does: every object, array, string, number,
// boolean and null is one node, while object keys are not. Depth counts the
// root value as one.
func jsonExtentOf(data []byte) jsonExtent {
	extent := jsonExtent{bytes: int64(len(data))}
	containers := make([]byte, 0, 16)
	key := false // the next string is an object key
	for index := 0; index < len(data); index++ {
		switch data[index] {
		case '{', '[':
			extent.nodes++
			containers = append(containers, data[index])
			extent.depth = max(extent.depth, len(containers))
			key = data[index] == '{'
		case '}', ']':
			containers = containers[:len(containers)-1]
		case ',':
			key = containers[len(containers)-1] == '{'
		case ':':
		case '"':
			index = jsonStringEnd(data, index)
			if key {
				key = false
				continue
			}
			extent.nodes++
			extent.depth = max(extent.depth, len(containers)+1)
		default: // A number, true, false or null ends before a separator.
			extent.nodes++
			extent.depth = max(extent.depth, len(containers)+1)
			for index+1 < len(data) && data[index+1] != ',' && data[index+1] != ']' && data[index+1] != '}' {
				index++
			}
		}
	}
	return extent
}

// jsonStringEnd returns the index of the quote closing the string that starts
// at start.
func jsonStringEnd(data []byte, start int) int {
	for index := start + 1; index < len(data); index++ {
		switch data[index] {
		case '\\':
			index++
		case '"':
			return index
		}
	}
	return len(data) - 1
}

// packageOverhead holds the bytes a canonical package adds to its members:
// the manifest and the stored-ZIP structure, with every member size at its
// widest decimal form.
type packageOverhead struct {
	task  int64 // one item-task package with its single task document
	round int64 // the Round worklist package with its four documents
}

var measuredPackageOverhead = sync.OnceValues(func() (packageOverhead, error) {
	sample := []byte("{}")
	widest := func(limit int) int64 { return int64(len(strconv.Itoa(limit)) - len(strconv.Itoa(len(sample)))) }
	task, _, err := BuildPackage(placeholderPackageID, PackageKindTask, "", []PackageInput{
		{ID: "task-document", Path: "task.json", MediaType: "application/json", Data: sample},
	})
	if err != nil {
		return packageOverhead{}, errors.Join(errors.New("measure item-task package overhead"), err)
	}
	members := []PackageInput{
		{ID: "coverage", Path: "coverage.json", MediaType: "application/json", Data: sample},
		{ID: "execution-manifest", Path: "execution.json", MediaType: "application/json", Data: sample},
		{ID: "inventory", Path: "inventory.json", MediaType: "application/json", Data: sample},
		{ID: "worklist", Path: "worklist.json", MediaType: "application/json", Data: sample},
	}
	round, _, err := BuildPackage(placeholderRoundID, PackageKindWorklist, "", members)
	if err != nil {
		return packageOverhead{}, errors.Join(errors.New("measure Round package overhead"), err)
	}
	return packageOverhead{
		task:  int64(len(task)-len(sample)) + widest(MaximumDocumentBytes),
		round: int64(len(round)-len(members)*len(sample)) + int64(len(members))*widest(MaximumMemberBytes),
	}, nil
})
