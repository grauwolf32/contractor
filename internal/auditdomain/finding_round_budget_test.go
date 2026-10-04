package auditdomain

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestJSONExtentMatchesStrictParser(t *testing.T) {
	revision := "r1"
	for name, value := range map[string]any{
		"scalar":  "plain",
		"number":  -12.5e3,
		"empty":   map[string]any{"object": map[string]any{}, "array": []any{}, "null": nil},
		"escapes": map[string]any{`ke"y\`: []any{`quote " and \ slash`, "<tag> &  ", "\x01\n\t", true, false}},
		"nested":  []any{[]any{[]any{map[string]any{"a": []any{1, 2, map[string]any{"b": "c"}}}}}, "d"},
		"task": ItemTask{
			Schema: TaskSchema, ItemKey: "item", Kind: "finding-verification", SubjectKey: "subject",
			WorkflowRole: "verify", SourceRef: contracts.ArtifactRef{Namespace: "ns", Name: "source", Revision: &revision},
			Finding: &FindingTask{ReceiptID: "receipt", Limitations: []string{"a", "b"}},
		},
		"inventory": testFindingInventoryDocument(t),
	} {
		t.Run(name, func(t *testing.T) {
			encoded, err := json.Marshal(value)
			if err != nil {
				t.Fatal(err)
			}
			decoder := json.NewDecoder(bytes.NewReader(encoded))
			decoder.UseNumber()
			nodes := 0
			parsed, err := readJSONValue(decoder, 1, &nodes, finiteJSONNumber)
			if err != nil {
				t.Fatal(err)
			}
			got := jsonExtentOf(encoded)
			if got.bytes != int64(len(encoded)) || got.nodes != int64(nodes) || got.depth != parsedDepth(parsed) {
				t.Fatalf("extent = %+v, want bytes=%d nodes=%d depth=%d", got, len(encoded), nodes, parsedDepth(parsed))
			}
		})
	}
}

func parsedDepth(value any) int {
	depth := 0
	switch typed := value.(type) {
	case map[string]any:
		for _, child := range typed {
			depth = max(depth, parsedDepth(child))
		}
	case []any:
		for _, child := range typed {
			depth = max(depth, parsedDepth(child))
		}
	default:
		return 1
	}
	return depth + 1
}

// TestFindingRoundBudgetBoundsEveryBuiltDocument builds a Round from the
// admitted proposals exactly as the next-Round builder does and checks that
// the budget is an upper bound for every bounded byte count and is exact for
// every JSON node count.
func TestFindingRoundBudgetBoundsEveryBuiltDocument(t *testing.T) {
	shape := testFindingRoundShape()
	budget, err := NewFindingRoundBudget(MaximumItems, shape)
	if err != nil {
		t.Fatal(err)
	}
	candidates := []FindingInventoryProposal{
		testBudgetProposal(t, "receipt-c", 3, func(document *FindingProposal) {
			document.Limitations = []string{"needs <live> & \"quoted\" confirmation", "partial-trace"}
			document.ProposedChecks[1].Objective = "Confirm   behavior with a \\ path."
		}),
		testBudgetProposal(t, "receipt-a", 1, func(document *FindingProposal) { document.Subject = nil }),
		testBudgetProposal(t, "receipt-b", 12, func(document *FindingProposal) {
			document.Preconditions = []string{strings.Repeat("x", 4096)}
		}),
	}
	checks := 0
	for _, candidate := range candidates {
		admission, err := budget.Admit(candidate)
		if err != nil || admission.Checks != len(candidate.SelectedCheckOrdinals) || admission.Unschedulable != "" {
			t.Fatalf("admit %s = (%d checks, %q, %v)", candidate.ReceiptID, admission.Checks, admission.Unschedulable, err)
		}
		checks += admission.Checks
	}
	if budget.Checks() != checks || budget.Full() {
		t.Fatalf("budget checks = %d full = %t, want %d and not full", budget.Checks(), budget.Full(), checks)
	}
	sort.Slice(candidates, func(i, j int) bool { return candidates[i].ReceiptID < candidates[j].ReceiptID })
	document := FindingInventoryDocument{Schema: FindingInventorySchema, Proposals: candidates}
	marshaled, err := json.Marshal(document)
	if err != nil {
		t.Fatal(err)
	}
	source, err := EncodeFindingInventory(document)
	if err != nil {
		t.Fatal(err)
	}
	round := budget.round
	if round.proposals.bytesTotal() != int64(len(marshaled)) || int64(len(source)) > round.proposals.bytesTotal() {
		t.Fatalf("proposal inventory bytes = %d (canonical %d), budget %d",
			len(marshaled), len(source), round.proposals.bytesTotal())
	}
	assertBudgetNodes(t, "proposal inventory", source, round.proposals)

	// The artifact store assigns shorter revisions than the provisional ones.
	assigned := "rev_" + strings.Repeat("a", 32)
	options := shape.Inventory
	options.SourceRef = contracts.ArtifactRef{
		Namespace: options.SourceRef.Namespace, Name: DeterministicID("proposal-inventory", "a", DigestBytes(source)),
		Revision: &assigned,
	}
	inventory, err := BuildFindingInventory(source, options)
	if err != nil {
		t.Fatal(err)
	}
	if len(inventory.Tasks) != checks {
		t.Fatalf("built %d items, want %d", len(inventory.Tasks), checks)
	}
	assertBudgetBytes(t, "canonical inventory", inventory.CanonicalInventory, round.canonical)
	coverage, err := EncodeCoverage(inventory.Coverage)
	if err != nil {
		t.Fatal(err)
	}
	assertBudgetBytes(t, "coverage", coverage, round.coverage)
	worklist, err := EncodeWorklist(inventory.Worklist)
	if err != nil {
		t.Fatal(err)
	}
	assertBudgetBytes(t, "worklist", worklist, round.worklist)
	dispatched := inventory.ExecutionManifest
	dispatched.Items = append([]ExecutionItem(nil), inventory.ExecutionManifest.Items...)
	var generated int64
	for index, task := range inventory.Tasks {
		dispatched.Items[index].TaskRef = &contracts.ArtifactRef{
			Namespace: shape.TaskNamespace, Name: task.Item.TaskPackageID, Revision: &assigned,
		}
		dispatched.Items[index].Inputs = shape.ExecutionInputs
		generated += int64(len(task.Package))
	}
	execution, err := EncodeExecutionManifest(dispatched)
	if err != nil {
		t.Fatal(err)
	}
	assertBudgetBytes(t, "execution manifest", execution, round.execution)
	if generated > round.generated {
		t.Fatalf("generated task packages = %d bytes, budget %d", generated, round.generated)
	}
	packageBytes, _, err := BuildPackage(
		DeterministicID("worklist", DigestBytes(execution), inventory.CanonicalInventoryDigest),
		PackageKindWorklist, "", []PackageInput{
			{ID: "coverage", Path: "coverage.json", MediaType: "application/json", Data: coverage},
			{ID: "execution-manifest", Path: "execution.json", MediaType: "application/json", Data: execution},
			{ID: "inventory", Path: "inventory.json", MediaType: "application/json", Data: inventory.CanonicalInventory},
			{ID: "worklist", Path: "worklist.json", MediaType: "application/json", Data: worklist},
		})
	if err != nil {
		t.Fatal(err)
	}
	archive := budget.overhead.round + round.canonical.bytesTotal() + round.coverage.bytesTotal() +
		round.worklist.bytesTotal() + round.execution.bytesTotal()
	if int64(len(packageBytes)) > archive {
		t.Fatalf("Round package = %d bytes, budget %d", len(packageBytes), archive)
	}
}

func assertBudgetBytes(t *testing.T, name string, encoded []byte, document boundedDocument) {
	t.Helper()
	if int64(len(encoded)) > document.bytesTotal() {
		t.Fatalf("%s = %d bytes, budget %d", name, len(encoded), document.bytesTotal())
	}
	assertBudgetNodes(t, name, encoded, document)
}

func assertBudgetNodes(t *testing.T, name string, encoded []byte, document boundedDocument) {
	t.Helper()
	actual := jsonExtentOf(encoded)
	if actual.nodes != document.shell.nodes+document.extent.nodes ||
		actual.depth != max(document.shell.depth, entryArrayDepth+document.extent.depth) {
		t.Fatalf("%s extent = %+v, budget nodes=%d depth=%d", name, actual,
			document.shell.nodes+document.extent.nodes, max(document.shell.depth, entryArrayDepth+document.extent.depth))
	}
}

func TestFindingRoundBudgetCapsChecksAtInventoryItemLimit(t *testing.T) {
	budget, err := NewFindingRoundBudget(10_000, testFindingRoundShape())
	if err != nil {
		t.Fatal(err)
	}
	var admitted []FindingInventoryProposal
	for index := range 9 {
		candidate := testBudgetProposal(t, fmt.Sprintf("receipt-%d", index), MaximumCoverageValues, nil)
		admission, err := budget.Admit(candidate)
		if err != nil || admission.Unschedulable != "" {
			t.Fatalf("admit proposal %d = (%d checks, %q, %v)", index, admission.Checks, admission.Unschedulable, err)
		}
		if want := min(MaximumCoverageValues, MaximumItems-index*MaximumCoverageValues); admission.Checks != max(want, 0) {
			t.Fatalf("proposal %d admitted %d checks, want %d", index, admission.Checks, max(want, 0))
		}
		if admission.Checks != 0 {
			admitted = append(admitted, admission.Proposal)
		}
	}
	if budget.Checks() != MaximumItems || !budget.Full() || len(admitted) != 8 {
		t.Fatalf("4608 eligible checks admitted %d into %d proposals, full=%t", budget.Checks(), len(admitted), budget.Full())
	}
	source, err := EncodeFindingInventory(FindingInventoryDocument{Schema: FindingInventorySchema, Proposals: admitted})
	if err != nil {
		t.Fatalf("admitted checks exceed the inventory: %v", err)
	}
	options := testFindingRoundShape().Inventory
	if _, err := BuildFindingInventory(source, options); err != nil {
		t.Fatalf("admitted checks cannot build a Round: %v", err)
	}
}

func TestFindingRoundBudgetSpreadsLargeProposalsAcrossRounds(t *testing.T) {
	// Three proposals of about 3 MiB together exceed one 8 MiB inventory.
	large := func(receiptID string) FindingInventoryProposal {
		return testBudgetProposal(t, receiptID, 2, func(document *FindingProposal) {
			document.Preconditions = make([]string, 48)
			for index := range document.Preconditions {
				document.Preconditions[index] = strings.Repeat(string(rune('a'+index%26)), MaximumStringBytes)
			}
		})
	}
	candidates := []FindingInventoryProposal{large("receipt-a"), large("receipt-b"), large("receipt-c")}
	first, err := NewFindingRoundBudget(MaximumItems, testFindingRoundShape())
	if err != nil {
		t.Fatal(err)
	}
	var rounds [][]FindingInventoryProposal
	var round []FindingInventoryProposal
	for _, candidate := range candidates {
		admission, err := first.Admit(candidate)
		if err != nil || admission.Unschedulable != "" {
			t.Fatalf("admit %s = (%d checks, %q, %v)", candidate.ReceiptID, admission.Checks, admission.Unschedulable, err)
		}
		if admission.Checks != 0 {
			round = append(round, admission.Proposal)
		}
	}
	rounds = append(rounds, round)
	if len(round) != 2 || !first.Full() {
		t.Fatalf("first Round admitted %d proposals, full=%t", len(round), first.Full())
	}
	second, err := NewFindingRoundBudget(MaximumItems, testFindingRoundShape())
	if err != nil {
		t.Fatal(err)
	}
	admission, err := second.Admit(candidates[2])
	if err != nil || admission.Checks != 2 || second.Full() {
		t.Fatalf("second Round admission = (%d checks, %v), full=%t", admission.Checks, err, second.Full())
	}
	rounds = append(rounds, []FindingInventoryProposal{admission.Proposal})
	for index, proposals := range rounds {
		if _, err := EncodeFindingInventory(FindingInventoryDocument{
			Schema: FindingInventorySchema, Proposals: proposals,
		}); err != nil {
			t.Fatalf("Round %d inventory: %v", index+2, err)
		}
	}
}

func TestFindingRoundBudgetReportsProposalThatCannotFitAnyRound(t *testing.T) {
	// The proposal itself is a valid 8 MiB document, so its inventory entry,
	// which adds the receipt and exact artifact identity, can never fit.
	oversized := testBudgetProposal(t, "receipt-oversized", 1, func(document *FindingProposal) {
		padDocument(t, document, MaximumDocumentBytes-16)
	})
	budget, err := NewFindingRoundBudget(MaximumItems, testFindingRoundShape())
	if err != nil {
		t.Fatal(err)
	}
	admission, err := budget.Admit(oversized)
	if err != nil || admission.Checks != 0 || admission.Unschedulable != "proposal_inventory.bytes" || budget.Full() {
		t.Fatalf("oversized admission = (%d checks, %q, %v), full=%t",
			admission.Checks, admission.Unschedulable, err, budget.Full())
	}
	admission, err = budget.Admit(testBudgetProposal(t, "receipt-small", 1, nil))
	if err != nil || admission.Checks != 1 {
		t.Fatalf("admission after an unschedulable proposal = (%d checks, %v)", admission.Checks, err)
	}
}

func TestFindingRoundBudgetAdmitsCheckPrefixes(t *testing.T) {
	limited, err := NewFindingRoundBudget(3, testFindingRoundShape())
	if err != nil {
		t.Fatal(err)
	}
	candidate := testBudgetProposal(t, "receipt-a", 5, nil)
	candidate.SelectedCheckOrdinals = []int{0, 2, 3, 4}
	admission, err := limited.Admit(candidate)
	if err != nil || admission.Checks != 3 || !limited.Full() ||
		fmt.Sprint(admission.Proposal.SelectedCheckOrdinals) != "[0 2 3]" {
		t.Fatalf("capacity-limited admission = (%v, %v), full=%t",
			admission.Proposal.SelectedCheckOrdinals, err, limited.Full())
	}
	entry, err := json.Marshal(admission.Proposal)
	if err != nil {
		t.Fatal(err)
	}
	if limited.round.proposals.extent.bytes != int64(len(entry)) ||
		limited.round.proposals.extent.nodes != jsonExtentOf(entry).nodes {
		t.Fatalf("prefix entry extent = %+v, want %+v", limited.round.proposals.extent, jsonExtentOf(entry))
	}

	// Every check repeats the proposal's limitations in the canonical
	// inventory and coverage, so the Round fills long before its item limit.
	verbose := testBudgetProposal(t, "receipt-verbose", 64, func(document *FindingProposal) {
		document.Limitations = make([]string, MaximumCoverageValues)
		for index := range document.Limitations {
			document.Limitations[index] = fmt.Sprintf("%03d-%s", index, strings.Repeat("l", MaximumCoverageValueBytes-4))
		}
	})
	budget, err := NewFindingRoundBudget(MaximumItems, testFindingRoundShape())
	if err != nil {
		t.Fatal(err)
	}
	admission, err = budget.Admit(verbose)
	if err != nil || admission.Checks < 1 || admission.Checks >= 64 || !budget.Full() {
		t.Fatalf("byte-limited admission = (%d checks, %v), full=%t", admission.Checks, err, budget.Full())
	}
	source, err := EncodeFindingInventory(FindingInventoryDocument{
		Schema: FindingInventorySchema, Proposals: []FindingInventoryProposal{admission.Proposal},
	})
	if err != nil {
		t.Fatal(err)
	}
	inventory, err := BuildFindingInventory(source, testFindingRoundShape().Inventory)
	if err != nil {
		t.Fatalf("admitted check prefix cannot build a Round: %v", err)
	}
	coverage, err := EncodeCoverage(inventory.Coverage)
	if err != nil {
		t.Fatal(err)
	}
	if len(inventory.CanonicalInventory)+len(coverage) > MaximumArchiveBytes {
		t.Fatalf("admitted prefix overflows the Round package: inventory=%d coverage=%d",
			len(inventory.CanonicalInventory), len(coverage))
	}
	rest := verbose
	rest.SelectedCheckOrdinals = verbose.SelectedCheckOrdinals[admission.Checks:]
	next, err := NewFindingRoundBudget(MaximumItems, testFindingRoundShape())
	if err != nil {
		t.Fatal(err)
	}
	if remaining, err := next.Admit(rest); err != nil || remaining.Checks < 1 || remaining.Unschedulable != "" {
		t.Fatalf("remaining checks admission = (%d checks, %v)", remaining.Checks, err)
	}
}

func TestFindingRoundBudgetRejectsInvalidCandidates(t *testing.T) {
	budget, err := NewFindingRoundBudget(MaximumItems, testFindingRoundShape())
	if err != nil {
		t.Fatal(err)
	}
	for name, ordinals := range map[string][]int{
		"empty": {}, "descending": {1, 0}, "duplicate": {0, 0}, "out of range": {2},
	} {
		candidate := testBudgetProposal(t, "receipt-a", 2, nil)
		candidate.SelectedCheckOrdinals = ordinals
		if _, err := budget.Admit(candidate); err == nil {
			t.Fatalf("%s ordinals were admitted", name)
		}
	}
	if budget.Checks() != 0 {
		t.Fatal("invalid candidates changed the budget")
	}
	shape := testFindingRoundShape()
	shape.TaskRevision = ""
	if _, err := NewFindingRoundBudget(MaximumItems, shape); err == nil {
		t.Fatal("budget accepted a task ref without a revision")
	}
}

func testFindingRoundShape() FindingRoundShape {
	revision := strings.Repeat("0", 64)
	inputRevision := "source-r1"
	namespace := DeterministicID("audit", "a")
	return FindingRoundShape{
		Inventory: InventoryOptions{
			Round: 2, ProfileMode: "finding-verification", WorkflowRole: "verify",
			SourceInputName: "proposal_inventory", ApprovalRequirement: ApprovalActiveCheck,
			SourceRef: contracts.ArtifactRef{
				Namespace: namespace, Name: DeterministicID("proposal-inventory", "a"), Revision: &revision,
			},
			Scope: map[string]string{"target": "service-a", "objective": "Verify the retained findings."},
		},
		TaskNamespace: namespace, TaskRevision: revision,
		ExecutionInputs: []ExactInput{{
			Name: "source", Digest: testDigest('e'),
			Ref: contracts.ArtifactRef{Namespace: "inputs", Name: "source", Revision: &inputRevision},
		}},
	}
}

func testBudgetProposal(
	t *testing.T, receiptID string, checks int, edit func(*FindingProposal),
) FindingInventoryProposal {
	t.Helper()
	document := FindingProposal{
		Schema: FindingProposalSchema, ClientKey: "candidate-" + receiptID, Title: "Authorization gap",
		Description:   "Ownership validation may be missing.",
		Subject:       &FindingSubject{Kind: "openapi-operation", Key: "get-widget"},
		Preconditions: []string{}, StandardRefs: []StandardReference{}, EvidenceIDs: []string{},
		ProposedChecks: make([]ProposedCheck, checks), SeveritySuggestion: "medium", Limitations: []string{},
	}
	ordinals := make([]int, checks)
	for index := range document.ProposedChecks {
		document.ProposedChecks[index] = ProposedCheck{
			Objective: fmt.Sprintf("Trace the ownership predicate on path %d.", index), Method: "static-trace",
		}
		ordinals[index] = index
	}
	if edit != nil {
		edit(&document)
	}
	encoded, err := EncodeFindingProposal(document)
	if err != nil {
		t.Fatal(err)
	}
	revision := "proposal-r1"
	return FindingInventoryProposal{
		ReceiptID: receiptID,
		Proposal: FindingInventoryArtifact{
			Ref:    contracts.ArtifactRef{Namespace: "audit-a", Name: "proposal-" + receiptID, Revision: &revision},
			Digest: DigestBytes(encoded), MediaType: JSONMediaType, SizeBytes: int64(len(encoded)),
		},
		Document: document, SelectedCheckOrdinals: ordinals,
	}
}

// padDocument fills preconditions until the encoded proposal is exactly size
// bytes. Plain letters encode one byte each; a precondition also costs its
// quotes and, after the first, a separator.
func padDocument(t *testing.T, document *FindingProposal, size int) {
	t.Helper()
	const chunk = 60_000
	document.Preconditions = []string{}
	encoded, err := EncodeFindingProposal(*document)
	if err != nil {
		t.Fatal(err)
	}
	for missing := size - len(encoded); missing > 0; {
		cost := 3
		if len(document.Preconditions) == 0 {
			cost = 2
		}
		length := min(missing-cost, chunk)
		if rest := missing - cost - length; rest > 0 && rest < 4 {
			length -= 4 // Leave room for one more non-empty precondition.
		}
		if length < 1 {
			t.Fatalf("cannot pad the proposal by %d bytes", missing)
		}
		document.Preconditions = append(document.Preconditions, strings.Repeat("p", length))
		missing -= length + cost
	}
	if encoded, err = EncodeFindingProposal(*document); err != nil || len(encoded) != size {
		t.Fatalf("padded proposal = %d bytes, %v; want %d", len(encoded), err, size)
	}
}
