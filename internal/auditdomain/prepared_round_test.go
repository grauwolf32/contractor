package auditdomain

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

func preparedRoundFixture(t *testing.T, count int) PreparedRoundPage {
	t.Helper()
	page := PreparedRoundPage{Schema: PreparedRoundPageSchema, AuditID: "audit", RoundID: "round-two", Generation: "generation", PreviousPageDigest: DigestBytes(nil), Items: []PreparedRoundItem{}}
	for index := range count {
		plan := acceptedPlanFixture(t)
		plan.Source.ReceiptID = fmt.Sprintf("receipt-%03d", index)
		plan.Source.Proposal.SizeBytes = 8
		plan.Source.RetainedProposal.SizeBytes = 8
		plan.ItemKey = FindingCheckItemKey(plan.Source.ReceiptID, 0)
		revision := "task-r1"
		task := contracts.ExactArtifact{Ref: contracts.ArtifactRef{Namespace: ArtifactNamespace("audit"), Name: fmt.Sprintf("task-%03d", index), Revision: &revision}, Digest: DigestBytes([]byte(fmt.Sprint(index))), MediaType: PackageMediaType, SizeBytes: 100}
		sourceRef := task.Ref.Clone()
		sourceRef.Name = fmt.Sprintf("source-%03d", index)
		item := PreparedRoundItem{ItemID: fmt.Sprintf("item-%03d", index), ItemKey: plan.ItemKey, Ordinal: index, Kind: "finding-verification", SubjectKey: plan.Source.ReceiptID, WorkflowRole: plan.Route.WorkflowRole, Task: task, InitialState: "ready", ApprovalKind: "none",
			Origin:          PreparedItemOrigin{Schema: ItemOriginSchema, SourceRef: sourceRef, SourceContentDigest: DigestBytes([]byte("source")), SourceMediaType: JSONMediaType, CanonicalInventoryDigest: DigestBytes([]byte("basis")), EntryKey: plan.ItemKey},
			Coverage:        RoundResultCoverage{Status: CoverageNotTested, Requested: []string{"verify"}, Completed: []string{}, Gaps: []string{}},
			ProposalSources: []PreparedProposalSource{{ReceiptID: plan.Source.ReceiptID, ProposedCheckOrdinal: 0, Proposal: plan.Source.RetainedProposal}},
			Execution:       PreparedExecutionItem{ExecutionItem: ExecutionItem{ItemKey: plan.ItemKey, Ordinal: index, SubjectKey: plan.Source.ReceiptID, TaskPackageID: fmt.Sprintf("task-package-%03d", index), TaskPackageDigest: task.Digest, TaskRef: &task.Ref, Inputs: []ExactInput{}}, AcceptedPlan: &plan}}
		page.Items = append(page.Items, item)
	}
	return page
}

func TestPreparedRoundDocumentsPinExactBoundedSelection(t *testing.T) {
	page := preparedRoundFixture(t, 2)
	data, err := EncodePreparedRoundPage(page)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := DecodePreparedRoundPage(data)
	if err != nil || len(decoded.Items) != 2 {
		t.Fatalf("page round trip = %+v %v", decoded, err)
	}
	digest := DigestBytes([]byte("pin"))
	root := PreparedRoundRoot{Schema: PreparedRoundRootSchema, AuditID: page.AuditID, RoundID: page.RoundID, Generation: page.Generation, ProfileDigest: digest, BaselineDigest: digest, PageCount: 1, ItemCount: 2, PagesDigest: DigestBytes(data)}
	encoded, err := EncodePreparedRoundRoot(root)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := DecodePreparedRoundRoot(encoded); err != nil {
		t.Fatal(err)
	}
	for _, old := range []string{`"source":null,`, `"source":{},`, `"selection_root":null,`} {
		if _, err := DecodePreparedRoundRoot(bytes.Replace(encoded, []byte(`"schema":`), []byte(old+`"schema":`), 1)); err == nil {
			t.Fatalf("removed root member accepted: %s", old)
		}
	}
	if _, err := DecodePreparedRoundPage(bytes.Replace(data, []byte(PreparedRoundPageSchema), []byte("contractor.audit.round-selection-page.v1"), 1)); err == nil {
		t.Fatal("removed selection schema accepted")
	}
	if _, err := DecodePreparedRoundRoot(bytes.Replace(encoded, []byte(`"schema":`), []byte(`"unknown":1,"schema":`), 1)); err == nil {
		t.Fatal("unknown root field accepted")
	}
	if _, err := DecodePreparedRoundPage(bytes.Replace(data, []byte(`"page_ordinal":0`), []byte(`"page_ordinal":0,"page_ordinal":0`), 1)); err == nil {
		t.Fatal("duplicate page field accepted")
	}
}

func TestPreparedRoundPageRejectsSourceRouteAndMembershipDrift(t *testing.T) {
	for _, test := range []struct {
		name   string
		mutate func(*PreparedRoundPage)
	}{
		{"missing plan", func(p *PreparedRoundPage) { p.Items[0].Execution.AcceptedPlan = nil }},
		{"route", func(p *PreparedRoundPage) { p.Items[0].WorkflowRole = "other" }},
		{"receipt", func(p *PreparedRoundPage) { p.Items[0].ProposalSources[0].ReceiptID = "foreign" }},
		{"source revision", func(p *PreparedRoundPage) { p.Items[0].ProposalSources[0].Proposal.Ref.Name = "foreign" }},
		{"task digest", func(p *PreparedRoundPage) { p.Items[0].Execution.TaskPackageDigest = DigestBytes([]byte("foreign")) }},
		{"global ordinal", func(p *PreparedRoundPage) { p.Items[0].Ordinal = 1 }},
		{"page ordinal", func(p *PreparedRoundPage) { p.Items[0].Execution.Ordinal = 1 }},
		{"partial chain", func(p *PreparedRoundPage) { p.PreviousPageDigest = DigestBytes([]byte("foreign")) }},
		{"approval", func(p *PreparedRoundPage) { p.Items[0].ApprovalKind = "active-check-approval" }},
		{"coverage", func(p *PreparedRoundPage) { p.Items[0].Coverage.Requested = []string{"replay"} }},
	} {
		t.Run(test.name, func(t *testing.T) {
			page := preparedRoundFixture(t, 1)
			test.mutate(&page)
			if _, err := EncodePreparedRoundPage(page); err == nil {
				t.Fatal("drift accepted")
			}
		})
	}
}

func TestPreparedRoundPageBoundsIncludeRelatedDescriptorsAndEncodedBytes(t *testing.T) {
	page := preparedRoundFixture(t, 102)
	if PreparedRoundPageDescriptors(page) != 510 {
		t.Fatal("exact descriptor accounting drifted")
	}
	if _, err := EncodePreparedRoundPage(page); err != nil {
		t.Fatal(err)
	}
	if _, err := EncodePreparedRoundPage(preparedRoundFixture(t, 103)); err == nil {
		t.Fatal("related descriptor overflow accepted")
	}
	if _, err := DecodePreparedRoundPage(bytes.Repeat([]byte(" "), MaximumPreparedRoundPageBytes+1)); err == nil {
		t.Fatal("encoded page overflow accepted")
	}
	page = preparedRoundFixture(t, 1)
	page.Items[0].Coverage.Gaps = []string{strings.Repeat("x", MaximumStringBytes+1)}
	if _, err := EncodePreparedRoundPage(page); err == nil {
		t.Fatal("oversized coverage metadata accepted")
	}
}

func TestPrivatePreparedExecutionRetainsFlattenedAuthority(t *testing.T) {
	projection := preparedRoundFixture(t, 1).Items[0].Execution
	encoded, err := canonicalJSON(projection)
	if err != nil {
		t.Fatal(err)
	}
	var decoded PreparedExecutionItem
	if err := strictjson.Decode(encoded, &decoded); err != nil {
		t.Fatal(err)
	}
	if err := ValidatePreparedExecutionItem(decoded); err != nil {
		t.Fatal(err)
	}
	var value map[string]any
	if err := json.Unmarshal(encoded, &value); err != nil {
		t.Fatal(err)
	}
	if value["accepted_plan"] == nil || value["item_key"] != projection.ItemKey || value["ExecutionItem"] != nil {
		t.Fatal("private projection embedding changed JSON shape")
	}
	compact, err := EncodeExecutionManifest(ExecutionManifest{Schema: ExecutionManifestSchema, Items: []ExecutionItem{projection.ExecutionItem}})
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(compact, []byte(`"accepted_plan"`)) || bytes.Contains(compact, []byte(`"dependencies"`)) {
		t.Fatal("private authority escaped into compact manifest")
	}
	decoded.AcceptedPlan = nil
	if ValidatePreparedExecutionItem(decoded) == nil {
		t.Fatal("private projection accepted missing plan")
	}
	decoded = projection
	copied := *decoded.AcceptedPlan
	copied.ItemKey = "different"
	decoded.AcceptedPlan = &copied
	if ValidatePreparedExecutionItem(decoded) == nil {
		t.Fatal("private projection accepted foreign plan")
	}
}
