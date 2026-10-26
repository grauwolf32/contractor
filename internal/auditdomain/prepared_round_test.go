package auditdomain

import (
	"bytes"
	"fmt"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func preparedRoundFixture(t *testing.T, count int) PreparedRoundPage {
	t.Helper()
	page := PreparedRoundPage{Schema: PreparedRoundPageSchema, AuditID: "audit", RoundID: "round-two", Generation: "generation", PreviousPageDigest: DigestBytes(nil), Items: []PreparedRoundItem{}}
	for index := range count {
		plan := acceptedPlanFixture(t)
		plan.Source.ReceiptID = fmt.Sprintf("receipt-%03d", index)
		plan.Source.Proposal.SizeBytes = 8
		plan.Source.RetainedProposal.SizeBytes = 8
		plan.ItemKey = findingCheckItemKey(plan.Source.ReceiptID, 0)
		revision := "task-r1"
		task := contracts.ExactArtifact{Ref: contracts.ArtifactRef{Namespace: ArtifactNamespace("audit"), Name: fmt.Sprintf("task-%03d", index), Revision: &revision}, Digest: DigestBytes([]byte(fmt.Sprint(index))), MediaType: PackageMediaType, SizeBytes: 100}
		sourceRef := task.Ref.Clone()
		sourceRef.Name = fmt.Sprintf("source-%03d", index)
		item := PreparedRoundItem{ItemID: fmt.Sprintf("item-%03d", index), ItemKey: plan.ItemKey, Ordinal: index, Kind: "finding-verification", SubjectKey: plan.Source.ReceiptID, WorkflowRole: plan.Route.WorkflowRole, Task: task, InitialState: "ready", ApprovalKind: "none",
			Origin:          PreparedItemOrigin{Schema: ItemOriginSchema, SourceRef: sourceRef, SourceContentDigest: DigestBytes([]byte("source")), SourceMediaType: JSONMediaType, CanonicalInventoryDigest: DigestBytes([]byte("basis")), EntryKey: plan.ItemKey},
			Coverage:        RoundResultCoverage{Status: CoverageNotTested, Requested: []string{"verify"}, Completed: []string{}, Gaps: []string{}},
			ProposalSources: []PreparedProposalSource{{ReceiptID: plan.Source.ReceiptID, ProposedCheckOrdinal: 0, Proposal: plan.Source.RetainedProposal}},
			Execution:       ExecutionItem{ItemKey: plan.ItemKey, Ordinal: index, SubjectKey: plan.Source.ReceiptID, TaskPackageID: fmt.Sprintf("task-package-%03d", index), TaskPackageDigest: task.Digest, TaskRef: &task.Ref, Inputs: []ExactInput{}, AcceptedPlan: &plan}}
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
	page.Schema = RoundSelectionPageSchema
	selectionPage, err := EncodeRoundSelectionPage(page)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := DecodePreparedRoundPage(selectionPage); err == nil {
		t.Fatal("selection page accepted as prepared membership")
	}
	digest := DigestBytes([]byte("pin"))
	selection := RoundSelectionRoot{Schema: RoundSelectionRootSchema, AuditID: page.AuditID, RoundID: page.RoundID, Generation: page.Generation, ProfileDigest: digest, BaselineDigest: digest, PageCount: 1, ItemCount: 2, PagesDigest: DigestBytes(selectionPage)}
	selectionBytes, err := EncodeRoundSelectionRoot(selection)
	if err != nil {
		t.Fatal(err)
	}
	revision := "selection-r1"
	source := contracts.ExactContent(contracts.ArtifactRef{Namespace: ArtifactNamespace("audit"), Name: "selection-root", Revision: &revision}, RoundSelectionRootMediaType, selectionBytes)
	root := PreparedRoundRoot{Schema: PreparedRoundRootSchema, AuditID: page.AuditID, RoundID: page.RoundID, Generation: page.Generation, ProfileDigest: digest, BaselineDigest: digest, Source: source, PageCount: 1, ItemCount: 2, PagesDigest: DigestBytes(data)}
	encoded, err := EncodePreparedRoundRoot(root)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := DecodePreparedRoundRoot(encoded); err != nil {
		t.Fatal(err)
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
