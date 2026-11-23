package findingexport

import (
	"bytes"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

func testSnapshot(t *testing.T, count int) Snapshot {
	t.Helper()
	request := Request{ClientKey: "selection", Format: "sarif", Profile: "generic", Findings: []Selection{}}
	for i := range count {
		request.Findings = append(request.Findings, Selection{FindingID: fmt.Sprintf("finding-%03d", i), Revision: 1})
	}
	s, err := NewSnapshot("audit-one", request, time.Date(2026, 10, 30, 12, 0, 0, 0, time.UTC))
	if err != nil {
		t.Fatal(err)
	}
	for i, selected := range s.Request.Findings {
		id := fmt.Sprintf("receipt-%03d", i)
		proposal := auditdomain.FindingProposal{Schema: auditdomain.FindingProposalSchema, ClientKey: id, Title: "Same title", Description: "Same description", Subject: &auditdomain.FindingSubject{Kind: "component", Key: "same"}, Preconditions: []string{}, StandardRefs: []auditdomain.StandardReference{}, EvidenceIDs: []string{}, ProposedChecks: []auditdomain.ProposedCheck{}, Limitations: []string{}}
		proposalBytes, err := auditdomain.EncodeFindingProposal(proposal)
		if err != nil {
			t.Fatal(err)
		}
		revision := "revision-1"
		doc := auditdomain.FindingCollectionDocument{Scope: auditdomain.FindingCollectionSource{Kind: "run", ID: "run"}, Ref: contracts.ArtifactRef{Namespace: "finding-proposals", Name: id, Revision: &revision}, Digest: contentdigest.Bytes(proposalBytes), MediaType: "application/json", SizeBytes: int64(len(proposalBytes))}
		doc.ID, err = auditdomain.FindingCollectionDocumentID(doc)
		if err != nil {
			t.Fatal(err)
		}
		err = s.AddReceipt(Receipt{ProposalArtifact: Evidence{Original: doc, Retained: doc}, ReceiptID: id, ProposalID: "proposal-" + id, Proposal: proposal, Evidence: []Evidence{}, Origin: Origin{RunID: "run", InvocationID: "invocation", StageExecutionID: "stage", AllocationID: "allocation", LogicalAgentName: "worker", WorkflowName: "review", WorkflowVersion: "1", WorkflowClosureDigest: contentdigest.Bytes([]byte("workflow"))}})
		if err != nil {
			t.Fatal(err)
		}
		if err := s.AddResult(Result{FindingID: selected.FindingID, Revision: 1, PresentationReceiptID: id, ReceiptIDs: []string{id}, Review: Review{State: nil}}); err != nil {
			t.Fatal(err)
		}
	}
	return s
}

func assignKey(t *testing.T, rows []Result, indices ...int) {
	t.Helper()
	value := strings.Repeat("a", 64)
	key, err := auditdomain.FindingGroupKey("project-one", "contractor/producer-issue/v1", value)
	if err != nil {
		t.Fatal(err)
	}
	for _, i := range indices {
		rows[i].PrimaryKey = &PrimaryKey{ProjectID: "project-one", Algorithm: "contractor/producer-issue/v1", Value: value, GroupKey: key, Policy: auditdomain.FindingCorrelationPolicy}
	}
}

func TestDuplicateParameterProjectsSelectedGroupsWithoutChangingReviews(t *testing.T) {
	s := testSnapshot(t, 6)
	assignKey(t, s.Results, 0)
	s.Results[1].Review = Review{State: findingStateForTest("duplicate"), Verdict: "duplicate", DuplicateTargetID: s.Results[0].FindingID}
	s.Results[2].Review = Review{State: findingStateForTest("duplicate"), Verdict: "duplicate", DuplicateTargetID: s.Results[1].FindingID}
	// Two selected duplicates share a target that is not selected. Neither
	// implicitly selects it nor loses its own independent triage.
	for _, i := range []int{3, 4} {
		s.Results[i].Review = Review{State: findingStateForTest("duplicate"), Verdict: "duplicate", DuplicateTargetID: "unselected"}
	}
	before, err := EncodeSnapshot(s)
	if err != nil {
		t.Fatal(err)
	}
	groups, err := s.ResultGroups()
	if err != nil {
		t.Fatal(err)
	}
	if len(groups) != 3 || !reflect.DeepEqual(groups[0].FindingIDs, []string{"finding-000", "finding-001", "finding-002"}) || !reflect.DeepEqual(groups[1].FindingIDs, []string{"finding-003", "finding-004"}) || groups[2].Result.FindingID != "finding-005" {
		t.Fatalf("groups: %+v", groups)
	}
	if groups[0].Result.FindingID != "finding-000" || !groups[1].Result.Review.IsState("duplicate") {
		t.Fatal("representative/rating changed")
	}
	after, _ := EncodeSnapshot(s)
	if !bytes.Equal(before, after) {
		t.Fatal("projection changed frozen input")
	}
	s.Request.IncludeDuplicates = true
	_, s.RequestDigest, err = s.Request.Normalize(s.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	groups, err = s.ResultGroups()
	if err != nil || len(groups) != 6 {
		t.Fatalf("include all: %d %v", len(groups), err)
	}
}

func TestDuplicateParameterDefaultAndReplayIdentity(t *testing.T) {
	base := `{"clientKey":"selection","format":"sarif","profile":"generic","findings":[]}`
	var omitted, explicit, included Request
	for _, tc := range []struct {
		data   string
		target *Request
	}{{base, &omitted}, {strings.Replace(base, `"findings"`, `"includeDuplicates":false,"findings"`, 1), &explicit}, {strings.Replace(base, `"findings"`, `"includeDuplicates":true,"findings"`, 1), &included}} {
		if err := strictjson.DecodeNonNull([]byte(tc.data), tc.target); err != nil {
			t.Fatal(err)
		}
	}
	for _, audit := range []string{"", "audit/foreign", "audit\nheader", strings.Repeat("a", 257)} {
		if _, _, err := omitted.Normalize(audit); err == nil {
			t.Fatalf("accepted invalid Audit ID %q", audit)
		}
	}
	_, a, _ := omitted.Normalize("audit-one")
	_, b, _ := explicit.Normalize("audit-one")
	_, c, _ := included.Normalize("audit-one")
	_, otherAudit, _ := omitted.Normalize("audit-two")
	if a != b || a == c || a == otherAudit {
		t.Fatal("incorrect normalized request identity")
	}
	for _, value := range []string{`null`, `"false"`, `0`, `[]`} {
		var r Request
		if strictjson.DecodeNonNull([]byte(strings.Replace(base, `"findings"`, `"includeDuplicates":`+value+`,"findings"`, 1)), &r) == nil {
			t.Fatalf("accepted %s", value)
		}
	}
	for _, data := range []string{strings.Replace(base, `"findings"`, `"IncludeDuplicates":false,"findings"`, 1), strings.Replace(base, `"findings"`, `"includeDuplicates":false,"includeDuplicates":true,"findings"`, 1)} {
		var r Request
		if strictjson.DecodeNonNull([]byte(data), &r) == nil {
			t.Fatal("accepted unknown/duplicate field")
		}
	}
}

func TestSnapshotCanonicalRoundTripAndSelectionBounds(t *testing.T) {
	for _, n := range []int{0, 1, 256} {
		s := testSnapshot(t, n)
		encoded, err := EncodeSnapshot(s)
		if err != nil {
			t.Fatal(err)
		}
		decoded, err := DecodeSnapshot(encoded)
		if err != nil {
			t.Fatal(err)
		}
		second, _ := EncodeSnapshot(decoded)
		if !bytes.Equal(encoded, second) {
			t.Fatal("unstable snapshot")
		}
		groups, err := decoded.ResultGroups()
		if err != nil || len(groups) != n {
			t.Fatalf("similar prose was merged: %d %v", len(groups), err)
		}
	}
	s := testSnapshot(t, 1)
	s.Request.Findings = append(s.Request.Findings, s.Request.Findings[0])
	if _, _, err := s.Request.Normalize(s.AuditID); err == nil {
		t.Fatal("accepted repeated selector")
	}
}

func TestSnapshotHistoricalExporterVersionIsProvenanceNotLiveCode(t *testing.T) {
	s := testSnapshot(t, 1)
	s.ExporterVersion = "0.9.0"
	encoded, err := EncodeSnapshot(s)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := DecodeSnapshot(encoded)
	if err != nil || decoded.ExporterVersion != "0.9.0" {
		t.Fatalf("frozen metadata required the live renderer version: %+v %v", decoded, err)
	}
}

func findingStateForTest(value string) *string { return &value }

// Frozen exports cannot smuggle a current assessment absent from the canonical
// model, even when their JSON is structurally valid and all source bytes exist.
func TestSnapshotRejectsInventedOrIncoherentCurrentAssessment(t *testing.T) {
	for _, review := range []Review{
		{State: nil, Verdict: "true_positive", Severity: "high"},
		{State: findingStateForTest("confirmed")},
		{State: findingStateForTest("confirmed"), Verdict: "false_positive"},
		{State: findingStateForTest("rejected"), Verdict: "false_positive", Severity: "high"},
		{State: findingStateForTest("proposed"), Verdict: "reopen", DuplicateTargetID: "other"},
		{State: findingStateForTest("duplicate"), Verdict: "duplicate"},
	} {
		snapshot := testSnapshot(t, 1)
		snapshot.Results[0].Review = review
		if _, err := EncodeSnapshot(snapshot); err == nil {
			t.Fatalf("invalid current assessment was frozen: %+v", review)
		}
	}
}
