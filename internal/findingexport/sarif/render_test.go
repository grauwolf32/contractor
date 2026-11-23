package sarif

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/findingexport"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
)

func noAnalysisSnapshot(t *testing.T, count int) findingexport.Snapshot {
	t.Helper()
	request := findingexport.Request{ClientKey: "snapshot", Format: "sarif", Profile: "generic", Findings: []findingexport.Selection{}}
	for i := range count {
		request.Findings = append(request.Findings, findingexport.Selection{FindingID: fmt.Sprintf("finding-%03d", i), Revision: 1})
	}
	snapshot, err := findingexport.NewSnapshot("audit", request, time.Date(2026, 10, 30, 12, 0, 0, 0, time.UTC))
	if err != nil {
		t.Fatal(err)
	}
	for i, selection := range snapshot.Request.Findings {
		id := fmt.Sprintf("receipt-%03d", i)
		proposal := auditdomain.FindingProposal{Schema: auditdomain.FindingProposalSchema, ClientKey: id, Title: "Possible issue", Description: "No structured analysis is available.", Preconditions: []string{}, StandardRefs: []auditdomain.StandardReference{}, EvidenceIDs: []string{}, ProposedChecks: []auditdomain.ProposedCheck{}, Limitations: []string{}}
		data, err := auditdomain.EncodeFindingProposal(proposal)
		if err != nil {
			t.Fatal(err)
		}
		revision := "r1"
		doc := auditdomain.FindingCollectionDocument{Scope: auditdomain.FindingCollectionSource{Kind: "run", ID: "run"}, Ref: contracts.ArtifactRef{Namespace: "finding-proposals", Name: id, Revision: &revision}, Digest: contentdigest.Bytes(data), MediaType: "application/json", SizeBytes: int64(len(data))}
		doc.ID, err = auditdomain.FindingCollectionDocumentID(doc)
		if err != nil {
			t.Fatal(err)
		}
		if err := snapshot.AddReceipt(findingexport.Receipt{ReceiptID: id, ProposalID: "proposal-" + id, ProposalArtifact: findingexport.Evidence{Original: doc, Retained: doc}, Proposal: proposal, Evidence: []findingexport.Evidence{}, Origin: findingexport.Origin{RunID: "run", InvocationID: "invocation", StageExecutionID: "stage", AllocationID: "allocation", LogicalAgentName: "worker", WorkflowName: "check", WorkflowVersion: "1", WorkflowClosureDigest: contentdigest.Bytes([]byte("workflow"))}}); err != nil {
			t.Fatal(err)
		}
		if err := snapshot.AddResult(findingexport.Result{FindingID: selection.FindingID, Revision: 1, PresentationReceiptID: id, ReceiptIDs: []string{id}, Review: findingexport.Review{State: nil}}); err != nil {
			t.Fatal(err)
		}
	}
	return snapshot
}

func TestRenderNoAnalysisIsDeterministicAndHonorsDuplicateParameter(t *testing.T) {
	snapshot := noAnalysisSnapshot(t, 2)
	snapshot.Results[1].Review = findingexport.Review{State: findingStateForTest("duplicate"), Verdict: "duplicate", DuplicateTargetID: snapshot.Results[0].FindingID}
	first, _, diagnostics := rendered(t, snapshot)
	if len(diagnostics.Items) == 0 {
		t.Fatal("missing diagnostics")
	}
	second, _, _ := rendered(t, snapshot)
	if !bytes.Equal(first, second) || first[len(first)-1] != '\n' {
		t.Fatal("unstable rendering")
	}
	var document struct {
		Version string `json:"version"`
		Runs    []struct {
			Results []struct{ Kind, Level string } `json:"results"`
		} `json:"runs"`
	}
	if err := json.Unmarshal(first, &document); err != nil || document.Version != "2.1.0" || len(document.Runs) != 1 || len(document.Runs[0].Results) != 1 || document.Runs[0].Results[0].Kind != "review" || document.Runs[0].Results[0].Level != "none" {
		t.Fatalf("unexpected projection: %+v %v", document, err)
	}
	snapshot.Request.IncludeDuplicates = true
	_, digest, err := snapshot.Request.Normalize(snapshot.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	snapshot.RequestDigest = digest
	expanded, _, _ := rendered(t, snapshot)
	if err := json.Unmarshal(expanded, &document); err != nil || len(document.Runs[0].Results) != 2 {
		t.Fatal("includeDuplicates did not retain results")
	}
}

func TestRenderBoundsRepeatedPresentationBeforeAccumulatingOutput(t *testing.T) {
	snapshot := noAnalysisSnapshot(t, 256)
	shared := &snapshot.Receipts[0]
	shared.Proposal.Title = strings.Repeat("T", 32000)
	shared.Proposal.Description = strings.Repeat("D", 64000)
	for i := range snapshot.Results {
		r := &snapshot.Results[i]
		r.PresentationReceiptID = shared.ReceiptID
		if i > 0 {
			r.ReceiptIDs = append(r.ReceiptIDs, shared.ReceiptID)
			sort.Strings(r.ReceiptIDs)
		}
	}
	if _, err := findingexport.EncodeSnapshot(snapshot); err != nil {
		t.Fatal(err)
	}
	if _, _, err := Render(snapshot); !errors.Is(err, findingexport.ErrLimit) {
		t.Fatalf("accepted oversized repeated output: %v", err)
	}
}

func TestRenderDoesNotClaimAnotherExporterVersion(t *testing.T) {
	s := noAnalysisSnapshot(t, 1)
	s.ExporterVersion = "0.9.0"
	if _, _, err := Render(s); err == nil {
		t.Fatal("rendered under another implementation version")
	}
}

func findingStateForTest(value string) *string { return &value }
