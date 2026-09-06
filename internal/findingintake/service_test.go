package findingintake

import (
	"errors"
	"fmt"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestCanonicalSubmissionPinsStableIdentityAndOptionalHypothesis(t *testing.T) {
	revision := "rev-1"
	input := testSubmission("worker-invocation-1", "candidate-1", []contracts.ArtifactRef{{
		Namespace: "worker", Name: "trace", Revision: &revision,
	}})
	canonical, err := canonicalize(input)
	if err != nil {
		t.Fatal(err)
	}
	if canonical.request.SubmissionID != StableSubmissionID(input.InvocationID, input.Proposal.ClientKey) ||
		canonical.digest == "" || string(canonical.proposalBytes) == "" {
		t.Fatalf("canonical finding submission = %+v", canonical)
	}
	changed := input
	changed.Proposal.Title = "Changed title"
	changedCanonical, err := canonicalize(changed)
	if err != nil {
		t.Fatal(err)
	}
	if changedCanonical.digest == canonical.digest {
		t.Fatal("changed finding content retained the same request digest")
	}
	forged := input
	forged.SubmissionID = "finding-forged"
	if _, err := canonicalize(forged); !errors.Is(err, ErrInvalid) {
		t.Fatalf("forged submission identity error = %v", err)
	}
}

func testSubmission(
	invocationID, clientKey string,
	evidence []contracts.ArtifactRef,
) Submission {
	ids := make([]string, len(evidence))
	for index := range ids {
		ids[index] = fmt.Sprintf("evidence-%d", index+1)
	}
	return Submission{
		APIVersion: APIVersion, InvocationID: invocationID,
		SubmissionID: StableSubmissionID(invocationID, clientKey),
		Proposal: auditdomain.FindingProposal{
			Schema: auditdomain.FindingProposalSchema, ClientKey: clientKey,
			Title: "Candidate", Description: "Candidate description",
			Subject:       auditdomain.FindingSubject{Kind: "code", Key: "handler"},
			Preconditions: []string{}, StandardRefs: []auditdomain.StandardReference{},
			EvidenceIDs: ids, ProposedChecks: []auditdomain.ProposedCheck{},
			Limitations: []string{},
		},
		EvidenceRefs: evidence,
	}
}
