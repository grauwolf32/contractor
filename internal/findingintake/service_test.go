package findingintake

import (
	"errors"
	"fmt"
	"strings"
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

func TestCanonicalSubmissionRejectsLimitationsThatCannotEnterAuditCoverage(t *testing.T) {
	for _, test := range []struct {
		name        string
		limitations []string
	}{
		{name: "duplicate", limitations: []string{"needs-live", "needs-live"}},
		{name: "over 512 bytes", limitations: []string{strings.Repeat("x", auditdomain.MaximumCoverageValueBytes+1)}},
		{name: "multibyte over 512 bytes", limitations: []string{strings.Repeat("é", auditdomain.MaximumCoverageValueBytes/2+1)}},
	} {
		t.Run(test.name, func(t *testing.T) {
			input := testSubmission("worker-invocation-1", "candidate-1", nil)
			input.Proposal.ProposedChecks = []auditdomain.ProposedCheck{{Objective: "Verify the finding", Method: "static"}}
			input.Proposal.Limitations = test.limitations
			legacy, err := auditdomain.EncodeFindingProposal(input.Proposal)
			if err != nil {
				t.Fatalf("legacy proposal codec rejected retained document: %v", err)
			}
			if _, err := auditdomain.DecodeFindingProposal(legacy); err != nil {
				t.Fatalf("legacy proposal codec could not read retained document: %v", err)
			}
			if _, err := canonicalize(input); !errors.Is(err, ErrInvalid) {
				t.Fatalf("invalid proposed-check limitations admitted: %v", err)
			}
		})
	}
	input := testSubmission("worker-invocation-1", "candidate-1", nil)
	input.Proposal.ProposedChecks = []auditdomain.ProposedCheck{{Objective: "Verify the finding", Method: "static"}}
	input.Proposal.Limitations = []string{strings.Repeat("x", auditdomain.MaximumCoverageValueBytes)}
	if _, err := canonicalize(input); err != nil {
		t.Fatalf("512-byte limitation rejected: %v", err)
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
			Subject:       &auditdomain.FindingSubject{Kind: "code", Key: "handler"},
			Preconditions: []string{}, StandardRefs: []auditdomain.StandardReference{},
			EvidenceIDs: ids, ProposedChecks: []auditdomain.ProposedCheck{},
			Limitations: []string{},
		},
		EvidenceRefs: evidence,
	}
}
