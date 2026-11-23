package findingintake

import (
	"encoding/json"
	"errors"
	"fmt"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
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
	if canonical.request.SubmissionID != stableSubmissionID(input.InvocationID, input.Proposal.ClientKey) ||
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
			encoded, err := auditdomain.EncodeFindingProposal(input.Proposal)
			if err != nil {
				t.Fatalf("proposal codec rejected retained document: %v", err)
			}
			if _, err := auditdomain.DecodeFindingProposal(encoded); err != nil {
				t.Fatalf("proposal codec could not read retained document: %v", err)
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
		SubmissionID: stableSubmissionID(invocationID, clientKey),
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

func TestSharedInformationalPayloadLimits(t *testing.T) {
	var fixture struct {
		MaxEvidence, MaxAuthoredBytes int
		Severities                    []string
		Payloads                      []struct {
			Name               string
			Sources, TextBytes int
			AuthoredBytes      int
			Valid              bool
		}
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "finding-fact-cases.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.MaxEvidence != MaxFactEvidenceRefs || fixture.MaxAuthoredBytes != MaxFactAuthoredBytes {
		t.Fatal("fact limit drift")
	}
	for _, c := range fixture.Payloads {
		t.Run(c.Name, func(t *testing.T) {
			refs := make([]contracts.ArtifactRef, c.Sources)
			revision := "r1"
			for i := range refs {
				refs[i] = contracts.ArtifactRef{Namespace: "proof", Name: fmt.Sprint("proof-", i), Revision: &revision}
			}
			input := testSubmission("invocation", "fact", refs)
			input.Proposal.SeveritySuggestion = "informational"
			input.Proposal.Subject = nil
			input.Proposal.Description = strings.Repeat("x", c.TextBytes)
			if c.AuthoredBytes > 0 {
				encoded, err := auditdomain.EncodeFindingProposal(input.Proposal)
				if err != nil {
					t.Fatal(err)
				}
				input.Proposal.Description = strings.Repeat("x", c.AuthoredBytes-len(encoded)+c.TextBytes)
			}
			_, err := canonicalize(input)
			if (err == nil) != c.Valid {
				t.Fatalf("valid=%t err=%v", c.Valid, err)
			}
		})
	}
	for _, severity := range fixture.Severities {
		input := testSubmission("invocation", "severity", []contracts.ArtifactRef{{Namespace: "proof", Name: "one", Revision: ptrString("r1")}})
		input.Proposal.SeveritySuggestion = severity
		if _, err := canonicalize(input); err != nil {
			t.Fatal(severity, err)
		}
	}
}

func ptrString(value string) *string { return &value }

func TestDirectSubmissionRejectsRemovedPublicationFlag(t *testing.T) {
	var input Submission
	if err := strictjson.Decode([]byte(`{"publishProblem":true}`), &input); err == nil {
		t.Fatal("removed field accepted")
	}
}

func TestDirectFindingSourcesFollowCommonModelVisibility(t *testing.T) {
	for _, ref := range []contracts.ArtifactRef{{Namespace: "worker", Name: "memory.private", Revision: ptrString("r1")}, {Namespace: "worker", Name: "http.body.private", Revision: ptrString("r1")}, {Namespace: "skills", Name: "trace", Revision: ptrString("r1")}} {
		input := testSubmission("invocation", "hidden", []contracts.ArtifactRef{ref})
		if _, err := canonicalize(input); !errors.Is(err, ErrInvalid) {
			t.Fatalf("hidden source accepted: %s/%s %v", ref.Namespace, ref.Name, err)
		}
	}
}
