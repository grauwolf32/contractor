package privateartifacts

import (
	"bytes"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/httpapi/artifacttransfer"
)

func TestFindingSubmissionDecoderDistinguishesNULEscapeFromLiteralText(t *testing.T) {
	for _, test := range []struct {
		name, description string
		invalid           bool
	}{
		{"NUL escape", "a\x00b", true},
		{"escaped backslash", `a\u0000b`, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			input := privateNULSubmission(test.description)
			body, err := json.Marshal(input)
			if err != nil {
				t.Fatal(err)
			}
			request := httptest.NewRequest(http.MethodPost, "/", bytes.NewReader(body))
			request, transfer, err := artifacttransfer.Acquire(httptest.NewRecorder(), request, artifactpgtest.NewStorage())
			if err != nil {
				t.Fatal(err)
			}
			defer transfer.Close()
			decoded, err := readFindingSubmission(request, transfer)
			if test.invalid {
				if !errors.Is(err, findingintake.ErrInvalid) {
					t.Fatalf("NUL submission decode error = %v", err)
				}
				return
			}
			if err != nil || decoded.Proposal.Description != test.description {
				t.Fatalf("literal text = %q, error = %v", decoded.Proposal.Description, err)
			}
			if _, err := auditdomain.EncodeFindingProposal(decoded.Proposal); err != nil {
				t.Fatalf("literal-text proposal is not valid: %v", err)
			}
		})
	}
}

func privateNULSubmission(description string) findingintake.Submission {
	return findingintake.Submission{
		APIVersion: findingintake.APIVersion, InvocationID: "worker-invocation-1",
		// Fixed Runtime derivation for this invocation and candidate-1; the
		// PostgreSQL test validates it with the production intake canonicalizer.
		SubmissionID: "finding-a7fcd33698853ef03c3d8706e4eb60c426bff8df3653e467f19c6c2f7aa2b948",
		Proposal: auditdomain.FindingProposal{
			Schema: auditdomain.FindingProposalSchema, ClientKey: "candidate-1",
			Title: "Candidate", Description: description,
			Subject:       &auditdomain.FindingSubject{Kind: "code", Key: "handler"},
			Preconditions: []string{}, StandardRefs: []auditdomain.StandardReference{},
			EvidenceIDs: []string{}, ProposedChecks: []auditdomain.ProposedCheck{},
			Limitations: []string{},
		},
		EvidenceRefs: []contracts.ArtifactRef{},
	}
}
