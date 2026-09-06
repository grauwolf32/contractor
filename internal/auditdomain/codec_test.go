package auditdomain

import (
	"bytes"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestAuditEnvelopeCodecsAndResultMembership(t *testing.T) {
	revision := "rev-1"
	manifest := ExecutionManifest{
		Schema: ExecutionManifestSchema,
		Items: []ExecutionItem{
			{ItemKey: "item-1", Ordinal: 0, SubjectKey: "subject-1", TaskPackageID: "task-1", TaskPackageDigest: testDigest('a'), TaskRef: &contracts.ArtifactRef{Namespace: "inputs", Name: "task", Revision: &revision}, Inputs: []ExactInput{}},
			{ItemKey: "item-2", Ordinal: 1, SubjectKey: "subject-2", TaskPackageID: "task-2", TaskPackageDigest: testDigest('b'), Inputs: []ExactInput{}},
		},
	}
	encoded, err := EncodeExecutionManifest(manifest)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := DecodeExecutionManifest(encoded)
	if err != nil || len(decoded.Items) != 2 {
		t.Fatalf("round trip = %+v, %v", decoded, err)
	}
	if !bytes.Equal(encoded, mustCanonical(t, decoded)) {
		t.Fatal("encoder did not produce canonical JSON")
	}
	digest, err := DigestExecutionManifest(manifest)
	if err != nil {
		t.Fatal(err)
	}
	valid := CheckResultSet{
		Schema: CheckResultsSchema, ExecutionManifestDigest: digest,
		Results: []CheckResult{
			{ItemKey: "item-2", SubjectKey: "subject-2", Assessment: "inconclusive", Summary: "not enough evidence", EvidenceIDs: []string{}, Coverage: ResultCoverage{Requested: []string{}, Completed: []string{}, Gaps: []string{"unknown"}}, Proposals: []ProposalSelection{}},
			{ItemKey: "item-1", SubjectKey: "subject-1", Assessment: "supported", Summary: "observed", EvidenceIDs: []string{"ev-2", "ev-1"}, Coverage: ResultCoverage{Requested: []string{"validation", "trace"}, Completed: []string{"trace", "validation"}, Gaps: []string{}}, Proposals: []ProposalSelection{{InvocationID: "invocation-1", ClientKey: "candidate-1"}}},
		},
	}
	if err := ValidateResultSet(valid, manifest); err != nil {
		t.Fatal(err)
	}
	resultBytes, err := EncodeCheckResultSet(valid)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := DecodeCheckResultSet(resultBytes); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name   string
		mutate func(*CheckResultSet)
	}{
		{name: "missing", mutate: func(value *CheckResultSet) { value.Results = value.Results[:1] }},
		{name: "duplicate", mutate: func(value *CheckResultSet) { value.Results[1] = value.Results[0] }},
		{name: "foreign", mutate: func(value *CheckResultSet) { value.Results[1].ItemKey = "item-foreign" }},
		{name: "extra", mutate: func(value *CheckResultSet) {
			value.Results = append(value.Results, value.Results[0])
			value.Results[2].ItemKey = "item-extra"
			value.Results[2].SubjectKey = "subject-extra"
		}},
		{name: "wrong subject", mutate: func(value *CheckResultSet) { value.Results[1].SubjectKey = "subject-wrong" }},
		{name: "wrong manifest", mutate: func(value *CheckResultSet) { value.ExecutionManifestDigest = testDigest('c') }},
		{name: "duplicate proposal", mutate: func(value *CheckResultSet) {
			value.Results[1].Proposals = []ProposalSelection{
				{InvocationID: "invocation-1", ClientKey: "candidate-1"},
				{InvocationID: "invocation-1", ClientKey: "candidate-1"},
			}
		}},
		{name: "unsorted proposals", mutate: func(value *CheckResultSet) {
			value.Results[1].Proposals = []ProposalSelection{
				{InvocationID: "invocation-2", ClientKey: "candidate-2"},
				{InvocationID: "invocation-1", ClientKey: "candidate-1"},
			}
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			candidate := valid
			candidate.Results = append([]CheckResult(nil), valid.Results...)
			for index := range candidate.Results {
				candidate.Results[index].Proposals = append([]ProposalSelection(nil), valid.Results[index].Proposals...)
			}
			test.mutate(&candidate)
			if err := ValidateResultSet(candidate, manifest); ErrorCode(err) != CodeResultSetInvalid {
				t.Fatalf("error = %v", err)
			}
		})
	}
}

func TestDispatchExecutionManifestRequiresExactTaskRefs(t *testing.T) {
	revision := "rev-1"
	manifest := ExecutionManifest{Schema: ExecutionManifestSchema, Items: []ExecutionItem{{
		ItemKey: "item-1", Ordinal: 0, SubjectKey: "subject-1", TaskPackageID: "task-1",
		TaskPackageDigest: testDigest('a'), Inputs: []ExactInput{},
	}}}
	if err := ValidateDispatchExecutionManifest(manifest); ErrorCode(err) != CodeReferenceInvalid {
		t.Fatalf("missing exact task ref error = %v", err)
	}
	manifest.Items[0].TaskRef = &contracts.ArtifactRef{Namespace: "inputs", Name: "task", Revision: &revision}
	if err := ValidateDispatchExecutionManifest(manifest); err != nil {
		t.Fatal(err)
	}
}

func TestAuditEnvelopeDecodeRejectsUnknownAndDuplicateKeys(t *testing.T) {
	digest := testDigest('a')
	unknown := []byte(`{"schema":"contractor.audit.check-results.v1","execution_manifest_digest":"` + digest + `","results":[],"unknown":true}`)
	if _, err := DecodeCheckResultSet(unknown); ErrorCode(err) != CodeInvalid {
		t.Fatalf("unknown-field error = %v", err)
	}
	duplicate := []byte(`{"schema":"contractor.audit.check-results.v1","schema":"contractor.audit.check-results.v1","execution_manifest_digest":"` + digest + `","results":[]}`)
	if _, err := DecodeCheckResultSet(duplicate); ErrorCode(err) != CodeInvalid {
		t.Fatalf("duplicate-field error = %v", err)
	}
	nestedDuplicate := []byte(`{"schema":"contractor.audit.check-results.v1","execution_manifest_digest":"` + digest + `","results":[{"item_key":"item","subject_key":"subject","assessment":"supported","summary":"one","summary":"two","evidence_ids":[],"coverage":{"requested":[],"completed":[],"gaps":[]},"proposals":[]}]}`)
	if _, err := DecodeCheckResultSet(nestedDuplicate); ErrorCode(err) != CodeInvalid {
		t.Fatalf("nested duplicate error = %v", err)
	}
	unpairedSurrogate := []byte(`{"schema":"contractor.audit.check-results.v1","execution_manifest_digest":"` + digest + `","results":[{"item_key":"item","subject_key":"subject","assessment":"supported","summary":"\ud800","evidence_ids":[],"coverage":{"requested":[],"completed":[],"gaps":[]},"proposals":[]}]}`)
	if _, err := DecodeCheckResultSet(unpairedSurrogate); ErrorCode(err) != CodeInvalid {
		t.Fatalf("unpaired surrogate error = %v", err)
	}
	validPair := []byte(`{"value":"\ud83d\ude00"}`)
	if _, err := parseStrictJSON(validPair); err != nil {
		t.Fatalf("valid surrogate pair rejected: %v", err)
	}
}

func TestAllAuditEnvelopeCodecsAcceptBoundedEmptyCollections(t *testing.T) {
	worklist := WorklistManifest{Schema: WorklistSchema, Round: 1, Items: []WorklistItem{}}
	if encoded, err := EncodeWorklist(worklist); err != nil || len(encoded) == 0 {
		t.Fatalf("empty worklist = %q, %v", encoded, err)
	}
	execution := ExecutionManifest{Schema: ExecutionManifestSchema, Items: []ExecutionItem{}}
	if encoded, err := EncodeExecutionManifest(execution); err != nil || len(encoded) == 0 {
		t.Fatalf("empty execution manifest = %q, %v", encoded, err)
	}
	coverage := CoverageEnvelope{Schema: CoverageSchema, Rows: []CoverageRow{}}
	if encoded, err := EncodeCoverage(coverage); err != nil || len(encoded) == 0 {
		t.Fatalf("empty coverage = %q, %v", encoded, err)
	}
	evidence := EvidenceEnvelope{Schema: EvidenceSchema, Evidence: []Evidence{}}
	if encoded, err := EncodeEvidence(evidence); err != nil || len(encoded) == 0 {
		t.Fatalf("empty evidence = %q, %v", encoded, err)
	}
	proposal := FindingProposal{
		Schema: FindingProposalSchema, ClientKey: "candidate-1", Title: "Candidate", Description: "Description",
		Subject: FindingSubject{Kind: "operation", Key: "op-1"}, Hypothesis: "A hypothesis.",
		Preconditions: []string{}, StandardRefs: []StandardReference{}, EvidenceIDs: []string{},
		ProposedChecks: []ProposedCheck{}, SeveritySuggestion: "", Limitations: []string{},
	}
	if encoded, err := EncodeFindingProposal(proposal); err != nil || len(encoded) == 0 {
		t.Fatalf("proposal = %q, %v", encoded, err)
	}
	proposal.Hypothesis = ""
	if encoded, err := EncodeFindingProposal(proposal); err != nil || bytes.Contains(encoded, []byte(`"hypothesis"`)) {
		t.Fatalf("optional proposal hypothesis = %q, %v", encoded, err)
	}
}

func testDigest(character byte) string {
	return "sha256:" + strings.Repeat(string(character), 64)
}

func mustCanonical(t *testing.T, value any) []byte {
	t.Helper()
	encoded, err := canonicalJSON(value)
	if err != nil {
		t.Fatal(err)
	}
	return encoded
}
