package auditdomain

import "testing"

func TestDecodeCheckResultPackageValidatesMemberAndEvidenceClosure(t *testing.T) {
	manifest := ExecutionManifest{Schema: ExecutionManifestSchema, Items: []ExecutionItem{{
		ItemKey: "check-1", Ordinal: 0, SubjectKey: "subject-1",
		TaskPackageID: "task-1", TaskPackageDigest: "sha256:" + repeatHex("a", 64),
	}}}
	digest, err := DigestExecutionManifest(manifest)
	if err != nil {
		t.Fatal(err)
	}
	results, err := EncodeCheckResultSet(CheckResultSet{
		Schema: CheckResultsSchema, ExecutionManifestDigest: digest,
		Results: []CheckResult{{
			ItemKey: "check-1", SubjectKey: "subject-1", Assessment: "inconclusive",
			Summary: "A bounded gap remains.", EvidenceIDs: []string{"ev-1"},
			Coverage:  ResultCoverage{Requested: []string{"source"}, Completed: []string{}, Gaps: []string{"missing-source"}},
			Proposals: []string{},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	evidence, err := EncodeEvidence(EvidenceEnvelope{Schema: EvidenceSchema, Evidence: []Evidence{{
		ID: "ev-1", Kind: "source", Summary: "Observed source", ContentMemberID: "ev-body",
	}}})
	if err != nil {
		t.Fatal(err)
	}
	payload, _, err := BuildPackage("result-1", PackageKindCheckResults, "", []PackageInput{
		{ID: CheckResultsMemberID, Path: "check-results.json", MediaType: JSONMediaType, Data: results},
		{ID: EvidenceMemberID, Path: "evidence.json", MediaType: JSONMediaType, Data: evidence},
		{ID: "ev-body", Path: "evidence/ev-1.txt", MediaType: "text/plain", Data: []byte("bounded evidence")},
	})
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := DecodeCheckResultPackage(payload)
	if err != nil || decoded.Package.Digest == "" || len(decoded.Evidence.Evidence) != 1 {
		t.Fatalf("decoded result package = (%+v, %v)", decoded, err)
	}
	if err := ValidateResultSet(decoded.Results, manifest); err != nil {
		t.Fatalf("result membership: %v", err)
	}

	extra, _, err := BuildPackage("result-extra", PackageKindCheckResults, "", []PackageInput{
		{ID: CheckResultsMemberID, Path: "check-results.json", MediaType: JSONMediaType, Data: results},
		{ID: EvidenceMemberID, Path: "evidence.json", MediaType: JSONMediaType, Data: evidence},
		{ID: "ev-body", Path: "evidence/ev-1.txt", MediaType: "text/plain", Data: []byte("bounded evidence")},
		{ID: "smuggled", Path: "extra.txt", MediaType: "text/plain", Data: []byte("not referenced")},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := DecodeCheckResultPackage(extra); ErrorCode(err) != CodeResultSetInvalid {
		t.Fatalf("unreferenced member error = %v", err)
	}
}

func repeatHex(value string, count int) string {
	result := ""
	for len(result) < count {
		result += value
	}
	return result[:count]
}
