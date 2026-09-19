package auditstore

import (
	"bytes"
	"encoding/json"
	"testing"
)

func TestCollectionPreparationCountsExactArtifactsOnceAndKeepsLinks(t *testing.T) {
	first := testExact("audit", "report", "v1")
	first.SizeBytes = 10
	second := testExact("audit", "report", "v2")
	second.SizeBytes = 20
	params := CollectParams{SourceOutput: &first, Retained: []ArtifactLink{{LogicalKey: "one", Artifact: first}, {LogicalKey: "alias", Artifact: first}, {LogicalKey: "next", Artifact: second}}, Items: []CollectionItem{{ExecutionItemID: "member"}}}
	prepared := prepareCollectionWrite(params)
	if prepared.retainedBytes != 30 {
		t.Fatalf("retained bytes=%d, want 30", prepared.retainedBytes)
	}
	var links []artifactLinkJSON
	if err := json.Unmarshal(prepared.links, &links); err != nil {
		t.Fatal(err)
	}
	if len(links) != 3 || links[1].LogicalKey != "alias" {
		t.Fatal("artifact accounting removed logical links")
	}
	if !bytes.Contains(prepared.items, []byte(`"finding_associations":[]`)) || !bytes.Contains(prepared.items, []byte(`"requested":[]`)) {
		t.Fatal("empty collections changed to null")
	}
	digest := *prepared.sourceDigest
	first.Digest = "mutated"
	if *prepared.sourceDigest != digest {
		t.Fatal("source digest aliases parameters")
	}
	empty := prepareCollectionWrite(CollectParams{})
	if string(empty.items) != "[]" || string(empty.links) != "[]" || string(empty.retained) != "[]" || empty.sourceDigest != nil || empty.sourceRef != nil {
		t.Fatal("empty projection changed")
	}
}
func TestExecutionPreparationDetachesOptionalIdentityAndPreservesOrder(t *testing.T) {
	round, attempt := "round", 2
	params := CreateExecutionIntentParams{RoundID: &round, RoleAttempt: &attempt, Members: []ExecutionMemberIntent{{ExecutionItemID: "second"}, {ExecutionItemID: "first"}}}
	prepared := prepareExecutionIntentWrite(params)
	round = "changed"
	attempt = 8
	params.Members[0].ExecutionItemID = "changed"
	var members []executionMemberJSON
	if err := json.Unmarshal(prepared.members, &members); err != nil {
		t.Fatal(err)
	}
	if *prepared.roundID != "round" || *prepared.roleAttempt != 2 || members[0].ExecutionItemID != "second" || members[1].ExecutionItemID != "first" || members[0].Inputs == nil {
		t.Fatal("prepared identity, member order or empty input shape changed")
	}
	empty := prepareExecutionIntentWrite(CreateExecutionIntentParams{})
	if empty.roundID != nil || empty.roleAttempt != nil || string(empty.members) != "[]" {
		t.Fatal("optional identity changed")
	}
}
