package pgstore

import (
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditstore"
)

func TestCollectionPreparationCountsExactArtifactsOnceAndKeepsLinks(t *testing.T) {
	first := testExact("audit", "report", "v1")
	first.SizeBytes = 10
	second := testExact("audit", "report", "v2")
	second.SizeBytes = 20
	params := auditstore.CollectParams{SourceOutput: &first, Retained: []auditstore.ArtifactLink{{Scope: "project", LogicalKey: "one", Artifact: first}, {Scope: "project", LogicalKey: "alias", Artifact: first}, {Scope: "project", LogicalKey: "next", Artifact: second}}, Tasks: []auditstore.TaskCollection{{ExecutionItemID: "member"}}}
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
	digest := *prepared.sourceDigest
	first.Digest = "mutated"
	if *prepared.sourceDigest != digest {
		t.Fatal("source digest aliases parameters")
	}
	empty := prepareCollectionWrite(auditstore.CollectParams{})
	if string(empty.items) != "[]" || string(empty.links) != "[]" || string(empty.retained) != "[]" || empty.sourceDigest != nil || empty.sourceRef != nil {
		t.Fatal("empty projection changed")
	}
}
