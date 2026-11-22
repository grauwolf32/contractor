package auditcontract

import (
	"encoding/json"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"testing"
)

func TestSharedAuditEvidenceSnapshotRefs(t *testing.T) {
	var cases []struct {
		Source   contracts.ArtifactRef
		Snapshot contracts.ArtifactRef
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "audit-evidence-snapshot-refs.json"), &cases); err != nil {
		t.Fatal(err)
	}
	if len(cases) == 0 {
		t.Fatal("empty snapshot fixture")
	}
	for _, c := range cases {
		ref, err := EvidenceSnapshotRef(c.Source)
		if err != nil || ref != c.Snapshot {
			t.Fatalf("snapshot %v %v", ref, err)
		}
	}
}
