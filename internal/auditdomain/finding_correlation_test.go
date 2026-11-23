package auditdomain

import (
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestSharedFindingCorrelationContracts(t *testing.T) {
	var fixture struct {
		Vectors []struct {
			ProjectID string
			Algorithm string
			Value     string
			GroupKey  string
		}
		Projections []struct {
			Name      string
			Valid     bool
			AuditID   string
			FindingID string
			Document  json.RawMessage
		}
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "public", "finding-correlation-cases.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	for _, v := range fixture.Vectors {
		key, err := FindingGroupKey(v.ProjectID, v.Algorithm, v.Value)
		if err != nil || key != v.GroupKey {
			t.Fatalf("group key=%s want=%s %v", key, v.GroupKey, err)
		}
	}
}
