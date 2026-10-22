package auditdomain

import (
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/strictjson"
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
	for _, c := range fixture.Projections {
		t.Run(c.Name, func(t *testing.T) {
			var value FindingCorrelation
			err := strictjson.DecodeNonNull(c.Document, &value)
			if err == nil {
				err = ValidateFindingCorrelation(value, c.AuditID, c.FindingID)
			}
			if (err == nil) != c.Valid {
				t.Fatalf("valid=%v want=%v err=%v", err == nil, c.Valid, err)
			}
		})
	}
}
