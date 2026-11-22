package auditcontract_test

import (
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

func TestSharedTaskStopReadContract(t *testing.T) {
	var fixture struct {
		Cases []struct {
			Name     string
			AuditID  string `json:"audit_id"`
			Checking bool
			Valid    bool
			Document string
		}
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "audit-stops.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	if len(fixture.Cases) == 0 {
		t.Fatal("no shared stop cases")
	}
	for _, test := range fixture.Cases {
		t.Run(test.Name, func(t *testing.T) {
			var stop auditcontract.TaskStop
			err := strictjson.Decode([]byte(test.Document), &stop)
			if err == nil {
				err = stop.Validate(test.AuditID, test.Checking)
			}
			if (err == nil) != test.Valid {
				t.Fatalf("valid=%t error=%v", test.Valid, err)
			}
		})
	}
}

func TestNormalizedJudgmentSubmission64AndStrictGeneration(t *testing.T) {
	input := auditcontract.JudgmentSubmission{APIVersion: "contractor/v1alpha1", InvocationID: "invocation", SubmissionID: "batch", Mode: "batch", Items: []auditcontract.JudgmentUpdate{}}
	for index := range 64 {
		input.Items = append(input.Items, auditcontract.JudgmentUpdate{Conclusion: auditcontract.Conclusion{ItemKey: auditcontract.ExecutionItemKey(index), Rationale: "Bounded exploratory observation", TestedConditions: []string{}, Evidence: []auditcontract.Material{}, Limitations: []string{}}})
	}
	if err := input.Validate(); err != nil {
		t.Fatal(err)
	}
	input.Items = append(input.Items, input.Items[0])
	if input.Validate() == nil {
		t.Fatal("over-bound duplicate batch accepted")
	}
	for _, body := range []string{
		`{"baseGeneration":1.0,"conclusion":{}}`,
		`{"conclusion":{}}`,
		`{"BaseGeneration":0,"conclusion":{}}`,
		`{"baseGeneration":0,"baseGeneration":1,"conclusion":{}}`,
	} {
		var item auditcontract.JudgmentUpdate
		if strictjson.Decode([]byte(body), &item) == nil {
			t.Fatalf("non-exact private generation accepted: %s", body)
		}
	}
}
