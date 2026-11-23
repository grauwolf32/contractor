package findingintake

import (
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

func TestFindingSubmissionRejectsNullHTTPSelection(t *testing.T) {
	for _, input := range []string{`{"httpEvidence":null}`, `{"httpEvidence":{},"unknown":true}`, `{"httpEvidence":{},"httpEvidence":{}}`} {
		var value Submission
		if err := strictjson.Decode([]byte(input), &value); err == nil {
			t.Fatal("invalid HTTP selection was accepted")
		}
	}
}

func TestSharedFindingHTTPSelectionOrigin(t *testing.T) {
	var fixture struct {
		Cases            []struct{ Value json.RawMessage }
		SelectionOrigins []struct {
			Name                    string
			Capture, Present, Valid bool
			Value                   any
		}
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "http-capture-cases.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	for _, row := range fixture.SelectionOrigins {
		t.Run(row.Name, func(t *testing.T) {
			base := testSubmission("current-invocation", "selected", nil)
			data, err := json.Marshal(base)
			if err != nil {
				t.Fatal(err)
			}
			var value map[string]any
			if err := json.Unmarshal(data, &value); err != nil {
				t.Fatal(err)
			}
			if row.Capture {
				value["httpEvidence"] = fixture.Cases[0].Value
			}
			if row.Present {
				value["httpEvidenceInvocationId"] = row.Value
			}
			data, err = json.Marshal(value)
			if err != nil {
				t.Fatal(err)
			}
			var input Submission
			err = strictjson.Decode(data, &input)
			if err == nil {
				_, err = canonicalize(input)
			}
			if (err == nil) != row.Valid {
				t.Fatalf("valid=%t: %v", row.Valid, err)
			}
		})
	}
}
