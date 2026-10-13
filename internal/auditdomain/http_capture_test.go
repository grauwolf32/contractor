package auditdomain

import (
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/strictjson"
)

func TestSharedHTTPFindingCapture(t *testing.T) {
	data := contractstest.ReadFile(t, "api", "testdata", "http-capture-cases.json")
	var fixtures struct {
		Limits map[string]int `json:"limits"`
		Cases  []struct {
			Name  string          `json:"name"`
			Valid bool            `json:"valid"`
			Value json.RawMessage `json:"value"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(data, &fixtures); err != nil {
		t.Fatal(err)
	}
	if fixtures.Limits["requestBodyBytes"] != MaximumHTTPRequestBodyBytes || fixtures.Limits["responseBodyBytes"] != MaximumHTTPResponseBodyBytes ||
		fixtures.Limits["headerBytes"] != MaximumHTTPHeaderBytes || fixtures.Limits["attempts"] != MaximumHTTPExchangeAttempts {
		t.Fatal("HTTP capture limits drifted")
	}
	for _, test := range fixtures.Cases {
		t.Run(test.Name, func(t *testing.T) {
			var value CapturedHTTPExchange
			err := strictjson.Decode(test.Value, &value)
			if (err == nil) != test.Valid {
				t.Fatalf("valid=%v, error=%v", test.Valid, err)
			}
			if err == nil {
				encoded, err := EncodeHTTPExchangeArtifact(value, "proposal-test", "receipt-test")
				if err != nil {
					t.Fatal(err)
				}
				var object map[string]any
				if err := json.Unmarshal(encoded, &object); err != nil {
					t.Fatal(err)
				}
				if object["schema"] != HTTPExchangeSchema || object["finding"].(map[string]any)["proposal_id"] != "proposal-test" {
					t.Fatal("missing finding association")
				}
			}
		})
	}
}
