package auditdomain

import (
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
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
				encoded, err := EncodeHTTPExchangeArtifact(value, "run-test", "invocation-test")
				if err != nil {
					t.Fatal(err)
				}
				var object map[string]any
				if err := json.Unmarshal(encoded, &object); err != nil {
					t.Fatal(err)
				}
				if object["schema"] != HTTPExchangeSchema || object["origin"].(map[string]any)["run_id"] != "run-test" || object["finding"] != nil {
					t.Fatal("missing independent origin")
				}
			}
		})
	}
}

func TestSharedStandaloneHTTPExchangeArtifacts(t *testing.T) {
	var fixture struct {
		Cases []struct {
			Name         string
			Valid        bool
			ArtifactName string `json:"artifact_name"`
			Value        json.RawMessage
		}
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "http-exchange-artifact-cases.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	for _, test := range fixture.Cases {
		t.Run(test.Name, func(t *testing.T) {
			value, err := DecodeHTTPExchangeArtifact(test.Value)
			if (err == nil) != test.Valid {
				t.Fatalf("valid=%t: %v", test.Valid, err)
			}
			if test.Valid && HTTPExchangeArtifactName(value.Origin.InvocationID, value.Capture.Source, value.Capture.RequestID) != test.ArtifactName {
				t.Fatal("natural HTTP evidence identity drifted")
			}
		})
	}
}
