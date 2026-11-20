package streamline

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestSharedModelResponseRecovery(t *testing.T) {
	var cases []struct {
		Name      string
		Response  json.RawMessage
		EmptyStop bool
		Accepted  bool
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "model-response-recovery.json"), &cases); err != nil {
		t.Fatal(err)
	}
	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			_, err := decodeChatResponse(c.Response)
			var rejected *gatewayResponseRejected
			empty := errors.As(err, &rejected) && rejected.emptyStop
			if empty != c.EmptyStop || (err == nil) != c.Accepted {
				t.Fatalf("error=%v empty=%v want=%+v", err, empty, c)
			}
		})
	}
}
