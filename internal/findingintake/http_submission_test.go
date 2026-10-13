package findingintake

import (
	"testing"

	"github.com/grauwolf32/contractor/internal/strictjson"
)

func TestFindingSubmissionRejectsNullHTTPSelection(t *testing.T) {
	for _, input := range []string{`{"httpEvidence":null}`, `{"httpEvidence":{},"unknown":true}`, `{"httpEvidence":{},"httpEvidence":{}}`} {
		var value Submission
		if err := strictjson.Decode([]byte(input), &value); err == nil {
			t.Fatal("invalid HTTP selection was accepted")
		}
	}
}
