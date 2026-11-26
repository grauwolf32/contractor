package gatewayrecovery

import (
	"bytes"
	"encoding/json"
	"slices"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"go.yaml.in/yaml/v4"
)

// participantUpdate is an update of
// api/testdata/v1alpha1/gateway-recovery-participant-cases.json.
type participantUpdate struct {
	Action            string  `json:"action"`
	Code              string  `json:"code"`
	RetryAfterSeconds float64 `json:"retryAfterSeconds"`
}

// TestSharedGatewayRecoveryParticipantCases holds the Server's side of the
// model recovery participant protocol to the cases the Runtime is held to as
// well (runtime/tests/test_shared_gateway_recovery_participant.py).
func TestSharedGatewayRecoveryParticipantCases(t *testing.T) {
	t.Parallel()
	var fixture struct {
		Description string `json:"description"`
		Actions     []struct {
			Action   string `json:"action"`
			Terminal bool   `json:"terminal"`
		} `json:"actions"`
		FailureCodes []string `json:"failureCodes"`
		Requests     []struct {
			Name    string          `json:"name"`
			Request json.RawMessage `json:"request"`
			Valid   bool            `json:"valid"`
		} `json:"requests"`
		Outcomes []struct {
			Name    string `json:"name"`
			Failure *struct {
				Code        string `json:"code"`
				Retryable   bool   `json:"retryable"`
				Abandoned   bool   `json:"abandoned"`
				ShouldRetry *bool  `json:"shouldRetry"`
			} `json:"failure"`
			Cancelled         bool              `json:"cancelled"`
			RetryAfterSeconds float64           `json:"retryAfterSeconds"`
			Update            participantUpdate `json:"update"`
			Resend            bool              `json:"resend"`
		} `json:"outcomes"`
		Redelivery []struct {
			Update      participantUpdate `json:"update"`
			Redelivered participantUpdate `json:"redelivered"`
		} `json:"redelivery"`
	}
	decoder := json.NewDecoder(bytes.NewReader(contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "gateway-recovery-participant-cases.json")))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&fixture); err != nil {
		t.Fatal(err)
	}
	if len(fixture.Actions) == 0 || len(fixture.FailureCodes) == 0 || len(fixture.Requests) == 0 ||
		len(fixture.Outcomes) == 0 || len(fixture.Redelivery) == 0 {
		t.Fatal("gateway recovery participant cases are incomplete")
	}

	for _, action := range fixture.Actions {
		request := Request{Model: "worker", RequestID: "request-1", Action: action.Action, Code: fixture.FailureCodes[0]}
		if err := request.Validate(); err != nil || request.Terminal() != action.Terminal {
			t.Errorf("action %q: valid = %v, terminal = %t, want valid and %t", action.Action, err, request.Terminal(), action.Terminal)
		}
	}
	// Every code the shared classifier marks retryable may be reported as
	// failed, and no permanent one may (SCH-R112).
	var classification struct {
		Cases []struct {
			Code      string `json:"code"`
			Retryable bool   `json:"retryable"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "gateway-failure-classification-cases.json"), &classification); err != nil {
		t.Fatal(err)
	}
	// The public RunRecovery.code reports the same causes (ERR-R14 sets the
	// vocabulary apart from the error code of the same name).
	var openapi struct {
		Components struct {
			Schemas map[string]struct {
				Properties map[string]struct {
					Enum []string `yaml:"enum"`
				} `yaml:"properties"`
			} `yaml:"schemas"`
		} `yaml:"components"`
	}
	if err := yaml.Unmarshal(contractstest.ReadFile(t, "api", "openapi", "contractor-public-v1.yaml"), &openapi); err != nil {
		t.Fatal(err)
	}
	public := slices.Sorted(slices.Values(openapi.Components.Schemas["RunRecovery"].Properties["code"].Enum))
	if shared := slices.Sorted(slices.Values(fixture.FailureCodes)); !slices.Equal(public, shared) {
		t.Errorf("RunRecovery.code enum = %v, shared recovery failure codes = %v", public, shared)
	}
	codes := append([]string{"gateway_request_rejected", "gateway_access_denied"}, fixture.FailureCodes...)
	for _, test := range classification.Cases {
		codes = append(codes, test.Code)
		if ValidFailure(test.Code) != test.Retryable {
			t.Errorf("classified %q (retryable %t) accepted by failed = %t", test.Code, test.Retryable, ValidFailure(test.Code))
		}
	}
	for _, code := range codes {
		if ValidFailure(code) != slices.Contains(fixture.FailureCodes, code) {
			t.Errorf("ValidFailure(%q) = %t, want the shared failure codes %q", code, ValidFailure(code), fixture.FailureCodes)
		}
	}

	for _, test := range fixture.Requests {
		var request Request
		err := strictjson.Decode(test.Request, &request)
		if err == nil {
			err = request.Validate()
		}
		if (err == nil) != test.Valid {
			t.Errorf("%s: request accepted = %t, want %t: %v", test.Name, err == nil, test.Valid, err)
		}
	}

	// Runtime owns outcome selection and redelivery. Server validates the
	// normalized requests it can receive for every recorded outcome.
	for _, test := range fixture.Outcomes {
		request := Request{Model: "worker", RequestID: "request-1", Action: test.Update.Action, Code: test.Update.Code, RetryAfterSeconds: test.Update.RetryAfterSeconds}
		if err := request.Validate(); err != nil {
			t.Errorf("%s: %v", test.Name, err)
		}
	}
}
