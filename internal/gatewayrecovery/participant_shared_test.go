package gatewayrecovery

import (
	"bytes"
	"encoding/json"
	"slices"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/strictjson"
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

	for _, test := range fixture.Outcomes {
		var failure *Failure
		switch {
		case test.Failure != nil:
			// readResponse folds an x-should-retry: false response into a
			// non-retryable failure before the update is chosen.
			retryable := test.Failure.Retryable && (test.Failure.ShouldRetry == nil || *test.Failure.ShouldRetry)
			failure = &Failure{Code: test.Failure.Code, Retryable: retryable, Abandoned: test.Failure.Abandoned}
		case test.Cancelled:
			// Do sees the ended invocation as a failed send.
			failure = &Failure{Code: "gateway_unavailable", Retryable: true}
		}
		update := attemptUpdate("request-1", failure, test.RetryAfterSeconds, test.Cancelled)
		got := participantUpdate{Action: update.Action, Code: update.Code, RetryAfterSeconds: update.RetryAfterSeconds}
		if got != test.Update || (update.Action == "failed") != test.Resend {
			t.Errorf("%s: update = %+v, resend = %t; want %+v, %t", test.Name, got, update.Action == "failed", test.Update, test.Resend)
		}
		update.Model = "worker"
		if err := update.Validate(); err != nil {
			t.Errorf("%s: update %+v is not accepted: %v", test.Name, update, err)
		}
	}

	for _, test := range fixture.Redelivery {
		update := Request{RequestID: "request-1", Action: test.Update.Action, Code: test.Update.Code, RetryAfterSeconds: test.Update.RetryAfterSeconds}
		redelivered := redelivery(update)
		got := participantUpdate{Action: redelivered.Action, Code: redelivered.Code, RetryAfterSeconds: redelivered.RetryAfterSeconds}
		if got != test.Redelivered || redelivered.RequestID != update.RequestID {
			t.Errorf("redelivery of %+v = %+v, want %+v", test.Update, redelivered, test.Redelivered)
		}
	}
}
