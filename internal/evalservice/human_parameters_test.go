package evalservice

import (
	"encoding/json"
	"errors"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/evaldomain"
)

func TestSharedHumanReviewParametersCannotReachFrozenPlanOrNativeExecution(t *testing.T) {
	draft, data, preflight := planInputs(t)
	draft.Checks[0].Parameters = map[string]string{"output": "report"}
	check := draft.Checks[0]
	for name, operation := range map[string]func() error{
		"registered validator": func() error { return validateNativeCheck(check) },
		"direct plan builder": func() error {
			_, err := BuildPlan("invalid-human-parameters", time.Unix(1, 0), draft, data, preflight)
			return err
		},
		"frozen policy read": func() error {
			raw, err := json.Marshal(preparedSetup{Checks: []evaldomain.Check{check}})
			if err != nil {
				return err
			}
			var setup preparedSetup
			return decodePreparedSetup(raw, &setup)
		},
		"native execution": func() error {
			_, err := (&nativeChecker{}).check(t.Context(), check)
			return err
		},
	} {
		t.Run(name, func(t *testing.T) {
			var failure *evaldomain.Error
			if err := operation(); !errors.As(err, &failure) || failure.Code != "eval_invalid" {
				t.Fatalf("unsupported human parameters accepted: %v", err)
			}
		})
	}
	for _, parameters := range []map[string]string{nil, {}} {
		draft.Checks[0].Parameters = parameters
		if _, err := BuildPlan("supported-human-parameters", time.Unix(1, 0), draft, data, preflight); err != nil {
			t.Fatal(err)
		}
		result, err := (&nativeChecker{}).check(t.Context(), draft.Checks[0])
		if err != nil || result.Status != "incomplete" {
			t.Fatalf("human assessment authority changed: %+v %v", result, err)
		}
	}
	for _, evaluator := range []string{"json-schema@1", "producer-review@2"} {
		check.Evaluator = evaluator
		if err := check.ValidateHumanReviewParameters(); err != nil {
			t.Fatalf("human-only rule changed another evaluator: %v", err)
		}
	}
}
