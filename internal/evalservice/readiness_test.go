package evalservice

import (
	"bytes"
	"encoding/json"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
)

func TestEvalReadinessRetainsEligibilityAndReportsObservedEqualityWithoutPrivateMaterial(t *testing.T) {
	draft, dataset, preflight := planInputs(t)
	preflight["a"].Cases[draft.CaseIDs[0]] = unavailable("Missing supported input")
	preflight["a"].Pins["instructions"] = observedPin(evaldomain.Digest([]byte("A instructions")))
	preflight["b"].Pins["instructions"] = observedPin(evaldomain.Digest([]byte("B instructions")))
	draft.Comparison.AllowedDifferences = append(draft.Comparison.AllowedDifferences, "unknown-model")
	bundle, err := BuildPlan("readiness-test", time.Now(), draft, dataset, preflight)
	if err != nil {
		t.Fatal(err)
	}
	view, err := nativeReadiness(evalstore.Plan{Document: bundle.Plan, Setup: bundle.Setup})
	if err != nil {
		t.Fatal(err)
	}
	if len(view.Arms) != 2 || view.Arms[0].Expected != 4 || view.Arms[0].Unsupported != 2 || view.Arms[1].Eligible != 4 {
		t.Fatalf("incorrect full matrix: %+v", view.Arms)
	}
	pins := map[string]ReadinessPin{}
	for _, pin := range view.Pins {
		pins[pin.Dimension] = pin
	}
	if pins["source"].Status != "equal" || !pins["source"].RequiredEqual || pins["instructions"].Status != "different" || pins["unknown-model"].Status != "unavailable" {
		t.Fatalf("incorrect pin coverage: %+v", pins)
	}
	raw, err := json.Marshal(view)
	if err != nil {
		t.Fatal(err)
	}
	if err := evaldomain.Validate("Readiness", raw); err != nil {
		t.Fatal(err)
	}
	for _, secret := range [][]byte{[]byte("PRIVATE_"), []byte("A instructions"), []byte("sha256:")} {
		if bytes.Contains(raw, secret) {
			t.Fatalf("readiness exposed private content: %s", raw)
		}
	}
}
