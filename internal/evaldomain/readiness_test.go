package evaldomain

import (
	"bytes"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/strictjson"
)

type readinessCase struct {
	Name       string          `json:"name"`
	Plan       json.RawMessage `json:"plan"`
	Comparison Comparison      `json:"comparison"`
	Readiness  json.RawMessage `json:"readiness"`
}

func readinessCases(t *testing.T) []readinessCase {
	t.Helper()
	var document struct {
		Cases []readinessCase `json:"cases"`
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "evals", "readiness-cases.json"), &document); err != nil {
		t.Fatal(err)
	}
	if len(document.Cases) == 0 {
		t.Fatal("readiness fixture has no cases")
	}
	return document.Cases
}

// Every implementation of native readiness (this function, the freeze that
// stores it and the backfill migration) must reproduce the shared fixture.
func TestSharedEvalNativeReadinessMatchesFixtures(t *testing.T) {
	for _, c := range readinessCases(t) {
		t.Run(c.Name, func(t *testing.T) {
			plan, err := Freeze("playground.plan/v1", c.Plan)
			if err != nil {
				t.Fatal(err)
			}
			got, err := NativeReadiness(plan, c.Comparison)
			if err != nil {
				t.Fatal(err)
			}
			var want Readiness
			if err = strictjson.Decode(c.Readiness, &want); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("readiness:\n got %+v\nwant %+v", got, want)
			}
			raw, err := json.Marshal(got)
			if err != nil {
				t.Fatal(err)
			}
			if err = Validate("Readiness", raw); err != nil {
				t.Fatal(err)
			}
			// Readiness reports equality, never the pinned values.
			for _, value := range []string{"sha256:", "tasks-a", "instructions-b", "zeta-b"} {
				if bytes.Contains(raw, []byte(value)) {
					t.Fatalf("readiness exposes pin value %q: %s", value, raw)
				}
			}
		})
	}
}

// ceilingPlan returns the conformance plan with dimensions pinned under each
// of the given variants: perArm dimensions of 128 characters per variant,
// none shared, so the readiness lists every one of them.
func ceilingPlan(t *testing.T, variants []string, perArm int) Frozen {
	t.Helper()
	var plan map[string]any
	if err := json.Unmarshal(fixture(t, "portable-plan"), &plan); err != nil {
		t.Fatal(err)
	}
	pins := map[string]any{}
	for _, variant := range variants {
		arm := map[string]any{}
		for i := range perArm {
			arm[fmt.Sprintf("%s%0127d", variant[:1], i)] = map[string]any{"value": "v", "origin": "observed"}
		}
		pins[variant] = arm
	}
	plan["pins"] = pins
	raw, err := json.Marshal(plan)
	if err != nil {
		t.Fatal(err)
	}
	frozen, err := Freeze("playground.plan/v1", raw)
	if err != nil {
		t.Fatal(err)
	}
	return frozen
}

// A native plan pins at most 1,024 dimensions for each of its two arms, so
// its readiness has at most 2,048 pins and 2 arms, whatever the member count
// (LIM-EVM-19). Even with every string at its schema maximum that encodes to
// 543,169 bytes, within the 1 MiB eval_frozen_plans_readiness_check bound.
// A plan that pins more dimensions under further variants yields a
// readiness the closed schema rejects, so the freeze cannot store it.
func TestSharedEvalNativeReadinessIsBoundedByPinnedDimensions(t *testing.T) {
	comparison := Comparison{Baseline: "a", Candidate: "b", RequiredEqual: []string{}, AllowedDifferences: []string{}}
	ceiling, err := NativeReadiness(ceilingPlan(t, []string{"a", "b"}, 1024), comparison)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(ceiling)
	if err != nil {
		t.Fatal(err)
	}
	if len(ceiling.Pins) != 2048 || len(ceiling.Arms) != 2 || ceiling.Arms[0].Expected != 4 {
		t.Fatalf("ceiling readiness has %d pins and arms %+v", len(ceiling.Pins), ceiling.Arms)
	}
	if err = Validate("Readiness", raw); err != nil {
		t.Fatalf("ceiling readiness of %d bytes: %v", len(raw), err)
	}

	widest := Readiness{Pins: make([]ReadinessPin, 2048), Arms: make([]ReadinessArm, 2)}
	for i := range widest.Pins {
		widest.Pins[i] = ReadinessPin{fmt.Sprintf("d%0127d", i), false, "producer-supplied", "producer-supplied", "unavailable"}
	}
	for i := range widest.Arms {
		widest.Arms[i] = ReadinessArm{strings.Repeat(string(rune('a'+i)), 128), 10000, 10000, 10000, 10000}
	}
	encoded, err := json.Marshal(widest)
	if err != nil {
		t.Fatal(err)
	}
	if err = Validate("Readiness", encoded); err != nil {
		t.Fatal(err)
	}
	if len(encoded) != 543169 || len(encoded) > 1<<20 || len(raw) > len(encoded) {
		t.Fatalf("widest readiness encodes to %d bytes and the ceiling plan's to %d", len(encoded), len(raw))
	}

	beyond, err := NativeReadiness(ceilingPlan(t, []string{"a", "b", "c"}, 1024), comparison)
	if err != nil {
		t.Fatal(err)
	}
	raw, err = json.Marshal(beyond)
	if err != nil {
		t.Fatal(err)
	}
	if err = Validate("Readiness", raw); !IsCode(err, "eval_invalid") {
		t.Fatalf("a readiness of %d pins was accepted: %v", len(beyond.Pins), err)
	}
}

func TestSharedEvalNativeReadinessRejectsAnotherPlanKind(t *testing.T) {
	registration, err := Freeze("ExternalRegistration", fixture(t, "registration"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err = NativeReadiness(registration, Comparison{}); !IsCode(err, "eval_invalid") {
		t.Fatalf("external registration readiness: %v", err)
	}
}
