package sitemapjoin

import (
	"errors"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/operations"
)

func TestModuleRejectsInvalidBindingsBeforeReadingInputs(t *testing.T) {
	revision := "selected"
	for _, change := range []struct {
		name  string
		apply func(*operations.Call)
	}{
		{"missing selected input", func(call *operations.Call) { delete(call.Inputs, "openapi") }},
		{"extra input", func(call *operations.Call) { call.Inputs["other"] = call.Inputs["openapi"] }},
		{"inexact input", func(call *operations.Call) {
			ref := call.Inputs["openapi"]
			ref.Revision = nil
			call.Inputs["openapi"] = ref
		}},
		{"non string argument", func(call *operations.Call) { call.Arguments["origin"] = true }},
		{"unknown argument", func(call *operations.Call) { call.Arguments["command"] = "run" }},
		{"unknown output", func(call *operations.Call) {
			call.Outputs["other"] = contracts.ArtifactRef{Namespace: "result", Name: "file"}
		}},
	} {
		t.Run(change.name, func(t *testing.T) {
			call := operations.Call{RunID: "run", StageExecutionID: "stage", ClaimID: "claim", Inputs: map[string]contracts.ArtifactRef{"sitemap": {Namespace: "inputs", Name: "sitemap", Revision: &revision}, "openapi": {Namespace: "inputs", Name: "openapi", Revision: &revision}}, Arguments: map[string]any{"origin": "https://example.test", "contract": "api"}, Outputs: map[string]contracts.ArtifactRef{}}
			change.apply(&call)
			for _, module := range []*Module{NewJoin(nil), NewApply(nil)} {
				_, err := module.Run(t.Context(), call)
				var failure *operations.Failure
				if !errors.As(err, &failure) || failure.Code != "invalid_request" || failure.Retryable {
					t.Fatalf("invalid binding reached domain service or became retryable: %v", err)
				}
			}
		})
	}
}
