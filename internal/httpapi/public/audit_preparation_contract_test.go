package public

import (
	"encoding/json"
	"os"
	"testing"
	"time"

	"github.com/getkin/kin-openapi/openapi3"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestPublicAuditPreparationContracts(t *testing.T) {
	document := loadPublicOpenAPI(t)
	validate := func(schema string, value any, valid bool) {
		t.Helper()
		encoded, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		var wire any
		if err := json.Unmarshal(encoded, &wire); err != nil {
			t.Fatal(err)
		}
		if err := document.Components.Schemas[schema].Value.VisitJSON(wire, openapi3.EnableJSONSchema2020()); (err == nil) != valid {
			t.Fatalf("%s valid=%t: %v", schema, valid, err)
		}
	}
	fixture, err := os.ReadFile("../../../api/testdata/audit-composition/public-cases.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		Name   string `json:"name"`
		Schema string `json:"schema"`
		Value  any    `json:"value"`
		Valid  bool   `json:"valid"`
	}
	if err := json.Unmarshal(fixture, &cases); err != nil {
		t.Fatal(err)
	}
	for _, tc := range cases {
		t.Run(tc.Name, func(t *testing.T) { validate(tc.Schema, tc.Value, tc.Valid) })
	}
	snapshot, err := config.Load("../../../configs", config.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	for _, profile := range snapshot.AuditProfiles() {
		validate("AuditProfile", auditProfileReadModel(auditservice.ProfileProjection{Profile: profile, Compatibility: auditservice.ProfileCompatibility(profile)}, true), true)
	}
	for _, tc := range []struct {
		value map[string]any
		valid bool
	}{
		{map[string]any{"source": "audit-input", "name": "source"}, true},
		{map[string]any{"source": "prepare-output", "name": "api", "role": "extract"}, true},
		{map[string]any{"source": "prepare-output", "name": "api"}, false},
		{map[string]any{"source": "audit-input", "name": "source", "role": "extract"}, false},
		{map[string]any{"source": "retained-output", "name": "api", "role": "extract"}, false},
		{map[string]any{"sourceInput": "source"}, false},
	} {
		validate("AuditInventoryArtifact", tc.value, tc.valid)
	}
	profile, err := snapshot.AuditProfile("openapi-sqlmap-scan@1")
	if err != nil {
		t.Fatal(err)
	}
	workflow, err := snapshot.Workflow("openapi-from-workspace@7")
	if err != nil {
		t.Fatal(err)
	}
	prepareBinding := auditProfileWorkflowResponse{
		Kind: config.AuditWorkflowPrepare, Workflow: workflow.Ref, MaxRunAttempts: 2,
		Inputs:     map[string]config.AuditWorkflowInputMapping{"source": {Source: config.AuditInputFromAudit, Name: "source"}},
		Parameters: map[string]config.AuditWorkflowParameterMapping{}, Outputs: map[string]string{"api": "openapi"},
	}
	validate("AuditProfileWorkflow", prepareBinding, true)
	prepareBinding.MaxRunAttempts = 0
	validate("AuditProfileWorkflow", prepareBinding, false)
	prepareBinding.MaxRunAttempts = 2
	prepareBinding.Kind = config.AuditWorkflowCheck
	validate("AuditProfileWorkflow", prepareBinding, false)

	// These DTOs specify the controller's future public contract. No prepare
	// Run is dispatched by this contract-only test or by the gated service.
	now := time.Now().UTC()
	revision := "source-r1"
	audit := auditResponse{
		AuditID: "audit_preparing", ProjectID: "project_example", Profile: auditProfileIdentityResponse{Name: profile.Ref.Name, Version: profile.Ref.Version, Digest: profile.Ref.Digest},
		Inputs: map[string]auditstore.ExactArtifact{"source": {Ref: contracts.ArtifactRef{Namespace: "sources", Name: "source", Revision: &revision}, Digest: auditHandlerDigest("source")}},
		Scope:  auditservice.Scope{}, RuntimeLabels: []string{}, State: auditstore.AuditActive, Phase: auditdomain.AuditPhasePreparing,
		Revision: 1, DispatchState: auditstore.DispatchOpen, HoldState: auditstore.HoldHeld,
		Limits:    auditstore.Limits{MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 16, MaxItemsTotal: 16, MaxSubmittedRuns: 50, MaxItemRunAttempts: 3, MaxEvidenceBytes: 64 << 20},
		CreatedAt: now, UpdatedAt: now,
		Preparation: &auditPreparationResponse{Roles: map[string]auditPreparationRoleResponse{"extract": {Status: auditdomain.PreparationPending, MaxAttempts: 2, Outputs: map[string]auditPreparationOutputResponse{}}}},
	}
	// Baseline is pinned before inventory and contains safe exact Runtime
	// identity, without a fabricated worklist.
	encoded, err := json.Marshal(audit)
	if err != nil {
		t.Fatal(err)
	}
	var wire map[string]any
	if err := json.Unmarshal(encoded, &wire); err != nil {
		t.Fatal(err)
	}
	wire["baseline"] = map[string]any{
		"inputs": wire["inputs"], "scope": map[string]any{}, "runtimeLabels": []any{}, "skills": []any{}, "standards": []any{},
		"runtimeConfig": map[string]any{"default": map[string]any{"label": "default", "explicit": false, "bindingRevision": 1, "config": map[string]any{"name": "runtime-example", "version": "1", "digest": auditHandlerDigest("runtime")}}, "labels": []any{}},
	}
	for _, state := range []string{"active", "paused", "cancelling", "cancelled", "failed", "deleting"} {
		wire["state"] = state
		validate("Audit", wire, true)
	}
	wire["state"] = "active"
	validate("AuditStartResponse", map[string]any{"audit": wire, "items": []any{}}, true)
	wire["currentRoundId"] = "fake_round"
	validate("Audit", wire, false)
	delete(wire, "currentRoundId")
	delete(wire, "preparation")
	validate("Audit", wire, false)
}
