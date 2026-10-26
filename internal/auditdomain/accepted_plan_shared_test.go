package auditdomain

import (
	"bytes"
	"encoding/json"
	"testing"

	jsonschema "github.com/santhosh-tekuri/jsonschema/v6"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestSharedAcceptedPlans(t *testing.T) {
	var fixture struct {
		MaximumCapabilities         int    `json:"maximumCapabilities"`
		MaximumCapabilityTools      int    `json:"maximumCapabilityTools"`
		MaximumPlanBytes            int    `json:"maximumPlanBytes"`
		CanonicalAcceptedPlanDigest string `json:"canonicalAcceptedPlanDigest"`
		CanonicalPlanDigest         string `json:"canonicalPlanDigest"`
		CanonicalDependenciesDigest string `json:"canonicalDependenciesDigest"`
		Methods                     []struct{ Method, Phase string }
		Base                        json.RawMessage
		Cases                       []struct {
			Name                      string
			Valid                     bool
			RefreshDependenciesDigest bool
			Patches                   []struct {
				Path   []any
				Value  any
				Delete bool
			}
		}
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "audit-check-plans", "accepted-plan-cases.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.MaximumPlanBytes != MaximumAcceptedPlanBytes || fixture.MaximumCapabilities != contracts.MaximumCheckCapabilities || fixture.MaximumCapabilityTools != contracts.MaximumCheckCapabilityTools {
		t.Fatal("shared maximum drifted")
	}
	for _, method := range fixture.Methods {
		key, ok := contracts.LookupCheckPlanMethod(contracts.FindingCheckPlanSchema, method.Method)
		if !ok || key.Phase != method.Phase {
			t.Fatal("method registry drifted")
		}
	}
	for _, test := range fixture.Cases {
		t.Run(test.Name, func(t *testing.T) {
			var value map[string]any
			_ = json.Unmarshal(fixture.Base, &value)
			for _, patch := range test.Patches {
				var parent any = value
				for _, part := range patch.Path[:len(patch.Path)-1] {
					parent = dependencyPathValue(parent, part)
				}
				last := patch.Path[len(patch.Path)-1].(string)
				if patch.Delete {
					delete(parent.(map[string]any), last)
				} else {
					parent.(map[string]any)[last] = patch.Value
				}
			}
			if test.RefreshDependenciesDigest {
				data, err := canonicalJSON(value["dependencies"])
				if err != nil {
					t.Fatal(err)
				}
				value["plan"].(map[string]any)["dependencies_digest"] = DigestBytes(data)
			}
			var deps *RetainedDependencyManifest
			depBytes, _ := json.Marshal(value["dependencies"])
			depErr := json.Unmarshal(depBytes, &deps)
			data, _ := json.Marshal(value["plan"])
			plan, err := DecodeAcceptedPlan(data, deps)
			valid := err == nil && depErr == nil
			if valid != test.Valid {
				t.Fatalf("accepted=%t want%t: %v %v", valid, test.Valid, err, depErr)
			}
			if test.Name == "complete-zero-size-and-zero-ordinal" {
				canonical, _ := EncodeAcceptedPlan(plan, deps)
				planBytes, _ := EncodeCheckPlan(plan.Plan)
				depsBytes, _ := EncodeRetainedDependencies(*deps)
				if DigestBytes(canonical) != fixture.CanonicalAcceptedPlanDigest || DigestBytes(planBytes) != fixture.CanonicalPlanDigest || DigestBytes(depsBytes) != fixture.CanonicalDependenciesDigest {
					t.Fatal("cross-language JCS digests differ")
				}
			}
		})
	}
}

func TestSharedAcceptedPlanPublishedSchema(t *testing.T) {
	var fixture struct {
		Base struct{ Plan json.RawMessage }
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "audit-check-plans", "accepted-plan-cases.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	compiler := jsonschema.NewCompiler()
	schema, err := compiler.Compile(contractstest.Path(t, "api", "audit-check-plans", "v1", "accepted-plan.schema.json"))
	if err != nil {
		t.Fatal(err)
	}
	value, err := jsonschema.UnmarshalJSON(bytes.NewReader(fixture.Base.Plan))
	if err != nil || schema.Validate(value) != nil {
		t.Fatal("valid shared plan fails published schema", err)
	}
}
