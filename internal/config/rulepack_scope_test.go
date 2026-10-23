package config

import (
	"go.yaml.in/yaml/v4"
	"testing"
)

func TestAuthoredRulepackScopeRejectsExplicitInvalidSelectors(t *testing.T) {
	for _, test := range []struct {
		value string
		valid bool
	}{{"user", true}, {"null", false}, {"false", false}, {"project", false}, {"[]", false}, {"{}", false}, {"''", false}} {
		var slot artifactSlotSource
		if err := yaml.Unmarshal([]byte("required: false\nmediaTypes: [application/yaml]\nsourceScope: "+test.value), &slot); err != nil {
			t.Fatal(err)
		}
		_, err := resolveArtifactSlotMap("spec.inputs", map[string]artifactSlotSource{"rulepack": slot}, false, false)
		if (err == nil) != test.valid {
			t.Fatalf("%s: %v", test.value, err)
		}
		if test.valid {
			for _, field := range []string{"spec.outputs", "spec.stages.scan.result.artifacts"} {
				if _, err := resolveArtifactSlotMap(field, map[string]artifactSlotSource{"rulepack": slot}, false, false); err == nil {
					t.Fatalf("scope accepted on %s", field)
				}
			}
		}
	}
}
