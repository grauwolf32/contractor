package config

import (
	"go.yaml.in/yaml/v4"
	"testing"
)

func TestMaterialSetSlotIsExplicitAndInputOnly(t *testing.T) {
	for _, test := range []struct {
		text  string
		valid bool
	}{
		{`required: false
mediaTypes: [text/plain]
materialSet: true`, true},
		{`required: false
mediaTypes: [text/plain]
materialSet: false`, false},
		{`required: false
mediaTypes: [text/plain]
materialSet: null`, false},
		{`required: false
mediaTypes: [text/plain]
materialSet: "true"`, false},
	} {
		t.Run(test.text, func(t *testing.T) {
			var slot artifactSlotSource
			if e := yaml.Unmarshal([]byte(test.text), &slot); e != nil {
				t.Fatal(e)
			}
			got, e := resolveArtifactSlotMap("spec.inputs", map[string]artifactSlotSource{"materials": slot}, false, false)
			if (e == nil) != test.valid {
				t.Fatalf("accepted=%v want=%v %v", e == nil, test.valid, e)
			}
			if test.valid && !got["materials"].MaterialSet {
				t.Fatal("material set flag lost")
			}
			for _, field := range []string{"spec.outputs", "result.artifacts"} {
				if _, e := resolveArtifactSlotMap(field, map[string]artifactSlotSource{"materials": slot}, false, field == "spec.outputs"); e == nil {
					t.Fatal("set admitted outside Workflow inputs", field)
				}
			}
		})
	}
}
