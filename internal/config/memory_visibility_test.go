package config

import (
	"strings"
	"testing"
)

func TestStageContextRejectsOnlyNonPurposeReservedMemoryBindings(t *testing.T) {
	required := true
	_, err := resolveStageContext(&stageContextSource{Artifacts: &map[string]contextArtifactSource{
		"note": {Namespace: "analysis", Name: "memory.note", Required: &required},
	}})
	if err == nil || !strings.Contains(err.Error(), "reserved Memory binding") {
		t.Fatalf("reserved Memory StageContext error = %v", err)
	}

	for _, namespace := range []string{"inputs", "outputs", "skills"} {
		t.Run(namespace, func(t *testing.T) {
			resolved, err := resolveStageContext(&stageContextSource{
				Artifacts: &map[string]contextArtifactSource{
					"ordinary": {
						Namespace: namespace, Name: "memory.ordinary", Required: &required,
					},
				},
			})
			if err != nil || resolved.Artifacts["ordinary"].Name != "memory.ordinary" {
				t.Fatalf("purpose Namespace StageContext = (%+v, %v)", resolved, err)
			}
		})
	}
}
