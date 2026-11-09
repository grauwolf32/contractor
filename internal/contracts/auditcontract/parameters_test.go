package auditcontract

import (
	"maps"
	"strings"
	"testing"
)

func TestLaunchParametersBindOnlyDeclaredWorkflowValues(t *testing.T) {
	p := Program{Parameters: map[string]ProgramParameter{"target": {Required: true}, "rules": {}}, Workflows: map[string]ProgramWorkflow{"scan": {Parameters: map[string]string{"server": "authored", "filter": "default"}, ParameterInputs: map[string]string{"server": "target", "filter": "rules"}}}}
	values := map[string]string{"target": "https://target.test"}
	bound, err := p.BindParameters(values)
	if err != nil || bound.Workflows["scan"].Parameters["server"] != values["target"] || bound.Workflows["scan"].Parameters["filter"] != "default" {
		t.Fatalf("binding: %+v %v", bound, err)
	}
	values["target"] = "mutated"
	bound.Workflows["scan"].Parameters["server"] = "mutated"
	if p.Workflows["scan"].Parameters["server"] != "authored" {
		t.Fatal("authored program was mutated")
	}
	values = map[string]string{"target": "https://target.test", "rules": ""}
	bound, err = p.BindParameters(values)
	if err != nil || bound.Workflows["scan"].Parameters["filter"] != "" {
		t.Fatal("explicit empty parameter did not replace literal default")
	}
	for _, values := range []map[string]string{nil, {"target": "ok", "unknown": "bad"}, {"target": strings.Repeat("я", 2049)}, {"target": "\x00"}, {"target": string([]byte{255})}} {
		if _, err := p.BindParameters(values); err == nil {
			t.Fatal("invalid launch value accepted")
		}
	}
	valid := map[string]string{"target": strings.Repeat("я", 2048)}
	if _, err := p.BindParameters(valid); err != nil {
		t.Fatal("UTF-8 byte boundary rejected")
	}
	if !maps.Equal(valid, map[string]string{"target": strings.Repeat("я", 2048)}) {
		t.Fatal("input changed")
	}
}
