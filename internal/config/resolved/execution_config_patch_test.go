package resolved

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

// TestExecutionConfigPatchRejectsRepeatedMembersItself pins that the patch
// types reject a repeated member at every depth even when their caller
// decodes with plain encoding/json, which keeps the last value.
func TestExecutionConfigPatchRejectsRepeatedMembersItself(t *testing.T) {
	t.Parallel()

	for name, raw := range map[string]string{
		"executionConfig":         `{"workers":{"modelPolicy":"a"},"workers":{"modelPolicy":"b"}}`,
		"selection patch":         `{"planner":{"modelPolicy":"a","modelPolicy":"b"}}`,
		"stages":                  `{"stages":{"copy":{"planner":{"llmGateway":"g"}},"copy":{"planner":{"llmGateway":"h"}}}}`,
		"stage patch":             `{"stages":{"copy":{"planner":{"llmGateway":"g"},"planner":{"llmGateway":"h"}}}}`,
		"stage agents":            `{"stages":{"copy":{"agents":{"builder":{"credential":null},"builder":{"credential":"c"}}}}}`,
		"stage agent selection":   `{"stages":{"copy":{"agents":{"builder":{"credential":null,"credential":"c"}}}}}`,
		"escaped member name":     `{"workers":{"modelPolicy":"a","modelPolicy":"b"}}`,
		"inside an unknown field": `{"unknown":{"x":1,"x":2}}`,
	} {
		var patch ExecutionConfigPatch
		err := json.Unmarshal([]byte(raw), &patch)
		if err == nil {
			t.Errorf("%s: json.Unmarshal(%s) accepted a repeated member", name, raw)
			continue
		}
		if name != "inside an unknown field" && !errors.Is(err, strictjson.ErrDuplicateKey) {
			t.Errorf("%s: error = %v, want ErrDuplicateKey", name, err)
		}
	}
}

// TestExecutionConfigPatchMembersRejectDataAfterTheValue pins that each patch
// object and string member decodes exactly one value.
func TestExecutionConfigPatchMembersRejectDataAfterTheValue(t *testing.T) {
	t.Parallel()

	var selection ExecutionSelectionPatch
	if err := selection.unmarshalJSON([]byte(`{"modelPolicy":"a"} {"modelPolicy":"b"}`), "workers"); err == nil {
		t.Error("a selection patch accepted a second value")
	}
	var stage StageExecutionConfigPatch
	if err := stage.unmarshalJSON([]byte(`{"planner":{"modelPolicy":"a"}} []`), "stages.copy"); err == nil {
		t.Error("a Stage patch accepted a second value")
	}
	if _, err := decodeOptionalString([]byte(`"a" "b"`), false, "workers.modelPolicy"); err == nil {
		t.Error("a string member accepted a second value")
	}
}

// TestExecutionConfigPatchDecodesValidValuesUnchanged pins that strict
// decoding keeps the tri-state credential and the canonical form the patch
// digests cover.
func TestExecutionConfigPatchDecodesValidValuesUnchanged(t *testing.T) {
	t.Parallel()

	raw := `{"stages":{"copy":{"planner":{"modelPolicy":"m"},"agents":{"builder":{"credential":null,"llmGateway":"g"}}},"empty":{}},` +
		`"workers":{"summarizerModelPolicy":"s","credential":"c"},"planner":{"llmGateway":"p"}}`
	var patch ExecutionConfigPatch
	if err := json.Unmarshal([]byte(raw), &patch); err != nil {
		t.Fatal(err)
	}
	if credential := patch.Stages["copy"].Agents["builder"].Credential; !credential.Present || !credential.Null {
		t.Fatalf("explicit credential null decoded as %#v", credential)
	}
	encoded, err := json.Marshal(patch)
	if err != nil {
		t.Fatal(err)
	}
	want := `{"planner":{"llmGateway":"p"},"stages":{"copy":{"agents":{"builder":{"credential":null,"llmGateway":"g"}},` +
		`"planner":{"modelPolicy":"m"}}},"workers":{"credential":"c","summarizerModelPolicy":"s"}}`
	if string(encoded) != want {
		t.Fatalf("canonical form = %s, want %s", encoded, want)
	}
}
