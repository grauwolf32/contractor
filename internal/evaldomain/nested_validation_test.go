package evaldomain

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestEvalCheckParameterNamesPreserveEvaluatorSpelling(t *testing.T) {
	check := Check{ID: "report-media", Evaluator: "media-type@1", Required: true}
	for _, name := range []string{"output", "mediaType", "custom_parameter", "", "bad key", "bad/key", strings.Repeat("x", 129)} {
		t.Run(name, func(t *testing.T) {
			check.Parameters = map[string]string{name: "text/markdown"}
			raw, err := json.Marshal(check)
			if err != nil {
				t.Fatal(err)
			}
			wantValid := name == "output" || name == "mediaType" || name == "custom_parameter"
			if err = Validate("Check", raw); (err == nil) != wantValid {
				t.Fatalf("parameter %q: want valid=%v, got %v", name, wantValid, err)
			}
		})
	}
	check.ID = "ReportMedia"
	check.Parameters = map[string]string{"output": "report", "mediaType": "text/markdown"}
	raw, err := json.Marshal(check)
	if err != nil {
		t.Fatal(err)
	}
	if err = Validate("Check", raw); err == nil {
		t.Fatal("parameter names must not relax check identifiers")
	}
}

func TestEvalDatasetOutputNamesDoNotSelectSemanticDTO(t *testing.T) {
	var dataset DatasetInput
	if err := DecodeInto("DatasetInput", fixture(t, "dataset"), &dataset); err != nil {
		t.Fatal(err)
	}
	dataset.Cases[0].Outputs = map[string]Output{
		"counts":     {MediaTypes: []string{"application/json"}, Required: true},
		"conclusion": {MediaTypes: []string{"text/markdown"}, Required: true},
	}
	data, err := json.Marshal(dataset)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if panicValue := recover(); panicValue != nil {
			t.Fatalf("valid role names caused a semantic validator panic: %v", panicValue)
		}
	}()
	if err := Validate("DatasetInput", data); err != nil {
		t.Fatalf("valid output roles were interpreted as another DTO: %v", err)
	}
}
