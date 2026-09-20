package evaldomain

import (
	"encoding/json"
	"testing"
)

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
