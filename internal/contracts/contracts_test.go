package contracts

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestDecodeStrictRejectsTrailingJSON(t *testing.T) {
	t.Parallel()

	input := append(contractstest.ReadFixture(t, "valid", "stage-content-request.json"), []byte(" {}")...)
	if _, err := DecodeStrict[StageContentRequest](input); err == nil {
		t.Fatal("trailing JSON value was accepted")
	}
}

func TestStageContentRequestDoesNotExposeSessionControls(t *testing.T) {
	t.Parallel()

	typeOfRequest := reflect.TypeOf(StageContentRequest{})
	for _, field := range []string{"Session", "SessionID", "SessionMode", "WorkerSessionMode"} {
		if _, ok := typeOfRequest.FieldByName(field); ok {
			t.Fatalf("StageContentRequest unexpectedly exposes %s", field)
		}
	}
}

func TestSchemaFilesContainJSONObjects(t *testing.T) {
	t.Parallel()

	matches, err := filepath.Glob(filepath.Join("..", "..", "api", "v1alpha1", "*.schema.json"))
	if err != nil {
		t.Fatalf("glob schemas: %v", err)
	}
	if len(matches) < 5 {
		t.Fatalf("found %d schema files, want at least 5", len(matches))
	}
	for _, path := range matches {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read %s: %v", path, err)
		}
		var value map[string]any
		if err := json.Unmarshal(data, &value); err != nil {
			t.Fatalf("parse %s: %v", path, err)
		}
		if value["$schema"] == nil {
			t.Fatalf("%s has no $schema", path)
		}
	}
}
