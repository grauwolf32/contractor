package contracts

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestValidGoldenFixtures(t *testing.T) {
	t.Parallel()

	cases := map[string]func([]byte) ([]byte, error){
		"agent-registration.json":           roundTrip[AgentRegistration],
		"agent-registration-response.json":  roundTrip[AgentRegistrationResponse],
		"agent-heartbeat.json":              roundTrip[AgentHeartbeat],
		"heartbeat-response.json":           roundTrip[HeartbeatResponse],
		"allocation-spec.json":              roundTrip[AllocationSpec],
		"finalize-allocation.json":          roundTrip[FinalizeAllocationRequest],
		"abort-allocation.json":             roundTrip[AbortAllocationRequest],
		"release-allocation.json":           roundTrip[ReleaseAllocationRequest],
		"artifact-read-result.json":         roundTrip[ArtifactReadResult],
		"stage-content-request.json":        roundTrip[StageContentRequest],
		"stage-content-result-success.json": roundTrip[StageContentResult],
		"stage-content-result-failure.json": roundTrip[StageContentResult],
	}

	for filename, decode := range cases {
		filename, decode := filename, decode
		t.Run(filename, func(t *testing.T) {
			t.Parallel()
			input := readFixture(t, "valid", filename)
			output, err := decode(input)
			if err != nil {
				t.Fatalf("decode valid fixture: %v", err)
			}
			assertSemanticJSONEqual(t, input, output)
		})
	}
}

func TestInvalidGoldenFixtures(t *testing.T) {
	t.Parallel()

	cases := map[string]func([]byte) error{
		"agent-registration-idle-with-allocation.json":   reject[AgentRegistration],
		"agent-heartbeat-missing-allocation.json":        reject[AgentHeartbeat],
		"heartbeat-response-unknown-action.json":         reject[HeartbeatResponse],
		"allocation-spec-bad-api-version.json":           reject[AllocationSpec],
		"stage-content-request-unknown-field.json":       reject[StageContentRequest],
		"stage-content-result-unversioned-artifact.json": reject[StageContentResult],
		"stage-content-result-success-with-error.json":   reject[StageContentResult],
		"artifact-read-result-unversioned.json":          reject[ArtifactReadResult],
	}

	for filename, decode := range cases {
		filename, decode := filename, decode
		t.Run(filename, func(t *testing.T) {
			t.Parallel()
			if err := decode(readFixture(t, "invalid", filename)); err == nil {
				t.Fatal("invalid fixture was accepted")
			}
		})
	}
}

func TestRuntimeSettingsRedactFormattingButSerializeOnWire(t *testing.T) {
	t.Parallel()

	const token = "recognizable-secret-token"
	settings := RuntimeSettings{
		LLMGatewayURL:         "https://gateway.example/v1",
		LLMGatewayToken:       NewSecretString(token),
		ArtifactAPIURL:        "https://server.example/private/v1",
		RequestTimeoutSeconds: 30,
	}
	formatted := fmt.Sprintf("%v %+v %#v %s", settings, settings, settings, settings.LLMGatewayToken)
	if strings.Contains(formatted, token) {
		t.Fatalf("formatted RuntimeSettings leaked token: %s", formatted)
	}
	wire, err := json.Marshal(settings)
	if err != nil {
		t.Fatalf("marshal RuntimeSettings: %v", err)
	}
	if !strings.Contains(string(wire), token) {
		t.Fatalf("wire JSON did not contain required private token: %s", wire)
	}
}

func TestDecodeStrictRejectsTrailingJSON(t *testing.T) {
	t.Parallel()

	input := append(readFixture(t, "valid", "stage-content-request.json"), []byte(" {}")...)
	if _, err := DecodeStrict[StageContentRequest](input); err == nil {
		t.Fatal("trailing JSON value was accepted")
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

func roundTrip[T Validatable](data []byte) ([]byte, error) {
	value, err := DecodeStrict[T](data)
	if err != nil {
		return nil, err
	}
	return json.Marshal(value)
}

func reject[T Validatable](data []byte) error {
	_, err := DecodeStrict[T](data)
	return err
}

func readFixture(t *testing.T, kind, filename string) []byte {
	t.Helper()
	path := filepath.Join("..", "..", "api", "testdata", "v1alpha1", kind, filename)
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture %s: %v", path, err)
	}
	return data
}

func assertSemanticJSONEqual(t *testing.T, left, right []byte) {
	t.Helper()
	var leftValue, rightValue any
	if err := json.Unmarshal(left, &leftValue); err != nil {
		t.Fatalf("decode left JSON: %v", err)
	}
	if err := json.Unmarshal(right, &rightValue); err != nil {
		t.Fatalf("decode right JSON: %v", err)
	}
	if !reflect.DeepEqual(leftValue, rightValue) {
		t.Fatalf("semantic JSON differs\nleft:  %s\nright: %s", left, right)
	}
}
