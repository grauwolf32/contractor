package contracts_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

type diagnosticTextCase struct {
	Name, Unit, Suffix string
	Count, Bytes       int
	Valid              bool
}

func TestSharedExecutionErrorMessageByteBounds(t *testing.T) {
	var cases struct {
		Cases []diagnosticTextCase `json:"cases"`
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "execution-error-cases.json"), &cases); err != nil {
		t.Fatal(err)
	}
	if len(cases.Cases) == 0 {
		t.Fatal("execution error cases are empty")
	}
	compiler, ids := privateSchemaCompiler(t)
	index := readFixtureIndex(t)
	for _, fixture := range []struct {
		name string
		path string
	}{
		{"allocation-final-response.json", "/report/worker/errors/0/message"},
		{"allocation-final-response.json", "/report/worker/toolCalls/0/error/message"},
		{"agent-state-snapshot.json", "/state/metrics/errors/0/message"},
		{"agent-state-snapshot.json", "/state/metrics/toolCalls/0/error/message"},
	} {
		entry := index.Valid[fixture.name]
		schema, err := compiler.Compile(ids[entry.Schema])
		if err != nil {
			t.Fatal(err)
		}
		raw := contractstest.ReadFixture(t, "valid", fixture.name)
		for _, item := range cases.Cases {
			t.Run(fixture.name+fixture.path+"/"+item.Name, func(t *testing.T) {
				message := strings.Repeat(item.Unit, item.Count) + item.Suffix
				if len(message) != item.Bytes {
					t.Fatalf("message size = %d, want %d", len(message), item.Bytes)
				}
				document := withValue(t, raw, fixture.path, message)
				if got := schema.Validate(decodeNumbers(t, document)) == nil; got != item.Valid {
					t.Errorf("schema accepted=%t, want %t", got, item.Valid)
				}
				if err := fixtureCodecs[entry.Type].privateReject(document); (err == nil) != item.Valid {
					t.Errorf("Go accepted=%t, want %t: %v", err == nil, item.Valid, err)
				}
			})
		}
	}
}

func TestSharedExecutionArgumentJSONByteBounds(t *testing.T) {
	var cases struct {
		Arguments []diagnosticTextCase `json:"argumentCases"`
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "execution-error-cases.json"), &cases); err != nil {
		t.Fatal(err)
	}
	if len(cases.Arguments) == 0 {
		t.Fatal("execution argument cases are empty")
	}
	index := readFixtureIndex(t)
	for _, fixture := range []struct{ name, path string }{
		{"allocation-final-response.json", "/report/worker/toolCalls/0/arguments"},
		{"agent-state-snapshot.json", "/state/metrics/toolCalls/0/arguments"},
	} {
		raw := contractstest.ReadFixture(t, "valid", fixture.name)
		for _, item := range cases.Arguments {
			arguments := map[string]any{"path": strings.Repeat(item.Unit, item.Count) + item.Suffix}
			if size, err := contracts.ResultJSONSize(arguments); err != nil || size != item.Bytes {
				t.Fatalf("%s size=%d, %v, want %d", item.Name, size, err, item.Bytes)
			}
			document := withValue(t, raw, fixture.path, arguments)
			if err := fixtureCodecs[index.Valid[fixture.name].Type].privateReject(document); (err == nil) != item.Valid {
				t.Errorf("%s %s accepted=%t, want %t: %v", fixture.name, item.Name, err == nil, item.Valid, err)
			}
		}
	}
}
