package a2a

import (
	"context"
	"encoding/json"
	"iter"
	"reflect"
	"strings"
	"testing"

	sdk "github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2asrv"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

// invokeOfficialSDK returns what the Invoker decodes when an official Go SDK
// server answers with one Agent message carrying data as its DataPart.
func invokeOfficialSDK(t *testing.T, data any, subtaskID string) (contracts.WorkerCompletion, error) {
	t.Helper()
	executor := a2asrv.AgentExecutorFunc(func(
		_ context.Context, _ *a2asrv.ExecutorContext,
	) iter.Seq2[sdk.Event, error] {
		return func(yield func(sdk.Event, error) bool) {
			raw, _ := json.Marshal(data)
			var value map[string]any
			_ = json.Unmarshal(raw, &value)
			part := sdk.NewDataPart(value)
			part.MediaType = workerCompletionMediaType
			message := sdk.NewMessage(sdk.MessageRoleAgent, part)
			message.TaskID, message.ContextID = "task-1", "context-1"
			yield(message, nil)
		}
	})
	invoker, handle, _ := nativeSDK(t, executor)
	request := stageRequest()
	request.SubtaskID = subtaskID
	return invoker.Invoke(t.Context(), "builder", handle, request)
}

// TestSharedWorkerResultCasesPassTheA2ADecodePath sends every shared Worker
// result case through the official SDK and the Invoker. The DataPart is
// decoded from the response bytes, so 2^53-1 keeps its digits, 2^53 is
// rejected rather than accepted, and each case gets the verdict of the
// private decoder.
func TestSharedWorkerResultCasesPassTheA2ADecodePath(t *testing.T) {
	var cases []struct {
		Name    string  `json:"name"`
		Unit    string  `json:"unit"`
		Repeat  int     `json:"repeat"`
		Prefix  string  `json:"prefix"`
		Padding int     `json:"padding"`
		Integer *uint64 `json:"integer"`
		Valid   bool    `json:"valid"`
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "worker-result-size-cases.json"), &cases); err != nil {
		t.Fatal(err)
	}
	base := contractstest.ReadFixture(t, "valid", "worker-completion-empty-observations.json")
	integers := map[uint64]bool{}
	for _, test := range cases {
		t.Run(test.Name, func(t *testing.T) {
			want, err := contracts.DecodePrivateStrict[contracts.WorkerCompletion](base)
			if err != nil {
				t.Fatal(err)
			}
			want.Result.Result = test.Prefix + strings.Repeat(test.Unit, test.Repeat)
			if test.Padding > 0 {
				revision := "r1"
				want.Result.Artifacts = map[string]contracts.ArtifactRef{
					strings.Repeat("p", test.Padding): {Namespace: "review", Name: "report", Revision: &revision},
				}
			}
			if test.Integer != nil {
				integers[*test.Integer] = test.Valid
				want.StateRevision = *test.Integer
				want.Result.Observations.Tools = map[string]contracts.ToolObservationCount{"read_file": {Calls: *test.Integer}}
			}
			got, err := invokeOfficialSDK(t, want, want.Result.SubtaskID)
			switch {
			case !test.Valid:
				assertPlannerCode(t, err, "invalid_worker_result")
			case err != nil:
				t.Fatalf("valid Worker result rejected: %v", err)
			case !reflect.DeepEqual(got, want):
				t.Fatalf("A2A decode changed the Worker result: state revision %d, want %d",
					got.StateRevision, want.StateRevision)
			}
		})
	}
	if valid, ok := integers[contracts.MaxExactInteger]; !ok || !valid {
		t.Fatal("the shared cases lost the valid 2^53-1 case")
	}
	if valid, ok := integers[contracts.MaxExactInteger+1]; !ok || valid {
		t.Fatal("the shared cases lost the invalid 2^53 case")
	}
}

// The A2A decode path rejects duplicate keys anywhere in the WorkerCompletion,
// maps the integral doubles of a google.protobuf.Value, as the Python Runtime
// renders them, to integers, and rejects every integer above 2^53-1, whether
// written exactly or as the double the Runtime's encoding would round it to.
func TestInvokerRejectsUnsafeProtobufNumbers(t *testing.T) {
	encoded, err := json.Marshal(successResult())
	if err != nil {
		t.Fatal(err)
	}
	for name, raw := range map[string]string{
		"fractional integer":    strings.Replace(string(encoded), `"stateRevision":7`, `"stateRevision":7.5`, 1),
		"integer beyond uint64": strings.Replace(string(encoded), `"stateRevision":7`, `"stateRevision":1.8446744073709552e+19`, 1),
		"2^53 as a double":      strings.Replace(string(encoded), `"stateRevision":7`, `"stateRevision":9007199254740992.0`, 1),
		"exact 2^53":            strings.Replace(string(encoded), `"stateRevision":7`, `"stateRevision":9007199254740992`, 1),
		"exact 2^53+1":          strings.Replace(string(encoded), `"stateRevision":7`, `"stateRevision":9007199254740993`, 1),
		"2^64-1":                strings.Replace(string(encoded), `"stateRevision":7`, `"stateRevision":18446744073709551615`, 1),
		"tool calls 2^53":       strings.Replace(string(encoded), `"tools":{}`, `"tools":{"grep":{"calls":9007199254740992.0,"failures":0.0}}`, 1),
	} {
		t.Run(name, func(t *testing.T) {
			if raw == string(encoded) {
				t.Fatal("the case did not change the WorkerCompletion")
			}
			_, err := invokeOfficialSDK(t, json.RawMessage(raw), "0")
			assertPlannerCode(t, err, "invalid_worker_result")
		})
	}

	maximum := strings.Replace(string(encoded), `"stateRevision":7`, `"stateRevision":9007199254740991.0`, 1)
	got, err := invokeOfficialSDK(t, json.RawMessage(maximum), "0")
	if err != nil || got.StateRevision != contracts.MaxExactInteger {
		t.Fatalf("2^53-1 as a double = (%d, %v)", got.StateRevision, err)
	}

	pythonRendering := strings.NewReplacer(`"stateRevision":7`, `"stateRevision":7.0`).Replace(string(encoded))
	pythonRendering = strings.Replace(pythonRendering, `"tools":{}`,
		`"tools":{"grep":{"calls":2.0,"failures":0.0},"read_file":{"calls":3e0,"failures":1.0}}`, 1)
	want := successResult()
	want.Result.Observations.Tools = map[string]contracts.ToolObservationCount{
		"grep": {Calls: 2}, "read_file": {Calls: 3, Failures: 1},
	}
	got, err = invokeOfficialSDK(t, json.RawMessage(pythonRendering), "0")
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("Runtime double rendering = (%+v, %v)", got, err)
	}
}
