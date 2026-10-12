package a2a

import (
	"bytes"
	"context"
	"encoding/json"
	"iter"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

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
			part := sdk.NewDataPart(data)
			part.MediaType = workerCompletionMediaType
			message := sdk.NewMessage(sdk.MessageRoleAgent, part)
			message.TaskID, message.ContextID = "task-1", "context-1"
			yield(message, nil)
		}
	})
	server := httptest.NewServer(a2asrv.NewJSONRPCHandler(a2asrv.NewHandler(executor)))
	defer server.Close()
	invoker, err := New(server.Client(), Options{PollInterval: time.Millisecond})
	if err != nil {
		t.Fatal(err)
	}
	request := stageRequest()
	request.SubtaskID = subtaskID
	return invoker.Invoke(t.Context(), "builder", workerHandle(server.URL), request)
}

// TestSharedWorkerResultCasesPassTheA2ADecodePath sends every shared Worker
// result case through the official SDK and the Invoker. The DataPart is
// decoded from the response bytes, so uint64_max keeps its digits and each
// case gets the verdict of the private decoder.
func TestSharedWorkerResultCasesPassTheA2ADecodePath(t *testing.T) {
	var cases []struct {
		Name      string `json:"name"`
		Unit      string `json:"unit"`
		Repeat    int    `json:"repeat"`
		Prefix    string `json:"prefix"`
		Padding   int    `json:"padding"`
		MaxUint64 bool   `json:"max_uint64"`
		Valid     bool   `json:"valid"`
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "worker-result-size-cases.json"), &cases); err != nil {
		t.Fatal(err)
	}
	base := contractstest.ReadFixture(t, "valid", "worker-completion-empty-observations.json")
	sawMaximum := false
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
			if test.MaxUint64 {
				sawMaximum = true
				want.StateRevision = ^uint64(0)
				want.Result.Observations.Tools = map[string]contracts.ToolObservationCount{"read_file": {Calls: ^uint64(0)}}
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
	if !sawMaximum {
		t.Fatal("the shared cases lost uint64_max")
	}
}

// The A2A decode path rejects duplicate keys anywhere in the WorkerCompletion,
// keeps integers beyond float64 precision, and maps the integral doubles of
// a google.protobuf.Value, as the Python Runtime renders them, to integers.
func TestInvokerDecodesTheDataPartFromResponseBytes(t *testing.T) {
	encoded, err := json.Marshal(successResult())
	if err != nil {
		t.Fatal(err)
	}
	for name, raw := range map[string]string{
		"duplicate top-level key": strings.Replace(string(encoded), `{"apiVersion"`, `{"stateRevision":1,"apiVersion"`, 1),
		"duplicate nested key":    strings.Replace(string(encoded), `"summarized":false`, `"summarized":true,"summarized":false`, 1),
		"duplicate escaped key":   strings.Replace(string(encoded), `"invocationId"`, `"invocationId":"other","invocationId"`, 1),
		"fractional integer":      strings.Replace(string(encoded), `"stateRevision":7`, `"stateRevision":7.5`, 1),
		"integer beyond uint64":   strings.Replace(string(encoded), `"stateRevision":7`, `"stateRevision":1.8446744073709552e+19`, 1),
	} {
		t.Run(name, func(t *testing.T) {
			if raw == string(encoded) {
				t.Fatal("the case did not change the WorkerCompletion")
			}
			_, err := invokeOfficialSDK(t, json.RawMessage(raw), "0")
			assertPlannerCode(t, err, "invalid_worker_result")
		})
	}

	beyondFloat := strings.Replace(string(encoded), `"stateRevision":7`, `"stateRevision":9007199254740993`, 1)
	got, err := invokeOfficialSDK(t, json.RawMessage(beyondFloat), "0")
	if err != nil || got.StateRevision != 9007199254740993 {
		t.Fatalf("2^53+1 = (%d, %v)", got.StateRevision, err)
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

func TestIntegralNumbersRewritesOnlyIntegralDoublesOutsideStrings(t *testing.T) {
	for input, want := range map[string]string{
		`{"a":7.0,"b":[1e3,-2.0,0.5],"c":"7.0","d":"\"8.0\\\\","e":12}`: `{"a":7,"b":[1000,-2,0.5],"c":"7.0","d":"\"8.0\\\\","e":12}`,
		`{"a":18446744073709551615,"b":1.8446744073709552e+19}`:         `{"a":18446744073709551615,"b":1.8446744073709552e+19}`,
		`{"a":9007199254740992.0,"b":1.5e1,"c":1e400,"d":true}`:         `{"a":9007199254740992,"b":15,"c":1e400,"d":true}`,
	} {
		if got := integralNumbers([]byte(input)); !bytes.Equal(got, []byte(want)) {
			t.Errorf("integralNumbers(%s) = %s, want %s", input, got, want)
		}
	}
}
