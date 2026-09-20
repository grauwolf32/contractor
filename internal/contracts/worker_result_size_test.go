package contracts

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

func TestWorkerResultSizeSharedPythonCases(t *testing.T) {
	raw, err := os.ReadFile("../../api/testdata/v1alpha1/worker-result-size-cases.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		Name      string `json:"name"`
		Unit      string `json:"unit"`
		Repeat    int    `json:"repeat"`
		Prefix    string `json:"prefix"`
		Padding   int    `json:"padding"`
		MaxUint64 bool   `json:"max_uint64"`
		Valid     bool   `json:"valid"`
		JSONBytes int    `json:"json_bytes"`
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatal(err)
	}
	base, err := os.ReadFile("../../api/testdata/v1alpha1/valid/worker-completion-empty-observations.json")
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range cases {
		t.Run(test.Name, func(t *testing.T) {
			var value WorkerCompletion
			if err := json.Unmarshal(base, &value); err != nil {
				t.Fatal(err)
			}
			value.Result.Result = test.Prefix + strings.Repeat(test.Unit, test.Repeat)
			if test.Padding > 0 {
				revision := "r1"
				value.Result.Artifacts = map[string]ArtifactRef{
					strings.Repeat("p", test.Padding): {Namespace: "review", Name: "report", Revision: &revision},
				}
			}
			if test.MaxUint64 {
				value.StateRevision = ^uint64(0)
				value.Result.Observations.Tools = map[string]ToolObservationCount{"read_file": {Calls: ^uint64(0)}}
			}
			encoded, err := json.Marshal(value)
			if err != nil {
				t.Fatal(err)
			}
			size, err := ResultJSONSize(value)
			if err != nil || size != test.JSONBytes {
				t.Fatalf("compact UTF-8 size = %d, %v; want %d", size, err, test.JSONBytes)
			}
			_, err = DecodePrivateStrict[WorkerCompletion](encoded)
			if (err == nil) != test.Valid {
				t.Fatalf("valid=%v: %v", test.Valid, err)
			}
		})
	}
}

func TestResultJSONSizeRejectsUnencodableValues(t *testing.T) {
	if _, err := ResultJSONSize(func() {}); err == nil {
		t.Fatal("unencodable result accepted")
	}
}
