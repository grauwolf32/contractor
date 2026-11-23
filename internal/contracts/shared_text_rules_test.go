package contracts

import (
	"bytes"
	"encoding/json"
	"slices"
	"strings"
	"testing"
	"unicode"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

// The tests in this file read the text-rule fixtures that
// runtime/tests/test_shared_text_rules.py reads too; make
// test-wire-cross-language runs both sides.

func TestSharedWorkspacePathCases(t *testing.T) {
	t.Parallel()
	target, state, cases := contractstest.WorkspacePathCases(t)
	if target != (contractstest.WorkspacePathRule{EmptyAllowed: true, MaxBytes: MaxWorkspaceTargetBytes, MaxComponents: MaxWorkspaceTargetComponents}) ||
		state != (contractstest.WorkspacePathRule{EmptyAllowed: false, MaxBytes: MaxWorkerStatePathBytes, MaxComponents: MaxWorkerStatePathComponents}) {
		t.Fatalf("shared path rule bounds = %+v, %+v; want the contracts bounds", target, state)
	}
	revision := "1"
	for _, test := range cases {
		// A workspace target is validated where an allocation pins it, and a
		// file the Worker reports reading follows the same rule.
		spec := AllocationWorkspaceSpec{Mode: WorkspaceModeDirect, Sources: []AllocationWorkspaceSource{{
			Artifact: ArtifactRef{Namespace: "inputs", Name: "source", Revision: &revision},
			Target:   test.Value,
		}}}
		if got := spec.Validate() == nil; got != test.WorkspaceTarget {
			t.Errorf("%s: workspace target accepted = %t, want %t", test.Name, got, test.WorkspaceTarget)
		}
		reported := test.Value != "" && test.WorkspaceTarget
		summary := WorkspaceObservationSummary{ReadFiles: 1, FilesRead: []string{test.Value}}
		if got := summary.validate() == nil; got != reported {
			t.Errorf("%s: reported file accepted = %t, want %t", test.Name, got, reported)
		}
		if got := RelativePathProblem(test.Value, state.MaxBytes, state.MaxComponents) == ""; got != test.WorkerStatePath {
			t.Errorf("%s: Worker State path accepted = %t, want %t", test.Name, got, test.WorkerStatePath)
		}
	}
}

// TestSharedBlankTextCases pins blank text to unicode.IsSpace and checks it
// on the hub audit's example: an objective of U+001C passes the Server and,
// through the same fixture, the Runtime.
func TestSharedBlankTextCases(t *testing.T) {
	t.Parallel()
	var cases struct {
		Description string   `json:"description"`
		Whitespace  []string `json:"whitespace"`
		Blank       []string `json:"blank"`
		Text        []string `json:"text"`
	}
	decoder := json.NewDecoder(bytes.NewReader(contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "blank-text-cases.json")))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&cases); err != nil {
		t.Fatal(err)
	}
	if len(cases.Whitespace) == 0 || len(cases.Blank) == 0 || len(cases.Text) == 0 {
		t.Fatal("shared blank text cases are empty")
	}
	var whitespace []string
	for character := rune(0); character <= unicode.MaxRune; character++ {
		if unicode.IsSpace(character) {
			whitespace = append(whitespace, string(character))
		}
	}
	if !slices.Equal(whitespace, cases.Whitespace) {
		t.Fatalf("unicode.IsSpace = %q, want the shared whitespace %q", whitespace, cases.Whitespace)
	}
	var request StageContentRequest
	if err := json.Unmarshal(contractstest.ReadFixture(t, "valid", "stage-content-request.json"), &request); err != nil {
		t.Fatal(err)
	}
	for _, value := range cases.Blank {
		request.Objective = value
		if strings.TrimSpace(value) != "" || ValidateOpaqueID("field", value) == nil || request.Validate() == nil {
			t.Errorf("blank %q was treated as text", value)
		}
	}
	for _, value := range cases.Text {
		request.Objective = value
		if strings.TrimSpace(value) == "" || ValidateOpaqueID("field", value) != nil {
			t.Errorf("text %q was treated as blank", value)
		}
		if err := request.Validate(); err != nil {
			t.Errorf("objective %q was rejected: %v", value, err)
		}
	}
}

func TestSharedJSONByteSizeCases(t *testing.T) {
	t.Parallel()
	var cases struct {
		Description string `json:"description"`
		Cases       []struct {
			Name  string          `json:"name"`
			Value json.RawMessage `json:"value"`
			Bytes *int            `json:"bytes"`
		} `json:"cases"`
	}
	decoder := json.NewDecoder(bytes.NewReader(contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "json-byte-size-cases.json")))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&cases); err != nil {
		t.Fatal(err)
	}
	if len(cases.Cases) == 0 {
		t.Fatal("shared JSON byte size cases are empty")
	}
	for _, test := range cases.Cases {
		if test.Name == "" || test.Value == nil || test.Bytes == nil {
			t.Fatalf("JSON byte size case %q needs a name, a value and bytes", test.Name)
		}
		var value any
		if err := json.Unmarshal(test.Value, &value); err != nil {
			t.Fatalf("%s: %v", test.Name, err)
		}
		if got, err := ResultJSONSize(value); err != nil || got != *test.Bytes {
			t.Errorf("%s: ResultJSONSize = %d, %v, want %d", test.Name, got, err, *test.Bytes)
		}
	}
}
