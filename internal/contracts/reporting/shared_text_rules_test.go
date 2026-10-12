package reporting

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

// TestSharedWorkerStatePathCases checks the Worker State path rule of
// api/testdata/v1alpha1/workspace-path-cases.json, which the Runtime's
// _require_state_workspace_path and project path normalizer apply too.
func TestSharedWorkerStatePathCases(t *testing.T) {
	t.Parallel()
	_, _, cases := contractstest.WorkspacePathCases(t)
	for _, test := range cases {
		if got := validateStateWorkspacePath(test.Value) == nil; got != test.WorkerStatePath {
			t.Errorf("%s: Worker State path accepted = %t, want %t", test.Name, got, test.WorkerStatePath)
		}
	}
}

// TestSharedWorkerStateBoundsCountCompactUTF8 decodes Worker State whose
// tool-call arguments and workspace path lists encode to exactly their bounds
// in compact UTF-8, with U+2028 counting three bytes as
// json-byte-size-cases.json pins. The Runtime accepts the same State
// (test_shared_worker_state_bounds); one more byte is over the bound on both
// sides.
func TestSharedWorkerStateBoundsCountCompactUTF8(t *testing.T) {
	t.Parallel()
	decode := func(t *testing.T, mutate func(state map[string]any)) error {
		t.Helper()
		var snapshot map[string]any
		if err := json.Unmarshal(contractstest.ReadFixture(t, "valid", "agent-state-snapshot.json"), &snapshot); err != nil {
			t.Fatal(err)
		}
		mutate(snapshot["state"].(map[string]any))
		data, err := json.Marshal(snapshot)
		if err != nil {
			t.Fatal(err)
		}
		_, err = contracts.DecodePrivateStrict[AgentStateSnapshot](data)
		return err
	}
	for _, test := range []struct {
		suffix string
		valid  bool
	}{{"aa", true}, {"aaa", false}} {
		// {"path":"..."} costs 11 bytes and 1361 U+2028 cost 4083.
		arguments := map[string]any{"path": strings.Repeat(" ", 1361) + test.suffix}
		if size, err := contracts.ResultJSONSize(arguments); err != nil || size != 4094+len(test.suffix) {
			t.Fatalf("arguments size = %d, %v", size, err)
		}
		err := decode(t, func(state map[string]any) {
			calls := state["metrics"].(map[string]any)["toolCalls"].([]any)
			calls[0].(map[string]any)["arguments"] = arguments
		})
		if (err == nil) != test.valid {
			t.Fatalf("Worker State with %d-byte arguments accepted = %t, want %t: %v", 4094+len(test.suffix), err == nil, test.valid, err)
		}
	}
	for _, test := range []struct {
		extra int
		valid bool
	}{{0, true}, {1, false}} {
		paths := sharedBoundPathList(maxStateWorkspacePathBytes + test.extra)
		if size, err := contracts.ResultJSONSize(paths); err != nil || size != maxStateWorkspacePathBytes+test.extra {
			t.Fatalf("scope path list size = %d, %v", size, err)
		}
		err := decode(t, func(state map[string]any) {
			state["currentInvocation"].(map[string]any)["workspace"].(map[string]any)["scopePaths"] = paths
		})
		if (err == nil) != test.valid {
			t.Fatalf("Worker State with a %d-byte scope list accepted = %t, want %t: %v", maxStateWorkspacePathBytes+test.extra, err == nil, test.valid, err)
		}
	}
}

// sharedBoundPathList returns sorted, unique Worker State paths whose compact
// UTF-8 JSON list is exactly size bytes: d/<5-digit index>/ followed by U+2028
// characters, which count three bytes each, and up to two "x" to reach the
// exact size. runtime/tests/test_shared_text_rules.py builds the same list.
func sharedBoundPathList(size int) []string {
	const fullBody, lastBodyLimit = 3 * 1300, 4000
	var paths []string
	for remaining := size - 2; remaining > 0; { // 2 for the brackets
		prefix := fmt.Sprintf("d/%05d/", len(paths))
		overhead := len(prefix) + 2 // quotes
		if len(paths) > 0 {
			overhead++ // comma
		}
		body := remaining - overhead
		if body > lastBodyLimit {
			body = fullBody
		}
		paths = append(paths, prefix+strings.Repeat("\u2028", body/3)+strings.Repeat("x", body%3))
		remaining -= overhead + body
	}
	return paths
}
