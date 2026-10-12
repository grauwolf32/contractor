package runstore

import (
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"

	"go.yaml.in/yaml/v4"
)

// runStateSet is a set of WorkflowRun states that SQL or the schema may spell
// out as a literal list. Every set derives from activeRunStates and
// terminalRunStates or names its states explicitly, so a state added to the
// definition changes the derived sets and fails each literal not updated
// with it. Removing a state needs no guard: a literal that still lists it
// only matches nothing.
type runStateSet struct {
	name, rule string
	states     []WorkflowRunState
}

func declaredRunStateSets() []runStateSet {
	activeWithout := func(excluded WorkflowRunState) []WorkflowRunState {
		return slices.DeleteFunc(slices.Clone(activeRunStates[:]), func(state WorkflowRunState) bool {
			return state == excluded
		})
	}
	return []runStateSet{
		{name: "all", rule: "the workflow_runs state CHECK", states: append(slices.Clone(activeRunStates[:]), terminalRunStates[:]...)},
		{name: "active", rule: "RUN-R30, the non-terminal states", states: activeRunStates[:]},
		{name: "terminal", rule: "RUN-R39", states: terminalRunStates[:]},
		{name: "cancellable", rule: "RUN-R41, every active state but cancelling", states: activeWithout(RunCancelling)},
		{name: "admitted", rule: "RUN-R28 and RUN-R30, admitted Runs that hold an invocation", states: []WorkflowRunState{RunRunning, RunWaiting}},
		{name: "gateway-routed", rule: "RUN-R30, Runs a blocked model route can hold", states: []WorkflowRunState{RunPending, RunRunning, RunWaiting}},
		{name: "cancellation-recorded", rule: "RUN-R40, states that carry a WorkflowRunCancellation", states: []WorkflowRunState{RunCancelling, RunCancelled}},
	}
}

// classifyRunStateList returns the declared set that list spells, "" when the
// list mixes in values that are not WorkflowRun states (it belongs to another
// lifecycle), or an error when it is a Run-state list no rule declares.
func classifyRunStateList(list []string) (string, error) {
	states := make([]WorkflowRunState, 0, len(list))
	for _, value := range list {
		state := WorkflowRunState(value)
		if !state.Valid() {
			return "", nil
		}
		states = append(states, state)
	}
	for _, set := range declaredRunStateSets() {
		if sameRunStates(states, set.states) {
			return set.name, nil
		}
	}
	return "", fmt.Errorf("Run-state list %v matches no declared set; spell one of declaredRunStateSets or declare the rule it follows", list)
}

func sameRunStates(left, right []WorkflowRunState) bool {
	sortedLeft, sortedRight := slices.Clone(left), slices.Clone(right)
	slices.Sort(sortedLeft)
	slices.Sort(sortedRight)
	return slices.Equal(sortedLeft, sortedRight)
}

// sqlValueList matches a parenthesized or bracketed list of at least two
// quoted lowercase values: IN ('a', 'b') or ARRAY['a'::text, 'b'::text].
var (
	sqlValueList = regexp.MustCompile(`[(\[]\s*'[a-z_-]+'(?:::text)?(?:\s*,\s*'[a-z_-]+'(?:::text)?)+\s*[)\]]`)
	sqlValue     = regexp.MustCompile(`'([a-z_-]+)'`)
)

// sqlValueLists returns each value list in text and the line, counted from
// zero, on which it starts.
func sqlValueLists(text string) (lists [][]string, lines []int) {
	for _, span := range sqlValueList.FindAllStringIndex(text, -1) {
		var list []string
		for _, value := range sqlValue.FindAllStringSubmatch(text[span[0]:span[1]], -1) {
			list = append(list, value[1])
		}
		lists = append(lists, list)
		lines = append(lines, strings.Count(text[:span[0]], "\n"))
	}
	return lists, lines
}

func TestRunStateSetsPartitionTheStates(t *testing.T) {
	all := []WorkflowRunState{
		RunInitializing, RunPending, RunRunning, RunWaiting, RunCancelling, RunSucceeded, RunFailed, RunCancelled,
	}
	if !sameRunStates(append(slices.Clone(activeRunStates[:]), terminalRunStates[:]...), all) {
		t.Fatalf("active %v and terminal %v do not partition %v", activeRunStates, terminalRunStates, all)
	}
	for _, state := range all {
		if !state.Valid() || RunLifecycleActive.Includes(state) == RunLifecycleTerminal.Includes(state) {
			t.Errorf("%s: valid=%v active=%v terminal=%v", state, state.Valid(),
				RunLifecycleActive.Includes(state), RunLifecycleTerminal.Includes(state))
		}
	}
	for _, state := range []WorkflowRunState{"", "paused", "Running"} {
		if state.Valid() || RunLifecycleActive.Includes(state) || RunLifecycleTerminal.Includes(state) {
			t.Errorf("%q is accepted as a WorkflowRun state", state)
		}
	}
	if got, err := classifyRunStateList([]string{"initializing", "running", "cancelling"}); err == nil {
		t.Errorf("the pre-pending active list classifies as %q", got)
	}
	if got, err := classifyRunStateList([]string{"preparing", "running"}); got != "" || err != nil {
		t.Errorf("a StageExecution list classifies as (%q, %v)", got, err)
	}
}

// TestRunStateSetsMatchOpenAPIAndGoSQL holds the public enums and every
// Run-state list written in production Go SQL to the one definition.
// TestPostgresRunStateSetsMatchSchema does the same for the schema.
func TestRunStateSetsMatchOpenAPIAndGoSQL(t *testing.T) {
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	t.Run("OpenAPI", func(t *testing.T) {
		data, err := os.ReadFile(filepath.Join(root, "api", "openapi", "contractor-public-v1.yaml"))
		if err != nil {
			t.Fatal(err)
		}
		var document struct {
			Components struct {
				Schemas map[string]struct {
					Enum []string `yaml:"enum"`
				} `yaml:"schemas"`
			} `yaml:"components"`
		}
		if err := yaml.Unmarshal(data, &document); err != nil {
			t.Fatal(err)
		}
		toStates := func(values []string) []WorkflowRunState {
			result := make([]WorkflowRunState, 0, len(values))
			for _, value := range values {
				result = append(result, WorkflowRunState(value))
			}
			return result
		}
		for schema, want := range map[string][]WorkflowRunState{
			"WorkflowRunState":            append(slices.Clone(activeRunStates[:]), terminalRunStates[:]...),
			"NonTerminalWorkflowRunState": activeRunStates[:],
		} {
			if got := toStates(document.Components.Schemas[schema].Enum); !sameRunStates(got, want) {
				t.Errorf("OpenAPI %s = %v, want %v", schema, got, want)
			}
		}
		lifecycles := document.Components.Schemas["WorkflowRunLifecycle"].Enum
		if !slices.Equal(lifecycles, []string{string(RunLifecycleActive), string(RunLifecycleTerminal)}) {
			t.Errorf("OpenAPI WorkflowRunLifecycle = %v", lifecycles)
		}
	})
	t.Run("GoSQL", func(t *testing.T) {
		found := map[string]int{}
		for _, dir := range []string{"cmd", "internal"} {
			err := filepath.WalkDir(filepath.Join(root, dir), func(path string, entry fs.DirEntry, err error) error {
				if err != nil {
					return err
				}
				if entry.IsDir() {
					if entry.Name() == "testdata" {
						return filepath.SkipDir
					}
					return nil
				}
				if !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
					return nil
				}
				fileSet := token.NewFileSet()
				file, err := parser.ParseFile(fileSet, path, nil, 0)
				if err != nil {
					return err
				}
				relative, _ := filepath.Rel(root, path)
				ast.Inspect(file, func(node ast.Node) bool {
					literal, ok := node.(*ast.BasicLit)
					if !ok || literal.Kind != token.STRING {
						return true
					}
					lists, lines := sqlValueLists(literal.Value)
					for index, list := range lists {
						name, err := classifyRunStateList(list)
						if err != nil {
							line := fileSet.Position(literal.Pos()).Line + lines[index]
							t.Errorf("%s:%d: %v", relative, line, err)
						}
						if name != "" {
							found[name]++
						}
					}
					return true
				})
				return nil
			})
			if err != nil {
				t.Fatal(err)
			}
		}
		// The queue, credential, Project-drain and claim-renewal reads spell the
		// active set; a scanner that finds none of them is broken.
		if found["active"] < 5 || found["terminal"] < 3 {
			t.Fatalf("found Run-state lists %v; the scan missed the known active and terminal lists", found)
		}
	})
}

// TestRunLifecycleMatchesSpecification compares runLifecycle with the
// WorkflowRun state diagram of docs/spec/execution/runs.md edge by edge
// (RUN-R26 to RUN-R46), so neither can gain or lose an edge alone. It also
// holds ValidateRunTransition to exactly the edges TransitionRun performs,
// and the operation-owned edges to the states RUN-R30, RUN-R41 and RUN-R46
// name.
func TestRunLifecycleMatchesSpecification(t *testing.T) {
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(filepath.Join(root, "docs", "spec", "execution", "runs.md"))
	if err != nil {
		t.Fatal(err)
	}
	const fence = "```mermaid\nstateDiagram-v2\n"
	section := string(data)
	if start := strings.Index(section, "### Run lifecycle"); start >= 0 {
		section = section[start:]
	} else {
		t.Fatal("runs.md has no Run lifecycle section")
	}
	start := strings.Index(section, fence)
	if start < 0 {
		t.Fatal("the Run lifecycle section has no state diagram")
	}
	diagram := section[start+len(fence):]
	diagram = diagram[:strings.Index(diagram, "```")]
	edgeLine := regexp.MustCompile(`^\s*(\S+)\s+-->\s+([^\s:]+)\s*(?::.*)?$`)
	specified := map[[2]WorkflowRunState]bool{}
	for _, line := range strings.Split(diagram, "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		match := edgeLine.FindStringSubmatch(line)
		if match == nil {
			t.Fatalf("state diagram line %q is not an edge", line)
		}
		if match[1] == "[*]" || match[2] == "[*]" {
			continue
		}
		from, to := WorkflowRunState(match[1]), WorkflowRunState(match[2])
		if !from.Valid() || !to.Valid() {
			t.Fatalf("state diagram edge %s -> %s names a state that is not a WorkflowRun state", from, to)
		}
		specified[[2]WorkflowRunState{from, to}] = true
	}
	if len(specified) == 0 {
		t.Fatal("the state diagram has no edge")
	}
	for edge := range specified {
		if _, ok := runLifecycle[edge]; !ok {
			t.Errorf("the specification has %s -> %s; runLifecycle does not", edge[0], edge[1])
		}
	}
	for edge := range runLifecycle {
		if !specified[edge] {
			t.Errorf("runLifecycle has %s -> %s; the specification does not", edge[0], edge[1])
		}
	}

	states := append(append(slices.Clone(activeRunStates[:]), terminalRunStates[:]...), "paused")
	for _, from := range states {
		for _, to := range states {
			operation, edge := runLifecycle[[2]WorkflowRunState{from, to}]
			want := edge && operation == transitionByTransitionRun
			if err := ValidateRunTransition(from, to); (err == nil) != want || err != nil && !errors.Is(err, ErrInvalid) {
				t.Errorf("ValidateRunTransition(%s, %s) = %v, want allowed=%t", from, to, err, want)
			}
		}
	}
	var cancellable []WorkflowRunState
	waitingEdges := 0
	for edge, operation := range runLifecycle {
		switch operation {
		case transitionByCancellation:
			if edge[1] != RunCancelling {
				t.Errorf("cancellation performs %s -> %s", edge[0], edge[1])
			}
			cancellable = append(cancellable, edge[0])
		case transitionByContinuation:
			if edge != [2]WorkflowRunState{RunFailed, RunPending} {
				t.Errorf("manual continuation performs %s -> %s", edge[0], edge[1])
			}
		case transitionByGatewayWait:
			// Only an admitted Run's invocation waits for its model route.
			if edge != [2]WorkflowRunState{RunRunning, RunWaiting} && edge != [2]WorkflowRunState{RunWaiting, RunRunning} {
				t.Errorf("the waiting operations perform %s -> %s", edge[0], edge[1])
			}
			waitingEdges++
		}
	}
	if waitingEdges != 2 {
		t.Errorf("the waiting operations perform %d edges, want running -> waiting and waiting -> running", waitingEdges)
	}
	for _, set := range declaredRunStateSets() {
		if set.name == "cancellable" && !sameRunStates(cancellable, set.states) {
			t.Errorf("cancellation leaves %v, want the cancellable states %v", cancellable, set.states)
		}
	}
}
