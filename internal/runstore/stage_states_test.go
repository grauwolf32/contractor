package runstore

import (
	"fmt"
	"path/filepath"
	"slices"
	"testing"
)

// stageStateSet is a set of StageExecution states that SQL or the schema may
// spell out as a literal list. The active, terminal and all sets derive from
// activeStageStates and terminalStageStates, so a state added to the
// definition changes them and fails each literal not updated with it; the
// named subsets list the states their rule names.
type stageStateSet struct {
	name, rule string
	states     []StageExecutionState
}

func declaredStageStateSets() []stageStateSet {
	return []stageStateSet{
		{
			name: "all", rule: "the stage_executions state CHECK and the OpenAPI StageExecutionState enum",
			states: append(slices.Clone(activeStageStates[:]), terminalStageStates[:]...),
		},
		{name: "active", rule: "SCH-R42, the non-terminal states; one per Run (SCH-R114)", states: activeStageStates[:]},
		{name: "terminal", rule: "SCH-R42", states: terminalStageStates[:]},
		{
			name: "continuable-outcome", rule: "SCH-R89 and RUN-R46, the outcomes an escalation or a manual continuation starts from",
			states: []StageExecutionState{StageFailed, StageInterrupted},
		},
		{
			name: "termination-phase", rule: "SCH-R42 and SCH-R100, the states that enter aborting and that a StageTermination records as its phase",
			states: []StageExecutionState{StagePreparing, StageRunning},
		},
	}
}

func stageStateStrings(states []StageExecutionState) []string {
	result := make([]string, 0, len(states))
	for _, state := range states {
		result = append(result, string(state))
	}
	return result
}

// classifyStageStateList returns the declared set that list spells, "" when
// the list belongs to another lifecycle (a value that is no StageExecution
// state, or a declared WorkflowRun set whose values are StageExecution
// states too), or an error when it is a StageExecution-state list no rule
// declares.
func classifyStageStateList(list []string) (string, error) {
	all := declaredStageStateSets()[0].states
	states := make([]StageExecutionState, 0, len(list))
	for _, value := range list {
		state := StageExecutionState(value)
		if !slices.Contains(all, state) {
			return "", nil
		}
		states = append(states, state)
	}
	for _, set := range declaredStageStateSets() {
		if sameStageStates(states, set.states) {
			return set.name, nil
		}
	}
	if name, err := classifyRunStateList(list); name != "" && err == nil {
		return "", nil
	}
	return "", fmt.Errorf(
		"StageExecution-state list %v matches no declared set; spell one of declaredStageStateSets or declare the rule it follows",
		list,
	)
}

func sameStageStates(left, right []StageExecutionState) bool {
	sortedLeft, sortedRight := slices.Clone(left), slices.Clone(right)
	slices.Sort(sortedLeft)
	slices.Sort(sortedRight)
	return slices.Equal(sortedLeft, sortedRight)
}

func TestStageStateSetsPartitionTheStates(t *testing.T) {
	all := []StageExecutionState{
		StagePreparing, StageRunning, StageFinalizing, StageAborting,
		StageSucceeded, StageFailed, StageInterrupted, StageCancelled,
	}
	if !sameStageStates(append(slices.Clone(activeStageStates[:]), terminalStageStates[:]...), all) {
		t.Fatalf("active %v and terminal %v do not partition %v", activeStageStates, terminalStageStates, all)
	}
	for _, state := range all {
		if state.Terminal() == slices.Contains(activeStageStates[:], state) {
			t.Errorf("%s: terminal=%v active=%v", state, state.Terminal(), slices.Contains(activeStageStates[:], state))
		}
	}
	for _, state := range []StageExecutionState{"", "pending", "Succeeded"} {
		if state.Terminal() {
			t.Errorf("%q is a terminal StageExecution state", state)
		}
	}
	for _, set := range declaredStageStateSets() {
		for _, state := range set.states {
			if !slices.Contains(all, state) {
				t.Errorf("declared set %s names %q, which is no StageExecution state", set.name, state)
			}
		}
	}
	for list, want := range map[string][]string{
		"undeclared":   {"preparing", "finalizing"},
		"stale active": {"preparing", "running", "finalizing"},
	} {
		if got, err := classifyStageStateList(want); err == nil {
			t.Errorf("the %s list %v classifies as %q", list, want, got)
		}
	}
	for _, list := range [][]string{{"pending", "running"}, {"succeeded", "failed", "cancelled"}} {
		if got, err := classifyStageStateList(list); got != "" || err != nil {
			t.Errorf("the WorkflowRun list %v classifies as (%q, %v)", list, got, err)
		}
	}
}

// TestStageStateSetsMatchOpenAPIAndGoSQL holds the public StageExecutionState
// enum and every StageExecution-state list written in production Go SQL to
// the one definition. TestPostgresStageExecutionStateSetsMatchSchema does the
// same for the schema.
func TestStageStateSetsMatchOpenAPIAndGoSQL(t *testing.T) {
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	t.Run("OpenAPI", func(t *testing.T) {
		enum := openAPIEnums(t, root)["StageExecutionState"]
		got := make([]StageExecutionState, 0, len(enum))
		for _, value := range enum {
			got = append(got, StageExecutionState(value))
		}
		if want := declaredStageStateSets()[0].states; !sameStageStates(got, want) {
			t.Errorf("OpenAPI StageExecutionState = %v, want %v", got, want)
		}
	})
	t.Run("GoSQL", func(t *testing.T) {
		found := map[string]int{}
		productionSQLValueLists(t, root, func(position string, list []string) {
			name, err := classifyStageStateList(list)
			if err != nil {
				t.Errorf("%s: %v", position, err)
			}
			if name != "" {
				found[name]++
			}
		})
		// The active-Stage, resumption and allocation-release reads spell the
		// active and terminal sets; a scanner that finds none of them is
		// broken.
		if found["active"] < 3 || found["terminal"] < 1 {
			t.Fatalf("found StageExecution-state lists %v; the scan missed the known active and terminal lists", found)
		}
	})
}
