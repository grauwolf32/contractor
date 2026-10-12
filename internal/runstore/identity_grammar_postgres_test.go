package runstore

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
)

// TestPostgresStoreRejectsIdentitiesOutsideThePublicGrammars is the
// regression for a run_id the store accepted as any non-blank text while
// ListNonTerminalRunIDsByCredential rejected it on read, which blocked the
// deletion of every credential that Run pinned. Run, Project, StageExecution
// and allocation identities now use the public ResourceId grammar, and Stage
// and logical Agent names the shared name grammar, when they are written.
func TestPostgresStoreRejectsIdentitiesOutsideThePublicGrammars(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	store := NewPostgresStore(pool)
	pinned := runtimeconfigtest.BuiltInRunSnapshot()
	pinned.LLMCredentialIDs = []string{"grammar-route"}
	runParams := func(runID string) CreateRunParams {
		params := testRunParams(runID)
		params.RuntimeConfig = pinned
		return params
	}

	longestRunID := strings.Repeat("r", contracts.MaxResourceIDLength)
	for _, runID := range []string{"run 1", "_run", ":run", "run/1", "run@1", "rün", longestRunID + "r"} {
		if _, err := createRun(ctx, store, runParams(runID)); !errors.Is(err, ErrInvalid) {
			t.Fatalf("CreateRun(%q) error = %v, want ErrInvalid", runID, err)
		}
	}
	project := "project 1"
	withProject := runParams("run-project-grammar")
	withProject.ProjectID = &project
	if _, err := createRun(ctx, store, withProject); !errors.Is(err, ErrInvalid) {
		t.Fatalf("CreateRun with Project %q error = %v, want ErrInvalid", project, err)
	}
	for _, runID := range []string{longestRunID, "run-grammar.1:a_b"} {
		if _, err := createRun(ctx, store, runParams(runID)); err != nil {
			t.Fatalf("CreateRun(%q): %v", runID, err)
		}
	}
	runIDs, err := store.ListNonTerminalRunIDsByCredential(ctx, "grammar-route", 10)
	if err != nil || len(runIDs) != 2 {
		t.Fatalf("credential Run references = (%v, %v), want both stored Runs", runIDs, err)
	}

	run := createTestRun(t, ctx, store, "run-stage-grammar")
	if _, err := admitTestRun(ctx, store, run.RunID); err != nil {
		t.Fatal(err)
	}
	stageParams := func(mutate func(*CreateStageExecutionParams)) CreateStageExecutionParams {
		params := CreateStageExecutionParams{
			StageExecutionID: "stage-grammar", RunID: run.RunID,
			StageName: "copy", Attempt: 1, StageSpecSchemaVersion: contracts.APIVersion,
			StageSpecSnapshot: json.RawMessage(`{}`), StageContextSchemaVersion: contracts.APIVersion,
			StageContext: StageContextSnapshot{},
		}
		mutate(&params)
		return params
	}
	for label, mutate := range map[string]func(*CreateStageExecutionParams){
		"StageExecution ID": func(params *CreateStageExecutionParams) { params.StageExecutionID = "stage 1" },
		"Stage name":        func(params *CreateStageExecutionParams) { params.StageName = "my stage" },
		"previous execution": func(params *CreateStageExecutionParams) {
			previous := "stage/0"
			params.Attempt, params.PreviousExecutionID = 2, &previous
		},
	} {
		if _, err := store.CreateStageExecution(ctx, stageParams(mutate)); !errors.Is(err, ErrInvalid) {
			t.Fatalf("CreateStageExecution with invalid %s error = %v, want ErrInvalid", label, err)
		}
	}
	execution, err := store.CreateStageExecution(ctx, stageParams(func(*CreateStageExecutionParams) {}))
	if err != nil {
		t.Fatal(err)
	}

	allocation := func(mutate func(*StageAllocation)) StageAllocation {
		value := StageAllocation{
			AllocationID: "allocation-grammar", StageExecutionID: execution.StageExecutionID,
			LogicalAgentName: "builder", Namespace: "builder",
			AgentTemplateRef: contracts.AgentTemplateRef{
				TemplateID: "builder", Version: "1", Digest: "sha256:" + strings.Repeat("a", 64),
			},
			WorkerRuntimeRef: contracts.WorkerRuntimeRef{RuntimeID: "adk", Version: "1"},
			RuntimeAgentID:   strings.Repeat("2", 64), RuntimeAgentInstanceID: "runtime-grammar",
			RuntimeAgentLabelRevision:         1,
			RuntimeConfigurationSchemaVersion: AllocationRuntimeConfigurationSchemaVersion,
			RuntimeConfiguration:              testAllocationRuntimeConfiguration(),
			PerformanceCollectionPolicy:       reporting.PerformanceCollectionDisabled,
		}
		mutate(&value)
		return value
	}
	for label, mutate := range map[string]func(*StageAllocation){
		"allocation ID":       func(value *StageAllocation) { value.AllocationID = "allocation 1" },
		"logical Agent name":  func(value *StageAllocation) { value.LogicalAgentName = "my builder" },
		"Runtime instance ID": func(value *StageAllocation) { value.RuntimeAgentInstanceID = "runtime 1" },
	} {
		if err := store.RecordStageAllocation(ctx, allocation(mutate)); !errors.Is(err, ErrInvalid) {
			t.Fatalf("RecordStageAllocation with invalid %s error = %v, want ErrInvalid", label, err)
		}
	}
	if err := store.RecordStageAllocation(ctx, allocation(func(*StageAllocation) {})); err != nil {
		t.Fatal(err)
	}
}
