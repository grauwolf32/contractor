package runstore

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestPostgresAllocationReadsEnforceWriteInvariants(t *testing.T) {
	ctx := t.Context()
	pool := isolatedRunStorePool(t, ctx)
	store := NewPostgresStore(pool)
	createPreparingStage(t, ctx, store, "run-allocation-validation", "stage-allocation-validation")
	valid := testStageAllocation("stage-allocation-validation", "allocation-validation")
	if err := store.RecordStageAllocation(ctx, valid); err != nil {
		t.Fatal(err)
	}
	// Simulate a broken writer or damaged row, past the database guards.
	unguardTable(t, ctx, pool, "stage_allocations")
	for _, test := range []struct {
		name   string
		change func(*StageAllocation)
	}{
		{"digest", func(a *StageAllocation) { a.AgentTemplateRef.Digest = "invalid" }},
		{"template identity", func(a *StageAllocation) { a.AgentTemplateRef.TemplateID = "" }},
		{"runtime version", func(a *StageAllocation) { a.WorkerRuntimeRef.Version = "" }},
		{"allocation identity", func(a *StageAllocation) { a.AllocationID = "invalid identity" }},
		{"logical Agent name", func(a *StageAllocation) { a.LogicalAgentName = "invalid name" }},
		{"namespace", func(a *StageAllocation) { a.Namespace = "invalid/namespace" }},
		{"Runtime instance identity", func(a *StageAllocation) { a.RuntimeAgentInstanceID = "invalid identity" }},
		{"completion namespace", func(a *StageAllocation) {
			revision := "revision"
			a.CompletionContract = &contracts.WorkerCompletionContract{
				Kind:              contracts.AuditCheckResultsV1,
				Task:              contracts.ArtifactRef{Namespace: "inputs", Name: "task", Revision: &revision},
				ExecutionManifest: contracts.ArtifactRef{Namespace: "inputs", Name: "manifest", Revision: &revision},
				ResultArtifact:    contracts.ArtifactRef{Namespace: "another-agent", Name: "result"},
			}
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			value := valid
			test.change(&value)
			if err := value.Validate(); !errors.Is(err, ErrInvalid) {
				t.Fatalf("write validation = %v, want ErrInvalid", err)
			}
			template, _ := json.Marshal(value.AgentTemplateRef)
			runtime, _ := json.Marshal(value.WorkerRuntimeRef)
			var completion []byte
			if value.CompletionContract != nil {
				completion, _ = json.Marshal(value.CompletionContract)
			}
			if _, err := pool.Exec(ctx, `UPDATE stage_allocations
SET allocation_id=$2, logical_agent_name=$3, namespace=$4,
    agent_template_ref=$5::jsonb, worker_runtime_ref=$6::jsonb,
    runtime_agent_instance_id=$7, completion_contract=$8::jsonb
WHERE stage_execution_id=$1`, value.StageExecutionID, value.AllocationID, value.LogicalAgentName,
				value.Namespace, template, runtime, value.RuntimeAgentInstanceID, completion); err != nil {
				t.Fatal(err)
			}
			for _, batch := range []bool{false, true} {
				var err error
				if batch {
					_, err = store.ListStageAllocationsBatch(ctx, []string{valid.StageExecutionID})
				} else {
					_, err = store.ListStageAllocations(ctx, valid.StageExecutionID)
				}
				if err == nil || errors.Is(err, ErrInvalid) {
					t.Fatalf("read batch=%t = %v, want a stored-data failure", batch, err)
				}
			}
		})
	}
}
