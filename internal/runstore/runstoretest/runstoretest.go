// Package runstoretest holds the WorkflowRun setup that tests of other
// packages share. It stores Runs only through the RunStore's production
// paths; production code never imports it.
package runstoretest

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// CreateRun stores params as a new initializing WorkflowRun through the
// public create-Run path, CreateRunIdempotent, with an idempotency key and a
// request digest derived from the Run ID. A Run ID that is already stored is
// runstore.ErrConflict, whichever owner stored it.
func CreateRun(
	ctx context.Context, store *runstore.PostgresStore, params runstore.CreateRunParams,
) (runstore.WorkflowRun, error) {
	digest := contentdigest.Bytes([]byte("runstoretest.CreateRun\x00" + params.RunID))
	run, inserted, err := store.CreateRunIdempotent(ctx, runstore.CreateRunIdempotentParams{
		CreateRunParams: params,
		IdempotencyKey:  "test-" + strings.TrimPrefix(digest, "sha256:"),
		RequestDigest:   digest,
	})
	if err != nil {
		return runstore.WorkflowRun{}, err
	}
	if !inserted {
		return runstore.WorkflowRun{}, fmt.Errorf("create WorkflowRun %q: %w", params.RunID, runstore.ErrConflict)
	}
	return run, nil
}

// CreateRunningStage stores and starts one minimal Stage through production
// RunStore paths. The caller has already admitted runID as running.
func CreateRunningStage(ctx context.Context, store *runstore.PostgresStore, runID, stageID, schemaVersion string) error {
	if _, err := store.CreateStageExecution(ctx, runstore.CreateStageExecutionParams{
		StageExecutionID: stageID, RunID: runID, StageName: "test", Attempt: 1,
		StageSpecSchemaVersion: schemaVersion, StageSpecSnapshot: json.RawMessage(`{"objective":"test"}`),
		StageContextSchemaVersion: schemaVersion,
		StageContext:              runstore.StageContextSnapshot{Parameters: map[string]string{}, Artifacts: map[string]runstore.PinnedContextArtifact{}},
	}); err != nil {
		return err
	}
	data, err := json.Marshal(map[string]string{
		"stageExecutionId": stageID, "sessionId": stageID + "-session", "invocationId": stageID + "-invocation",
	})
	if err != nil {
		return err
	}
	return store.StartPlanner(ctx, runstore.StartPlannerParams{
		StageExecutionID: stageID, SessionID: stageID + "-session", InvocationID: stageID + "-invocation",
		StateSchemaVersion: schemaVersion, InitialState: json.RawMessage(`{"status":"test"}`),
		RunEvent: runstore.RunEventAppend{EventID: stageID + "-started", EventSchemaVersion: schemaVersion, Kind: runstore.RunEventPlannerStarted, Data: data},
		Reason:   runstore.Reason{Code: "planner_started"},
	})
}
