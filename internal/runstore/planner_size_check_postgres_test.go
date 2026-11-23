package runstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

// expandingState is compact JSON within the submitted 2 MiB pre-check whose
// jsonb text, which adds a space after every comma, exceeds the stored bound
// planner_session_state_size of the same size.
func expandingState(t *testing.T) json.RawMessage {
	t.Helper()
	state := json.RawMessage(`{"a":[` + strings.TrimSuffix(strings.Repeat("0,", 1_000_000), ",") + `]}`)
	if len(state) > maxRunEventDataBytes || len(state)+999_999 <= maxRunEventDataBytes {
		t.Fatalf("state of %d bytes does not straddle the bound", len(state))
	}
	return state
}

// A stored size CHECK that rejects a Planner session write past the
// submitted-JSON pre-check, on the session state or on the Run event of the
// same statement, returns ErrInvalid like the pre-check, keeps the
// PostgreSQL cause and writes nothing (PER-R13).
func TestPostgresPlannerSizeCheckRejectionIsInvalid(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	store := NewPostgresStore(pool, artifactpgtest.NewStorage())
	const stageID = "stage-size"
	createPreparingStage(t, ctx, store, "run-size", stageID)

	before := readStartPlannerEffects(t, ctx, pool, stageID)
	params := testStartPlannerParams(t, stageID, "session-size")
	params.InitialState = expandingState(t)
	err := store.StartPlanner(ctx, params)
	if !errors.Is(err, ErrInvalid) || persistencepostgres.Classify(err) != persistencepostgres.ClassInvalid ||
		persistencepostgres.ConstraintName(err) != "planner_session_state_size" {
		t.Fatalf("oversized StartPlanner = %v, want ErrInvalid wrapping planner_session_state_size", err)
	}
	if after := readStartPlannerEffects(t, ctx, pool, stageID); after != before {
		t.Fatalf("effects after a rejected start = %+v, want %+v", after, before)
	}

	params = testStartPlannerParams(t, stageID, "session-size")
	if err := store.StartPlanner(ctx, params); err != nil {
		t.Fatal(err)
	}
	started := readStartPlannerEffects(t, ctx, pool, stageID)
	identity := plannerRunEventData{
		StageExecutionID: stageID, SessionID: params.SessionID, InvocationID: params.InvocationID,
	}
	err = store.AppendPlannerEvent(ctx, AppendPlannerEventParams{
		SessionID: params.SessionID, StageExecutionID: stageID, InvocationID: params.InvocationID,
		SequenceNumber: 2, NewStateSchemaVersion: contracts.APIVersion, NewState: expandingState(t),
		RunEvent: RunEventAppend{
			EventID: "session-size-request", EventSchemaVersion: contracts.APIVersion,
			Kind: RunEventPlannerRequestRecorded, Data: mustRunEventJSON(t, identity),
		},
	})
	if !errors.Is(err, ErrInvalid) || persistencepostgres.Classify(err) != persistencepostgres.ClassInvalid ||
		persistencepostgres.ConstraintName(err) != "planner_session_state_size" {
		t.Fatalf("oversized AppendPlannerEvent = %v, want ErrInvalid wrapping planner_session_state_size", err)
	}
	if after := readStartPlannerEffects(t, ctx, pool, stageID); after != started {
		t.Fatalf("effects after a rejected append = %+v, want %+v", after, started)
	}
	session, err := store.GetPlannerSession(ctx, params.SessionID)
	if err != nil || session.NextEventSequence != 2 {
		t.Fatalf("Planner session after a rejected append = (%+v, %v)", session.NextEventSequence, err)
	}

	// The Run event of the same append meets workflow_run_events_data_check,
	// with its production measure and a bound at the compact data's length,
	// which the stored jsonb text outgrows.
	data := mustRunEventJSON(t, identity)
	if _, err := pool.Exec(ctx, fmt.Sprintf(`
ALTER TABLE workflow_run_events DROP CONSTRAINT workflow_run_events_data_check,
ADD CONSTRAINT workflow_run_events_data_check
    CHECK (jsonb_typeof(data) = 'object' AND octet_length(data::text) <= %d) NOT VALID`, len(data))); err != nil {
		t.Fatal(err)
	}
	err = store.AppendPlannerEvent(ctx, AppendPlannerEventParams{
		SessionID: params.SessionID, StageExecutionID: stageID, InvocationID: params.InvocationID,
		SequenceNumber: 2, NewStateSchemaVersion: contracts.APIVersion, NewState: json.RawMessage(`{"step":1}`),
		RunEvent: RunEventAppend{
			EventID: "session-size-event", EventSchemaVersion: contracts.APIVersion,
			Kind: RunEventPlannerRequestRecorded, Data: data,
		},
	})
	if !errors.Is(err, ErrInvalid) || persistencepostgres.Classify(err) != persistencepostgres.ClassInvalid ||
		persistencepostgres.ConstraintName(err) != "workflow_run_events_data_check" {
		t.Fatalf("append of a widened Run event = %v, want ErrInvalid wrapping workflow_run_events_data_check", err)
	}
	if after := readStartPlannerEffects(t, ctx, pool, stageID); after != started {
		t.Fatalf("effects after a rejected Run event = %+v, want %+v", after, started)
	}
	session, err = store.GetPlannerSession(ctx, params.SessionID)
	if err != nil || session.NextEventSequence != 2 || string(session.State) != `{"step": 0}` {
		t.Fatalf("Planner session after a rejected Run event = (%d %s, %v)", session.NextEventSequence, session.State, err)
	}
}
