package runstore

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strconv"
	"strings"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/jackc/pgx/v5"
)

const (
	maxRunEventDataBytes               = 2 * 1024 * 1024
	maxRunEventIdentityBytes           = 512
	maxRunEventSubtasks                = 32
	maxRunEventSubtaskObjectiveBytes   = 8 * 1024
	maxRunEventSubtaskInstructionBytes = 32 * 1024
	maxRunEventWorkerNameBytes         = 128
	maxRunEventFunctions               = 32
)

var validRunEventKinds = map[RunEventKind]bool{
	RunEventPlannerStarted:           true,
	RunEventPlannerRequestRecorded:   true,
	RunEventPlannerActivity:          true,
	RunEventPlannerPlanChanged:       true,
	RunEventPlannerCurrentChanged:    true,
	RunEventPlannerDispatchSelected:  true,
	RunEventPlannerDispatchStarted:   true,
	RunEventPlannerDispatchCompleted: true,
	RunEventPlannerFinishRequested:   true,
	RunEventPlannerCompleted:         true,
	RunEventPlannerFailed:            true,
}

func validateRunEventAppend(value RunEventAppend) error {
	for field, current := range map[string]string{
		"Run event ID": value.EventID, "Run event schema version": value.EventSchemaVersion,
	} {
		if err := validateOpaque(field, current); err != nil {
			return err
		}
	}
	if !validRunEventKinds[value.Kind] {
		return invalidf("unknown WorkflowRun event kind %q", value.Kind)
	}
	if value.EventSchemaVersion != contracts.APIVersion {
		return invalidf("unsupported WorkflowRun event schema version %q", value.EventSchemaVersion)
	}
	if len(value.Data) > maxRunEventDataBytes {
		return invalidf("WorkflowRun event data exceeds its bounded contract")
	}
	data, err := decodePlannerRunEventData(value.Data)
	if err != nil {
		return err
	}
	return validatePlannerRunEventData(value.Kind, data)
}

func validatePlannerRunEventIdentity(
	value RunEventAppend,
	stageExecutionID string,
	sessionID string,
	invocationID string,
) error {
	data, err := decodePlannerRunEventData(value.Data)
	if err != nil {
		return err
	}
	if data.StageExecutionID != stageExecutionID || data.SessionID != sessionID ||
		data.InvocationID != invocationID {
		return invalidf("WorkflowRun Planner event identity differs from its durable session")
	}
	return nil
}

type plannerRunEventData struct {
	StageExecutionID string                    `json:"stageExecutionId"`
	SessionID        string                    `json:"sessionId"`
	InvocationID     string                    `json:"invocationId"`
	Plan             *plannerRunPlanProjection `json:"plan,omitempty"`
	PlanRevision     *uint64                   `json:"planRevision,omitempty"`
	SubtaskID        string                    `json:"subtaskId,omitempty"`
	CallID           string                    `json:"callId,omitempty"`
	WorkerName       string                    `json:"workerName,omitempty"`
	Outcome          string                    `json:"outcome,omitempty"`
	Code             string                    `json:"code,omitempty"`
	Activity         *plannerRunActivity       `json:"activity,omitempty"`
}

type plannerRunPlanProjection struct {
	Revision         uint64                    `json:"revision"`
	Subtasks         []plannerRunSubtask       `json:"subtasks"`
	CurrentSubtaskID string                    `json:"currentSubtaskId,omitempty"`
	ActiveDispatch   *plannerRunActiveDispatch `json:"activeDispatch,omitempty"`
}

type plannerRunSubtask struct {
	ID           string `json:"id"`
	Objective    string `json:"objective"`
	Instructions string `json:"instructions"`
	Status       string `json:"status"`
}

type plannerRunActiveDispatch struct {
	CallID     string `json:"callId"`
	SubtaskID  string `json:"subtaskId"`
	WorkerName string `json:"workerName"`
}

type plannerRunActivity struct {
	Kind              string   `json:"kind"`
	Author            string   `json:"author"`
	FunctionCalls     []string `json:"functionCalls"`
	FunctionResults   []string `json:"functionResults"`
	InputTokens       int64    `json:"inputTokens,omitempty"`
	OutputTokens      int64    `json:"outputTokens,omitempty"`
	SkipSummarization bool     `json:"skipSummarization,omitempty"`
	Escalate          bool     `json:"escalate,omitempty"`
	Truncated         bool     `json:"truncated,omitempty"`
}

func decodePlannerRunEventData(data json.RawMessage) (plannerRunEventData, error) {
	if len(data) == 0 || len(data) > maxRunEventDataBytes {
		return plannerRunEventData{}, invalidf("WorkflowRun event data exceeds its bounded contract")
	}
	var result plannerRunEventData
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&result); err != nil {
		return plannerRunEventData{}, invalidf("WorkflowRun event data has an invalid closed schema: %v", err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		return plannerRunEventData{}, invalidf("WorkflowRun event data contains trailing JSON")
	}
	return result, nil
}

func validatePlannerRunEventData(kind RunEventKind, data plannerRunEventData) error {
	for name, value := range map[string]string{
		"stageExecutionId": data.StageExecutionID,
		"sessionId":        data.SessionID,
		"invocationId":     data.InvocationID,
	} {
		if strings.TrimSpace(value) == "" || len(value) > maxRunEventIdentityBytes {
			return invalidf("WorkflowRun Planner event %s is invalid", name)
		}
	}
	noDetails := func() bool {
		return data.Plan == nil && data.PlanRevision == nil && data.SubtaskID == "" &&
			data.CallID == "" && data.WorkerName == "" && data.Outcome == "" &&
			data.Code == "" && data.Activity == nil
	}
	noPlanOrActivity := func() bool { return data.Plan == nil && data.Activity == nil }
	switch kind {
	case RunEventPlannerStarted, RunEventPlannerRequestRecorded:
		if !noDetails() {
			return invalidf("WorkflowRun %s event contains unexpected fields", kind)
		}
	case RunEventPlannerActivity:
		if data.Activity == nil || data.Plan != nil || data.PlanRevision != nil ||
			data.SubtaskID != "" || data.CallID != "" || data.WorkerName != "" ||
			data.Outcome != "" || data.Code != "" {
			return invalidf("WorkflowRun Planner activity event has invalid fields")
		}
		if err := validatePlannerRunActivity(*data.Activity); err != nil {
			return err
		}
	case RunEventPlannerPlanChanged:
		if data.Plan == nil || data.PlanRevision != nil || data.SubtaskID != "" ||
			data.CallID != "" || data.WorkerName != "" || data.Outcome != "" ||
			data.Code != "" || data.Activity != nil {
			return invalidf("WorkflowRun Planner plan-changed event has invalid fields")
		}
		if err := validatePlannerRunPlan(*data.Plan); err != nil {
			return err
		}
		if data.Plan.ActiveDispatch != nil {
			return invalidf("WorkflowRun Planner plan-changed event cannot contain an active dispatch")
		}
	case RunEventPlannerCurrentChanged:
		if !noPlanOrActivity() || data.PlanRevision == nil || *data.PlanRevision == 0 ||
			!validRunSubtaskID(data.SubtaskID, true) || data.CallID != "" ||
			data.WorkerName != "" || data.Outcome != "" || data.Code != "" {
			return invalidf("WorkflowRun Planner current-changed event has invalid fields")
		}
	case RunEventPlannerDispatchSelected:
		if !noPlanOrActivity() || data.PlanRevision == nil || *data.PlanRevision == 0 ||
			!validRunSubtaskID(data.SubtaskID, false) || !validRunDispatchCallID(data.CallID) ||
			!validRunWorkerName(data.WorkerName) || data.Outcome != "" || data.Code != "" {
			return invalidf("WorkflowRun Planner dispatch-selected event has invalid fields")
		}
	case RunEventPlannerDispatchStarted:
		if data.Plan == nil || data.PlanRevision != nil || data.Activity != nil ||
			!validRunSubtaskID(data.SubtaskID, false) || !validRunDispatchCallID(data.CallID) ||
			!validRunWorkerName(data.WorkerName) || data.Outcome != "" || data.Code != "" {
			return invalidf("WorkflowRun Planner dispatch-started event has invalid fields")
		}
		if err := validatePlannerRunPlan(*data.Plan); err != nil {
			return err
		}
		active := data.Plan.ActiveDispatch
		if active == nil || active.SubtaskID != data.SubtaskID || active.CallID != data.CallID ||
			active.WorkerName != data.WorkerName {
			return invalidf("WorkflowRun Planner dispatch-started details differ from its plan")
		}
	case RunEventPlannerDispatchCompleted:
		if data.Plan == nil || data.PlanRevision != nil || data.Activity != nil ||
			!validRunSubtaskID(data.SubtaskID, false) || !validRunDispatchCallID(data.CallID) ||
			!validRunWorkerName(data.WorkerName) ||
			(data.Outcome != "succeeded" && data.Outcome != "failed") || data.Code != "" {
			return invalidf("WorkflowRun Planner dispatch-completed event has invalid fields")
		}
		if err := validatePlannerRunPlan(*data.Plan); err != nil {
			return err
		}
		if data.Plan.ActiveDispatch != nil || !plannerRunSubtaskHasStatus(*data.Plan, data.SubtaskID, data.Outcome) {
			return invalidf("WorkflowRun Planner dispatch-completed details differ from its plan")
		}
	case RunEventPlannerFinishRequested:
		if !noPlanOrActivity() || data.PlanRevision == nil || *data.PlanRevision == 0 || data.SubtaskID != "" ||
			data.CallID != "" || data.WorkerName != "" || data.Code != "" ||
			(data.Outcome != string(contracts.StageSucceeded) && data.Outcome != string(contracts.StageFailed)) {
			return invalidf("WorkflowRun Planner finish-requested event has invalid fields")
		}
	case RunEventPlannerCompleted:
		if data.Plan != nil || data.PlanRevision != nil || data.SubtaskID != "" ||
			data.CallID != "" || data.WorkerName != "" || data.Code != "" || data.Activity != nil ||
			(data.Outcome != string(contracts.StageSucceeded) && data.Outcome != string(contracts.StageFailed)) {
			return invalidf("WorkflowRun Planner completed event has invalid fields")
		}
	case RunEventPlannerFailed:
		if data.Plan != nil || data.PlanRevision != nil || data.SubtaskID != "" ||
			data.CallID != "" || data.WorkerName != "" || data.Activity != nil ||
			data.Outcome != "failure" || !validRunStableName(data.Code, 128) {
			return invalidf("WorkflowRun Planner failed event has invalid fields")
		}
	default:
		return invalidf("unknown WorkflowRun event kind %q", kind)
	}
	return nil
}

func validatePlannerRunPlan(plan plannerRunPlanProjection) error {
	if plan.Revision == 0 || len(plan.Subtasks) == 0 || len(plan.Subtasks) > maxRunEventSubtasks {
		return invalidf("WorkflowRun Planner plan revision or subtask count is invalid")
	}
	firstUnresolved := ""
	running := 0
	for index, subtask := range plan.Subtasks {
		if subtask.ID != strconv.Itoa(index) || strings.TrimSpace(subtask.Objective) == "" ||
			len(subtask.Objective) > maxRunEventSubtaskObjectiveBytes ||
			strings.TrimSpace(subtask.Instructions) == "" ||
			len(subtask.Instructions) > maxRunEventSubtaskInstructionBytes {
			return invalidf("WorkflowRun Planner subtask is invalid")
		}
		switch subtask.Status {
		case "pending", "succeeded", "failed":
		case "running":
			running++
		default:
			return invalidf("WorkflowRun Planner subtask status is invalid")
		}
		if firstUnresolved == "" && (subtask.Status == "pending" || subtask.Status == "running") {
			firstUnresolved = subtask.ID
		}
	}
	if plan.CurrentSubtaskID != firstUnresolved {
		return invalidf("WorkflowRun Planner current subtask is invalid")
	}
	if plan.ActiveDispatch == nil {
		if running != 0 {
			return invalidf("WorkflowRun Planner running subtask has no active dispatch")
		}
		return nil
	}
	active := plan.ActiveDispatch
	if running != 1 || active.SubtaskID != plan.CurrentSubtaskID ||
		!validRunDispatchCallID(active.CallID) || !validRunWorkerName(active.WorkerName) ||
		!plannerRunSubtaskHasStatus(plan, active.SubtaskID, "running") {
		return invalidf("WorkflowRun Planner active dispatch is invalid")
	}
	return nil
}

func validatePlannerRunActivity(activity plannerRunActivity) error {
	if activity.Kind != "adk_event" || activity.InputTokens < 0 || activity.OutputTokens < 0 {
		return invalidf("WorkflowRun Planner activity counters are invalid")
	}
	switch activity.Author {
	case "user", "streamline_planner", "router_planner", "other":
	default:
		return invalidf("WorkflowRun Planner activity author is invalid")
	}
	for _, values := range [][]string{activity.FunctionCalls, activity.FunctionResults} {
		if len(values) > maxRunEventFunctions {
			return invalidf("WorkflowRun Planner activity function list is too large")
		}
		for _, value := range values {
			if !validRunStableName(value, 128) {
				return invalidf("WorkflowRun Planner activity function name is invalid")
			}
		}
	}
	return nil
}

func validRunSubtaskID(value string, allowEmpty bool) bool {
	if value == "" {
		return allowEmpty
	}
	if len(value) > 2 {
		return false
	}
	for _, current := range value {
		if current < '0' || current > '9' {
			return false
		}
	}
	return true
}

func validRunDispatchCallID(value string) bool {
	if !strings.HasPrefix(value, "dispatch-") || len(value) > 64 {
		return false
	}
	digits := strings.TrimPrefix(value, "dispatch-")
	if digits == "" {
		return false
	}
	for _, current := range digits {
		if current < '0' || current > '9' {
			return false
		}
	}
	return true
}

func validRunWorkerName(value string) bool {
	return strings.TrimSpace(value) != "" && len(value) <= maxRunEventWorkerNameBytes
}

func validRunStableName(value string, limit int) bool {
	if strings.TrimSpace(value) == "" || len(value) > limit {
		return false
	}
	for _, current := range value {
		if current >= 'a' && current <= 'z' || current >= 'A' && current <= 'Z' ||
			current >= '0' && current <= '9' || strings.ContainsRune("._:-", current) {
			continue
		}
		return false
	}
	return true
}

func plannerRunSubtaskHasStatus(plan plannerRunPlanProjection, id, status string) bool {
	for _, subtask := range plan.Subtasks {
		if subtask.ID == id {
			return subtask.Status == status
		}
	}
	return false
}

func (s *PostgresStore) GetRunEventCursor(
	ctx context.Context, runID string,
) (WorkflowRunEventCursor, error) {
	if err := validateOpaque("runID", runID); err != nil {
		return WorkflowRunEventCursor{}, err
	}
	var result WorkflowRunEventCursor
	err := s.db.QueryRow(ctx, `
SELECT run_event_generation, next_run_event_sequence - 1
FROM workflow_runs
WHERE run_id = $1`, runID).Scan(&result.Generation, &result.Sequence)
	if errors.Is(err, pgx.ErrNoRows) {
		return WorkflowRunEventCursor{}, fmt.Errorf("get WorkflowRun event cursor %q: %w", runID, ErrNotFound)
	}
	if err != nil {
		return WorkflowRunEventCursor{}, fmt.Errorf("get WorkflowRun event cursor %q: %w", runID, err)
	}
	return result, nil
}

func (s *PostgresStore) ListRunEvents(
	ctx context.Context, runID string, afterSequence int64, limit int,
) ([]WorkflowRunEvent, error) {
	if err := validateOpaque("runID", runID); err != nil {
		return nil, err
	}
	if afterSequence < 0 || limit < 1 || limit > 1000 {
		return nil, invalidf("WorkflowRun event cursor or limit is invalid")
	}
	rows, err := s.db.Query(ctx, `
SELECT run_id, sequence_number, event_id, event_schema_version, kind, data, occurred_at
FROM workflow_run_events
WHERE run_id = $1 AND sequence_number > $2
ORDER BY sequence_number
LIMIT $3`, runID, afterSequence, limit)
	if err != nil {
		return nil, fmt.Errorf("list WorkflowRun events %q: %w", runID, err)
	}
	defer rows.Close()
	result := make([]WorkflowRunEvent, 0)
	for rows.Next() {
		var event WorkflowRunEvent
		var data []byte
		if err := rows.Scan(
			&event.RunID, &event.SequenceNumber, &event.EventID,
			&event.EventSchemaVersion, &event.Kind, &data, &event.OccurredAt,
		); err != nil {
			return nil, fmt.Errorf("scan WorkflowRun event %q: %w", runID, err)
		}
		event.Data = append(json.RawMessage(nil), data...)
		if event.SequenceNumber <= afterSequence || !validRunEventKinds[event.Kind] ||
			validateRunEventAppend(RunEventAppend{
				EventID: event.EventID, EventSchemaVersion: event.EventSchemaVersion,
				Kind: event.Kind, Data: event.Data,
			}) != nil {
			return nil, fmt.Errorf("persisted WorkflowRun event is invalid")
		}
		result = append(result, event)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate WorkflowRun events %q: %w", runID, err)
	}
	return result, nil
}
