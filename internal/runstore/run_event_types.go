package runstore

import (
	"encoding/json"
	"time"
)

type RunEventKind string

const (
	RunEventPlannerStarted           RunEventKind = "planner.started"
	RunEventPlannerRequestRecorded   RunEventKind = "planner.request_recorded"
	RunEventPlannerActivity          RunEventKind = "planner.activity"
	RunEventPlannerPlanChanged       RunEventKind = "planner.plan_changed"
	RunEventPlannerCurrentChanged    RunEventKind = "planner.current_changed"
	RunEventPlannerDispatchSelected  RunEventKind = "planner.dispatch_selected"
	RunEventPlannerDispatchStarted   RunEventKind = "planner.dispatch_started"
	RunEventPlannerDispatchCompleted RunEventKind = "planner.dispatch_completed"
	RunEventPlannerFinishRequested   RunEventKind = "planner.finish_requested"
	RunEventPlannerCompleted         RunEventKind = "planner.completed"
	RunEventPlannerFailed            RunEventKind = "planner.failed"
	RunEventLifecycleChanged         RunEventKind = "lifecycle.changed"
)

// RunEventAppend is the validated event half of a Planner/session mutation.
// The store allocates its per-Run sequence in the same SQL statement.
type RunEventAppend struct {
	EventID            string
	EventSchemaVersion string
	Kind               RunEventKind
	Data               json.RawMessage
}

type WorkflowRunEvent struct {
	RunID              string
	SequenceNumber     int64
	EventID            string
	EventSchemaVersion string
	Kind               RunEventKind
	Data               json.RawMessage
	OccurredAt         time.Time
}

// WorkflowRunEventPage is one bounded read of a Run's committed events after
// a cursor (API-R110): the contiguous sequence prefix within the requested
// event count and the page data budget.
type WorkflowRunEventPage struct {
	Events []WorkflowRunEvent
	// More reports that a committed event follows the last returned one. A
	// page with More holds at least one event, so the next read continues
	// after the last returned sequence and makes progress. Without More the
	// read reached the end of the committed history.
	More bool
}

type WorkflowRunEventCursor struct {
	Generation string
	Sequence   int64
}
