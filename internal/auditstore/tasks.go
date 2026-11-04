package auditstore

import (
	"encoding/json"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
)

type ActivateParams struct {
	OwnerID          string
	AuditID          string
	ExpectedRevision uint64
	BaselineSnapshot json.RawMessage
	DeadlineAt       time.Time
	InitialRetained  []ArtifactLink
	IdempotencyKey   string
	RequestDigest    string
}

// TaskState records selection and collection progress. Technical execution
// state is read from the ordinary Run associated with an attempt.
type TaskState string

const (
	TaskReady    TaskState = "ready"
	TaskSelected TaskState = "selected"
	TaskSettled  TaskState = "settled"
)

// Task retains one immutable accepted document and its source identity. The
// source key reconciles the same producer/list/entry without creating work.
type Task struct {
	Attention     bool
	Hypothesis    *auditcontract.Hypothesis
	Document      auditcontract.Task
	SourceKey     string
	Source        TaskSource
	Ordinal       int
	Iteration     int
	BatchDigest   string
	State         TaskState
	Attempts      int
	LastExecution *string
	Result        *auditcontract.Material
	GapCode       *string
	CreatedAt     time.Time
	UpdatedAt     time.Time
}

type TaskPage struct {
	Tasks       []Task
	NextOrdinal *int
}

type ListTasksParams struct {
	HypothesisID string
	OwnerID      string
	AuditID      string
	State        *TaskState
	AfterOrdinal int
	Limit        int
}

// AcceptedTask is the immutable work admitted by one common mutation, after
// its exact selected inputs have been retained under the normal Audit scope.
type AcceptedTask struct {
	Attention  bool                      `json:"attention"`
	Hypothesis *auditcontract.Hypothesis `json:"hypothesis_snapshot,omitempty"`
	Document   auditcontract.Task        `json:"document"`
	Source     TaskSource                `json:"source"`
}

// TaskSource distinguishes explicit work from one exact accepted producer
// output. Its producer-local key has no authority outside this source.
type TaskSource struct {
	ThroughOrdinal int                     `json:"through_ordinal,omitempty"`
	Kind           string                  `json:"kind"`
	Key            string                  `json:"key"`
	ExecutionID    string                  `json:"execution_id,omitempty"`
	Output         *auditcontract.Material `json:"output,omitempty"`
}

type AcceptTasksParams struct {
	ContinuationRevision uint64
	HypothesisWrites     []HypothesisWrite
	AssertionSnapshots   map[string]auditcontract.Hypothesis
	Attention            []string
	Hypotheses           map[string]string
	OwnerID              string
	AuditID              string
	ExpectedRevision     uint64
	Claim                *ControllerClaim
	IdempotencyKey       string
	RequestDigest        string
	Tasks                []AcceptedTask
	InitialProgress      *InitialWorkProgress
	PreparedProgress     *PreparedWorkProgress
}

type TaskAcceptance struct {
	TaskIDs       []string `json:"task_ids"`
	CreatedCount  int      `json:"created_count"`
	AuditRevision uint64   `json:"audit_revision"`
}

type InitialWorkProgress struct {
	From int
	To   int
}

type PreparedWorkProgress struct {
	ExecutionID string
	From        int
	To          int
}

// SelectTasksParams reserves one ordinary Run over an ordered compatible
// batch. The owner derives attempts and submission identity from accepted work.
type SelectTasksParams struct {
	Claim   ControllerClaim
	TaskIDs []string
}

// TaskCollection records the retained ordinary output or a truthful gap of
// one attempt. Run technical outcome remains on its execution/Run association.
type TaskCollection struct {
	ExecutionItemID string                  `json:"execution_item_id"`
	TaskID          string                  `json:"task_id"`
	Result          *auditcontract.Material `json:"task_result,omitempty"`
	GapCode         *string                 `json:"gap_code,omitempty"`
	Retryable       bool                    `json:"retryable"`
}

type PreparedSourceParams struct {
	Claim       ControllerClaim
	ExecutionID string
	Source      auditcontract.Material
	Count       int
}

type PreparedSourceCursor struct {
	Offset    int
	Count     int
	Accepting bool
}

type TaskIntakeResult struct {
	Complete bool
	Changed  bool
	GapCode  string
}

func RemainingTaskCapacity(audit Audit) int {
	return max(0, min(audit.Limits.MaxItemsTotal-audit.TaskCount,
		(audit.Limits.MaxRounds-audit.TaskIteration)*audit.Limits.MaxItemsPerRound+audit.Limits.MaxItemsPerRound-audit.IterationTaskCount))
}

// ContinuationSource is the bounded read projection of one settled Task.
// Payloads remain exact ordinary Artifacts; this is not another result store.
type ContinuationSource struct {
	TaskID     string
	Ordinal    int
	Hypothesis *auditcontract.Hypothesis
	Result     *auditcontract.Material
	GapCode    *string
}
