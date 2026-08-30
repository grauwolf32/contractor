// Package public implements the authenticated single-user HTTP API. Scope IDs
// are derived from authentication and route-owned Run records, never accepted
// as arbitrary request fields.
package public

import (
	"context"
	"errors"
	"log/slog"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

type RunReader interface {
	GetRun(context.Context, string) (runstore.WorkflowRun, error)
	ListStageExecutions(context.Context, string) ([]runstore.StageExecution, error)
	ListStageTransitionDecisions(context.Context, string) ([]runstore.StageTransitionDecision, error)
	RequestRunCancellation(context.Context, string, runstore.WorkflowRunCancellation) (runstore.WorkflowRun, error)
}

type RunWriter interface {
	CreateRun(context.Context, runstore.CreateRunParams) (runstore.WorkflowRun, error)
	CreateRunIdempotent(
		context.Context,
		runstore.CreateRunIdempotentParams,
	) (runstore.WorkflowRun, bool, error)
	TransitionRun(
		context.Context,
		string,
		runstore.WorkflowRunState,
		runstore.WorkflowRunState,
		runstore.Reason,
	) (runstore.WorkflowRun, error)
}

type MetricsReader interface {
	GetStageMetrics(context.Context, string) (telemetry.StageMetricsRecord, error)
}

// UnitOfWork supplies transaction-bound Run and Artifact stores. The callback
// commits only when it returns nil.
type UnitOfWork interface {
	Do(context.Context, func(RunWriter, *artifacts.Service) error) error
}

type RunNotifier interface {
	Wake()
}

type RunCancellationNotifier interface {
	Cancel(string)
}

type Dependencies struct {
	Config       *config.Snapshot
	Credentials  config.CredentialLookup
	Runs         RunReader
	Metrics      MetricsReader
	Artifacts    *artifacts.Service
	Transactions UnitOfWork
	BearerToken  contracts.SecretString
	UserID       string
	NewID        func(string) (string, error)
	NewRequestID func() (string, error)
	RunNotifier  RunNotifier
	Now          func() time.Time
	Logger       *slog.Logger
}

var errInvalidRequest = errors.New("invalid public API request")

type createRunRequest struct {
	Workflow        string                           `json:"workflow"`
	Parameters      map[string]string                `json:"parameters"`
	Artifacts       map[string]contracts.ArtifactRef `json:"artifacts"`
	ExecutionConfig config.ExecutionConfigPatch      `json:"executionConfig"`
}

type createRunResponse struct {
	RunID string                    `json:"runId"`
	State runstore.WorkflowRunState `json:"state"`
}

type cancelRunRequest struct {
	Reason *string `json:"reason"`
}

type cancelRunResponse struct {
	RunID        string                            `json:"runId"`
	State        runstore.WorkflowRunState         `json:"state"`
	Cancellation *runstore.WorkflowRunCancellation `json:"cancellation,omitempty"`
}

type artifactWriteResponse struct {
	Artifact  contracts.ArtifactRef `json:"artifact"`
	MediaType string                `json:"mediaType"`
	Size      int64                 `json:"size"`
}

type runStatusResponse struct {
	RunID        string                            `json:"runId"`
	Workflow     string                            `json:"workflow"`
	State        runstore.WorkflowRunState         `json:"state"`
	Cancellation *runstore.WorkflowRunCancellation `json:"cancellation,omitempty"`
	Attempts     []stageAttemptResponse            `json:"attempts"`
	Transitions  []stageTransitionResponse         `json:"transitions"`
	Outputs      map[string]contracts.ArtifactRef  `json:"outputs"`
}

type stageAttemptResponse struct {
	StageExecutionID    string                        `json:"stageExecutionId"`
	Stage               string                        `json:"stage"`
	Attempt             int                           `json:"attempt"`
	PreviousExecutionID *string                       `json:"previousExecutionId,omitempty"`
	ExecutionConfig     stageExecutionConfigResponse  `json:"executionConfig"`
	State               runstore.StageExecutionState  `json:"state"`
	Result              *contracts.StageContentResult `json:"result,omitempty"`
	Termination         *runstore.StageTermination    `json:"termination,omitempty"`
	Metrics             *telemetry.Summary            `json:"metrics,omitempty"`
}

type stageExecutionConfigResponse struct {
	Variant           runstore.StageExecutionConfigVariant           `json:"variant"`
	EscalationOrdinal *int                                           `json:"escalationOrdinal,omitempty"`
	Ref               *config.ExecutionConfigRef                     `json:"ref,omitempty"`
	Planner           *consumerExecutionConfigRefsResponse           `json:"planner,omitempty"`
	Agents            map[string]consumerExecutionConfigRefsResponse `json:"agents"`
}

type consumerExecutionConfigRefsResponse struct {
	ModelPolicy contracts.ModelPolicyRef      `json:"modelPolicy"`
	LLMGateway  contracts.LLMGatewayConfigRef `json:"llmGateway"`
	Credential  *contracts.LLMCredentialRef   `json:"credential,omitempty"`
}

type stageTransitionResponse struct {
	SourceExecutionID   string                         `json:"sourceExecutionId"`
	Action              runstore.StageTransitionAction `json:"action"`
	TargetStage         *string                        `json:"targetStage,omitempty"`
	TargetExecutionID   *string                        `json:"targetExecutionId,omitempty"`
	EscalationOrdinal   *int                           `json:"escalationOrdinal,omitempty"`
	EscalationExhausted bool                           `json:"escalationExhausted"`
	DecidedAt           time.Time                      `json:"decidedAt"`
}

type errorResponse struct {
	Code      string `json:"code"`
	Message   string `json:"message"`
	Retryable bool   `json:"retryable"`
	RequestID string `json:"requestId"`
}
