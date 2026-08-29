// Package public implements the authenticated single-user HTTP API. Scope IDs
// are derived from authentication and route-owned Run records, never accepted
// as arbitrary request fields.
package public

import (
	"context"
	"errors"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
)

type RunReader interface {
	GetRun(context.Context, string) (runstore.WorkflowRun, error)
	ListStageExecutions(context.Context, string) ([]runstore.StageExecution, error)
}

type RunWriter interface {
	CreateRun(context.Context, runstore.CreateRunParams) (runstore.WorkflowRun, error)
	TransitionRun(
		context.Context,
		string,
		runstore.WorkflowRunState,
		runstore.WorkflowRunState,
		runstore.Reason,
	) (runstore.WorkflowRun, error)
}

// UnitOfWork supplies transaction-bound Run and Artifact stores. The callback
// commits only when it returns nil.
type UnitOfWork interface {
	Do(context.Context, func(RunWriter, *artifacts.Service) error) error
}

type Dependencies struct {
	Config       *config.Snapshot
	Runs         RunReader
	Artifacts    *artifacts.Service
	Transactions UnitOfWork
	BearerToken  contracts.SecretString
	UserID       string
	NewID        func(string) (string, error)
	NewRequestID func() (string, error)
}

var errInvalidRequest = errors.New("invalid public API request")

type createRunRequest struct {
	Workflow   string                           `json:"workflow"`
	Parameters map[string]string                `json:"parameters"`
	Artifacts  map[string]contracts.ArtifactRef `json:"artifacts"`
}

type createRunResponse struct {
	RunID string                    `json:"runId"`
	State runstore.WorkflowRunState `json:"state"`
}

type artifactWriteResponse struct {
	Artifact  contracts.ArtifactRef `json:"artifact"`
	MediaType string                `json:"mediaType"`
	Size      int64                 `json:"size"`
}

type runStatusResponse struct {
	RunID    string                           `json:"runId"`
	Workflow string                           `json:"workflow"`
	State    runstore.WorkflowRunState        `json:"state"`
	Attempts []stageAttemptResponse           `json:"attempts"`
	Outputs  map[string]contracts.ArtifactRef `json:"outputs"`
}

type stageAttemptResponse struct {
	StageExecutionID string                        `json:"stageExecutionId"`
	Stage            string                        `json:"stage"`
	Attempt          int                           `json:"attempt"`
	State            runstore.StageExecutionState  `json:"state"`
	Result           *contracts.StageContentResult `json:"result,omitempty"`
	Termination      *runstore.StageTermination    `json:"termination,omitempty"`
}

type errorResponse struct {
	Code      string `json:"code"`
	Message   string `json:"message"`
	Retryable bool   `json:"retryable"`
}
