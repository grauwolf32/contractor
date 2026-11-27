package controlplane

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"regexp"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/mtls"
)

var agentStateETagPattern = regexp.MustCompile(`^"contractor-agent-state-v1-[1-9][0-9]*"$`)

func (c *RuntimeControlClient) ReadWorkerState(
	ctx context.Context,
	handle contracts.WorkerHandle,
	ifNoneMatch string,
) (reporting.WorkerStateReadResult, error) {
	if ifNoneMatch != "" && !agentStateETagPattern.MatchString(ifNoneMatch) {
		return reporting.WorkerStateReadResult{}, workerStateReadError(0, "worker_state_etag_invalid", false)
	}
	target, err := c.workerStateEndpoint(handle)
	if err != nil {
		return reporting.WorkerStateReadResult{}, workerStateReadError(0, "worker_state_endpoint_invalid", false)
	}
	return c.readWorkerState(ctx, target, handle.RuntimeAgentID, ifNoneMatch)
}

func (c *RuntimeControlClient) readWorkerState(ctx context.Context, target, runtimeAgentID, ifNoneMatch string) (reporting.WorkerStateReadResult, error) {
	headers := map[string]string{}
	if ifNoneMatch != "" {
		headers["If-None-Match"] = ifNoneMatch
	}
	response, err := c.doRequest(
		ctx, http.MethodGet, target, nil, runtimeAgentID, headers,
	)
	if err != nil {
		code := "worker_state_transport_failed"
		if ctx.Err() != nil {
			code = "worker_state_cancelled"
		}
		return reporting.WorkerStateReadResult{}, workerStateReadError(0, code, true)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK && response.StatusCode != http.StatusNotModified {
		apiErr := decodeRuntimeError(response)
		if typed, ok := apiErr.(*RuntimeAPIError); ok {
			return reporting.WorkerStateReadResult{}, workerStateReadError(
				typed.StatusCode, typed.Code, typed.Retryable,
			)
		}
		return reporting.WorkerStateReadResult{}, workerStateReadError(
			response.StatusCode, "worker_state_response_invalid", false,
		)
	}
	if !oneHeaderValue(response, "Cache-Control", "private, no-cache") {
		return reporting.WorkerStateReadResult{}, workerStateReadError(
			response.StatusCode, "worker_state_response_invalid", false,
		)
	}
	etagValues := response.Header.Values("ETag")
	if len(etagValues) != 1 || !agentStateETagPattern.MatchString(etagValues[0]) {
		return reporting.WorkerStateReadResult{}, workerStateReadError(
			response.StatusCode, "worker_state_response_invalid", false,
		)
	}
	etag := etagValues[0]
	if response.StatusCode == http.StatusNotModified {
		if ifNoneMatch == "" || etag != ifNoneMatch {
			return reporting.WorkerStateReadResult{}, workerStateReadError(
				response.StatusCode, "worker_state_response_invalid", false,
			)
		}
		data, readErr := io.ReadAll(io.LimitReader(response.Body, 1))
		if readErr != nil || len(data) != 0 {
			return reporting.WorkerStateReadResult{}, workerStateReadError(
				response.StatusCode, "worker_state_response_invalid", false,
			)
		}
		return reporting.WorkerStateReadResult{ETag: etag, NotModified: true}, nil
	}
	if ifNoneMatch != "" && etag == ifNoneMatch {
		return reporting.WorkerStateReadResult{}, workerStateReadError(
			response.StatusCode, "worker_state_response_invalid", false,
		)
	}
	if err := requireJSONContentType(response); err != nil {
		return reporting.WorkerStateReadResult{}, workerStateReadError(
			response.StatusCode, "worker_state_response_invalid", false,
		)
	}
	data, err := readBoundedBody(response.Body, reporting.MaxAgentStateSnapshotBytes)
	if err != nil {
		return reporting.WorkerStateReadResult{}, workerStateReadError(
			response.StatusCode, "worker_state_response_invalid", false,
		)
	}
	snapshot, err := contracts.DecodePrivateStrict[reporting.AgentStateSnapshot](data)
	if err != nil {
		return reporting.WorkerStateReadResult{}, workerStateReadError(
			response.StatusCode, "worker_state_response_invalid", false,
		)
	}
	expectedETag := fmt.Sprintf(
		"\"contractor-agent-state-v1-%d\"", snapshot.State.StateRevision,
	)
	if etag != expectedETag {
		return reporting.WorkerStateReadResult{}, workerStateReadError(
			response.StatusCode, "worker_state_response_invalid", false,
		)
	}
	return reporting.WorkerStateReadResult{Snapshot: &snapshot, ETag: etag}, nil
}

func (c *RuntimeControlClient) workerStateEndpoint(handle contracts.WorkerHandle) (string, error) {
	if !runtimePathIDPattern.MatchString(handle.AllocationID) {
		return "", errors.New("allocation ID is not a safe URL path segment")
	}
	interfaces, ok := handle.AgentCard["supportedInterfaces"].([]any)
	if !ok || len(interfaces) != 1 {
		return "", errors.New("WorkerHandle has no exact A2A endpoint")
	}
	current, ok := interfaces[0].(map[string]any)
	if !ok {
		return "", errors.New("WorkerHandle has no exact A2A endpoint")
	}
	rawEndpoint, ok := current["url"].(string)
	if !ok {
		return "", errors.New("WorkerHandle has no exact A2A endpoint")
	}
	if current["protocolBinding"] != "GRPC" || current["tenant"] != handle.AllocationID {
		return "", errors.New("WorkerHandle A2A interface is invalid")
	}
	authority, err := mtls.GRPCAuthority("https://" + rawEndpoint)
	if err != nil || authority != rawEndpoint {
		return "", errors.New("WorkerHandle A2A endpoint is invalid")
	}
	return "https://" + authority + "/private/v1/allocations/" + handle.AllocationID + "/agent-state", nil
}

func oneHeaderValue(response *http.Response, name, expected string) bool {
	values := response.Header.Values(name)
	return len(values) == 1 && values[0] == expected
}

func workerStateReadError(statusCode int, code string, retryable bool) error {
	return &reporting.WorkerStateReadError{StatusCode: statusCode, Code: code, Retryable: retryable}
}
