package controlplane

import (
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/mtls"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/requestid"
)

const maxRuntimeResponseBytes = 1 << 20

var (
	runtimePathIDPattern  = regexp.MustCompile(`^[A-Za-z0-9_-]+$`)
	agentStateETagPattern = regexp.MustCompile(`^"contractor-agent-state-v1-[1-9][0-9]*"$`)
)

type WorkerStateReader = planner.WorkerStateReader
type WorkerStateReadResult = planner.WorkerStateReadResult
type WorkerStateReadError = planner.WorkerStateReadError

var _ planner.WorkerStateReader = (*RuntimeControlClient)(nil)

type RuntimeAPIError struct {
	StatusCode int
	Code       string
	Retryable  bool
}

func (e *RuntimeAPIError) Error() string {
	return fmt.Sprintf("Runtime Agent returned HTTP %d (%s)", e.StatusCode, e.Code)
}

// RuntimeControlClient implements the Runtime Agent HTTP protocol, including
// principal binding and validation of allocation responses and Worker State.
type RuntimeControlClient struct {
	client       *http.Client
	tlsConfig    *tls.Config
	timeout      time.Duration
	requireHTTPS bool
}

func NewMTLSRuntimeControlClient(files mtls.Files, timeout time.Duration) (*RuntimeControlClient, error) {
	if timeout <= 0 {
		return nil, errors.New("Runtime Agent request timeout must be positive")
	}
	tlsConfig, err := mtls.ControlPlaneEndpointClientConfig(files)
	if err != nil {
		return nil, fmt.Errorf("build Runtime Agent TLS client: %w", err)
	}
	return &RuntimeControlClient{
		tlsConfig: tlsConfig, timeout: timeout, requireHTTPS: true,
	}, nil
}

// NewRuntimeControlClient accepts an injected client for focused protocol
// tests. Production code must use NewMTLSRuntimeControlClient.
func NewRuntimeControlClient(client *http.Client) (*RuntimeControlClient, error) {
	if client == nil {
		return nil, errors.New("HTTP client is required")
	}
	clone := *client
	clone.CheckRedirect = func(_ *http.Request, _ []*http.Request) error {
		return errors.New("Runtime Agent redirects are not allowed")
	}
	return &RuntimeControlClient{client: &clone}, nil
}

func (c *RuntimeControlClient) Prepare(
	ctx context.Context,
	reservation Reservation,
	settings contracts.WorkerExecutionSettings,
) (contracts.WorkerHandle, error) {
	if !contracts.SupportsWorkerCompletion(reservation.CompletionCapabilities, reservation.CompletionContract) {
		return contracts.WorkerHandle{}, fmt.Errorf("%w: Runtime does not support completion contract", ErrInvalidRequest)
	}
	if err := contracts.ValidateWorkerCompletionSelection(reservation.CompletionContract, reservation.Grant.Namespace, reservation.AgentTemplate); err != nil {
		return contracts.WorkerHandle{}, err
	}
	resolvedSkills := reservation.ResolvedSkills
	if resolvedSkills == nil && len(reservation.AgentTemplate.Skills) == 0 {
		resolvedSkills = []contracts.ResolvedSkill{}
	}
	spec := contracts.AllocationSpec{
		CompletionContract: contracts.CloneWorkerCompletionContract(reservation.CompletionContract),
		APIVersion:         contracts.APIVersion, AllocationID: reservation.Grant.AllocationID,
		RunID: reservation.Grant.RunID, StageExecutionID: reservation.Grant.StageExecutionID,
		LogicalAgentName: reservation.Grant.LogicalAgentName, Namespace: reservation.Grant.Namespace,
		WorkerSessionMode: reservation.WorkerSessionMode,
		RunMetadataLabels: reservation.RunMetadataLabels.Clone(),
		LeaseExpiresAt:    wireTime(reservation.LeaseExpiresAt), AgentTemplate: cloneAgentTemplate(reservation.AgentTemplate),
		ResolvedSkills: contracts.CloneResolvedSkills(resolvedSkills),
		ModelPolicy:    cloneModelPolicy(settings.ModelPolicy), RuntimeSettings: settings.RuntimeSettings,
		ResolvedRuntimeConfigProvenance: settings.ResolvedRuntimeConfigProvenance,
		Workspace:                       contracts.CloneAllocationWorkspaceSpec(reservation.Workspace),
		PerformanceMetrics:              clonePerformanceMetricsRequest(reservation.PerformanceMetrics),
	}
	request := contracts.PrepareAllocationRequest{APIVersion: contracts.APIVersion, Spec: spec}
	if err := request.Validate(); err != nil {
		return contracts.WorkerHandle{}, fmt.Errorf("build prepare request: %w", err)
	}
	var response contracts.PrepareAllocationResponse
	if err := c.postJSON(
		ctx, reservation.ControlURL, reservation.Grant.AllocationID,
		reservation.Grant.RuntimeAgentID, "prepare", request, &response, true,
	); err != nil {
		return contracts.WorkerHandle{}, err
	}
	if err := validateWorkerHandle(response.WorkerHandle, reservation, settings.RuntimeSettings); err != nil {
		return contracts.WorkerHandle{}, err
	}
	response.WorkerHandle.RuntimeAgentID = reservation.Grant.RuntimeAgentID
	return response.WorkerHandle, nil
}

func (c *RuntimeControlClient) Finalize(
	ctx context.Context,
	reservation Reservation,
	finalizationID string,
	deadline time.Time,
) (contracts.AllocationFinalReport, error) {
	request := contracts.FinalizeAllocationRequest{
		APIVersion: contracts.APIVersion, AllocationID: reservation.Grant.AllocationID,
		FinalizationID: finalizationID, Deadline: deadline,
	}
	if err := request.Validate(); err != nil {
		return contracts.AllocationFinalReport{}, fmt.Errorf("build finalize request: %w", err)
	}
	var response contracts.AllocationFinalResponse
	if err := c.postJSON(
		ctx, reservation.ControlURL, reservation.Grant.AllocationID,
		reservation.Grant.RuntimeAgentID, "finalize", request, &response, false,
	); err != nil {
		return contracts.AllocationFinalReport{}, err
	}
	sanitizeRuntimeAdapterMetrics(&response.Report.Runtime, reservation)
	sanitizeRuntimeResources(&response.Report.Runtime, reservation)
	if err := response.Validate(); err != nil {
		return contracts.AllocationFinalReport{}, errors.New("Runtime Agent returned a lifecycle response that violates the contract")
	}
	if response.Report.AllocationID != reservation.Grant.AllocationID {
		return contracts.AllocationFinalReport{}, errors.New("Runtime Agent final report identifies another allocation")
	}
	return response.Report, nil
}

func (c *RuntimeControlClient) Abort(
	ctx context.Context,
	reservation Reservation,
	abortID string,
	reason contracts.TerminationError,
	deadline time.Time,
) (contracts.AllocationFinalReport, error) {
	request := contracts.AbortAllocationRequest{
		APIVersion: contracts.APIVersion, AllocationID: reservation.Grant.AllocationID,
		AbortID: abortID, Reason: reason, Deadline: deadline,
	}
	if err := request.Validate(); err != nil {
		return contracts.AllocationFinalReport{}, fmt.Errorf("build abort request: %w", err)
	}
	var response contracts.AllocationFinalResponse
	if err := c.postJSON(
		ctx, reservation.ControlURL, reservation.Grant.AllocationID,
		reservation.Grant.RuntimeAgentID, "abort", request, &response, false,
	); err != nil {
		return contracts.AllocationFinalReport{}, err
	}
	sanitizeRuntimeAdapterMetrics(&response.Report.Runtime, reservation)
	sanitizeRuntimeResources(&response.Report.Runtime, reservation)
	if err := response.Validate(); err != nil {
		return contracts.AllocationFinalReport{}, errors.New("Runtime Agent returned a lifecycle response that violates the contract")
	}
	if response.Report.AllocationID != reservation.Grant.AllocationID {
		return contracts.AllocationFinalReport{}, errors.New("Runtime Agent abort report identifies another allocation")
	}
	return response.Report, nil
}

func (c *RuntimeControlClient) Release(ctx context.Context, reservation Reservation) error {
	request := contracts.ReleaseAllocationRequest{
		APIVersion: contracts.APIVersion, AllocationID: reservation.Grant.AllocationID,
	}
	if err := request.Validate(); err != nil {
		return fmt.Errorf("build release request: %w", err)
	}
	target, err := c.endpoint(reservation.ControlURL, reservation.Grant.AllocationID, "release")
	if err != nil {
		return err
	}
	body, err := json.Marshal(request)
	if err != nil {
		return fmt.Errorf("encode release request: %w", err)
	}
	response, err := c.do(ctx, target, body, reservation.Grant.RuntimeAgentID)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusNoContent {
		return decodeRuntimeError(response)
	}
	limited := io.LimitReader(response.Body, 1)
	data, err := io.ReadAll(limited)
	if err != nil {
		return errors.New("read Runtime Agent release response")
	}
	if len(data) != 0 {
		return errors.New("Runtime Agent release response must be empty")
	}
	return nil
}

func (c *RuntimeControlClient) ReadWorkerState(
	ctx context.Context,
	handle contracts.WorkerHandle,
	ifNoneMatch string,
) (WorkerStateReadResult, error) {
	if ifNoneMatch != "" && !agentStateETagPattern.MatchString(ifNoneMatch) {
		return WorkerStateReadResult{}, workerStateReadError(0, "worker_state_etag_invalid", false)
	}
	target, err := c.workerStateEndpoint(handle)
	if err != nil {
		return WorkerStateReadResult{}, workerStateReadError(0, "worker_state_endpoint_invalid", false)
	}
	headers := map[string]string{}
	if ifNoneMatch != "" {
		headers["If-None-Match"] = ifNoneMatch
	}
	response, err := c.doRequest(
		ctx, http.MethodGet, target, nil, handle.RuntimeAgentID, headers,
	)
	if err != nil {
		code := "worker_state_transport_failed"
		if ctx.Err() != nil {
			code = "worker_state_cancelled"
		}
		return WorkerStateReadResult{}, workerStateReadError(0, code, true)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK && response.StatusCode != http.StatusNotModified {
		apiErr := decodeRuntimeError(response)
		if typed, ok := apiErr.(*RuntimeAPIError); ok {
			return WorkerStateReadResult{}, workerStateReadError(
				typed.StatusCode, typed.Code, typed.Retryable,
			)
		}
		return WorkerStateReadResult{}, workerStateReadError(
			response.StatusCode, "worker_state_response_invalid", false,
		)
	}
	if !oneHeaderValue(response, "Cache-Control", "private, no-cache") {
		return WorkerStateReadResult{}, workerStateReadError(
			response.StatusCode, "worker_state_response_invalid", false,
		)
	}
	etagValues := response.Header.Values("ETag")
	if len(etagValues) != 1 || !agentStateETagPattern.MatchString(etagValues[0]) {
		return WorkerStateReadResult{}, workerStateReadError(
			response.StatusCode, "worker_state_response_invalid", false,
		)
	}
	etag := etagValues[0]
	if response.StatusCode == http.StatusNotModified {
		if ifNoneMatch == "" || etag != ifNoneMatch {
			return WorkerStateReadResult{}, workerStateReadError(
				response.StatusCode, "worker_state_response_invalid", false,
			)
		}
		data, readErr := io.ReadAll(io.LimitReader(response.Body, 1))
		if readErr != nil || len(data) != 0 {
			return WorkerStateReadResult{}, workerStateReadError(
				response.StatusCode, "worker_state_response_invalid", false,
			)
		}
		return WorkerStateReadResult{ETag: etag, NotModified: true}, nil
	}
	if ifNoneMatch != "" && etag == ifNoneMatch {
		return WorkerStateReadResult{}, workerStateReadError(
			response.StatusCode, "worker_state_response_invalid", false,
		)
	}
	if err := requireJSONContentType(response); err != nil {
		return WorkerStateReadResult{}, workerStateReadError(
			response.StatusCode, "worker_state_response_invalid", false,
		)
	}
	data, err := readBoundedBody(response.Body, contracts.MaxAgentStateSnapshotBytes)
	if err != nil {
		return WorkerStateReadResult{}, workerStateReadError(
			response.StatusCode, "worker_state_response_invalid", false,
		)
	}
	snapshot, err := contracts.DecodeStrict[contracts.AgentStateSnapshot](data)
	if err != nil {
		return WorkerStateReadResult{}, workerStateReadError(
			response.StatusCode, "worker_state_response_invalid", false,
		)
	}
	expectedETag := fmt.Sprintf(
		"\"contractor-agent-state-v1-%d\"", snapshot.State.StateRevision,
	)
	if etag != expectedETag {
		return WorkerStateReadResult{}, workerStateReadError(
			response.StatusCode, "worker_state_response_invalid", false,
		)
	}
	return WorkerStateReadResult{Snapshot: &snapshot, ETag: etag}, nil
}

func (c *RuntimeControlClient) postJSON(
	ctx context.Context,
	baseURL string,
	allocationID string,
	runtimeAgentID string,
	operation string,
	request any,
	response contracts.Validatable,
	validateResponse bool,
) error {
	target, err := c.endpoint(baseURL, allocationID, operation)
	if err != nil {
		return err
	}
	body, err := json.Marshal(request)
	if err != nil {
		return fmt.Errorf("encode Runtime Agent request: %w", err)
	}
	defer clear(body)
	httpResponse, err := c.do(ctx, target, body, runtimeAgentID)
	if err != nil {
		return err
	}
	defer httpResponse.Body.Close()
	if httpResponse.StatusCode < 200 || httpResponse.StatusCode >= 300 {
		return decodeRuntimeError(httpResponse)
	}
	if err := requireJSONContentType(httpResponse); err != nil {
		return err
	}
	data, err := readBoundedRuntimeBody(httpResponse.Body)
	if err != nil {
		return err
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(response); err != nil {
		return errors.New("Runtime Agent returned an invalid lifecycle response")
	}
	if err := ensureRuntimeJSONEOF(decoder); err != nil {
		return err
	}
	if validateResponse {
		if err := response.Validate(); err != nil {
			return errors.New("Runtime Agent returned a lifecycle response that violates the contract")
		}
	}
	return nil
}

func sanitizeRuntimeAdapterMetrics(report *contracts.RuntimeReport, reservation Reservation) {
	expected := map[contracts.RuntimeAdapterRef]struct{}{}
	if reservation.ResolvedRuntimeConfig != nil {
		for _, ref := range reservation.ResolvedRuntimeConfig.RequiredRuntimeAdapters {
			expected[ref] = struct{}{}
		}
	}
	if report.Adapters == nil {
		report.Adapters = map[contracts.RuntimeAdapterRef]contracts.RuntimeAdapterMetrics{}
	}
	for ref, metrics := range report.Adapters {
		_, selected := expected[ref]
		if !selected || metrics.Validate() != nil {
			delete(report.Adapters, ref)
			report.Complete = false
		}
	}
	for ref := range expected {
		if _, reported := report.Adapters[ref]; !reported {
			report.Complete = false
		}
	}
}

func clonePerformanceMetricsRequest(source *contracts.PerformanceMetricsRequest) *contracts.PerformanceMetricsRequest {
	if source == nil {
		return nil
	}
	result := *source
	return &result
}

// Resource telemetry is optional and must never poison Worker truth. Only a
// Server-pinned request may contribute measurements. A malformed requested
// block is retained as one bounded diagnostic sentinel, not raw input.
func sanitizeRuntimeResources(report *contracts.RuntimeReport, reservation Reservation) {
	if reservation.PerformanceCollectionPolicy != contracts.PerformanceCollectionRequested {
		report.Resources = nil
		report.ResourcesError = nil
		return
	}
	if report.ResourcesError != nil ||
		(report.Resources != nil && report.Resources.Validate() != nil) {
		reason := contracts.ResourceInvalidReport
		report.Resources = &contracts.RuntimeResources{
			Version: contracts.PerformanceMetricsVersion, Scope: "runtime_process",
			Status: contracts.ResourceUnavailable, Reason: &reason,
		}
		report.ResourcesError = nil
	}
}

func (c *RuntimeControlClient) do(
	ctx context.Context, target string, body []byte, runtimeAgentID string,
) (*http.Response, error) {
	return c.doRequest(ctx, http.MethodPost, target, body, runtimeAgentID, nil)
}

func (c *RuntimeControlClient) doRequest(
	ctx context.Context,
	method string,
	target string,
	body []byte,
	runtimeAgentID string,
	headers map[string]string,
) (*http.Response, error) {
	var reader io.Reader
	if body != nil {
		reader = bytes.NewReader(body)
	}
	request, err := http.NewRequestWithContext(ctx, method, target, reader)
	if err != nil {
		return nil, errors.New("build Runtime Agent request")
	}
	if body != nil {
		request.Header.Set("Content-Type", "application/json")
	}
	request.Header.Set("Accept", "application/json")
	request.Header.Set(requestid.Header, requestid.Ensure(ctx))
	for name, value := range headers {
		request.Header.Set(name, value)
	}
	client := c.client
	if c.tlsConfig != nil {
		bound, bindErr := mtls.BindRuntimeAgentPrincipal(c.tlsConfig, runtimeAgentID)
		if bindErr != nil {
			return nil, errors.New("Runtime Agent principal binding is invalid")
		}
		client = &http.Client{
			Transport: &http.Transport{
				TLSClientConfig: bound, ForceAttemptHTTP2: false, DisableKeepAlives: true,
				TLSHandshakeTimeout: c.timeout, ResponseHeaderTimeout: c.timeout,
			},
			Timeout: c.timeout,
			CheckRedirect: func(_ *http.Request, _ []*http.Request) error {
				return errors.New("Runtime Agent redirects are not allowed")
			},
		}
	}
	if client == nil {
		return nil, errors.New("Runtime Agent HTTP client is unavailable")
	}
	response, err := client.Do(request)
	if err != nil {
		return nil, fmt.Errorf("call Runtime Agent: %w", err)
	}
	return response, nil
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
	parsed, err := url.Parse(rawEndpoint)
	if err != nil || parsed.Host == "" || parsed.User != nil || parsed.RawQuery != "" ||
		parsed.Fragment != "" || parsed.RawPath != "" {
		return "", errors.New("WorkerHandle A2A endpoint is invalid")
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return "", errors.New("WorkerHandle A2A endpoint scheme is invalid")
	}
	if c.requireHTTPS && parsed.Scheme != "https" {
		return "", errors.New("WorkerHandle A2A endpoint must use HTTPS")
	}
	suffix := "/private/v1/allocations/" + handle.AllocationID + "/a2a"
	if !strings.HasSuffix(parsed.Path, suffix) {
		return "", errors.New("WorkerHandle A2A endpoint does not match allocation")
	}
	parsed.Path = strings.TrimSuffix(parsed.Path, suffix) +
		"/private/v1/allocations/" + handle.AllocationID + "/agent-state"
	return parsed.String(), nil
}

func (c *RuntimeControlClient) endpoint(baseURL, allocationID, operation string) (string, error) {
	parsed, err := url.Parse(baseURL)
	if err != nil || parsed.Host == "" || parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" {
		return "", errors.New("Runtime Agent control URL is invalid")
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return "", errors.New("Runtime Agent control URL must use HTTP or HTTPS")
	}
	if c.requireHTTPS && parsed.Scheme != "https" {
		return "", errors.New("Runtime Agent control URL must use HTTPS")
	}
	if !runtimePathIDPattern.MatchString(allocationID) {
		return "", errors.New("allocation ID is not a safe URL path segment")
	}
	parsed.Path = strings.TrimRight(parsed.Path, "/") + "/private/v1/allocations/" + allocationID + "/" + operation
	parsed.RawPath = ""
	return parsed.String(), nil
}

func validateWorkerHandle(
	handle contracts.WorkerHandle,
	reservation Reservation,
	settings contracts.RuntimeSettings,
) error {
	if handle.AllocationID != reservation.Grant.AllocationID ||
		handle.AgentTemplateRef != reservation.AgentTemplate.Ref ||
		handle.WorkerRuntimeRef != reservation.AgentTemplate.Runtime ||
		!handle.LeaseExpiresAt.Equal(wireTime(reservation.LeaseExpiresAt)) {
		return errors.New("Runtime Agent returned a WorkerHandle for different resolved inputs")
	}
	encoded, err := json.Marshal(handle)
	if err != nil {
		return errors.New("Runtime Agent returned an unencodable WorkerHandle")
	}
	handleStrings := workerHandleStringValues(handle)
	for _, secret := range runtimeSettingSecrets(settings) {
		_, exact := handleStrings[secret]
		if secret != "" && (exact || (len([]byte(secret)) >= 16 && bytes.Contains(encoded, []byte(secret)))) {
			return errors.New("Runtime Agent exposed RuntimeSettings secret in WorkerHandle")
		}
	}
	if err := validateA2AAgentCard(
		handle.AgentCard, reservation.Grant.AllocationID, reservation.A2AURL,
	); err != nil {
		return err
	}
	return nil
}

func workerHandleStringValues(handle contracts.WorkerHandle) map[string]struct{} {
	result := map[string]struct{}{
		handle.AllocationID: {}, handle.AgentTemplateRef.TemplateID: {},
		handle.AgentTemplateRef.Version: {}, handle.AgentTemplateRef.Digest: {},
		handle.WorkerRuntimeRef.RuntimeID: {}, handle.WorkerRuntimeRef.Version: {},
	}
	collectStringValues(result, handle.AgentCard)
	return result
}

func collectStringValues(result map[string]struct{}, value any) {
	switch typed := value.(type) {
	case string:
		result[typed] = struct{}{}
	case []any:
		for _, item := range typed {
			collectStringValues(result, item)
		}
	case map[string]any:
		for _, item := range typed {
			collectStringValues(result, item)
		}
	}
}

func runtimeSettingSecrets(settings contracts.RuntimeSettings) []string {
	result := make([]string, 0, 36)
	if settings.LLMGatewayToken != nil {
		result = append(result, settings.LLMGatewayToken.Reveal())
	}
	if settings.Telemetry != nil {
		for _, value := range settings.Telemetry.Headers {
			result = append(result, value.Reveal())
		}
	}
	if settings.HTTPProxy != nil {
		if settings.HTTPProxy.BasicAuth != nil {
			result = append(result,
				settings.HTTPProxy.BasicAuth.Username.Reveal(),
				settings.HTTPProxy.BasicAuth.Password.Reveal(),
			)
		}
		if settings.HTTPProxy.BearerToken != nil {
			result = append(result, settings.HTTPProxy.BearerToken.Reveal())
		}
	}
	if settings.Caido != nil && settings.Caido.BearerToken != nil {
		result = append(result, settings.Caido.BearerToken.Reveal())
	}
	if settings.HTTPOriginTarget != nil {
		if settings.HTTPOriginTarget.BasicAuth != nil {
			result = append(result,
				settings.HTTPOriginTarget.BasicAuth.Username.Reveal(),
				settings.HTTPOriginTarget.BasicAuth.Password.Reveal(),
			)
		}
		if settings.HTTPOriginTarget.BearerToken != nil {
			result = append(result, settings.HTTPOriginTarget.BearerToken.Reveal())
		}
	}
	return result
}

const (
	stageContentMediaType     = "application/vnd.contractor.stage-content+json"
	workerCompletionMediaType = "application/vnd.contractor.worker-completion+json"
)

func validateA2AAgentCard(card map[string]any, allocationID, registeredURL string) error {
	interfaces, ok := card["supportedInterfaces"].([]any)
	if !ok || len(interfaces) != 1 {
		return errors.New("Runtime Agent returned an incompatible A2A Agent Card")
	}
	current, ok := interfaces[0].(map[string]any)
	if !ok || current["protocolBinding"] != "JSONRPC" || current["protocolVersion"] != "1.0" ||
		current["tenant"] != allocationID {
		return errors.New("Runtime Agent returned an incompatible A2A Agent Card")
	}
	cardURL, ok := current["url"].(string)
	if !ok || !isExpectedA2AEndpoint(cardURL, registeredURL, allocationID) {
		return errors.New("Runtime Agent returned an A2A Agent Card for another endpoint")
	}
	if !oneStringValue(card["defaultInputModes"], stageContentMediaType) ||
		!oneStringValue(card["defaultOutputModes"], workerCompletionMediaType) {
		return errors.New("Runtime Agent returned incompatible A2A content modes")
	}
	skills, ok := card["skills"].([]any)
	if !ok || len(skills) != 1 {
		return errors.New("Runtime Agent returned an incompatible A2A skill set")
	}
	skill, ok := skills[0].(map[string]any)
	if !ok || skill["id"] != "contractor_stage_content" ||
		!oneStringValue(skill["inputModes"], stageContentMediaType) ||
		!oneStringValue(skill["outputModes"], workerCompletionMediaType) {
		return errors.New("Runtime Agent returned an incompatible A2A skill set")
	}
	securitySchemes, ok := card["securitySchemes"].(map[string]any)
	if !ok {
		return errors.New("Runtime Agent A2A Agent Card does not declare mutual TLS")
	}
	mutualTLS, ok := securitySchemes["mutualTLS"].(map[string]any)
	if !ok {
		return errors.New("Runtime Agent A2A Agent Card does not declare mutual TLS")
	}
	if _, ok := mutualTLS["mtlsSecurityScheme"].(map[string]any); !ok {
		return errors.New("Runtime Agent A2A Agent Card does not declare mutual TLS")
	}
	securityRequirements, ok := card["securityRequirements"].([]any)
	if !ok || len(securityRequirements) != 1 {
		return errors.New("Runtime Agent A2A Agent Card does not require mutual TLS")
	}
	requirement, ok := securityRequirements[0].(map[string]any)
	if !ok {
		return errors.New("Runtime Agent A2A Agent Card does not require mutual TLS")
	}
	requiredSchemes, ok := requirement["schemes"].(map[string]any)
	if !ok || len(requiredSchemes) != 1 {
		return errors.New("Runtime Agent A2A Agent Card does not require mutual TLS")
	}
	if _, ok := requiredSchemes["mutualTLS"].(map[string]any); !ok {
		return errors.New("Runtime Agent A2A Agent Card does not require mutual TLS")
	}
	return nil
}

func oneStringValue(value any, expected string) bool {
	values, ok := value.([]any)
	return ok && len(values) == 1 && values[0] == expected
}

func isExpectedA2AEndpoint(cardURL, registeredURL, allocationID string) bool {
	card, cardErr := url.Parse(cardURL)
	registered, registeredErr := url.Parse(registeredURL)
	if cardErr != nil || registeredErr != nil || !card.IsAbs() || !registered.IsAbs() ||
		card.User != nil || card.RawQuery != "" || card.Fragment != "" ||
		registered.User != nil || registered.RawQuery != "" || registered.Fragment != "" ||
		!runtimePathIDPattern.MatchString(allocationID) {
		return false
	}
	expected := strings.TrimRight(registeredURL, "/") +
		"/private/v1/allocations/" + allocationID + "/a2a"
	return cardURL == expected
}

func decodeRuntimeError(response *http.Response) error {
	if err := requireJSONContentType(response); err != nil {
		return &RuntimeAPIError{StatusCode: response.StatusCode, Code: "invalid_error_response"}
	}
	data, err := readBoundedRuntimeBody(response.Body)
	if err != nil {
		return &RuntimeAPIError{StatusCode: response.StatusCode, Code: "invalid_error_response"}
	}
	var value privateErrorResponse
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&value); err != nil || strings.TrimSpace(value.Code) == "" ||
		strings.TrimSpace(value.Message) == "" || ensureRuntimeJSONEOF(decoder) != nil {
		return &RuntimeAPIError{StatusCode: response.StatusCode, Code: "invalid_error_response"}
	}
	return &RuntimeAPIError{
		StatusCode: response.StatusCode, Code: value.Code, Retryable: value.Retryable,
	}
}

func requireJSONContentType(response *http.Response) error {
	values := response.Header.Values("Content-Type")
	if len(values) != 1 {
		return errors.New("Runtime Agent response must have one JSON Content-Type")
	}
	mediaType, parameters, err := mime.ParseMediaType(values[0])
	if err != nil || mediaType != "application/json" || len(parameters) > 1 {
		return errors.New("Runtime Agent response Content-Type is invalid")
	}
	if len(parameters) == 1 {
		charset, ok := parameters["charset"]
		if !ok || !strings.EqualFold(charset, "utf-8") {
			return errors.New("Runtime Agent response Content-Type parameter is invalid")
		}
	}
	return nil
}

func readBoundedRuntimeBody(body io.Reader) ([]byte, error) {
	return readBoundedBody(body, maxRuntimeResponseBytes)
}

func readBoundedBody(body io.Reader, limit int) ([]byte, error) {
	data, err := io.ReadAll(io.LimitReader(body, int64(limit)+1))
	if err != nil {
		return nil, errors.New("read Runtime Agent response")
	}
	if len(data) > limit {
		return nil, errors.New("Runtime Agent response is too large")
	}
	return data, nil
}

func oneHeaderValue(response *http.Response, name, expected string) bool {
	values := response.Header.Values(name)
	return len(values) == 1 && values[0] == expected
}

func workerStateReadError(statusCode int, code string, retryable bool) error {
	return &WorkerStateReadError{StatusCode: statusCode, Code: code, Retryable: retryable}
}

func ensureRuntimeJSONEOF(decoder *json.Decoder) error {
	var trailing json.RawMessage
	if err := decoder.Decode(&trailing); errors.Is(err, io.EOF) {
		return nil
	}
	return errors.New("Runtime Agent returned multiple JSON values")
}

// Python datetime and PostgreSQL both retain microseconds. Normalize private
// wire timestamps at the Go boundary so an echoed deadline remains exact.
func wireTime(value time.Time) time.Time {
	return value.UTC().Truncate(time.Microsecond)
}

var _ RuntimeLifecycle = (*RuntimeControlClient)(nil)
