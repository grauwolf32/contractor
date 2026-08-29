package controlplane

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
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
)

const maxRuntimeResponseBytes = 1 << 20

var runtimePathIDPattern = regexp.MustCompile(`^[A-Za-z0-9_-]+$`)

type RuntimeLifecycle interface {
	Prepare(context.Context, Reservation, contracts.RuntimeSettings) (contracts.WorkerHandle, error)
	Finalize(context.Context, Reservation, string, time.Time) (contracts.ExecutionReport, error)
	Abort(context.Context, Reservation, string, contracts.TerminationError, time.Time) (contracts.ExecutionReport, error)
	Release(context.Context, Reservation) error
}

type RuntimeAPIError struct {
	StatusCode int
	Code       string
	Retryable  bool
}

func (e *RuntimeAPIError) Error() string {
	return fmt.Sprintf("Runtime Agent returned HTTP %d (%s)", e.StatusCode, e.Code)
}

type RuntimeControlClient struct {
	client       *http.Client
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
	client := &http.Client{
		Transport: &http.Transport{
			TLSClientConfig:       tlsConfig,
			ForceAttemptHTTP2:     false,
			MaxIdleConnsPerHost:   2,
			IdleConnTimeout:       30 * time.Second,
			TLSHandshakeTimeout:   timeout,
			ResponseHeaderTimeout: timeout,
		},
		Timeout: timeout,
		CheckRedirect: func(_ *http.Request, _ []*http.Request) error {
			return errors.New("Runtime Agent redirects are not allowed")
		},
	}
	return &RuntimeControlClient{client: client, requireHTTPS: true}, nil
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
	settings contracts.RuntimeSettings,
) (contracts.WorkerHandle, error) {
	spec := contracts.AllocationSpec{
		APIVersion: contracts.APIVersion, AllocationID: reservation.Grant.AllocationID,
		RunID: reservation.Grant.RunID, StageExecutionID: reservation.Grant.StageExecutionID,
		LogicalAgentName: reservation.Grant.LogicalAgentName, Namespace: reservation.Grant.Namespace,
		LeaseExpiresAt: wireTime(reservation.LeaseExpiresAt), AgentTemplate: cloneAgentTemplate(reservation.AgentTemplate),
		RuntimeSettings: settings,
	}
	request := contracts.PrepareAllocationRequest{APIVersion: contracts.APIVersion, Spec: spec}
	if err := request.Validate(); err != nil {
		return contracts.WorkerHandle{}, fmt.Errorf("build prepare request: %w", err)
	}
	var response contracts.PrepareAllocationResponse
	if err := c.postJSON(ctx, reservation.ControlURL, reservation.Grant.AllocationID, "prepare", request, &response); err != nil {
		return contracts.WorkerHandle{}, err
	}
	if err := validateWorkerHandle(response.WorkerHandle, reservation, settings); err != nil {
		return contracts.WorkerHandle{}, err
	}
	return response.WorkerHandle, nil
}

func (c *RuntimeControlClient) Finalize(
	ctx context.Context,
	reservation Reservation,
	finalizationID string,
	deadline time.Time,
) (contracts.ExecutionReport, error) {
	request := contracts.FinalizeAllocationRequest{
		APIVersion: contracts.APIVersion, AllocationID: reservation.Grant.AllocationID,
		FinalizationID: finalizationID, Deadline: deadline,
	}
	if err := request.Validate(); err != nil {
		return contracts.ExecutionReport{}, fmt.Errorf("build finalize request: %w", err)
	}
	var response contracts.AllocationFinalResponse
	if err := c.postJSON(ctx, reservation.ControlURL, reservation.Grant.AllocationID, "finalize", request, &response); err != nil {
		return contracts.ExecutionReport{}, err
	}
	if response.Report.AllocationID != reservation.Grant.AllocationID {
		return contracts.ExecutionReport{}, errors.New("Runtime Agent final report identifies another allocation")
	}
	return response.Report, nil
}

func (c *RuntimeControlClient) Abort(
	ctx context.Context,
	reservation Reservation,
	abortID string,
	reason contracts.TerminationError,
	deadline time.Time,
) (contracts.ExecutionReport, error) {
	request := contracts.AbortAllocationRequest{
		APIVersion: contracts.APIVersion, AllocationID: reservation.Grant.AllocationID,
		AbortID: abortID, Reason: reason, Deadline: deadline,
	}
	if err := request.Validate(); err != nil {
		return contracts.ExecutionReport{}, fmt.Errorf("build abort request: %w", err)
	}
	var response contracts.AllocationFinalResponse
	if err := c.postJSON(ctx, reservation.ControlURL, reservation.Grant.AllocationID, "abort", request, &response); err != nil {
		return contracts.ExecutionReport{}, err
	}
	if response.Report.AllocationID != reservation.Grant.AllocationID {
		return contracts.ExecutionReport{}, errors.New("Runtime Agent abort report identifies another allocation")
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
	response, err := c.do(ctx, target, body)
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

func (c *RuntimeControlClient) postJSON(
	ctx context.Context,
	baseURL string,
	allocationID string,
	operation string,
	request any,
	response contracts.Validatable,
) error {
	target, err := c.endpoint(baseURL, allocationID, operation)
	if err != nil {
		return err
	}
	body, err := json.Marshal(request)
	if err != nil {
		return fmt.Errorf("encode Runtime Agent request: %w", err)
	}
	httpResponse, err := c.do(ctx, target, body)
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
	if err := response.Validate(); err != nil {
		return errors.New("Runtime Agent returned a lifecycle response that violates the contract")
	}
	return nil
}

func (c *RuntimeControlClient) do(ctx context.Context, target string, body []byte) (*http.Response, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, target, bytes.NewReader(body))
	if err != nil {
		return nil, errors.New("build Runtime Agent request")
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", "application/json")
	response, err := c.client.Do(request)
	if err != nil {
		return nil, fmt.Errorf("call Runtime Agent: %w", err)
	}
	return response, nil
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
	secret := settings.LLMGatewayToken.Reveal()
	if secret != "" && bytes.Contains(encoded, []byte(secret)) {
		return errors.New("Runtime Agent exposed RuntimeSettings secret in WorkerHandle")
	}
	protocol, ok := handle.AgentCard["protocolVersion"].(string)
	if !ok || protocol != "1.0" {
		return errors.New("Runtime Agent returned an incompatible A2A Agent Card")
	}
	cardURL, ok := handle.AgentCard["url"].(string)
	if !ok || !sameEndpointOrigin(cardURL, reservation.A2AURL) {
		return errors.New("Runtime Agent returned an A2A Agent Card for another endpoint")
	}
	return nil
}

func sameEndpointOrigin(cardURL, registeredURL string) bool {
	card, cardErr := url.Parse(cardURL)
	registered, registeredErr := url.Parse(registeredURL)
	return cardErr == nil && registeredErr == nil && card.IsAbs() && registered.IsAbs() &&
		card.Scheme == registered.Scheme && card.Host == registered.Host &&
		strings.HasPrefix(card.Path, strings.TrimRight(registered.Path, "/")+"/")
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
	data, err := io.ReadAll(io.LimitReader(body, maxRuntimeResponseBytes+1))
	if err != nil {
		return nil, errors.New("read Runtime Agent response")
	}
	if len(data) > maxRuntimeResponseBytes {
		return nil, errors.New("Runtime Agent response is too large")
	}
	return data, nil
}

func ensureRuntimeJSONEOF(decoder *json.Decoder) error {
	var trailing json.RawMessage
	if err := decoder.Decode(&trailing); errors.Is(err, io.EOF) {
		return nil
	}
	return errors.New("Runtime Agent returned multiple JSON values")
}

type AllocationRegistry interface {
	SetWriteFence(string) error
	Release(string) error
}

type RuntimeBatchController struct {
	runtime        RuntimeLifecycle
	registry       AllocationRegistry
	now            func() time.Time
	newID          func(string) (string, error)
	cleanupTimeout time.Duration
}

type RuntimeBatchOptions struct {
	Now            func() time.Time
	NewID          func(string) (string, error)
	CleanupTimeout time.Duration
}

func NewRuntimeBatchController(
	runtime RuntimeLifecycle,
	registry AllocationRegistry,
	options RuntimeBatchOptions,
) (*RuntimeBatchController, error) {
	if runtime == nil || registry == nil {
		return nil, errors.New("Runtime lifecycle and allocation registry are required")
	}
	if options.Now == nil {
		options.Now = time.Now
	}
	if options.NewID == nil {
		options.NewID = lifecycleID
	}
	if options.CleanupTimeout == 0 {
		options.CleanupTimeout = 10 * time.Second
	}
	if options.CleanupTimeout <= 0 {
		return nil, errors.New("cleanup timeout must be positive")
	}
	return &RuntimeBatchController{
		runtime: runtime, registry: registry, now: options.Now,
		newID: options.NewID, cleanupTimeout: options.CleanupTimeout,
	}, nil
}

func (c *RuntimeBatchController) PrepareAll(
	ctx context.Context,
	reservations []Reservation,
	settings contracts.RuntimeSettings,
) (map[string]contracts.WorkerHandle, error) {
	if err := validateReservationBatch(reservations); err != nil {
		return nil, err
	}
	handles := make(map[string]contracts.WorkerHandle, len(reservations))
	for _, reservation := range reservations {
		handle, err := c.runtime.Prepare(ctx, reservation, settings)
		if err != nil {
			prepareErr := fmt.Errorf("prepare logical Agent %q: %w", reservation.Grant.LogicalAgentName, err)
			cleanupErr := c.cleanupFailedPrepare(reservations)
			return nil, errors.Join(prepareErr, cleanupErr)
		}
		handles[reservation.Grant.LogicalAgentName] = handle
	}
	return handles, nil
}

func (c *RuntimeBatchController) FinalizeAll(
	ctx context.Context,
	reservations []Reservation,
	finalizationID string,
	deadline time.Time,
) (map[string]contracts.ExecutionReport, error) {
	if err := validateReservationBatch(reservations); err != nil {
		return nil, err
	}
	var failures []error
	for _, reservation := range reservations {
		if err := c.registry.SetWriteFence(reservation.Grant.AllocationID); err != nil {
			failures = append(failures, fmt.Errorf("fence allocation %q: %w", reservation.Grant.AllocationID, err))
		}
	}
	reports := make(map[string]contracts.ExecutionReport, len(reservations))
	for _, reservation := range reservations {
		report, err := c.runtime.Finalize(ctx, reservation, finalizationID, deadline)
		if err != nil {
			failures = append(failures, fmt.Errorf("finalize allocation %q: %w", reservation.Grant.AllocationID, err))
			continue
		}
		reports[reservation.Grant.LogicalAgentName] = report
	}
	return reports, errors.Join(failures...)
}

func (c *RuntimeBatchController) AbortAll(
	ctx context.Context,
	reservations []Reservation,
	abortID string,
	reason contracts.TerminationError,
	deadline time.Time,
) (map[string]contracts.ExecutionReport, error) {
	if err := validateReservationBatch(reservations); err != nil {
		return nil, err
	}
	var failures []error
	reports := make(map[string]contracts.ExecutionReport, len(reservations))
	for _, reservation := range reservations {
		if err := c.registry.SetWriteFence(reservation.Grant.AllocationID); err != nil {
			failures = append(failures, fmt.Errorf("fence allocation %q: %w", reservation.Grant.AllocationID, err))
		}
		report, err := c.runtime.Abort(ctx, reservation, abortID, reason, deadline)
		if err != nil {
			failures = append(failures, fmt.Errorf("abort allocation %q: %w", reservation.Grant.AllocationID, err))
			continue
		}
		reports[reservation.Grant.LogicalAgentName] = report
	}
	return reports, errors.Join(failures...)
}

func (c *RuntimeBatchController) ReleaseAll(ctx context.Context, reservations []Reservation) error {
	if err := validateReservationBatch(reservations); err != nil {
		return err
	}
	var failures []error
	for _, reservation := range reservations {
		if err := c.runtime.Release(ctx, reservation); err != nil {
			failures = append(failures, fmt.Errorf("release Runtime Agent allocation %q: %w", reservation.Grant.AllocationID, err))
			continue
		}
		if err := c.registry.Release(reservation.Grant.AllocationID); err != nil {
			failures = append(failures, fmt.Errorf("release registry allocation %q: %w", reservation.Grant.AllocationID, err))
		}
	}
	return errors.Join(failures...)
}

func (c *RuntimeBatchController) cleanupFailedPrepare(reservations []Reservation) error {
	ctx, cancel := context.WithTimeout(context.Background(), c.cleanupTimeout)
	defer cancel()
	reason := contracts.TerminationError{
		Code: "allocation_batch_preparation_failed", Message: "Stage allocation batch preparation failed", Retryable: true,
	}
	deadline := c.now().Add(c.cleanupTimeout)
	var failures []error
	for _, reservation := range reservations {
		allocationID := reservation.Grant.AllocationID
		if err := c.registry.SetWriteFence(allocationID); err != nil {
			failures = append(failures, fmt.Errorf("fence failed prepare allocation %q: %w", allocationID, err))
		}
		abortID, err := c.newID("abort_")
		if err != nil {
			failures = append(failures, fmt.Errorf("create cleanup abort ID: %w", err))
		} else if _, err := c.runtime.Abort(ctx, reservation, abortID, reason, deadline); err != nil {
			failures = append(failures, fmt.Errorf("abort failed prepare allocation %q: %w", allocationID, err))
		}
		if err := c.runtime.Release(ctx, reservation); err != nil {
			failures = append(failures, fmt.Errorf("release failed prepare Runtime Agent allocation %q: %w", allocationID, err))
		}
		if err := c.registry.Release(allocationID); err != nil {
			failures = append(failures, fmt.Errorf("release failed prepare registry allocation %q: %w", allocationID, err))
		}
	}
	return errors.Join(failures...)
}

func validateReservationBatch(reservations []Reservation) error {
	if len(reservations) == 0 {
		return errors.New("allocation reservation batch must not be empty")
	}
	allocationIDs := make(map[string]struct{}, len(reservations))
	logicalNames := make(map[string]struct{}, len(reservations))
	for _, reservation := range reservations {
		if strings.TrimSpace(reservation.Grant.AllocationID) == "" || strings.TrimSpace(reservation.Grant.LogicalAgentName) == "" {
			return errors.New("allocation reservation is incomplete")
		}
		if _, duplicate := allocationIDs[reservation.Grant.AllocationID]; duplicate {
			return errors.New("allocation reservation batch contains duplicate allocation IDs")
		}
		if _, duplicate := logicalNames[reservation.Grant.LogicalAgentName]; duplicate {
			return errors.New("allocation reservation batch contains duplicate logical Agent names")
		}
		allocationIDs[reservation.Grant.AllocationID] = struct{}{}
		logicalNames[reservation.Grant.LogicalAgentName] = struct{}{}
	}
	return nil
}

func lifecycleID(prefix string) (string, error) {
	buffer := make([]byte, 16)
	if _, err := rand.Read(buffer); err != nil {
		return "", err
	}
	return prefix + hex.EncodeToString(buffer), nil
}

// Python datetime and PostgreSQL both retain microseconds. Normalize private
// wire timestamps at the Go boundary so an echoed deadline remains exact.
func wireTime(value time.Time) time.Time {
	return value.UTC().Truncate(time.Microsecond)
}

var _ RuntimeLifecycle = (*RuntimeControlClient)(nil)
