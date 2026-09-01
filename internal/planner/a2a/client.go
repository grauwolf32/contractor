// Package a2a adapts the official A2A Go SDK to Contractor's framework-neutral
// WorkerInvoker boundary.
package a2a

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	sdk "github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2aclient"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/mtls"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/requestid"
)

const (
	stageContentMediaType = "application/vnd.contractor.stage-content+json"
	maxAgentCardBytes     = 1 << 20
	maxA2AResponseBytes   = 1 << 20
	defaultPollInterval   = 100 * time.Millisecond
)

type Options struct {
	PollInterval time.Duration
}

type protocolClient interface {
	SendMessage(context.Context, *sdk.SendMessageRequest) (sdk.SendMessageResult, error)
	GetTask(context.Context, *sdk.GetTaskRequest) (*sdk.Task, error)
	Destroy() error
}

type clientBuilder func(context.Context, *sdk.AgentCard) (protocolClient, error)

type Invoker struct {
	build        clientBuilder
	pollInterval time.Duration
	requireHTTPS bool
	tlsConfig    *tls.Config
	timeout      time.Duration
}

// New accepts an injected HTTP client for tests and local composition. Private
// production traffic must use NewMTLS.
func New(httpClient *http.Client, options Options) (*Invoker, error) {
	return newInvoker(httpClient, options, false)
}

func NewMTLS(files mtls.Files, timeout time.Duration, options Options) (*Invoker, error) {
	if timeout <= 0 {
		return nil, fmt.Errorf("A2A request timeout must be positive")
	}
	tlsConfig, err := mtls.ControlPlaneEndpointClientConfig(files)
	if err != nil {
		return nil, fmt.Errorf("build A2A mTLS client: %w", err)
	}
	httpClient := &http.Client{
		Transport: &http.Transport{
			TLSClientConfig:       tlsConfig,
			ForceAttemptHTTP2:     false,
			MaxIdleConnsPerHost:   2,
			IdleConnTimeout:       30 * time.Second,
			TLSHandshakeTimeout:   timeout,
			ResponseHeaderTimeout: timeout,
		},
		Timeout: timeout,
	}
	result, err := newInvoker(httpClient, options, true)
	if err != nil {
		return nil, err
	}
	result.tlsConfig = tlsConfig
	result.timeout = timeout
	return result, nil
}

func newInvoker(
	httpClient *http.Client, options Options, requireHTTPS bool,
) (*Invoker, error) {
	if httpClient == nil {
		return nil, fmt.Errorf("A2A HTTP client is required")
	}
	if options.PollInterval == 0 {
		options.PollInterval = defaultPollInterval
	}
	if options.PollInterval < 0 {
		return nil, fmt.Errorf("A2A poll interval must be positive")
	}
	builder := sdkClientBuilder(cloneBoundedHTTPClient(httpClient))
	return &Invoker{
		build: builder, pollInterval: options.PollInterval, requireHTTPS: requireHTTPS,
	}, nil
}

func sdkClientBuilder(client *http.Client) clientBuilder {
	return func(ctx context.Context, card *sdk.AgentCard) (protocolClient, error) {
		factory := a2aclient.NewFactory(
			a2aclient.WithDefaultsDisabled(),
			a2aclient.WithJSONRPCTransport(client),
			a2aclient.WithConfig(a2aclient.Config{
				AcceptedOutputModes: []string{stageContentMediaType},
				PreferredTransports: []sdk.TransportProtocol{sdk.TransportProtocolJSONRPC},
			}),
		)
		return factory.CreateFromCard(ctx, card)
	}
}

func (i *Invoker) Invoke(
	ctx context.Context,
	binding string,
	handle contracts.WorkerHandle,
	request contracts.StageContentRequest,
) (contracts.StageContentResult, error) {
	if strings.TrimSpace(binding) == "" || strings.TrimSpace(handle.AllocationID) == "" {
		return contracts.StageContentResult{}, planner.NewError(
			"invalid_worker_handle", "Worker binding and allocation identity are required", false, nil,
		)
	}
	if err := request.Validate(); err != nil {
		return contracts.StageContentResult{}, planner.NewError(
			"invalid_stage_content", "StageContentRequest violates its contract", false, err,
		)
	}
	card, err := decodeCard(handle, i.requireHTTPS)
	if err != nil {
		return contracts.StageContentResult{}, err
	}
	builder := i.build
	if i.tlsConfig != nil {
		bound, bindErr := mtls.BindRuntimeAgentPrincipal(i.tlsConfig, handle.RuntimeAgentID)
		if bindErr != nil {
			return contracts.StageContentResult{}, planner.NewError(
				"invalid_worker_handle", "Worker handle has no valid Runtime Agent principal", false, nil,
			)
		}
		transport := &http.Transport{
			TLSClientConfig: bound, ForceAttemptHTTP2: false, DisableKeepAlives: true,
			TLSHandshakeTimeout: i.timeout, ResponseHeaderTimeout: i.timeout,
		}
		client := &http.Client{Transport: transport, Timeout: i.timeout}
		builder = sdkClientBuilder(cloneBoundedHTTPClient(client))
	}
	client, buildErr := builder(ctx, card)
	if buildErr != nil {
		return contracts.StageContentResult{}, transportError(ctx, buildErr)
	}
	defer func() { _ = client.Destroy() }()

	part := sdk.NewDataPart(request)
	part.MediaType = stageContentMediaType
	message := sdk.NewMessage(sdk.MessageRoleUser, part)
	response, sendErr := client.SendMessage(ctx, &sdk.SendMessageRequest{
		Tenant:  handle.AllocationID,
		Message: message,
		Config: &sdk.SendMessageConfig{
			AcceptedOutputModes: []string{stageContentMediaType},
			ReturnImmediately:   false,
		},
	})
	if sendErr != nil {
		return contracts.StageContentResult{}, transportError(ctx, sendErr)
	}
	return i.resolve(ctx, client, handle.AllocationID, response)
}

func (i *Invoker) resolve(
	ctx context.Context,
	client protocolClient,
	allocationID string,
	response sdk.SendMessageResult,
) (contracts.StageContentResult, error) {
	for {
		switch current := response.(type) {
		case *sdk.Message:
			return decodeResultMessage(current)
		case *sdk.Task:
			message, wait, err := taskMessage(current)
			if err != nil {
				return contracts.StageContentResult{}, err
			}
			if message != nil {
				result, err := decodeResultMessage(message)
				if err != nil {
					return contracts.StageContentResult{}, err
				}
				if current.Status.State == sdk.TaskStateFailed &&
					result.Outcome != contracts.StageFailed {
					return contracts.StageContentResult{}, planner.NewError(
						"invalid_a2a_response",
						"Failed A2A Task carried a successful Contractor result",
						false,
						nil,
					)
				}
				return result, nil
			}
			if !wait {
				return contracts.StageContentResult{}, planner.NewError(
					"invalid_a2a_response", "Terminal A2A Task has no Contractor result", false, nil,
				)
			}
			timer := time.NewTimer(i.pollInterval)
			select {
			case <-ctx.Done():
				timer.Stop()
				return contracts.StageContentResult{}, transportError(ctx, ctx.Err())
			case <-timer.C:
			}
			historyLength := 1
			next, getErr := client.GetTask(ctx, &sdk.GetTaskRequest{
				Tenant: allocationID, ID: current.ID, HistoryLength: &historyLength,
			})
			if getErr != nil {
				return contracts.StageContentResult{}, transportError(ctx, getErr)
			}
			response = next
		default:
			return contracts.StageContentResult{}, planner.NewError(
				"invalid_a2a_response", "Worker returned an unsupported A2A response", false, nil,
			)
		}
	}
}

func decodeCard(handle contracts.WorkerHandle, requireHTTPS bool) (*sdk.AgentCard, error) {
	encoded, err := json.Marshal(handle.AgentCard)
	if err != nil || len(encoded) == 0 || len(encoded) > maxAgentCardBytes {
		return nil, planner.NewError(
			"invalid_worker_handle", "Worker Agent Card is invalid", false, err,
		)
	}
	// Python's protobuf JSON encoder emits an empty StringList as `{}`. The Go
	// SDK models the same A2A security scope list as `[]`; normalize only that
	// cross-SDK representation after the original card passed Control Plane
	// endpoint and mTLS declaration checks.
	var normalized map[string]any
	if err := json.Unmarshal(encoded, &normalized); err != nil {
		return nil, planner.NewError(
			"invalid_worker_handle", "Worker Agent Card is invalid", false, err,
		)
	}
	normalizeEmptySecurityScopes(normalized)
	encoded, err = json.Marshal(normalized)
	if err != nil || len(encoded) > maxAgentCardBytes {
		return nil, planner.NewError(
			"invalid_worker_handle", "Worker Agent Card is invalid", false, err,
		)
	}
	var card sdk.AgentCard
	if err := json.Unmarshal(encoded, &card); err != nil {
		return nil, planner.NewError(
			"invalid_worker_handle", "Worker Agent Card is invalid", false, err,
		)
	}
	if len(card.SupportedInterfaces) != 1 || card.SupportedInterfaces[0] == nil {
		return nil, planner.NewError(
			"invalid_worker_handle", "Worker Agent Card has no exact A2A interface", false, nil,
		)
	}
	endpoint := card.SupportedInterfaces[0]
	parsed, parseErr := url.Parse(endpoint.URL)
	if parseErr != nil || parsed.Host == "" || parsed.User != nil || parsed.RawQuery != "" ||
		parsed.Fragment != "" || endpoint.ProtocolBinding != sdk.TransportProtocolJSONRPC ||
		endpoint.ProtocolVersion != sdk.Version || endpoint.Tenant != handle.AllocationID ||
		(requireHTTPS && parsed.Scheme != "https") ||
		(!requireHTTPS && parsed.Scheme != "http" && parsed.Scheme != "https") {
		return nil, planner.NewError(
			"invalid_worker_handle", "Worker Agent Card interface is incompatible", false, parseErr,
		)
	}
	return &card, nil
}

func normalizeEmptySecurityScopes(card map[string]any) {
	requirements, ok := card["securityRequirements"].([]any)
	if !ok {
		return
	}
	for _, rawRequirement := range requirements {
		requirement, ok := rawRequirement.(map[string]any)
		if !ok {
			continue
		}
		schemes, ok := requirement["schemes"].(map[string]any)
		if !ok {
			continue
		}
		for name, rawScopes := range schemes {
			if scopes, ok := rawScopes.(map[string]any); ok && len(scopes) == 0 {
				schemes[name] = []any{}
			}
		}
	}
}

func taskMessage(task *sdk.Task) (*sdk.Message, bool, error) {
	if task == nil || task.ID == "" {
		return nil, false, planner.NewError(
			"invalid_a2a_response", "Worker returned an invalid A2A Task", false, nil,
		)
	}
	switch task.Status.State {
	case sdk.TaskStateSubmitted, sdk.TaskStateWorking:
		return nil, true, nil
	case sdk.TaskStateInputRequired:
		return nil, false, planner.NewError(
			"worker_input_required", "Worker requires input unsupported by passthrough@1", false, nil,
		)
	case sdk.TaskStateAuthRequired:
		return nil, false, planner.NewError(
			"worker_auth_required", "Worker requires authentication unsupported by passthrough@1", false, nil,
		)
	case sdk.TaskStateCanceled:
		return nil, false, planner.NewError(
			"worker_task_cancelled", "Worker A2A Task was cancelled", true, nil,
		)
	case sdk.TaskStateRejected:
		return nil, false, planner.NewError(
			"worker_task_rejected", "Worker rejected the A2A Task", false, nil,
		)
	case sdk.TaskStateCompleted, sdk.TaskStateFailed:
		if task.Status.Message != nil {
			return task.Status.Message, false, nil
		}
		for index := len(task.History) - 1; index >= 0; index-- {
			if task.History[index] != nil && task.History[index].Role == sdk.MessageRoleAgent {
				return task.History[index], false, nil
			}
		}
		if task.Status.State == sdk.TaskStateFailed {
			return nil, false, planner.NewError(
				"worker_task_failed", "Worker A2A Task failed without a Contractor result", true, nil,
			)
		}
		return nil, false, nil
	default:
		return nil, false, planner.NewError(
			"invalid_a2a_response", "Worker returned an unknown A2A Task state", false, nil,
		)
	}
}

func decodeResultMessage(message *sdk.Message) (contracts.StageContentResult, error) {
	if message == nil || message.Role != sdk.MessageRoleAgent || len(message.Parts) != 1 ||
		message.Parts[0] == nil {
		return contracts.StageContentResult{}, planner.NewError(
			"invalid_a2a_response", "Worker response must be one Agent DataPart", false, nil,
		)
	}
	part := message.Parts[0]
	if part.MediaType != "" && part.MediaType != stageContentMediaType {
		return contracts.StageContentResult{}, planner.NewError(
			"invalid_a2a_response", "Worker response has an incompatible media type", false, nil,
		)
	}
	data, ok := part.Content.(sdk.Data)
	if !ok {
		return contracts.StageContentResult{}, planner.NewError(
			"invalid_a2a_response", "Worker response must be one Agent DataPart", false, nil,
		)
	}
	encoded, err := json.Marshal(data.Value)
	if err != nil || len(encoded) == 0 || len(encoded) > maxA2AResponseBytes {
		return contracts.StageContentResult{}, planner.NewError(
			"invalid_a2a_response", "Worker Contractor payload is invalid or oversized", false, err,
		)
	}
	result, err := contracts.DecodeStrict[contracts.StageContentResult](encoded)
	if err != nil {
		return contracts.StageContentResult{}, planner.NewError(
			"invalid_worker_result", "Worker returned an invalid StageContentResult", false, err,
		)
	}
	return result, nil
}

func transportError(ctx context.Context, cause error) *planner.Error {
	if errors.Is(ctx.Err(), context.DeadlineExceeded) || errors.Is(cause, context.DeadlineExceeded) {
		return planner.NewError(
			"worker_deadline_exceeded", "Worker invocation deadline expired", true, cause,
		)
	}
	if errors.Is(ctx.Err(), context.Canceled) || errors.Is(cause, context.Canceled) {
		return planner.NewError(
			"planner_cancelled", "Planner invocation was cancelled", true, cause,
		)
	}
	return planner.NewError(
		"worker_unavailable", "Worker A2A endpoint is unavailable", true, cause,
	)
}

func cloneBoundedHTTPClient(input *http.Client) *http.Client {
	result := *input
	base := input.Transport
	if base == nil {
		base = http.DefaultTransport
	}
	result.Transport = boundedRoundTripper{base: base, limit: maxA2AResponseBytes}
	result.CheckRedirect = func(_ *http.Request, _ []*http.Request) error {
		return errors.New("A2A redirects are not allowed")
	}
	return &result
}

type boundedRoundTripper struct {
	base  http.RoundTripper
	limit int64
}

func (t boundedRoundTripper) RoundTrip(request *http.Request) (*http.Response, error) {
	outgoing := request.Clone(request.Context())
	outgoing.Header = request.Header.Clone()
	outgoing.Header.Set(requestid.Header, requestid.Ensure(request.Context()))
	response, err := t.base.RoundTrip(outgoing)
	if err != nil {
		return nil, err
	}
	if response.ContentLength > t.limit {
		_ = response.Body.Close()
		return nil, fmt.Errorf("A2A response exceeds its limit")
	}
	response.Body = struct {
		io.Reader
		io.Closer
	}{Reader: io.LimitReader(response.Body, t.limit+1), Closer: response.Body}
	return response, nil
}
