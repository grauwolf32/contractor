package telemetry

import (
	"bytes"
	"context"
	"crypto/rand"
	"errors"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"runtime/debug"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/contracts"
	collectortracev1 "go.opentelemetry.io/proto/otlp/collector/trace/v1"
	commonv1 "go.opentelemetry.io/proto/otlp/common/v1"
	resourcev1 "go.opentelemetry.io/proto/otlp/resource/v1"
	tracev1 "go.opentelemetry.io/proto/otlp/trace/v1"
	"google.golang.org/protobuf/proto"
)

const (
	maxPlannerPendingSpans   = 2048
	maxPlannerPendingBytes   = 2 * 1024 * 1024
	maxPlannerResponseBytes  = 64 * 1024
	maxPlannerAttributeBytes = 256
	maxPlannerResourceValues = 65
)

var safePlannerIdentifier = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,255}$`)
var plannerHeaderName = regexp.MustCompile(`^[!#$%&'*+.^_` + "`" + `|~0-9A-Za-z-]+$`)

var allowedPlannerSpanNames = map[PlannerSpanName]struct{}{
	PlannerSpanInvocation: {}, PlannerSpanSession: {}, PlannerSpanModel: {},
	PlannerSpanWorker: {}, PlannerSpanSubtask: {}, PlannerSpanFinish: {}, PlannerSpanError: {},
}

var allowedPlannerOutcomes = map[string]struct{}{
	"cancelled": {}, "failed": {}, "recovered": {}, "rejected": {},
	"succeeded": {}, "unavailable": {},
}

type OTLPHTTPPlannerAdapterFactory struct{}

func NewOTLPHTTPPlannerAdapterFactory() *OTLPHTTPPlannerAdapterFactory {
	return &OTLPHTTPPlannerAdapterFactory{}
}

func (*OTLPHTTPPlannerAdapterFactory) Ref() string { return PlannerAdapterOTLPHTTP }

func (*OTLPHTTPPlannerAdapterFactory) Create(
	settings PlannerAdapterSettings,
) (PlannerTelemetry, error) {
	if err := validatePlannerAdapterSettings(settings); err != nil {
		return nil, err
	}
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.Proxy = nil
	transport.DisableCompression = true
	transport.MaxConnsPerHost = 1
	transport.MaxIdleConnsPerHost = 1
	client := &http.Client{
		Transport: transport,
		CheckRedirect: func(*http.Request, []*http.Request) error {
			return errors.New("OTLP redirects are disabled")
		},
	}
	traceID := make([]byte, 16)
	if _, err := rand.Read(traceID); err != nil {
		return nil, errors.New("Planner telemetry identity is unavailable")
	}
	secrets := make([]string, 0, len(settings.Headers)+1)
	secrets = append(secrets, settings.Endpoint)
	for _, value := range settings.Headers {
		secrets = append(secrets, value.Reveal())
	}
	result := &otlpHTTPPlannerTelemetry{
		endpoint: settings.Endpoint, headers: settings.Headers,
		flushTimeout: settings.FlushTimeout, resource: settings.Resource.clone(),
		client: client, ownedTransport: true, traceID: traceID,
		secretValues: secrets, spans: make([]*tracev1.Span, 0, 32),
	}
	result.pendingBytes = proto.Size(result.resourceProto())
	return result, nil
}

type otlpHTTPPlannerTelemetry struct {
	mu             sync.Mutex
	flushMu        sync.Mutex
	endpoint       string
	headers        map[string]contracts.SecretString
	flushTimeout   time.Duration
	resource       PlannerResource
	client         *http.Client
	ownedTransport bool
	traceID        []byte
	secretValues   []string
	spans          []*tracev1.Span
	pendingBytes   int
	queueOverflow  bool
	closed         bool
}

type otlpPlannerInstrumentation struct{ owner *otlpHTTPPlannerTelemetry }

type otlpPlannerSpan struct {
	owner      *otlpHTTPPlannerTelemetry
	name       PlannerSpanName
	attributes PlannerSpanAttributes
	startedAt  time.Time
	once       sync.Once
}

func (o *otlpHTTPPlannerTelemetry) Instrumentation() PlannerInstrumentation {
	return otlpPlannerInstrumentation{owner: o}
}

func (o *otlpHTTPPlannerTelemetry) FlushTimeout() time.Duration { return o.flushTimeout }

func (i otlpPlannerInstrumentation) StartSpan(
	name PlannerSpanName, attributes PlannerSpanAttributes,
) PlannerSpan {
	if _, ok := allowedPlannerSpanNames[name]; !ok {
		name = PlannerSpanError
	}
	return &otlpPlannerSpan{
		owner: i.owner, name: name, attributes: attributes, startedAt: time.Now(),
	}
}

func (s *otlpPlannerSpan) End(outcome string, attributes PlannerSpanAttributes) {
	s.once.Do(func() {
		mergePlannerSpanAttributes(&s.attributes, attributes)
		s.owner.enqueue(s.name, s.startedAt, time.Now(), outcome, s.attributes)
	})
}

func (o *otlpHTTPPlannerTelemetry) enqueue(
	name PlannerSpanName,
	startedAt time.Time,
	finishedAt time.Time,
	outcome string,
	attributes PlannerSpanAttributes,
) {
	if _, ok := allowedPlannerOutcomes[outcome]; !ok {
		outcome = "failed"
	}
	spanID := make([]byte, 8)
	if _, err := rand.Read(spanID); err != nil {
		return
	}
	if finishedAt.Before(startedAt) {
		finishedAt = startedAt
	}
	o.mu.Lock()
	if o.closed {
		o.mu.Unlock()
		return
	}
	traceID := append([]byte(nil), o.traceID...)
	secrets := append([]string(nil), o.secretValues...)
	o.mu.Unlock()
	values := plannerAttributeValues(attributes, secrets)
	values["outcome"] = outcome
	values["duration.ms"] = max(int64(0), finishedAt.Sub(startedAt).Milliseconds())
	status := tracev1.Status_STATUS_CODE_OK
	if outcome != "succeeded" && outcome != "recovered" {
		status = tracev1.Status_STATUS_CODE_ERROR
	}
	span := &tracev1.Span{
		TraceId: traceID, SpanId: spanID,
		Name: string(name), Kind: tracev1.Span_SPAN_KIND_INTERNAL,
		StartTimeUnixNano: uint64(startedAt.UnixNano()),
		EndTimeUnixNano:   uint64(finishedAt.UnixNano()),
		Attributes:        keyValues(values), Status: &tracev1.Status{Code: status},
	}
	size := proto.Size(span)
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.closed {
		return
	}
	if len(o.spans) >= maxPlannerPendingSpans || o.pendingBytes+size > maxPlannerPendingBytes {
		o.queueOverflow = true
		return
	}
	o.spans = append(o.spans, span)
	o.pendingBytes += size
}

func (o *otlpHTTPPlannerTelemetry) Flush(ctx context.Context) PlannerExportResult {
	o.flushMu.Lock()
	defer o.flushMu.Unlock()
	o.mu.Lock()
	if o.closed {
		o.mu.Unlock()
		return PlannerExportResult{}
	}
	overflow := o.queueOverflow
	o.queueOverflow = false
	if len(o.spans) == 0 {
		o.mu.Unlock()
		if overflow {
			return PlannerExportResult{Attempted: true, ErrorCode: "queue_overflow"}
		}
		return PlannerExportResult{}
	}
	request := &collectortracev1.ExportTraceServiceRequest{ResourceSpans: []*tracev1.ResourceSpans{{
		Resource: o.resourceProto(),
		ScopeSpans: []*tracev1.ScopeSpans{{
			Scope: &commonv1.InstrumentationScope{
				Name: "contractor.server.planner", Version: plannerServiceVersion(),
			},
			Spans: append([]*tracev1.Span(nil), o.spans...),
		}},
	}}}
	exportedCount := len(o.spans)
	payload, err := proto.Marshal(request)
	endpoint := o.endpoint
	headers := make(map[string]string, len(o.headers))
	for name, value := range o.headers {
		headers[name] = value.Reveal()
	}
	defer func() {
		for name := range headers {
			delete(headers, name)
		}
	}()
	client := o.client
	o.mu.Unlock()
	if err != nil || len(payload) > maxPlannerPendingBytes+128*1024 {
		return PlannerExportResult{Attempted: true, ErrorCode: "request_failed"}
	}
	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(payload))
	if err != nil {
		return PlannerExportResult{Attempted: true, ErrorCode: "request_failed"}
	}
	for name, value := range headers {
		httpRequest.Header.Set(name, value)
	}
	defer func() {
		for name := range httpRequest.Header {
			httpRequest.Header.Del(name)
		}
	}()
	httpRequest.Header.Set("Content-Type", "application/x-protobuf")
	httpRequest.Header.Set("Accept", "application/x-protobuf")
	response, err := client.Do(httpRequest)
	if err != nil {
		code := "delivery_failed"
		if errors.Is(ctx.Err(), context.DeadlineExceeded) {
			code = "flush_timeout"
		}
		return PlannerExportResult{Attempted: true, ErrorCode: code}
	}
	defer response.Body.Close()
	body, readErr := io.ReadAll(io.LimitReader(response.Body, maxPlannerResponseBytes+1))
	if readErr != nil || len(body) > maxPlannerResponseBytes || response.StatusCode < 200 || response.StatusCode >= 300 {
		return PlannerExportResult{Attempted: true, ErrorCode: "delivery_failed"}
	}
	if len(body) != 0 {
		var decoded collectortracev1.ExportTraceServiceResponse
		if proto.Unmarshal(body, &decoded) != nil {
			return PlannerExportResult{Attempted: true, ErrorCode: "delivery_failed"}
		}
	}
	o.mu.Lock()
	if o.closed {
		o.mu.Unlock()
		return PlannerExportResult{Attempted: true, ErrorCode: "delivery_failed"}
	}
	if exportedCount <= len(o.spans) {
		o.spans = append([]*tracev1.Span(nil), o.spans[exportedCount:]...)
	} else {
		o.spans = nil
	}
	o.pendingBytes = proto.Size(o.resourceProto())
	for _, span := range o.spans {
		o.pendingBytes += proto.Size(span)
	}
	o.mu.Unlock()
	if overflow {
		return PlannerExportResult{Attempted: true, ErrorCode: "queue_overflow"}
	}
	return PlannerExportResult{Attempted: true, Succeeded: true}
}

func (o *otlpHTTPPlannerTelemetry) Close() {
	o.flushMu.Lock()
	defer o.flushMu.Unlock()
	o.mu.Lock()
	if o.closed {
		o.mu.Unlock()
		return
	}
	o.closed = true
	for name := range o.headers {
		delete(o.headers, name)
	}
	o.endpoint = ""
	o.secretValues = nil
	o.traceID = nil
	o.spans = nil
	o.resource = PlannerResource{}
	client := o.client
	owned := o.ownedTransport
	o.client = nil
	o.mu.Unlock()
	if owned && client != nil {
		client.CloseIdleConnections()
	}
}

func (o *otlpHTTPPlannerTelemetry) resourceProto() *resourcev1.Resource {
	values := map[string]any{
		"service.name":                      "contractor-server-planner",
		"service.version":                   plannerServiceVersion(),
		"contractor.run.id":                 o.resource.RunID,
		"contractor.stage_execution.id":     o.resource.StageExecutionID,
		"contractor.planner.ref":            o.resource.PlannerRef,
		"contractor.model.alias":            o.resource.ModelAlias,
		"contractor.model_policy.ref":       o.resource.ModelPolicyRef,
		"contractor.llm_gateway.ref":        o.resource.LLMGatewayRef,
		"contractor.runtime.config_refs":    boundedStrings(o.resource.RuntimeConfigRefs, maxPlannerResourceValues),
		"contractor.runtime.config_digests": boundedStrings(o.resource.RuntimeConfigDigests, maxPlannerResourceValues),
		"contractor.run.labels":             boundedStrings(o.resource.RunLabels, maxPlannerResourceValues),
	}
	if o.resource.LLMCredentialID != "" {
		values["contractor.llm_credential.id"] = o.resource.LLMCredentialID
	}
	if o.resource.RuntimeCredentialID != "" {
		values["contractor.runtime_credential.id"] = o.resource.RuntimeCredentialID
	}
	return &resourcev1.Resource{Attributes: keyValues(safeMap(values, o.secretValues))}
}

func plannerServiceVersion() string {
	if build, ok := debug.ReadBuildInfo(); ok && build.Main.Version != "" && build.Main.Version != "(devel)" {
		if safePlannerIdentifier.MatchString(build.Main.Version) {
			return build.Main.Version
		}
	}
	return "devel"
}

func plannerAttributeValues(attributes PlannerSpanAttributes, secrets []string) map[string]any {
	result := make(map[string]any, 8)
	for key, value := range map[string]string{
		"operation.kind": attributes.Operation, "planner.session_id": attributes.SessionID,
		"model.alias": attributes.ModelAlias, "tool.name": attributes.ToolName,
		"worker.name": attributes.WorkerName, "subtask.id": attributes.SubtaskID,
		"error.type": attributes.ErrorCode,
	} {
		if safe := safePlannerString(value, secrets); safe != "" {
			result[key] = safe
		}
	}
	if attributes.PlanRevision != 0 {
		result["plan.revision"] = int64(attributes.PlanRevision)
	}
	return result
}

func safeMap(values map[string]any, secrets []string) map[string]any {
	result := make(map[string]any, len(values))
	for key, value := range values {
		switch current := value.(type) {
		case string:
			if safe := safePlannerString(current, secrets); safe != "" {
				result[key] = safe
			}
		case []string:
			safeValues := make([]string, 0, len(current))
			for _, item := range current {
				if safe := safePlannerString(item, secrets); safe != "" {
					safeValues = append(safeValues, safe)
				}
			}
			result[key] = safeValues
		case int64:
			result[key] = current
		}
	}
	return result
}

func safePlannerString(value string, secrets []string) string {
	if value == "" || !utf8.ValidString(value) || !safePlannerIdentifier.MatchString(value) {
		return ""
	}
	for _, secret := range secrets {
		if secret != "" && strings.Contains(value, secret) {
			return ""
		}
	}
	if len(value) <= maxPlannerAttributeBytes {
		return value
	}
	value = value[:maxPlannerAttributeBytes]
	for !utf8.ValidString(value) {
		value = value[:len(value)-1]
	}
	return value
}

func boundedStrings(values []string, maximum int) []string {
	values = sortedUnique(values)
	if len(values) > maximum {
		values = values[:maximum]
	}
	return values
}

func keyValues(values map[string]any) []*commonv1.KeyValue {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	result := make([]*commonv1.KeyValue, 0, len(keys))
	for _, key := range keys {
		var value *commonv1.AnyValue
		switch current := values[key].(type) {
		case string:
			value = &commonv1.AnyValue{Value: &commonv1.AnyValue_StringValue{StringValue: current}}
		case int64:
			value = &commonv1.AnyValue{Value: &commonv1.AnyValue_IntValue{IntValue: current}}
		case []string:
			items := make([]*commonv1.AnyValue, 0, len(current))
			for _, item := range current {
				items = append(items, &commonv1.AnyValue{Value: &commonv1.AnyValue_StringValue{StringValue: item}})
			}
			value = &commonv1.AnyValue{Value: &commonv1.AnyValue_ArrayValue{ArrayValue: &commonv1.ArrayValue{Values: items}}}
		}
		if value != nil {
			result = append(result, &commonv1.KeyValue{Key: key, Value: value})
		}
	}
	return result
}

func validatePlannerAdapterSettings(settings PlannerAdapterSettings) error {
	parsed, err := url.Parse(settings.Endpoint)
	if err != nil || parsed == nil || !parsed.IsAbs() || parsed.Opaque != "" ||
		parsed.Host == "" || parsed.Hostname() == "" || parsed.User != nil ||
		parsed.RawQuery != "" || parsed.ForceQuery || parsed.Fragment != "" || parsed.RawFragment != "" ||
		(parsed.Scheme != "http" && parsed.Scheme != "https") {
		return errors.New("Planner telemetry endpoint is invalid")
	}
	if settings.FlushTimeout < time.Second || settings.FlushTimeout > 10*time.Second {
		return errors.New("Planner telemetry flush timeout is invalid")
	}
	if settings.Resource.RunID == "" || settings.Resource.StageExecutionID == "" ||
		settings.Resource.PlannerRef == "" || len(settings.Headers) > 32 {
		return errors.New("Planner telemetry settings are incomplete")
	}
	seenHeaders := make(map[string]struct{}, len(settings.Headers))
	totalHeaderBytes := 0
	for name, value := range settings.Headers {
		lower := strings.ToLower(name)
		secret := value.Reveal()
		_, duplicate := seenHeaders[lower]
		seenHeaders[lower] = struct{}{}
		totalHeaderBytes += len(secret)
		if name == "" || len(name) > 64 || !plannerHeaderName.MatchString(name) ||
			secret == "" || len(secret) > 4*1024 || totalHeaderBytes > 16*1024 ||
			duplicate || forbiddenPlannerHeader(lower) || !validPlannerHeaderValue(secret) {
			return errors.New("Planner telemetry headers are invalid")
		}
	}
	return nil
}

func forbiddenPlannerHeader(name string) bool {
	switch name {
	case "accept", "connection", "content-length", "content-type", "forwarded", "host",
		"keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer",
		"transfer-encoding", "upgrade", "via", "x-real-ip":
		return true
	default:
		return strings.HasPrefix(name, "x-forwarded-") || strings.HasPrefix(name, "proxy-")
	}
}

func validPlannerHeaderValue(value string) bool {
	if !utf8.ValidString(value) {
		return false
	}
	for _, raw := range []byte(value) {
		if raw < 0x20 && raw != '\t' || raw == 0x7f {
			return false
		}
	}
	return true
}
