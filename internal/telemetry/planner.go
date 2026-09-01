// Package telemetry owns supplementary, content-free Server-side telemetry.
// It deliberately has no dependency on Planner implementations or global
// OpenTelemetry process state.
package telemetry

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
)

const PlannerAdapterOTLPHTTP = "otlp-http@1"

// PlannerSpanName is a closed vocabulary. Implementations replace an unknown
// value with PlannerSpanError rather than exporting caller-controlled names.
type PlannerSpanName string

const (
	PlannerSpanInvocation PlannerSpanName = "contractor.planner.invocation"
	PlannerSpanSession    PlannerSpanName = "contractor.planner.session"
	PlannerSpanModel      PlannerSpanName = "contractor.planner.model"
	PlannerSpanWorker     PlannerSpanName = "contractor.planner.worker"
	PlannerSpanSubtask    PlannerSpanName = "contractor.planner.subtask"
	PlannerSpanFinish     PlannerSpanName = "contractor.planner.finish"
	PlannerSpanError      PlannerSpanName = "contractor.planner.error"
)

// PlannerSpanAttributes intentionally cannot represent prompts, responses,
// objectives, instructions, tool arguments/results, artifact values, URLs or
// credentials. Exporters additionally bound and sanitize every string.
type PlannerSpanAttributes struct {
	Operation    string
	SessionID    string
	ModelAlias   string
	ToolName     string
	WorkerName   string
	SubtaskID    string
	ErrorCode    string
	PlanRevision uint64
}

type PlannerSpan interface {
	End(outcome string, attributes PlannerSpanAttributes)
}

type PlannerInstrumentation interface {
	StartSpan(PlannerSpanName, PlannerSpanAttributes) PlannerSpan
}

type noopInstrumentation struct{}
type noopSpan struct{}

func (noopInstrumentation) StartSpan(PlannerSpanName, PlannerSpanAttributes) PlannerSpan {
	return noopSpan{}
}
func (noopSpan) End(string, PlannerSpanAttributes) {}

func NoopPlannerInstrumentation() PlannerInstrumentation { return noopInstrumentation{} }

type PlannerResource struct {
	RunID                string
	StageExecutionID     string
	PlannerRef           string
	ModelAlias           string
	ModelPolicyRef       string
	LLMGatewayRef        string
	LLMCredentialID      string
	RuntimeCredentialID  string
	RuntimeConfigRefs    []string
	RuntimeConfigDigests []string
	RunLabels            []string
}

func (r PlannerResource) clone() PlannerResource {
	result := r
	result.RuntimeConfigRefs = append([]string(nil), r.RuntimeConfigRefs...)
	result.RuntimeConfigDigests = append([]string(nil), r.RuntimeConfigDigests...)
	result.RunLabels = append([]string(nil), r.RunLabels...)
	return result
}

type PlannerAdapterSettings struct {
	Endpoint     string
	Headers      map[string]contracts.SecretString
	FlushTimeout time.Duration
	Resource     PlannerResource
}

func (s PlannerAdapterSettings) clone() PlannerAdapterSettings {
	result := s
	result.Headers = make(map[string]contracts.SecretString, len(s.Headers))
	for name, value := range s.Headers {
		result.Headers[name] = value
	}
	result.Resource = s.Resource.clone()
	return result
}

type PlannerExportResult struct {
	Attempted bool
	Succeeded bool
	ErrorCode string
}

type PlannerTelemetry interface {
	Instrumentation() PlannerInstrumentation
	FlushTimeout() time.Duration
	Flush(context.Context) PlannerExportResult
	Close()
}

type PlannerAdapterFactory interface {
	Ref() string
	Create(PlannerAdapterSettings) (PlannerTelemetry, error)
}

// PlannerAdapterRegistry is immutable after construction and is shared by
// RuntimeConfig publication validation and Scheduler invocation construction.
type PlannerAdapterRegistry struct {
	factories map[string]PlannerAdapterFactory
}

func NewPlannerAdapterRegistry(factories ...PlannerAdapterFactory) (*PlannerAdapterRegistry, error) {
	if len(factories) == 0 {
		return nil, errors.New("Planner telemetry adapter registry is empty")
	}
	result := &PlannerAdapterRegistry{factories: make(map[string]PlannerAdapterFactory, len(factories))}
	for _, factory := range factories {
		if factory == nil || factory.Ref() == "" {
			return nil, errors.New("Planner telemetry adapter factory is invalid")
		}
		if _, duplicate := result.factories[factory.Ref()]; duplicate {
			return nil, fmt.Errorf("duplicate Planner telemetry adapter ref %q", factory.Ref())
		}
		result.factories[factory.Ref()] = factory
	}
	return result, nil
}

func NewBuiltinPlannerAdapterRegistry() (*PlannerAdapterRegistry, error) {
	return NewPlannerAdapterRegistry(NewOTLPHTTPPlannerAdapterFactory())
}

func (r *PlannerAdapterRegistry) SupportsPlannerTelemetryAdapter(ref string) bool {
	if r == nil {
		return false
	}
	_, ok := r.factories[ref]
	return ok
}

func (r *PlannerAdapterRegistry) Create(
	ref string, settings PlannerAdapterSettings,
) (PlannerTelemetry, error) {
	if r == nil {
		return nil, errors.New("Planner telemetry adapter registry is unavailable")
	}
	factory, ok := r.factories[ref]
	if !ok {
		return nil, errors.New("Planner telemetry adapter is unavailable")
	}
	return factory.Create(settings.clone())
}

func sortedUnique(values []string) []string {
	result := append([]string(nil), values...)
	sort.Strings(result)
	write := 0
	for _, value := range result {
		if write != 0 && result[write-1] == value {
			continue
		}
		result[write] = value
		write++
	}
	return result[:write]
}

func mergePlannerSpanAttributes(target *PlannerSpanAttributes, source PlannerSpanAttributes) {
	if source.Operation != "" {
		target.Operation = source.Operation
	}
	if source.SessionID != "" {
		target.SessionID = source.SessionID
	}
	if source.ModelAlias != "" {
		target.ModelAlias = source.ModelAlias
	}
	if source.ToolName != "" {
		target.ToolName = source.ToolName
	}
	if source.WorkerName != "" {
		target.WorkerName = source.WorkerName
	}
	if source.SubtaskID != "" {
		target.SubtaskID = source.SubtaskID
	}
	if source.ErrorCode != "" {
		target.ErrorCode = source.ErrorCode
	}
	if source.PlanRevision != 0 {
		target.PlanRevision = source.PlanRevision
	}
}
