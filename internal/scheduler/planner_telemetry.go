package scheduler

import (
	"context"
	"errors"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

// Planner telemetry uses pinned Run provenance; allocation and claim ownership
// remain with Scheduler progression.
func (s *Scheduler) newPlannerTelemetry(
	ctx context.Context,
	run runstore.WorkflowRun,
	execution runstore.StageExecution,
	reservations []controlplane.Reservation,
	plannerRef string,
	modelAccess *planner.ModelAccess,
) telemetry.PlannerTelemetry {
	if s.options.PlannerTelemetry == nil || len(reservations) == 0 {
		return nil
	}
	selected := reservations[0].ResolvedRuntimeConfig
	for _, reservation := range reservations {
		if reservation.ResolvedRuntimeConfig != nil &&
			reservation.ResolvedRuntimeConfig.PlannerTelemetry != nil &&
			!plannerTelemetryOriginAllowed(reservation.ResolvedRuntimeConfig.Origins.PlannerTelemetry) {
			s.options.Logger.Warn(
				"Planner telemetry rejected a non-Run origin",
				"stage_execution_id", execution.StageExecutionID,
			)
			return nil
		}
	}
	for _, reservation := range reservations[1:] {
		if !samePlannerTelemetrySelection(selected, reservation.ResolvedRuntimeConfig) {
			s.options.Logger.Warn(
				"Planner telemetry selection differs across allocations",
				"stage_execution_id", execution.StageExecutionID,
			)
			return nil
		}
	}
	if selected == nil || selected.PlannerTelemetry == nil {
		return nil
	}
	configuration := *selected.PlannerTelemetry
	headers := make(map[string]contracts.SecretString)
	if configuration.Credential != "" {
		if selected.PlannerRuntimeCredential == nil ||
			selected.PlannerRuntimeCredential.CredentialID != configuration.Credential ||
			selected.PlannerRuntimeCredential.Kind != contracts.RuntimeCredentialOTLPHeaders {
			s.options.Logger.Warn(
				"Planner telemetry credential provenance is unavailable",
				"stage_execution_id", execution.StageExecutionID,
			)
			return nil
		}
		err := s.useRuntimeCredential(
			ctx, configuration.Credential,
			[]contracts.RuntimeCredentialKind{contracts.RuntimeCredentialOTLPHeaders},
			func(kind contracts.RuntimeCredentialKind, plaintext []byte) error {
				var material struct {
					Headers map[string]string `json:"headers"`
				}
				if kind != contracts.RuntimeCredentialOTLPHeaders || decodeRuntimeCredential(plaintext, &material) != nil {
					return errors.New("invalid OTLP credential material")
				}
				for name, value := range material.Headers {
					headers[name] = contracts.NewSecretString(value)
				}
				return nil
			},
		)
		if err != nil {
			for name := range headers {
				delete(headers, name)
			}
			s.options.Logger.Warn(
				"Planner telemetry credential is unavailable",
				"stage_execution_id", execution.StageExecutionID,
			)
			return nil
		}
	}
	resource := telemetry.PlannerResource{
		RunID: run.RunID, StageExecutionID: execution.StageExecutionID, PlannerRef: plannerRef,
		RuntimeCredentialID:  configuration.Credential,
		RuntimeConfigRefs:    plannerRuntimeConfigRefs(run.RuntimeConfig),
		RuntimeConfigDigests: plannerRuntimeConfigDigests(run.RuntimeConfig),
		RunLabels:            run.RuntimeConfig.ExplicitLabels(),
	}
	if modelAccess != nil {
		resource.ModelAlias = modelAccess.ModelPolicy.Model
		resource.ModelPolicyRef = modelAccess.ModelPolicy.Ref.PolicyID + "@" + modelAccess.ModelPolicy.Ref.Version
		resource.LLMGatewayRef = modelAccess.LLMGateway.Ref.GatewayID + "@" + modelAccess.LLMGateway.Ref.Version
		if modelAccess.Credential != nil {
			resource.LLMCredentialID = modelAccess.Credential.CredentialID
		}
	}
	created, err := s.options.PlannerTelemetry.Create(configuration.Adapter, telemetry.PlannerAdapterSettings{
		Endpoint: configuration.Endpoint, Headers: headers,
		CaptureContent: configuration.CaptureContent,
		FlushTimeout:   time.Duration(configuration.FlushTimeoutSeconds) * time.Second,
		Resource:       resource, RunMetadataLabels: run.MetadataLabels.Clone(),
	})
	for name := range headers {
		delete(headers, name)
	}
	if err != nil {
		s.options.Logger.Warn(
			"Planner telemetry adapter could not be created",
			"stage_execution_id", execution.StageExecutionID,
		)
		return nil
	}
	return created
}

func plannerTelemetryOriginAllowed(origin *runtimeconfig.RuntimeFieldOrigin) bool {
	return origin != nil && (origin.Layer == runtimeconfig.LayerDefault || origin.Layer == runtimeconfig.LayerRunLabels)
}

func samePlannerTelemetrySelection(
	left, right *runtimeconfig.ResolvedRuntimeConfig,
) bool {
	if left == nil || right == nil {
		return left == right
	}
	if (left.PlannerTelemetry == nil) != (right.PlannerTelemetry == nil) ||
		(left.PlannerRuntimeCredential == nil) != (right.PlannerRuntimeCredential == nil) {
		return false
	}
	if left.PlannerTelemetry != nil && *left.PlannerTelemetry != *right.PlannerTelemetry {
		return false
	}
	return left.PlannerRuntimeCredential == nil ||
		*left.PlannerRuntimeCredential == *right.PlannerRuntimeCredential
}

func plannerRuntimeConfigRefs(snapshot runtimeconfig.RunSnapshot) []string {
	result := make([]string, 0, len(snapshot.Labels)+1)
	result = append(result, snapshot.Default.Config.Name+"@"+snapshot.Default.Config.Version)
	for _, pinned := range snapshot.Labels {
		result = append(result, pinned.Config.Name+"@"+pinned.Config.Version)
	}
	return result
}

func plannerRuntimeConfigDigests(snapshot runtimeconfig.RunSnapshot) []string {
	result := make([]string, 0, len(snapshot.Labels)+1)
	result = append(result, snapshot.Default.Config.Digest)
	for _, pinned := range snapshot.Labels {
		result = append(result, pinned.Config.Digest)
	}
	return result
}

func (s *Scheduler) flushPlannerTelemetry(
	ctx context.Context,
	instance telemetry.PlannerTelemetry,
	stageDeadline time.Time,
	stageExecutionID string,
) *telemetry.PlannerExportResult {
	if instance == nil {
		return nil
	}
	bound := instance.FlushTimeout()
	if s.options.FinalizationTimeout < bound {
		bound = s.options.FinalizationTimeout
	}
	if remaining := stageDeadline.Sub(s.now()); remaining < bound {
		bound = remaining
	}
	result := telemetry.PlannerExportResult{Attempted: true, ErrorCode: "flush_timeout"}
	if bound > 0 {
		flushContext, cancel := context.WithTimeout(ctx, bound)
		result = normalizePlannerExportResult(instance.Flush(flushContext))
		cancel()
	}
	if result.Attempted && !result.Succeeded {
		s.options.Logger.Warn(
			"Planner telemetry export failed",
			"stage_execution_id", stageExecutionID,
			"error_code", result.ErrorCode,
		)
	}
	return &result
}

func normalizePlannerExportResult(result telemetry.PlannerExportResult) telemetry.PlannerExportResult {
	if !result.Attempted {
		return telemetry.PlannerExportResult{}
	}
	if result.Succeeded {
		return telemetry.PlannerExportResult{Attempted: true, Succeeded: true}
	}
	switch result.ErrorCode {
	case "delivery_failed", "flush_timeout", "queue_overflow", "request_failed":
		return result
	default:
		return telemetry.PlannerExportResult{Attempted: true, ErrorCode: "request_failed"}
	}
}
