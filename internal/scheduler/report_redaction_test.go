package scheduler

import (
	"context"
	"encoding/json"
	"errors"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

type normalizingReportStore struct{ *memorySchedulerStore }

func (s *normalizingReportStore) RecordStageExecutionReport(
	ctx context.Context, params runstore.RecordStageExecutionReportParams,
) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	report, err := telemetry.NewPolicy(params.Secrets...).NormalizeAllocationReport(params.Report)
	if err != nil {
		return err
	}
	params.Report = report
	params.Secrets = nil
	return s.memorySchedulerStore.RecordStageExecutionReport(ctx, params)
}

func TestReportPersistenceReresolvesAllocationCredentialsAcrossSchedulers(t *testing.T) {
	harness := newSchedulerHarness(t)
	const (
		llmSecret        = "llm-report-secret"
		otlpSecret       = "otlp-report-secret"
		proxySecret      = "proxy-report-secret"
		caidoSecret      = "caido-report-secret"
		originUserSecret = "origin-report-username"
		originPassSecret = "origin-report-password"
		staticSecret     = "static-report-secret"
	)
	harness.scheduler.options.Credentials = credentialResolverFunc(func(
		context.Context, contracts.LLMCredentialRef, contracts.LLMGatewayConfigRef,
	) (contracts.SecretString, error) {
		return contracts.NewSecretString(llmSecret), nil
	})
	type runtimeMaterial struct {
		kind     contracts.RuntimeCredentialKind
		document string
	}
	materials := map[string]runtimeMaterial{
		"otlp":   {contracts.RuntimeCredentialOTLPHeaders, `{"headers":{"x-otlp-token":"` + otlpSecret + `"}}`},
		"proxy":  {contracts.RuntimeCredentialProxyBearer, `{"token":"` + proxySecret + `"}`},
		"caido":  {contracts.RuntimeCredentialCaidoBearer, `{"token":"` + caidoSecret + `"}`},
		"origin": {contracts.RuntimeCredentialOriginBasic, `{"username":"` + originUserSecret + `","password":"` + originPassSecret + `"}`},
	}
	harness.scheduler.options.RuntimeCredentials = runtimeCredentialResolverFunc(func(
		_ context.Context,
		credentialID string,
		_ []contracts.RuntimeCredentialKind,
		consumer func(contracts.RuntimeCredentialKind, []byte) error,
	) error {
		material, ok := materials[credentialID]
		if !ok {
			return errors.New("unknown report credential")
		}
		return consumer(material.kind, []byte(material.document))
	})
	harness.scheduler.options.TelemetrySecrets = []string{staticSecret}

	stage := harness.workflow.Stages[harness.workflow.EntryStage]
	binding := stage.Agents["builder"]
	binding.Template.Toolsets = append(binding.Template.Toolsets, contracts.ToolsetSelection{
		Ref: contracts.ToolsetRef{ToolsetID: "http-tools", Version: "1"}, Tools: []string{"http_request"},
	})
	stage.Agents["builder"] = binding
	reservations := schedulerTestReservations(t, stage)
	if len(reservations) != 1 {
		t.Fatalf("reservations = %d", len(reservations))
	}
	reservation := &reservations[0]
	reservation.Grant.AllocationID = "allocation-report-redaction"
	resolved := reservation.ResolvedRuntimeConfig
	resolved.WorkerTelemetry = &runtimeconfig.TelemetryConfig{
		Adapter: string(contracts.RuntimeAdapterOTLPHTTP), Endpoint: "https://otlp.example/v1/traces",
		Credential: "otlp", FlushTimeoutSeconds: 1,
	}
	resolved.HTTPProxy = &runtimeconfig.HTTPProxyConfig{
		Adapter: string(contracts.RuntimeAdapterHTTPProxy), ProxyURL: "https://proxy.example",
		Credential: "proxy", Targets: []string{string(contracts.ProxyTargetToolHTTP)},
	}
	resolved.Caido = &runtimeconfig.CaidoConfig{
		Adapter: string(contracts.RuntimeAdapterCaidoGraphQL), Endpoint: "https://caido.example/api",
		Credential: "caido", RequestTimeoutSeconds: 5,
	}
	run := harness.store.run
	run.ProjectHTTPTarget = &contracts.HTTPOriginTargetRef{
		URL: "https://target.example/api",
		Credential: &contracts.RuntimeCredentialRef{
			CredentialID: "origin", Kind: contracts.RuntimeCredentialOriginBasic,
		},
	}
	prepared, err := harness.scheduler.materializeRuntimeSettings(t.Context(), resolved.Clone())
	if err != nil {
		t.Fatal(err)
	}
	settings := map[string]contracts.WorkerExecutionSettings{
		"builder": {RuntimeSettings: prepared},
	}
	clearWorkerExecutionSettings(settings)
	if settings["builder"].RuntimeSettings.LLMGatewayToken != nil {
		t.Fatal("prepared credential was retained after Worker preparation")
	}

	execution := runstore.StageExecution{
		StageExecutionID: "stage-report-redaction", CreatedAt: harness.clock.now,
	}
	harness.store.allocations = []runstore.StageAllocation{{
		AllocationID: reservation.Grant.AllocationID, StageExecutionID: execution.StageExecutionID,
		LogicalAgentName: "builder",
	}}
	report := schedulerTestAllocationReport(reservation.Grant.AllocationID, harness.clock.now)
	secretValues := []string{
		llmSecret, otlpSecret, proxySecret, caidoSecret, originUserSecret, originPassSecret, staticSecret,
	}
	for index, secret := range secretValues {
		report.Worker.ToolCalls = append(report.Worker.ToolCalls, contracts.ToolCallRecord{
			CallID: "secret-call-" + strconv.Itoa(index), Tool: "probe",
			Arguments: map[string]any{"detail": "before " + secret + " after"},
			Outcome:   contracts.ToolCallSucceeded,
		})
		report.Worker.Errors = append(report.Worker.Errors, contracts.ExecutionError{
			Code: "provider_error", Message: "before " + secret + " after",
		})
	}

	store := &normalizingReportStore{harness.store}
	other, err := New(
		store, harness.persistence, harness.artifacts, harness.allocator, harness.workers,
		harness.planners, harness.scheduler.options,
	)
	if err != nil {
		t.Fatal(err)
	}
	other.persistReports(t.Context(), run, executableWorkflow{stage: stage}, execution, reservations,
		map[string]contracts.AllocationFinalReport{"builder": report})
	if len(harness.store.reports) != 1 {
		t.Fatalf("persisted report count = %d", len(harness.store.reports))
	}
	stored := harness.store.reports[0].Report
	encoded, err := json.Marshal(stored)
	if err != nil {
		t.Fatal(err)
	}
	for index, secret := range secretValues {
		if strings.Contains(string(encoded), secret) {
			t.Fatalf("credential %d remained in persisted report", index)
		}
		if got := stored.Worker.ToolCalls[index].Arguments["detail"]; got != "before [REDACTED] after" {
			t.Fatalf("tool argument %d = %v", index, got)
		}
		if got := stored.Worker.Errors[index].Message; got != "before [REDACTED] after" {
			t.Fatalf("error message %d = %q", index, got)
		}
	}
}

func TestReportPersistenceContinuesWhenCredentialResolutionFails(t *testing.T) {
	harness := newSchedulerHarness(t)
	harness.scheduler.options.OperationTimeout = 20 * time.Millisecond
	reservations := schedulerTestReservations(t, harness.workflow.Stages[harness.workflow.EntryStage])
	reservations[0].Grant.AllocationID = "allocation-unresolved-report"
	harness.store.allocations = []runstore.StageAllocation{{
		AllocationID:     reservations[0].Grant.AllocationID,
		StageExecutionID: "stage-unresolved-report", LogicalAgentName: "builder",
	}}
	harness.scheduler.options.Credentials = credentialResolverFunc(func(
		ctx context.Context, _ contracts.LLMCredentialRef, _ contracts.LLMGatewayConfigRef,
	) (contracts.SecretString, error) {
		<-ctx.Done()
		return contracts.SecretString{}, ctx.Err()
	})
	store := &normalizingReportStore{harness.store}
	other, err := New(store, harness.persistence, harness.artifacts, harness.allocator,
		harness.workers, harness.planners, harness.scheduler.options)
	if err != nil {
		t.Fatal(err)
	}
	execution := runstore.StageExecution{
		StageExecutionID: "stage-unresolved-report", CreatedAt: harness.clock.now,
	}
	other.persistReports(t.Context(), harness.store.run,
		executableWorkflow{stage: harness.workflow.Stages[harness.workflow.EntryStage]},
		execution, reservations, map[string]contracts.AllocationFinalReport{
			"builder": schedulerTestAllocationReport(reservations[0].Grant.AllocationID, harness.clock.now),
		})
	if len(harness.store.reports) != 1 {
		t.Fatalf("credential failure blocked report persistence: reports = %d", len(harness.store.reports))
	}
}
