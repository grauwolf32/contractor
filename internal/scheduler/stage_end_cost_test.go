package scheduler

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

// The Stage-end tests (SCH-R178, SCH-R51) count each credential resolution
// and Runtime credential decryption by credential ID, and record every report
// write and StageMetrics rebuild in order, so a regression that resolves the
// credentials again or rebuilds twice in one phase fails on the count.

const (
	stageEndWorkerCredential = "development-worker"
	stageEndOTLPCredential   = "stage-end-otlp"
	stageEndLLMSecret        = "stage-end-llm-secret-value"
	stageEndOTLPSecret       = "stage-end-otlp-secret-value"
)

var stageEndSecrets = []string{stageEndLLMSecret, stageEndOTLPSecret}

type stageEndCredentials struct {
	llm, runtime map[string]int
	failLLM      bool
}

func (c *stageEndCredentials) ResolveLLMCredential(
	_ context.Context, ref llmgateway.LLMCredentialRef, _ llmgateway.LLMGatewayConfigRef,
) (contracts.SecretString, error) {
	c.llm[ref.CredentialID]++
	if c.failLLM {
		return contracts.SecretString{}, credentials.ErrNotFound
	}
	if ref.CredentialID != stageEndWorkerCredential {
		return contracts.NewSecretString("stage-end-planner-token"), nil
	}
	return contracts.NewSecretString(stageEndLLMSecret), nil
}

func (c *stageEndCredentials) UsePlaintext(
	_ context.Context, credentialID string, _ []runtimesettings.RuntimeCredentialKind,
	consumer func(runtimesettings.RuntimeCredentialKind, []byte) error,
) error {
	c.runtime[credentialID]++
	return consumer(runtimesettings.RuntimeCredentialOTLPHeaders,
		[]byte(`{"headers":{"x-otlp-token":"`+stageEndOTLPSecret+`"}}`))
}

// stageEndReports records the order of report writes and StageMetrics
// rebuilds in front of another ReportStore.
type stageEndReports struct {
	inner ReportStore
	ops   []string
}

func (r *stageEndReports) RecordAllocationReport(ctx context.Context, envelope telemetry.AllocationReportEnvelope) error {
	r.ops = append(r.ops, "allocation_report")
	return r.inner.RecordAllocationReport(ctx, envelope)
}

func (r *stageEndReports) RecordPlannerReport(ctx context.Context, envelope telemetry.PlannerReportEnvelope) error {
	r.ops = append(r.ops, "planner_report")
	return r.inner.RecordPlannerReport(ctx, envelope)
}

func (r *stageEndReports) RebuildStageMetrics(
	ctx context.Context, stageExecutionID, schemaVersion string,
) (telemetry.StageMetricsRecord, error) {
	r.ops = append(r.ops, "rebuild")
	return r.inner.RebuildStageMetrics(ctx, stageExecutionID, schemaVersion)
}

// stageEndWorkers returns, from finalize and abort, a complete report for
// every live allocation whose tool calls and errors carry the Worker's secret
// values, as a Worker that echoed its credentials would. beforeFinalize, when
// set, observes the Stage when its finalization begins.
type stageEndWorkers struct {
	*memoryWorkers
	secrets        []string
	beforeFinalize func()
}

func (w stageEndWorkers) FinalizeAll(
	ctx context.Context, reservations []controlplane.Reservation, finalizationID string, deadline time.Time,
) (map[string]reporting.AllocationFinalReport, error) {
	if w.beforeFinalize != nil {
		w.beforeFinalize()
	}
	reports, err := w.memoryWorkers.FinalizeAll(ctx, reservations, finalizationID, deadline)
	for name, report := range reports {
		reports[name] = withStageEndSecrets(report, w.secrets)
	}
	return reports, err
}

func (w stageEndWorkers) AbortAll(
	ctx context.Context, reservations []controlplane.Reservation, abortID string,
	reason contracts.TerminationError, deadline time.Time,
) (map[string]reporting.AllocationFinalReport, error) {
	if _, err := w.memoryWorkers.AbortAll(ctx, reservations, abortID, reason, deadline); err != nil {
		return nil, err
	}
	reports := make(map[string]reporting.AllocationFinalReport, len(reservations))
	for _, reservation := range reservations {
		reports[reservation.Grant.LogicalAgentName] = withStageEndSecrets(
			schedulerTestAllocationReport(reservation.Grant.AllocationID, w.clock.now), w.secrets,
		)
	}
	return reports, nil
}

func withStageEndSecrets(report reporting.AllocationFinalReport, secrets []string) reporting.AllocationFinalReport {
	for index, secret := range secrets {
		report.Worker.ToolCalls = append(report.Worker.ToolCalls, reporting.ToolCallRecord{
			CallID: "secret-call-" + strconv.Itoa(index), Tool: "probe",
			Arguments: map[string]any{"detail": "before " + secret + " after"},
			Outcome:   reporting.ToolCallSucceeded,
		})
		report.Worker.Errors = append(report.Worker.Errors, reporting.ExecutionError{
			Code: "provider_error", Message: "before " + secret + " after",
		})
	}
	return report
}

// requireStageEndRedaction checks the exact redacted text of every secret the
// Worker echoed in a persisted report.
func requireStageEndRedaction(t *testing.T, report reporting.AllocationFinalReport, secrets []string) {
	t.Helper()
	encoded, err := json.Marshal(report)
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Worker.ToolCalls) != len(secrets) || len(report.Worker.Errors) != len(secrets) {
		t.Fatalf("persisted report = %s", encoded)
	}
	for index, secret := range secrets {
		if strings.Contains(string(encoded), secret) {
			t.Fatalf("secret %d remained in the persisted report: %s", index, encoded)
		}
		if got := report.Worker.ToolCalls[index].Arguments["detail"]; got != "before [REDACTED] after" {
			t.Fatalf("tool argument %d = %v", index, got)
		}
		if got := report.Worker.Errors[index].Message; got != "before [REDACTED] after" {
			t.Fatalf("error message %d = %q", index, got)
		}
	}
}

// newStageEndHarness returns a harness whose Worker uses an LLM credential
// and an OTLP header credential, whose Runtime reports echo both secrets and
// whose report store normalizes like PostgreSQL and records its operations.
func newStageEndHarness(t *testing.T) (*schedulerHarness, *stageEndCredentials, *stageEndReports) {
	t.Helper()
	harness := newSchedulerHarness(t)
	selection := harness.workflow.Stages[harness.workflow.EntryStage].ExecutionConfig.Agents["builder"]
	resolved, err := schedulerTestResolvedWorkerConfig(selection, false)
	if err != nil {
		t.Fatal(err)
	}
	resolved.WorkerTelemetry = &runtimeconfig.TelemetryConfig{
		Adapter: string(runtimesettings.RuntimeAdapterOTLPHTTP), Endpoint: "https://otlp.example/v1/traces",
		Credential: stageEndOTLPCredential, FlushTimeoutSeconds: 1,
	}
	harness.allocator.resolvedRuntimeConfig = &resolved
	stageCredentials := &stageEndCredentials{llm: map[string]int{}, runtime: map[string]int{}}
	reports := &stageEndReports{inner: &normalizingReportStore{harness.store}}
	options := harness.scheduler.options
	options.Credentials, options.RuntimeCredentials, options.Reports = stageCredentials, stageCredentials, reports
	scheduler, err := New(harness.store, harness.persistence, harness.artifacts, harness.allocator,
		stageEndWorkers{memoryWorkers: harness.workers, secrets: stageEndSecrets}, harness.planners, options)
	if err != nil {
		t.Fatal(err)
	}
	harness.scheduler = scheduler
	return harness, stageCredentials, reports
}

func requireNoPreparedSecrets(t *testing.T, s *Scheduler) {
	t.Helper()
	s.preparedSecrets.mu.Lock()
	defer s.preparedSecrets.mu.Unlock()
	if s.preparedSecrets.order.Len() != 0 || len(s.preparedSecrets.entries) != 0 {
		t.Fatalf("preparation secrets outlived the Stage lifecycle: %d Stages", s.preparedSecrets.order.Len())
	}
}

// An uninterrupted Stage decrypts each credential once, at preparation; its
// report persistence redacts with those values. The Planner completion and
// the final persistence each rebuild StageMetrics once.
func TestUninterruptedStageRedactsReportsWithPreparationSecrets(t *testing.T) {
	harness, stageCredentials, reports := newStageEndHarness(t)
	var duringPlanner int
	onRun := harness.planners.onRun
	harness.planners.onRun = func() {
		onRun()
		harness.scheduler.preparedSecrets.mu.Lock()
		duringPlanner = harness.scheduler.preparedSecrets.order.Len()
		harness.scheduler.preparedSecrets.mu.Unlock()
	}

	worked, err := harness.scheduler.RunOnce(t.Context())
	if err != nil || !worked || harness.store.run.State != runstore.RunSucceeded {
		t.Fatalf("RunOnce = (%v, %v), run=%s", worked, err, harness.store.run.State)
	}
	if duringPlanner != 1 {
		t.Fatalf("Stages with preparation secrets during the Planner = %d, want 1", duringPlanner)
	}
	if stageCredentials.llm[stageEndWorkerCredential] != 1 || stageCredentials.runtime[stageEndOTLPCredential] != 1 {
		t.Fatalf("credential resolutions = LLM %v, Runtime %v; want one each, at preparation",
			stageCredentials.llm, stageCredentials.runtime)
	}
	want := []string{"planner_report", "rebuild", "planner_report", "allocation_report", "rebuild"}
	if !slices.Equal(reports.ops, want) {
		t.Fatalf("report operations = %v, want Planner completion %v and final persistence %v",
			reports.ops, want[:2], want[2:])
	}
	if len(harness.store.reports) != 1 {
		t.Fatalf("persisted allocation reports = %d", len(harness.store.reports))
	}
	requireStageEndRedaction(t, harness.store.reports[0].Report, stageEndSecrets)
	requireNoPreparedSecrets(t, harness.scheduler)
}

// Credentials that become unavailable after preparation do not weaken the
// redaction of the Worker's report: it redacts the values the Worker received
// without resolving them again.
func TestPreparationSecretsRedactReportsAfterCredentialLoss(t *testing.T) {
	harness, stageCredentials, _ := newStageEndHarness(t)
	onRun := harness.planners.onRun
	harness.planners.onRun = func() {
		onRun()
		stageCredentials.failLLM = true
	}

	worked, err := harness.scheduler.RunOnce(t.Context())
	if err != nil || !worked || harness.store.run.State != runstore.RunSucceeded {
		t.Fatalf("RunOnce = (%v, %v), run=%s", worked, err, harness.store.run.State)
	}
	if stageCredentials.llm[stageEndWorkerCredential] != 1 {
		t.Fatalf("Worker LLM credential resolutions = %d, want 1", stageCredentials.llm[stageEndWorkerCredential])
	}
	requireStageEndRedaction(t, harness.store.reports[0].Report, stageEndSecrets)
	requireNoPreparedSecrets(t, harness.scheduler)
}

// A credential that cannot be resolved at preparation keeps nothing: the
// Stage aborts, its report persistence attempts the resolution itself as
// before, and the report is still persisted.
func TestPreparationCredentialFailureKeepsNoSecrets(t *testing.T) {
	harness, stageCredentials, reports := newStageEndHarness(t)
	stageCredentials.failLLM = true

	if _, err := harness.scheduler.RunOnce(t.Context()); err != nil {
		t.Fatal(err)
	}
	stage := harness.store.stages[0]
	if stage.Termination == nil || stage.Termination.Code != "worker_execution_config_unavailable" {
		t.Fatalf("Stage = %+v, want the preparation credential failure", stage)
	}
	if stageCredentials.llm[stageEndWorkerCredential] != 2 {
		t.Fatalf("Worker LLM credential resolutions = %d, want preparation and report persistence",
			stageCredentials.llm[stageEndWorkerCredential])
	}
	if !slices.Equal(reports.ops, []string{"allocation_report", "rebuild"}) || len(harness.store.reports) != 1 {
		t.Fatalf("report operations = %v, reports = %d", reports.ops, len(harness.store.reports))
	}
	requireNoPreparedSecrets(t, harness.scheduler)
}

// A failed report write changes neither the Stage outcome nor the single
// rebuild of the phase, and the Stage's preparation secrets still end with
// its lifecycle.
func TestReportWriteFailureKeepsStageEndGuarantees(t *testing.T) {
	harness, stageCredentials, reports := newStageEndHarness(t)
	harness.store.reportError = errors.New("telemetry database unavailable")

	worked, err := harness.scheduler.RunOnce(t.Context())
	if err != nil || !worked || harness.store.run.State != runstore.RunSucceeded ||
		harness.store.stages[0].State != runstore.StageSucceeded {
		t.Fatalf("RunOnce = (%v, %v), run=%s", worked, err, harness.store.run.State)
	}
	if stageCredentials.llm[stageEndWorkerCredential] != 1 {
		t.Fatalf("Worker LLM credential resolutions = %d, want 1", stageCredentials.llm[stageEndWorkerCredential])
	}
	want := []string{"planner_report", "rebuild", "planner_report", "allocation_report", "rebuild"}
	if !slices.Equal(reports.ops, want) || len(harness.store.reports) != 0 {
		t.Fatalf("report operations = %v, reports = %d; want %v", reports.ops, len(harness.store.reports), want)
	}
	requireNoPreparedSecrets(t, harness.scheduler)
}

// A Scheduler that restarts directly into finalizing or aborting has no
// preparation secrets: it resolves the pinned credentials once per live
// report, records the Planner placeholder and rebuilds StageMetrics once,
// and the report is redacted exactly as on the uninterrupted path.
func TestRestartIntoTerminalPhaseResolvesReportSecretsAgain(t *testing.T) {
	for _, state := range []runstore.StageExecutionState{runstore.StageFinalizing, runstore.StageAborting} {
		t.Run(string(state), func(t *testing.T) {
			harness, stageCredentials, reports := newStageEndHarness(t)
			execution := harness.persistedExecution(t, state)
			if state == runstore.StageFinalizing {
				result := harness.planners.result.Clone()
				finalizationID, deadline := "finalization-restart", harness.clock.now.Add(time.Minute)
				execution.CandidateResultSchemaVersion = stringPointer(contracts.APIVersion)
				execution.CandidateResult = &result
				execution.FinalizationID = &finalizationID
				execution.FinalizationDeadline = &deadline
			} else {
				harness.requestCancellation("restart into aborting")
				abortID, deadline := "abort-restart", harness.clock.now.Add(time.Minute)
				execution.TerminationSchemaVersion = stringPointer(contracts.APIVersion)
				execution.Termination = &runstore.StageTermination{
					Outcome: runstore.TerminationCancelled, Code: runstore.CancellationUserRequested,
					Message: "restart into aborting", Phase: runstore.TerminationRunning,
					OccurredAt: harness.clock.now.Add(-time.Second),
				}
				execution.AbortID = &abortID
				execution.AbortDeadline = &deadline
			}
			harness.store.stages = []runstore.StageExecution{execution}
			harness.persistence.stageStates = []runstore.StageExecutionState{state}
			harness.installRecordedReservation(execution.StageExecutionID)

			worked, err := harness.scheduler.RunOnce(t.Context())
			if err != nil || !worked {
				t.Fatalf("RunOnce = (%v, %v)", worked, err)
			}
			if harness.store.run.State != runstore.RunSucceeded && harness.store.run.State != runstore.RunCancelled {
				t.Fatalf("restarted Run = %s", harness.store.run.State)
			}
			if stageCredentials.llm[stageEndWorkerCredential] != 1 || stageCredentials.runtime[stageEndOTLPCredential] != 1 {
				t.Fatalf("credential resolutions = LLM %v, Runtime %v; want one each, at report persistence",
					stageCredentials.llm, stageCredentials.runtime)
			}
			if !slices.Equal(reports.ops, []string{"planner_report", "allocation_report", "rebuild"}) {
				t.Fatalf("report operations = %v, want one rebuild after the placeholder and the report", reports.ops)
			}
			if len(harness.store.plannerReports) != 1 || harness.store.plannerReports[0].Report.Complete {
				t.Fatalf("Planner reports = %+v, want the incomplete placeholder", harness.store.plannerReports)
			}
			requireStageEndRedaction(t, harness.store.reports[0].Report, stageEndSecrets)
			requireNoPreparedSecrets(t, harness.scheduler)
		})
	}
}

// The values kept from preparation and the values a restarted Scheduler
// resolves again produce byte-identical persisted reports for every
// credential-bearing setting, including the Project HTTP target credential.
func TestPreparationAndRecoveredReportRedactionAreIdentical(t *testing.T) {
	harness := newSchedulerHarness(t)
	const (
		llmSecret        = "llm-identical-secret"
		otlpSecret       = "otlp-identical-secret"
		proxySecret      = "proxy-identical-secret"
		caidoSecret      = "caido-identical-secret"
		originUserSecret = "origin-identical-username"
		originPassSecret = "origin-identical-password"
		staticSecret     = "static-identical-secret"
	)
	llmCalls, runtimeCalls := 0, 0
	harness.scheduler.options.Credentials = credentialResolverFunc(func(
		context.Context, llmgateway.LLMCredentialRef, llmgateway.LLMGatewayConfigRef,
	) (contracts.SecretString, error) {
		llmCalls++
		return contracts.NewSecretString(llmSecret), nil
	})
	materials := map[string][2]string{
		"otlp":   {string(runtimesettings.RuntimeCredentialOTLPHeaders), `{"headers":{"x-otlp-token":"` + otlpSecret + `"}}`},
		"proxy":  {string(runtimesettings.RuntimeCredentialProxyBearer), `{"token":"` + proxySecret + `"}`},
		"caido":  {string(runtimesettings.RuntimeCredentialCaidoBearer), `{"token":"` + caidoSecret + `"}`},
		"origin": {string(runtimesettings.RuntimeCredentialOriginBasic), `{"username":"` + originUserSecret + `","password":"` + originPassSecret + `"}`},
	}
	harness.scheduler.options.RuntimeCredentials = runtimeCredentialResolverFunc(func(
		_ context.Context, credentialID string, _ []runtimesettings.RuntimeCredentialKind,
		consumer func(runtimesettings.RuntimeCredentialKind, []byte) error,
	) error {
		runtimeCalls++
		material, ok := materials[credentialID]
		if !ok {
			return errors.New("unknown report credential")
		}
		return consumer(runtimesettings.RuntimeCredentialKind(material[0]), []byte(material[1]))
	})
	harness.scheduler.options.TelemetrySecrets = []string{staticSecret}
	stage := harness.workflow.Stages[harness.workflow.EntryStage]
	binding := stage.Agents["builder"]
	binding.Template.Toolsets = append(binding.Template.Toolsets, contracts.ToolsetSelection{
		Ref: contracts.ToolsetRef{ToolsetID: "http-tools", Version: "1"}, Tools: []string{"http_request"},
	})
	stage.Agents["builder"] = binding
	reservations := schedulerTestReservations(t, stage)
	reservations[0].Grant.AllocationID = "allocation-identical"
	resolved := reservations[0].ResolvedRuntimeConfig
	resolved.WorkerTelemetry = &runtimeconfig.TelemetryConfig{
		Adapter: string(runtimesettings.RuntimeAdapterOTLPHTTP), Endpoint: "https://otlp.example/v1/traces",
		Credential: "otlp", FlushTimeoutSeconds: 1,
	}
	resolved.HTTPProxy = &runtimeconfig.HTTPProxyConfig{
		Adapter: string(runtimesettings.RuntimeAdapterHTTPProxy), ProxyURL: "https://proxy.example",
		Credential: "proxy", Targets: []string{string(runtimesettings.ProxyTargetToolHTTP)},
	}
	resolved.Caido = &runtimeconfig.CaidoConfig{
		Adapter: string(runtimesettings.RuntimeAdapterCaidoGraphQL), Endpoint: "https://caido.example/api",
		Credential: "caido", RequestTimeoutSeconds: 5,
	}
	run := harness.store.run
	run.ProjectHTTPTarget = &runtimesettings.HTTPOriginTargetRef{
		URL: "https://target.example/api",
		Credential: &runtimesettings.RuntimeCredentialRef{
			CredentialID: "origin", Kind: runtimesettings.RuntimeCredentialOriginBasic,
		},
	}
	execution := runstore.StageExecution{StageExecutionID: "stage-identical", CreatedAt: harness.clock.now}
	harness.store.allocations = []runstore.StageAllocation{{
		AllocationID: reservations[0].Grant.AllocationID, StageExecutionID: execution.StageExecutionID,
		LogicalAgentName: "builder",
	}}
	secretValues := []string{
		llmSecret, otlpSecret, proxySecret, caidoSecret, originUserSecret, originPassSecret, staticSecret,
	}
	report := schedulerTestAllocationReport(reservations[0].Grant.AllocationID, harness.clock.now)
	for index, secret := range secretValues {
		report.Worker.ToolCalls = append(report.Worker.ToolCalls, reporting.ToolCallRecord{
			CallID: "secret-call-" + strconv.Itoa(index), Tool: "probe",
			Arguments: map[string]any{"detail": "before " + secret + " after"}, Outcome: reporting.ToolCallSucceeded,
		})
	}
	reports := map[string]reporting.AllocationFinalReport{"builder": report}
	store := &normalizingReportStore{harness.store}
	persist := func(s *Scheduler) []byte {
		t.Helper()
		harness.store.reports = nil
		s.persistReports(t.Context(), run, executableWorkflow{stage: stage}, execution, reservations, reports)
		if len(harness.store.reports) != 1 {
			t.Fatalf("persisted reports = %d", len(harness.store.reports))
		}
		encoded, err := json.Marshal(harness.store.reports[0].Report)
		if err != nil {
			t.Fatal(err)
		}
		for index, secret := range secretValues {
			if strings.Contains(string(encoded), secret) {
				t.Fatalf("secret %d remained in the persisted report", index)
			}
		}
		return encoded
	}

	prepared, err := New(store, harness.persistence, harness.artifacts, harness.allocator, harness.workers,
		harness.planners, withReports(harness.scheduler.options, store))
	if err != nil {
		t.Fatal(err)
	}
	settings, err := prepared.workerExecutionSettingsForRun(t.Context(), run, stage, reservations)
	if err != nil {
		t.Fatal(err)
	}
	prepared.preparedSecrets.store(execution.StageExecutionID, reservations, settings)
	clearWorkerExecutionSettings(settings)
	if llmCalls != 1 || runtimeCalls != 4 {
		t.Fatalf("preparation resolutions = LLM %d, Runtime %d", llmCalls, runtimeCalls)
	}
	fromPreparation := persist(prepared)
	if llmCalls != 1 || runtimeCalls != 4 {
		t.Fatalf("report persistence with preparation secrets resolved again: LLM %d, Runtime %d", llmCalls, runtimeCalls)
	}

	restarted, err := New(store, harness.persistence, harness.artifacts, harness.allocator, harness.workers,
		harness.planners, withReports(harness.scheduler.options, store))
	if err != nil {
		t.Fatal(err)
	}
	recovered := persist(restarted)
	if llmCalls != 2 || runtimeCalls != 8 {
		t.Fatalf("restarted report persistence resolutions = LLM %d, Runtime %d", llmCalls, runtimeCalls)
	}
	if string(fromPreparation) != string(recovered) {
		t.Fatalf("redaction differs:\npreparation: %s\nrecovered:   %s", fromPreparation, recovered)
	}
}

// The cache holds at most LIM-SCH-19 Stages, evicting the least recently
// stored first; it hands out copies, keeps only allocations it was given
// settings for, and forgets a Stage on request.
func TestPreparedSecretCacheIsBounded(t *testing.T) {
	cache := newPreparedSecretCache(preparedSecretStages)
	settings := map[string]runtimesettings.WorkerExecutionSettings{"builder": {
		RuntimeSettings: runtimesettings.RuntimeSettings{LLMGatewayToken: new(contracts.NewSecretString("cache-secret"))},
	}}
	reservation := func(stage int) []controlplane.Reservation {
		return []controlplane.Reservation{
			{Grant: controlplane.AllocationGrant{AllocationID: fmt.Sprintf("allocation-%d", stage), LogicalAgentName: "builder"}},
			{Grant: controlplane.AllocationGrant{AllocationID: fmt.Sprintf("unsettled-%d", stage), LogicalAgentName: "reviewer"}},
		}
	}
	for stage := range preparedSecretStages + 1 {
		cache.store(fmt.Sprintf("stage-%d", stage), reservation(stage), settings)
	}
	if cache.order.Len() != preparedSecretStages || preparedSecretStages != 128 {
		t.Fatalf("cached Stages = %d, bound = %d", cache.order.Len(), preparedSecretStages)
	}
	if _, kept := cache.lookup("stage-0", "allocation-0"); kept {
		t.Fatal("least recently stored Stage was not evicted")
	}
	values, kept := cache.lookup("stage-1", "allocation-1")
	if !kept || !slices.Equal(values, []string{"cache-secret"}) {
		t.Fatalf("kept values = %v, %v", values, kept)
	}
	values[0] = "changed"
	if again, _ := cache.lookup("stage-1", "allocation-1"); again[0] != "cache-secret" {
		t.Fatal("lookup exposed the cached values")
	}
	if _, kept := cache.lookup("stage-1", "unsettled-1"); kept {
		t.Fatal("an allocation without settings has kept values")
	}
	cache.forget("stage-1")
	if _, kept := cache.lookup("stage-1", "allocation-1"); kept || cache.order.Len() != preparedSecretStages-1 {
		t.Fatal("forgotten Stage is still cached")
	}
	var absent *preparedSecretCache
	absent.store("stage", reservation(0), settings)
	absent.forget("stage")
	if _, kept := absent.lookup("stage", "allocation-0"); kept {
		t.Fatal("an absent cache returned values")
	}
}

// No form of the cache, its entries or the list elements that hold them
// reveals a kept value: not fmt with any verb, not encoding/json and not
// log/slog with either standard handler (SEC-R88).
func TestPreparedSecretTypesNeverRevealValues(t *testing.T) {
	const secret = "formatting-secret-value"
	cache := newPreparedSecretCache(preparedSecretStages)
	cache.store("stage-format", []controlplane.Reservation{{Grant: controlplane.AllocationGrant{
		AllocationID: "allocation-format", LogicalAgentName: "builder",
	}}}, map[string]runtimesettings.WorkerExecutionSettings{"builder": {
		RuntimeSettings: runtimesettings.RuntimeSettings{LLMGatewayToken: new(contracts.NewSecretString(secret))},
	}})
	if values, kept := cache.lookup("stage-format", "allocation-format"); !kept || !slices.Equal(values, []string{secret}) {
		t.Fatalf("fixture keeps %v, %v", values, kept)
	}
	element := cache.order.Front()
	entry := element.Value.(*preparedSecretEntry)
	// The cache, its entries and every value that holds them redact
	// themselves as a whole, so every verb is safe. The kept values alone are
	// contracts.SecretString, which redacts the string verbs (SEC-R88).
	allVerbs := []string{"%v", "%+v", "%#v", "%s", "%q", "%x", "%d", "%t", "%o"}
	stringVerbs := []string{"%v", "%+v", "%#v", "%s", "%q", "%x"}
	for _, test := range []struct {
		name  string
		value any
		verbs []string
	}{
		{"entry", *entry, allVerbs},
		{"entry pointer", entry, allVerbs},
		{"list element", element, allVerbs},
		{"element copy", *element, allVerbs},
		{"cache", cache, allVerbs},
		{"cache struct", reflect.ValueOf(cache).Elem(), allVerbs},
		{"cache entries", cache.entries, allVerbs},
		{"cache order", cache.order, allVerbs},
		{"wrapping error", fmt.Errorf("wrapped: %v", entry), allVerbs},
		{"kept values", entry.allocations, stringVerbs},
	} {
		for _, verb := range test.verbs {
			if formatted := fmt.Sprintf(verb, test.value); strings.Contains(formatted, secret) ||
				strings.Contains(formatted, fmt.Sprintf("%x", secret)) {
				t.Fatalf("%s formatted with %s reveals the secret: %s", test.name, verb, formatted)
			}
		}
		encoded, err := json.Marshal(test.value)
		if err != nil || bytes.Contains(encoded, []byte(secret)) {
			t.Fatalf("%s encodes as (%s, %v)", test.name, encoded, err)
		}
		var logs bytes.Buffer
		for _, handler := range []slog.Handler{slog.NewTextHandler(&logs, nil), slog.NewJSONHandler(&logs, nil)} {
			slog.New(handler).Info("prepared secrets", "value", test.value)
		}
		if bytes.Contains(logs.Bytes(), []byte(secret)) {
			t.Fatalf("%s logs as %s", test.name, logs.Bytes())
		}
	}
	if got := fmt.Sprint(entry); got != `preparedSecretEntry("stage-format", 1 allocations, [REDACTED])` {
		t.Fatalf("entry formats as %q", got)
	}
}

// A Stage that ends through failActiveRun is never released by its own lane
// (SCH-R102), so its preparation secrets end only when terminal release
// recovery selects it (SCH-R178, LIFE-SCH-6).
func TestTerminalReleaseRecoveryForgetsPreparationSecrets(t *testing.T) {
	harness := newSchedulerHarness(t)
	execution := harness.persistedExecution(t, runstore.StageRunning)
	harness.store.stages = []runstore.StageExecution{execution}
	harness.installRecordedReservation(execution.StageExecutionID)
	reservations := append([]controlplane.Reservation(nil), harness.allocator.cached...)
	allocationID := reservations[0].Grant.AllocationID
	harness.scheduler.preparedSecrets.store(execution.StageExecutionID, reservations,
		map[string]runtimesettings.WorkerExecutionSettings{reservations[0].Grant.LogicalAgentName: {
			RuntimeSettings: runtimesettings.RuntimeSettings{LLMGatewayToken: new(contracts.NewSecretString("recovery-secret"))},
		}})

	if err := harness.scheduler.failInvalidRunState(t.Context(), "run-1", errReviewStateInvalid); err != nil {
		t.Fatal(err)
	}
	if harness.store.run.State != runstore.RunFailed || harness.store.stages[0].State != runstore.StageInterrupted ||
		harness.workers.releaseCalls != 0 {
		t.Fatalf("ended Run = %s, Stage = %s, releases = %d", harness.store.run.State,
			harness.store.stages[0].State, harness.workers.releaseCalls)
	}
	if _, kept := harness.scheduler.preparedSecrets.lookup(execution.StageExecutionID, allocationID); !kept {
		t.Fatal("the ending released the secrets itself; this test no longer covers recovery")
	}

	worked, err := harness.scheduler.recoverTerminalRelease(t.Context())
	if err != nil || !worked || harness.workers.releaseCalls != 1 {
		t.Fatalf("terminal release recovery = (%v, %v), releases = %d", worked, err, harness.workers.releaseCalls)
	}
	requireNoPreparedSecrets(t, harness.scheduler)
}
