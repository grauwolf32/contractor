package scan

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
)

func TestFactoryFreezesAllInputsBeforeDispatchAndProducesDeterministicReport(t *testing.T) {
	first := newFactoryHarness(t, 3)
	result := first.run(t)
	if result.Outcome != contracts.StageSucceeded || len(first.invoker.calls) != 3 {
		t.Fatalf("result=%+v calls=%d", result, len(first.invoker.calls))
	}
	report := first.report(t, result)
	if !report.Coverage.Complete || report.Coverage.Completed != 3 || report.Coverage.Selected != 3 || len(report.Jobs) != 3 {
		t.Fatalf("aggregate = %+v", report)
	}
	seenOutputs := map[string]bool{}
	for index, call := range first.invoker.calls {
		if call.binding != "sqlmap" || call.request.SubtaskID != first.sessions.state.Jobs[index].ID || len(call.request.Artifacts) != 1 ||
			!reflect.DeepEqual(call.request.Artifacts, first.sessions.state.Jobs[index].InputArtifacts) {
			t.Fatalf("Worker call %d differs from frozen job: %+v", index, call)
		}
		target := call.request.ResultArtifacts["report"]
		key := factoryRefKey(target)
		if target.Namespace != "sqlmap" || target.Revision != nil || seenOutputs[key] {
			t.Fatalf("job output target is not unique and stable: %+v", target)
		}
		seenOutputs[key] = true
		if report.Jobs[index].Report == nil || report.Jobs[index].Report.ValidateExact() != nil ||
			report.Jobs[index].Report.Name != target.Name || report.Jobs[index].Report.Namespace != target.Namespace {
			t.Fatalf("job report does not retain exact target revision: %+v", report.Jobs[index])
		}
	}
	if first.sessions.initializeCalls != 1 || first.sessions.completeCalls != 1 {
		t.Fatalf("journal calls = init:%d complete:%d", first.sessions.initializeCalls, first.sessions.completeCalls)
	}
	second := newFactoryHarness(t, 3)
	other := second.run(t)
	if !reflect.DeepEqual(first.invoker.calls, second.invoker.calls) ||
		!bytes.Equal(first.artifacts.payload(result.Artifacts["report"]).Data, second.artifacts.payload(other.Artifacts["report"]).Data) {
		t.Fatal("same exact source and stage produced different Worker requests or aggregate bytes")
	}
}

func TestFactoryCompletedRecoveryDoesNotReadSourceOrInvokeWorkers(t *testing.T) {
	h := newFactoryHarness(t, 2)
	want := h.run(t)
	h.invoker.calls = nil
	h.artifacts.reads = nil
	h.artifacts.creates = nil
	got := h.run(t)
	if !reflect.DeepEqual(got, want) || len(h.invoker.calls) != 0 || len(h.artifacts.reads) != 0 || len(h.artifacts.creates) != 0 {
		t.Fatalf("completed recovery repeated work: result=%+v invokes=%d reads=%d writes=%d", got, len(h.invoker.calls), len(h.artifacts.reads), len(h.artifacts.creates))
	}
}

func TestFactoryRejectsContextBindingSubstitution(t *testing.T) {
	h := newFactoryHarness(t, 1)
	for _, ref := range h.invocation.Context.Artifacts {
		ref.Name = "different-exact-artifact"
	}
	if _, err := h.factory.Create(h.invocation); err == nil {
		t.Fatal("accepted another artifact in the declared context slot")
	}
}

func TestFactoryRecoveredCompletedAndUnknownJobsNeverDispatch(t *testing.T) {
	h := newFactoryHarness(t, 3)
	h.freeze(t)
	done := &h.sessions.state.Jobs[0]
	done.Status = planner.ScanJobCompleted
	reportRef := h.artifacts.put(contracts.ArtifactRef{Namespace: "sqlmap", Name: "previous-report"}, artifacts.Payload{MediaType: "application/json", Data: []byte(`{"evidence":"retained"}`)})
	done.Report = &reportRef
	h.sessions.state.Jobs[1].Status, h.sessions.state.Jobs[1].Code = planner.ScanJobUnknown, "scan_outcome_unknown"
	pendingID := h.sessions.state.Jobs[2].ID
	result := h.run(t)
	report := h.report(t, result)
	if len(h.invoker.calls) != 1 || h.invoker.calls[0].request.SubtaskID != pendingID || report.Coverage.Completed != 2 || report.Coverage.Unknown != 1 || report.Coverage.Complete {
		t.Fatalf("recovery dispatch/coverage = calls:%+v coverage:%+v", h.invoker.calls, report.Coverage)
	}
	if !reflect.DeepEqual(report.Jobs[0].Report, &reportRef) || h.sessions.initializeCalls != 1 {
		t.Fatal("recovery replaced completed evidence or reinitialized plan")
	}
}

func TestFactoryLostJobClaimDoesNotInvoke(t *testing.T) {
	for _, claimErr := range []error{nil, errors.New("durable claim acknowledgement lost")} {
		h := newFactoryHarness(t, 2)
		h.sessions.rejectClaim, h.sessions.claimErr = true, claimErr
		_, err := h.instance(t).Run(t.Context())
		if err == nil || planner.FailureFrom(err).Code != "scan_job_claim_unavailable" || len(h.invoker.calls) != 0 || h.sessions.completeCalls != 0 {
			t.Fatalf("lost claim invoked Worker: err=%v calls=%d completion=%d", err, len(h.invoker.calls), h.sessions.completeCalls)
		}
	}
}

func TestFactoryFailedResultWriteStopsFollowingDispatch(t *testing.T) {
	h := newFactoryHarness(t, 3)
	h.sessions.finishErr = errors.New("durable result write unavailable")
	_, err := h.instance(t).Run(t.Context())
	if err == nil || planner.FailureFrom(err).Code != "scan_job_result_unavailable" || len(h.invoker.calls) != 1 || h.sessions.completeCalls != 0 {
		t.Fatalf("result-write failure = err:%v calls:%d completion:%d", err, len(h.invoker.calls), h.sessions.completeCalls)
	}
}

func TestFactoryCorruptedFrozenPlanRefusesDispatch(t *testing.T) {
	h := newFactoryHarness(t, 2)
	h.freeze(t)
	key := factoryRefKey(*h.sessions.state.Plan)
	corrupt := h.artifacts.values[key]
	corrupt.Data = append(corrupt.Data, '\n')
	h.artifacts.values[key] = corrupt
	_, err := h.instance(t).Run(t.Context())
	if err == nil || planner.FailureFrom(err).Code != "scan_saved_plan_invalid" || len(h.invoker.calls) != 0 {
		t.Fatalf("corrupt frozen plan = err:%v invokes:%d", err, len(h.invoker.calls))
	}
}

func TestFactoryRecoveryKeepsExactInputsWhenBindingHeadChanges(t *testing.T) {
	h := newFactoryHarness(t, 2)
	h.freeze(t)
	inputs := cloneFactoryValue(h.sessions.state.Jobs)
	refs := []contracts.ArtifactRef{*h.invocation.Context.Artifacts["requests"], *h.sessions.state.Plan}
	for _, job := range inputs {
		for _, ref := range job.InputArtifacts {
			refs = append(refs, ref)
		}
	}
	for _, ref := range refs {
		ref.Revision = nil
		h.artifacts.put(ref, artifacts.Payload{MediaType: "application/json", Data: []byte(`{"changed":true}`)})
	}
	result := h.run(t)
	if result.Outcome != contracts.StageSucceeded || len(h.invoker.calls) != 2 {
		t.Fatalf("exact revision recovery failed: %+v", result)
	}
	for i, call := range h.invoker.calls {
		if !reflect.DeepEqual(call.request.Artifacts, inputs[i].InputArtifacts) {
			t.Fatal("recovery followed the current input binding instead of its persisted revision")
		}
	}
}

func TestFactoryTransportFailureMarksUnknownAndUnstartedIncomplete(t *testing.T) {
	h := newFactoryHarness(t, 3)
	h.invoker.invokeErr = errors.New("transport lost after dispatch, private token must not escape")
	result := h.run(t)
	report := h.report(t, result)
	if result.Outcome != contracts.StageFailed || len(h.invoker.calls) != 1 || report.Coverage.Unknown != 1 || report.Coverage.Incomplete != 2 || report.Coverage.Complete {
		t.Fatalf("uncertain transport = outcome:%s calls:%d coverage:%+v", result.Outcome, len(h.invoker.calls), report.Coverage)
	}
	encoded, _ := json.Marshal(report)
	if strings.Contains(string(encoded)+result.Summary, "private token") {
		t.Fatal("transport diagnostics leaked into persisted report")
	}
}

func TestFactorySuccessfulTruncatedOutputRemainsIncomplete(t *testing.T) {
	for _, field := range []string{"stdoutTruncated", "scanComplete", "workerObservations"} {
		t.Run(field, func(t *testing.T) {
			h := newFactoryHarness(t, 1)
			if field == "workerObservations" {
				h.invoker.truncated = true
			} else {
				h.invoker.observation[field] = field != "scanComplete"
			}
			result := h.run(t)
			report := h.report(t, result)
			if result.Outcome != contracts.StageFailed || report.Coverage.Completed != 0 || report.Coverage.Incomplete != 1 || report.Coverage.Complete || report.Jobs[0].Report == nil {
				t.Fatalf("truncated success = %+v", report)
			}
		})
	}
}

func TestFactoryExpiredDeadlineSkipsEveryWorkerButPersistsCoverage(t *testing.T) {
	h := newFactoryHarness(t, 2)
	h.invocation.Deadline = time.Now().Add(-time.Second)
	result := h.run(t)
	report := h.report(t, result)
	if len(h.invoker.calls) != 0 || len(h.sessions.claimed) != 0 || report.Coverage.Incomplete != 2 || result.Outcome != contracts.StageFailed {
		t.Fatalf("deadline handling = calls:%d claims:%d coverage:%+v", len(h.invoker.calls), len(h.sessions.claimed), report.Coverage)
	}
}

func TestFactoryRetainsFailedToolReportWithoutRepeatingScanner(t *testing.T) {
	for _, test := range []struct {
		name       string
		code       string
		omitReport bool
		status     string
	}{
		{name: "unavailable", code: "scanner_unavailable", status: planner.ScanJobUnavailable},
		{name: "timeout", code: "scan_timeout", status: planner.ScanJobIncomplete},
		{name: "incomplete", code: "scan_incomplete", status: planner.ScanJobIncomplete},
		{name: "missing report", code: "scan_failed", omitReport: true, status: planner.ScanJobFailed},
	} {
		t.Run(test.name, func(t *testing.T) {
			h := newFactoryHarness(t, 1)
			h.invoker.failureCode = "tool_execution_failed"
			h.invoker.omitReport = test.omitReport
			h.invoker.observation = map[string]any{"status": "failed", "exitCode": 1, "errorCode": test.code}
			result := h.run(t)
			report := h.report(t, result)
			if len(h.invoker.calls) != 1 || report.Jobs[0].Status != test.status || report.Coverage.Complete || result.Outcome != contracts.StageFailed {
				t.Fatalf("failed tool result = calls:%d report:%+v result:%s", len(h.invoker.calls), report, result.Outcome)
			}
			if (report.Jobs[0].Report == nil) != test.omitReport {
				t.Fatalf("persisted failure evidence mismatch: %+v", report.Jobs[0])
			}
		})
	}
}

type factoryHarness struct {
	t          *testing.T
	factory    *Factory
	invocation planner.Invocation
	artifacts  *factoryArtifacts
	sessions   *factorySessions
	invoker    *factoryInvoker
}

func newFactoryHarness(t *testing.T, count int) *factoryHarness {
	t.Helper()
	store := &factoryArtifacts{values: map[string]artifacts.Payload{}, current: map[string]contracts.ArtifactRef{}}
	requestSet := contracts.HTTPRequestSet{
		SchemaVersion: 1, Source: contracts.RequestSetSource{Artifact: factoryExactRef("inputs", "openapi"), ContentDigest: "sha256:" + strings.Repeat("b", 64)},
		PreparationDigest: "sha256:" + strings.Repeat("c", 64), Requests: []contracts.RequestSetEntry{}, Gaps: []contracts.PreparationGap{},
		Coverage: contracts.RequestSetCoverage{Operations: count, Prepared: count, Complete: true},
	}
	for i := range count {
		request := contracts.PreparedHTTPRequest{Method: "GET", URL: fmt.Sprintf("https://target.invalid/items?id=%d", i), Headers: []contracts.HTTPRequestHeader{}, Body: ""}
		digest, err := contracts.RequestContentDigest(request)
		if err != nil {
			t.Fatal(err)
		}
		requestSet.Requests = append(requestSet.Requests, contracts.RequestSetEntry{
			ID: "request-" + strings.TrimPrefix(digest, "sha256:"), ContentDigest: digest, Request: request,
			Origins: []contracts.RequestOrigin{{Pointer: fmt.Sprintf("#/paths/~1items%d/get", i)}},
		})
	}
	sort.Slice(requestSet.Requests, func(i, j int) bool { return requestSet.Requests[i].ID < requestSet.Requests[j].ID })
	data, err := contracts.MarshalHTTPRequestSet(requestSet)
	if err != nil {
		t.Fatal(err)
	}
	source := factoryExactRef("inputs", "requests")
	store.values[factoryRefKey(source)] = artifacts.Payload{MediaType: contracts.HTTPRequestSetMediaType, Data: data}
	template := contracts.ResolvedAgentTemplate{
		Ref:         contracts.AgentTemplateRef{TemplateID: "sqlmap-scan", Version: "1", Digest: "sha256:" + strings.Repeat("a", 64)},
		Description: "Fixed SQLMap Worker", Runtime: contracts.WorkerRuntimeRef{RuntimeID: "tool", Version: "1"},
		SandboxProfile: contracts.SandboxProfileRef{SandboxProfileID: "local-workdir", Version: "1"},
		Toolsets:       []contracts.ToolsetSelection{{Ref: contracts.ToolsetRef{ToolsetID: "scan", Version: "1"}, Tools: []string{"scan_sqlmap"}}},
		Execution:      &contracts.ToolExecutionConfig{Tool: "scan_sqlmap", Arguments: map[string]contracts.ToolArgumentBinding{"request_ref": {Source: "artifact", Name: "request"}}, ResultArtifact: "report", TimeoutSeconds: 10},
	}
	invocation := planner.Invocation{
		StageExecutionID: "stage-scan", RunID: "run-scan", SchedulerClaimID: "claim-scan", Deadline: time.Now().Add(time.Minute),
		Stage: workflowconfig.ResolvedStage{
			Objective: "Execute bounded scan jobs", Instructions: contracts.ResolvedInstructions{Text: "Use only the fixed scanner configuration"},
			Planner: workflowconfig.PlannerRef{PlannerID: "scan-plan", Version: "1"},
			ScanPlan: &contracts.ScanPlanPolicy{InputArtifact: "requests", MaxInputs: 100, MaxJobs: 100, MaxTotalSeconds: 3600,
				Tools: []contracts.ScanToolPolicy{{Worker: "sqlmap", MaxJobs: 100, MaxTotalSeconds: 3600, TestParameters: []string{"id"}}}},
			Agents:  map[string]workflowconfig.ResolvedAgentBinding{"sqlmap": {Template: template, Namespace: "sqlmap"}},
			Context: workflowconfig.StageContext{Artifacts: map[string]workflowconfig.ContextArtifact{"requests": {Namespace: "inputs", Name: "requests", Required: true}}},
			Result:  workflowconfig.StageResultContract{Artifacts: map[string]workflowconfig.ArtifactSlot{"report": {Required: true, MediaTypes: []string{"application/json"}, From: &workflowconfig.ArtifactBinding{Namespace: "scan-results", Name: "aggregate"}}}},
		},
		Context: planner.StageContext{Artifacts: map[string]*contracts.ArtifactRef{"requests": &source}},
		Workers: map[string]contracts.WorkerHandle{"sqlmap": {AllocationID: "allocation-scan", AgentTemplateRef: template.Ref, WorkerRuntimeRef: template.Runtime, AgentCard: map[string]any{"name": "sqlmap"}, LeaseExpiresAt: time.Now().Add(time.Minute)}},
	}
	sessions := &factorySessions{t: t, artifacts: store, expectedJobs: count}
	invoker := &factoryInvoker{t: t, artifacts: store, sessions: sessions, observation: map[string]any{"status": "completed", "exitCode": 0, "scanComplete": true}}
	factory, err := NewFactory(sessions, invoker, store, store)
	if err != nil {
		t.Fatal(err)
	}
	return &factoryHarness{t: t, factory: factory, invocation: invocation, artifacts: store, sessions: sessions, invoker: invoker}
}

func (h *factoryHarness) instance(t *testing.T) planner.Planner {
	t.Helper()
	instance, err := h.factory.Create(h.invocation)
	if err != nil {
		t.Fatal(err)
	}
	return instance
}

func (h *factoryHarness) run(t *testing.T) contracts.StageContentResult {
	t.Helper()
	result, err := h.instance(t).Run(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	return result
}

func (h *factoryHarness) freeze(t *testing.T) {
	t.Helper()
	h.sessions.rejectClaim = true
	if _, err := h.instance(t).Run(t.Context()); err == nil || h.sessions.state.Plan == nil {
		t.Fatalf("failed to freeze plan without dispatch: %v", err)
	}
	h.sessions.rejectClaim = false
	h.sessions.claimed = nil
}

func (h *factoryHarness) report(t *testing.T, result contracts.StageContentResult) Report {
	t.Helper()
	var report Report
	if err := json.Unmarshal(h.artifacts.payload(result.Artifacts["report"]).Data, &report); err != nil {
		t.Fatal(err)
	}
	return report
}

type factoryArtifacts struct {
	values  map[string]artifacts.Payload
	current map[string]contracts.ArtifactRef
	reads   []contracts.ArtifactRef
	creates []contracts.ArtifactRef
}

func (s *factoryArtifacts) Read(ctx context.Context, _ string, ref contracts.ArtifactRef, limit int) (artifacts.Payload, error) {
	s.reads = append(s.reads, planner.CloneArtifactRef(ref))
	payload, exists := s.values[factoryRefKey(ref)]
	if !exists || ref.ValidateExact() != nil || len(payload.Data) > limit || ctx.Err() != nil {
		return artifacts.Payload{}, errors.New("exact artifact unavailable")
	}
	return artifacts.Payload{MediaType: payload.MediaType, Data: append([]byte(nil), payload.Data...)}, nil
}

func (s *factoryArtifacts) Create(ctx context.Context, _ string, target contracts.ArtifactRef, payload artifacts.Payload) (contracts.ArtifactRef, error) {
	if target.Revision != nil || ctx.Err() != nil {
		return contracts.ArtifactRef{}, errors.New("invalid artifact create")
	}
	s.creates = append(s.creates, target)
	if ref, exists := s.current[factoryRefKey(target)]; exists {
		if current := s.payload(ref); current.MediaType != payload.MediaType || !bytes.Equal(current.Data, payload.Data) {
			return contracts.ArtifactRef{}, errors.New("immutable artifact conflict")
		}
		return planner.CloneArtifactRef(ref), nil
	}
	return s.put(target, payload), nil
}

func (s *factoryArtifacts) Inspect(_ context.Context, _ string, ref contracts.ArtifactRef) (planner.ArtifactMetadata, error) {
	payload, exists := s.values[factoryRefKey(ref)]
	if !exists || ref.ValidateExact() != nil {
		return planner.ArtifactMetadata{}, errors.New("exact report unavailable")
	}
	return planner.ArtifactMetadata{MediaType: payload.MediaType}, nil
}

func (s *factoryArtifacts) Resolve(ctx context.Context, _ string, target contracts.ArtifactRef) (contracts.ArtifactRef, error) {
	ref, exists := s.current[factoryRefKey(target)]
	if !exists || target.Revision != nil || ctx.Err() != nil {
		return contracts.ArtifactRef{}, errors.New("report metadata unavailable")
	}
	return planner.CloneArtifactRef(ref), nil
}

func (s *factoryArtifacts) put(target contracts.ArtifactRef, payload artifacts.Payload) contracts.ArtifactRef {
	checksum := sha256.Sum256(append([]byte(target.Namespace+"/"+target.Name+"/"+payload.MediaType), payload.Data...))
	revision := "rev-" + hex.EncodeToString(checksum[:])
	ref := planner.CloneArtifactRef(target)
	ref.Revision = &revision
	s.current[factoryRefKey(target)] = ref
	s.values[factoryRefKey(ref)] = artifacts.Payload{MediaType: payload.MediaType, Data: append([]byte(nil), payload.Data...)}
	return ref
}

func (s *factoryArtifacts) payload(ref contracts.ArtifactRef) artifacts.Payload {
	return s.values[factoryRefKey(ref)]
}

func factoryRefKey(ref contracts.ArtifactRef) string {
	key := ref.Namespace + "/" + ref.Name
	if ref.Revision != nil {
		key += "@" + *ref.Revision
	}
	return key
}

func factoryExactRef(namespace, name string) contracts.ArtifactRef {
	revision := "rev-source"
	return contracts.ArtifactRef{Namespace: namespace, Name: name, Revision: &revision}
}

type factorySessions struct {
	t               *testing.T
	artifacts       *factoryArtifacts
	expectedJobs    int
	state           planner.ScanState
	completion      *planner.Completion
	initializeCalls int
	completeCalls   int
	claimed         []string
	rejectClaim     bool
	claimErr        error
	finishErr       error
}

func (s *factorySessions) BeginScan(_ context.Context, stage, claim string) (planner.ScanSessionStart, error) {
	return planner.ScanSessionStart{
		Identity: planner.ScanSessionIdentity{SessionIdentity: planner.SessionIdentity{StageExecutionID: stage, SessionID: "session-scan", InvocationID: "invocation-scan"}, SchedulerClaimID: claim},
		Invoke:   s.completion == nil, State: cloneFactoryValue(s.state), Completion: cloneFactoryValue(s.completion),
	}, nil
}

func (s *factorySessions) InitializeScan(_ context.Context, _ planner.ScanSessionIdentity, state planner.ScanState) error {
	s.initializeCalls++
	s.state = cloneFactoryValue(state)
	s.assertAllInputsPersisted()
	return nil
}

func (s *factorySessions) assertAllInputsPersisted() {
	s.t.Helper()
	if s.state.Plan == nil || s.state.Plan.ValidateExact() != nil || len(s.state.Jobs) != s.expectedJobs || len(s.artifacts.payload(*s.state.Plan).Data) == 0 {
		s.t.Fatal("dispatch/journal initialization preceded the full frozen plan")
	}
	for _, job := range s.state.Jobs {
		for _, ref := range job.InputArtifacts {
			if ref.ValidateExact() != nil || len(s.artifacts.payload(ref).Data) == 0 {
				s.t.Fatal("dispatch/journal initialization preceded another job's exact input")
			}
		}
	}
}

func (s *factorySessions) ClaimScanJob(_ context.Context, _ planner.ScanSessionIdentity, id string) (bool, error) {
	s.assertAllInputsPersisted()
	s.claimed = append(s.claimed, id)
	if s.rejectClaim {
		return false, s.claimErr
	}
	for i := range s.state.Jobs {
		if s.state.Jobs[i].ID == id && s.state.Jobs[i].Status == planner.ScanJobPending {
			s.state.Jobs[i].Status = planner.ScanJobStarted
			return true, nil
		}
	}
	return false, nil
}

func (s *factorySessions) FinishScanJob(_ context.Context, _ planner.ScanSessionIdentity, record planner.ScanJobRecord) error {
	if s.finishErr != nil {
		return s.finishErr
	}
	for i := range s.state.Jobs {
		if s.state.Jobs[i].ID == record.ID {
			s.state.Jobs[i] = cloneFactoryValue(record)
			return nil
		}
	}
	return errors.New("unknown job")
}

func (s *factorySessions) CompleteScan(_ context.Context, _ planner.ScanSessionIdentity, completion planner.Completion) error {
	s.completeCalls++
	s.completion = cloneFactoryValue(&completion)
	return nil
}

type factoryCall struct {
	binding string
	request contracts.StageContentRequest
}

type factoryInvoker struct {
	tool        string
	t           *testing.T
	artifacts   *factoryArtifacts
	sessions    *factorySessions
	calls       []factoryCall
	invokeErr   error
	observation map[string]any
	truncated   bool
	failureCode string
	omitReport  bool
}

func (w *factoryInvoker) Invoke(ctx context.Context, binding string, _ contracts.WorkerHandle, request contracts.StageContentRequest) (contracts.WorkerCompletion, error) {
	w.sessions.assertAllInputsPersisted()
	if err := request.Validate(); err != nil {
		w.t.Fatal(err)
	}
	found := false
	for _, job := range w.sessions.state.Jobs {
		if job.ID == request.SubtaskID && job.Status == planner.ScanJobStarted {
			found = true
		}
	}
	if !found {
		w.t.Fatal("Worker invoked before durable started intent")
	}
	w.calls = append(w.calls, factoryCall{binding: binding, request: planner.CloneStageRequest(request)})
	if w.invokeErr != nil {
		return contracts.WorkerCompletion{}, w.invokeErr
	}
	tool := w.tool
	if tool == "" {
		tool = "scan_sqlmap"
	}
	data, _ := json.Marshal(map[string]any{
		"schemaVersion": 1, "tool": tool, "inputDigest": "sha256:" + strings.Repeat("d", 64),
		"inputArtifacts": request.Artifacts, "observation": w.observation,
	})
	var ref contracts.ArtifactRef
	if !w.omitReport {
		var err error
		ref, err = w.artifacts.Create(ctx, "run-scan", request.ResultArtifacts["report"], artifacts.Payload{MediaType: "application/json", Data: data})
		if err != nil {
			return contracts.WorkerCompletion{}, err
		}
	}
	if w.failureCode != "" {
		return contracts.WorkerCompletion{
			APIVersion: contracts.APIVersion, InvocationID: "worker-invocation", StateRevision: 1,
			Failure: &contracts.WorkerFailure{Code: w.failureCode, Message: "Scanner execution failed", Retryable: false},
		}, nil
	}
	return contracts.WorkerCompletion{
		APIVersion: contracts.APIVersion, InvocationID: "worker-invocation", StateRevision: 1,
		Result: &contracts.WorkerResult{
			SubtaskID: request.SubtaskID, Result: "Scanner output persisted", Artifacts: map[string]contracts.ArtifactRef{"report": ref},
			Observations: contracts.WorkerObservations{Profile: contracts.WorkerObservationProfileLeanV1, Tools: map[string]contracts.ToolObservationCount{tool: {Calls: 1}}, Truncated: w.truncated},
		},
	}, nil
}

func cloneFactoryValue[T any](value T) T {
	data, err := json.Marshal(value)
	if err != nil {
		panic(err)
	}
	var copy T
	if err := json.Unmarshal(data, &copy); err != nil {
		panic(err)
	}
	return copy
}
