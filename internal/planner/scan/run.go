package scan

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"reflect"
	"strconv"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	contractscan "github.com/grauwolf32/contractor/internal/contracts/scan"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/scanplan"
	"github.com/grauwolf32/contractor/internal/stageresult"
)

// Allow journal writes to finish after invocation cancellation, but never wait
// indefinitely for storage during recovery. This is not a scanner time budget.
const journalIOTimeout = 5 * time.Second

func (p *execution) Run(ctx context.Context) (stageresult.StageContentResult, error) {
	empty := stageresult.StageContentResult{}
	start, err := p.factory.sessions.BeginScan(ctx, p.invocation.StageExecutionID, p.invocation.SchedulerClaimID)
	if err != nil {
		return empty, p.scanError("scan_session_unavailable", err)
	}
	if start.Completion != nil {
		result, err := planner.RecoverCompletion(
			ctx, p.invocation.RunID, p.invocation.Stage.Result.Artifacts, *start.Completion, p.factory.inspector,
		)
		return result, p.resultError(err)
	}
	if !start.Invoke {
		return empty, p.scanError("scan_session_unavailable", nil)
	}
	if p.invocation.Stage.ScanPreparation != nil {
		if err := p.loadAuditInputs(ctx); err != nil {
			return empty, err
		}
		history, err := p.auditHistory(ctx)
		if err != nil {
			return empty, err
		}
		recover := false
		for _, attempt := range history {
			if attempt.StageExecutionID == p.invocation.StageExecutionID {
				continue
			}
			if !attempt.Terminal {
				return empty, p.scanError("scan_prior_attempt_active", nil)
			}
			recover = recover || planner.ScanAttemptNeedsRecovery(attempt)
		}
		if recover || !p.audit.task.Runnable {
			return p.finishAudit(ctx, start.Identity, history)
		}
	}
	plan, state, err := p.prepare(ctx, start)
	if err != nil {
		return empty, err
	}
	stopCode := ""
	for i, job := range plan.Jobs {
		record := state.Jobs[i]
		if record.Status != planner.ScanJobPending {
			continue
		}
		if stopCode == "" && (ctx.Err() != nil || !time.Now().Before(p.invocation.Deadline)) {
			stopCode = "scan_deadline_exceeded"
		}
		if stopCode != "" {
			record.Status, record.Code = planner.ScanJobIncomplete, stopCode
		} else {
			won, err := p.factory.sessions.ClaimScanJob(ctx, start.Identity, job.ID)
			if err != nil || !won {
				return empty, p.scanError("scan_job_claim_unavailable", err)
			}
			record.Status = planner.ScanJobStarted
			deadline := minTime(p.invocation.Deadline, time.Now().Add(time.Duration(job.TimeoutSeconds)*time.Second))
			request, err := p.request(job, record, deadline)
			if err != nil {
				record.Status, record.Code = planner.ScanJobFailed, "scan_invalid_worker_result"
			} else {
				callCtx, cancel := context.WithDeadline(ctx, deadline)
				completion, invokeErr := p.factory.invoker.Invoke(callCtx, job.Worker, p.invocation.Workers[job.Worker].Clone(), request.Clone())
				cancel()
				if invokeErr != nil {
					record.Status, record.Code = planner.ScanJobUnknown, "scan_outcome_unknown"
					stopCode = "scan_outcome_unknown"
				} else {
					record = p.observe(ctx, job, request, record, completion)
					if record.Status == planner.ScanJobUnknown {
						stopCode = "scan_outcome_unknown"
					}
				}
			}
		}
		writeCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), journalIOTimeout)
		err := p.factory.sessions.FinishScanJob(writeCtx, start.Identity, record)
		cancel()
		if err != nil {
			return empty, p.scanError("scan_job_result_unavailable", err)
		}
		state.Jobs[i] = record
	}
	return p.finish(ctx, start.Identity, plan, state)
}

func (p *execution) prepare(ctx context.Context, start planner.ScanSessionStart) (scanplan.ScanPlan, planner.ScanState, error) {
	emptyPlan, emptyState := scanplan.ScanPlan{}, planner.ScanState{}
	policy := *p.invocation.Stage.ScanPlan
	source := p.invocation.Context.Artifacts[policy.InputArtifact]
	if source == nil {
		return emptyPlan, emptyState, p.scanError("scan_source_unavailable", nil)
	}
	payload, err := p.factory.artifacts.Read(ctx, p.invocation.RunID, *source, scanplan.MaxPlanBytes)
	if err != nil {
		return emptyPlan, emptyState, p.scanError("scan_source_unavailable", err)
	}
	if p.audit != nil {
		prepared, preparedPayload, err := p.auditPlanInput(ctx)
		if err != nil {
			return emptyPlan, emptyState, err
		}
		source, payload = &prepared, preparedPayload
		policy.Tools = append([]contractscan.ScanToolPolicy{}, policy.Tools...)
		policy.Tools[0].TestParameters = append([]string{}, p.audit.task.TestParameters...)
	}
	bindings := map[string]scanplan.ToolBinding{}
	for name, binding := range p.invocation.Stage.Agents {
		tool := scanplan.ToolBinding{Namespace: binding.Namespace, Template: binding.Template}
		if p.audit != nil && p.audit.selection != nil {
			selection := p.audit.selection
			tool.Parameters = map[string]string{"template_ids": selection.TemplateIDs, "tags": selection.Tags,
				"severity": selection.Severity, "rate_limit": strconv.Itoa(selection.RateLimit)}
			if argument, exists := binding.Template.Execution.Arguments["rulepack_ref"]; exists {
				ref := p.invocation.Context.Artifacts[argument.Name]
				if ref == nil && !argument.Optional {
					return emptyPlan, emptyState, p.scanError("audit_scan_input_missing", nil)
				}
				if ref != nil {
					tool.Artifacts = map[string]contracts.ArtifactRef{argument.Name: ref.Clone()}
				}
			}
		}
		bindings[name] = tool
	}
	wordlists := map[string]contracts.ArtifactRef{}
	for _, tool := range policy.Tools {
		if tool.WordlistArtifact != "" {
			ref := p.invocation.Context.Artifacts[tool.WordlistArtifact]
			if ref == nil {
				return emptyPlan, emptyState, p.scanError("scan_source_unavailable", nil)
			}
			wordlists[tool.WordlistArtifact] = ref.Clone()
		}
	}
	plan, err := scanplan.BuildPlan(scanplan.PlanInput{Artifact: *source, MediaType: payload.MediaType, Data: payload.Data}, policy, bindings, wordlists)
	if err != nil {
		return emptyPlan, emptyState, p.scanError("scan_plan_invalid", err)
	}
	data, err := scanplan.MarshalPlan(plan)
	if err != nil {
		return emptyPlan, emptyState, p.scanError("scan_plan_invalid", err)
	}
	state := start.State
	if state.Plan != nil {
		frozen, err := p.factory.artifacts.Read(ctx, p.invocation.RunID, *state.Plan, scanplan.MaxPlanBytes)
		if err != nil {
			return emptyPlan, emptyState, p.scanError("scan_saved_plan_unavailable", err)
		}
		if frozen.MediaType != scanplan.PlanMediaType || state.PlanDigest != contentdigest.Bytes(data) || !bytes.Equal(frozen.Data, data) || len(state.Jobs) != len(plan.Jobs) {
			return emptyPlan, emptyState, p.scanError("scan_saved_plan_invalid", nil)
		}
		for i, job := range plan.Jobs {
			record := state.Jobs[i]
			if record.ID != job.ID || record.Worker != job.Worker {
				return emptyPlan, emptyState, p.scanError("scan_saved_plan_invalid", nil)
			}
			if err := p.validateSavedInputs(ctx, job, record); err != nil {
				var readError *savedInputReadError
				if errors.As(err, &readError) {
					return emptyPlan, emptyState, p.scanError("scan_saved_input_unavailable", err)
				}
				return emptyPlan, emptyState, p.scanError("scan_saved_input_invalid", err)
			}
		}
		return plan, state, nil
	}
	planRef, err := p.factory.artifacts.Create(ctx, p.invocation.RunID, p.internalTarget("plan", plan.ID), artifacts.Payload{MediaType: scanplan.PlanMediaType, Data: data})
	if err != nil {
		return emptyPlan, emptyState, p.scanError("scan_plan_write_failed", err)
	}
	state = planner.ScanState{Plan: &planRef, PlanDigest: contentdigest.Bytes(data), Jobs: []planner.ScanJobRecord{}}
	for _, job := range plan.Jobs {
		inputs, err := p.materialize(ctx, job)
		if err != nil {
			return emptyPlan, emptyState, p.scanError("scan_input_write_failed", err)
		}
		state.Jobs = append(state.Jobs, planner.ScanJobRecord{ID: job.ID, Worker: job.Worker, Status: planner.ScanJobPending, InputArtifacts: inputs})
	}
	if err := p.factory.sessions.InitializeScan(ctx, start.Identity, state); err != nil {
		return emptyPlan, emptyState, p.scanError("scan_plan_record_failed", err)
	}
	return plan, state, nil
}

func (p *execution) validateSavedInputs(ctx context.Context, job scanplan.ScanJob, record planner.ScanJobRecord) error {
	expected := map[string]contracts.ArtifactRef{}
	for name, ref := range job.Artifacts {
		expected[name] = ref.Clone()
	}
	if job.Request != nil {
		binding := job.Execution.Arguments["request_ref"]
		ref, ok := record.InputArtifacts[binding.Name]
		target := p.internalTarget("request", job.ID)
		if !ok || ref.ValidateExact() != nil || ref.Namespace != target.Namespace || ref.Name != target.Name {
			return fmt.Errorf("saved scan input binding differs")
		}
		payload, err := p.factory.artifacts.Read(ctx, p.invocation.RunID, ref, 256*1024)
		if err != nil {
			return &savedInputReadError{cause: err}
		}
		data, err := strictjson.Canonical(job.Request)
		if err != nil || payload.MediaType != "application/vnd.contractor.http-request+json" || !bytes.Equal(payload.Data, data) {
			return fmt.Errorf("saved scan input content differs")
		}
		expected[binding.Name] = ref
	}
	if !reflect.DeepEqual(record.InputArtifacts, expected) {
		return fmt.Errorf("saved scan input references differ")
	}
	return nil
}

func (p *execution) materialize(ctx context.Context, job scanplan.ScanJob) (map[string]contracts.ArtifactRef, error) {
	inputs := map[string]contracts.ArtifactRef{}
	for name, ref := range job.Artifacts {
		inputs[name] = ref.Clone()
	}
	if job.Request != nil {
		data, err := strictjson.Canonical(job.Request)
		if err != nil {
			return nil, err
		}
		ref, err := p.factory.artifacts.Create(ctx, p.invocation.RunID, p.internalTarget("request", job.ID), artifacts.Payload{MediaType: "application/vnd.contractor.http-request+json", Data: data})
		if err != nil {
			return nil, err
		}
		binding := job.Execution.Arguments["request_ref"]
		if binding.Source != "artifact" || binding.Name == "" {
			return nil, fmt.Errorf("scan request binding is invalid")
		}
		inputs[binding.Name] = ref
	}
	return inputs, nil
}

func (p *execution) request(job scanplan.ScanJob, record planner.ScanJobRecord, deadline time.Time) (contracts.StageContentRequest, error) {
	request := contracts.StageContentRequest{APIVersion: contracts.APIVersion, SubtaskID: job.ID, Objective: p.invocation.Stage.Objective, Instructions: p.invocation.Stage.Instructions.Text, Parameters: job.Parameters, Artifacts: record.InputArtifacts, ResultArtifacts: map[string]contracts.ArtifactRef{job.Execution.ResultArtifact: p.jobOutput(job)}, Deadline: deadline}
	if _, selected := p.invocation.Stage.Result.Artifacts["materials"]; selected {
		for _, name := range []string{"candidates", "evidence"} {
			ref := p.jobOutput(job)
			ref.Name += "." + name
			request.ResultArtifacts[name] = ref
		}
	}
	if err := request.Validate(); err != nil {
		return contracts.StageContentRequest{}, err
	}
	return request, nil
}

func (p *execution) internalTarget(kind, id string) contracts.ArtifactRef {
	return contracts.ArtifactRef{Namespace: p.invocation.Stage.Result.Artifacts["report"].From.Namespace, Name: kind + "." + stableHash(p.invocation.StageExecutionID+"\x00"+id)}
}

func (p *execution) jobOutput(job scanplan.ScanJob) contracts.ArtifactRef {
	return contracts.ArtifactRef{Namespace: job.Namespace, Name: "scan." + stableHash(p.invocation.StageExecutionID+"\x00"+job.ID)}
}

func stableHash(value string) string {
	sum := sha256.Sum256([]byte(value))
	return hex.EncodeToString(sum[:])
}
func minTime(a, b time.Time) time.Time {
	if a.Before(b) {
		return a
	}
	return b
}
