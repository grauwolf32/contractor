package scan

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"reflect"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/scanplan"
)

func (p *execution) Run(ctx context.Context) (contracts.StageContentResult, error) {
	empty := contracts.StageContentResult{}
	start, err := p.factory.sessions.BeginScan(ctx, p.invocation.StageExecutionID, p.invocation.SchedulerClaimID)
	if err != nil {
		return empty, scanError("scan_session_unavailable", err)
	}
	if start.Completion != nil {
		return p.recoverCompletion(ctx, *start.Completion)
	}
	if !start.Invoke {
		return empty, scanError("scan_session_unavailable", nil)
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
				return empty, scanError("scan_job_claim_unavailable", err)
			}
			record.Status = planner.ScanJobStarted
			request, err := p.request(job, record)
			if err != nil {
				record.Status, record.Code = planner.ScanJobFailed, "scan_invalid_worker_result"
			} else {
				deadline := minTime(p.invocation.Deadline, time.Now().Add(time.Duration(job.TimeoutSeconds)*time.Second))
				callCtx, cancel := context.WithDeadline(ctx, deadline)
				completion, invokeErr := p.factory.invoker.Invoke(callCtx, job.Worker, planner.CloneWorkerHandle(p.invocation.Workers[job.Worker]), planner.CloneStageRequest(request))
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
		writeCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
		err := p.factory.sessions.FinishScanJob(writeCtx, start.Identity, record)
		cancel()
		if err != nil {
			return empty, scanError("scan_job_result_unavailable", err)
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
		return emptyPlan, emptyState, scanError("scan_source_unavailable", nil)
	}
	payload, err := p.factory.artifacts.Read(ctx, p.invocation.RunID, *source, scanplan.MaxPlanBytes)
	if err != nil {
		return emptyPlan, emptyState, scanError("scan_source_unavailable", err)
	}
	bindings := map[string]scanplan.ToolBinding{}
	for name, binding := range p.invocation.Stage.Agents {
		bindings[name] = scanplan.ToolBinding{Namespace: binding.Namespace, Template: binding.Template}
	}
	wordlists := map[string]contracts.ArtifactRef{}
	for _, tool := range policy.Tools {
		if tool.WordlistArtifact != "" {
			ref := p.invocation.Context.Artifacts[tool.WordlistArtifact]
			if ref == nil {
				return emptyPlan, emptyState, scanError("scan_source_unavailable", nil)
			}
			wordlists[tool.WordlistArtifact] = planner.CloneArtifactRef(*ref)
		}
	}
	plan, err := scanplan.BuildPlan(scanplan.PlanInput{Artifact: *source, MediaType: payload.MediaType, Data: payload.Data}, policy, bindings, wordlists)
	if err != nil {
		return emptyPlan, emptyState, scanError("scan_plan_invalid", err)
	}
	data, err := scanplan.MarshalPlan(plan)
	if err != nil {
		return emptyPlan, emptyState, scanError("scan_plan_invalid", err)
	}
	state := start.State
	if state.Plan != nil {
		frozen, err := p.factory.artifacts.Read(ctx, p.invocation.RunID, *state.Plan, scanplan.MaxPlanBytes)
		if err != nil || frozen.MediaType != scanplan.PlanMediaType || state.PlanDigest != "sha256:"+stableHash(string(data)) || !bytes.Equal(frozen.Data, data) || len(state.Jobs) != len(plan.Jobs) {
			return emptyPlan, emptyState, scanError("scan_saved_plan_invalid", err)
		}
		for i, job := range plan.Jobs {
			record := state.Jobs[i]
			if record.ID != job.ID || record.Worker != job.Worker {
				return emptyPlan, emptyState, scanError("scan_saved_plan_invalid", nil)
			}
			if err := p.validateSavedInputs(ctx, job, record); err != nil {
				return emptyPlan, emptyState, scanError("scan_saved_input_invalid", err)
			}
		}
		return plan, state, nil
	}
	planRef, err := p.factory.artifacts.Create(ctx, p.invocation.RunID, p.internalTarget("plan", plan.ID), artifacts.Payload{MediaType: scanplan.PlanMediaType, Data: data})
	if err != nil {
		return emptyPlan, emptyState, scanError("scan_plan_write_failed", err)
	}
	state = planner.ScanState{Plan: &planRef, PlanDigest: "sha256:" + stableHash(string(data)), Jobs: []planner.ScanJobRecord{}}
	for _, job := range plan.Jobs {
		inputs, err := p.materialize(ctx, job)
		if err != nil {
			return emptyPlan, emptyState, scanError("scan_input_write_failed", err)
		}
		state.Jobs = append(state.Jobs, planner.ScanJobRecord{ID: job.ID, Worker: job.Worker, Status: planner.ScanJobPending, InputArtifacts: inputs})
	}
	if err := p.factory.sessions.InitializeScan(ctx, start.Identity, state); err != nil {
		return emptyPlan, emptyState, scanError("scan_plan_record_failed", err)
	}
	return plan, state, nil
}

func (p *execution) validateSavedInputs(ctx context.Context, job scanplan.ScanJob, record planner.ScanJobRecord) error {
	expected := map[string]contracts.ArtifactRef{}
	for name, ref := range job.Artifacts {
		expected[name] = planner.CloneArtifactRef(ref)
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
			return err
		}
		data, err := contracts.MarshalPrivateCanonical(job.Request)
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
		inputs[name] = planner.CloneArtifactRef(ref)
	}
	if job.Request != nil {
		data, err := contracts.MarshalPrivateCanonical(job.Request)
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

func (p *execution) request(job scanplan.ScanJob, record planner.ScanJobRecord) (contracts.StageContentRequest, error) {
	request := contracts.StageContentRequest{APIVersion: contracts.APIVersion, SubtaskID: job.ID, Objective: p.invocation.Stage.Objective, Instructions: p.invocation.Stage.Instructions.Text, Parameters: job.Parameters, Artifacts: record.InputArtifacts, ResultArtifacts: map[string]contracts.ArtifactRef{job.Execution.ResultArtifact: p.jobOutput(job)}}
	if err := request.Validate(); err != nil {
		return contracts.StageContentRequest{}, err
	}
	data, err := contracts.MarshalPrivateCanonical(request)
	if err != nil || len(data) > contracts.MaxStageRequestBytes {
		return contracts.StageContentRequest{}, fmt.Errorf("scan Worker request exceeds its bound")
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
