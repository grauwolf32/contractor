package session

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strings"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
)

type scanPersistentState struct {
	OwnerClaimID string            `json:"ownerClaimId"`
	State        planner.ScanState `json:"state"`
}

type scanEvent struct {
	Kind       string                 `json:"kind"`
	Job        *planner.ScanJobRecord `json:"job,omitempty"`
	Plan       *contracts.ArtifactRef `json:"plan,omitempty"`
	PlanDigest string                 `json:"planDigest,omitempty"`
}

var _ planner.ScanSessionService = (*Service)(nil)

func (s *Service) BeginScan(ctx context.Context, stageID, claimID string) (planner.ScanSessionStart, error) {
	if !validScanName(claimID, 256) {
		return planner.ScanSessionStart{}, fmt.Errorf("scan Scheduler claim is invalid")
	}
	started, err := s.Begin(ctx, stageID)
	if err != nil && !errors.Is(err, planner.ErrInvocationInProgress) {
		return planner.ScanSessionStart{}, err
	}
	identity := planner.ScanSessionIdentity{SessionIdentity: started.Identity, SchedulerClaimID: claimID}
	if errors.Is(err, planner.ErrInvocationInProgress) {
		execution, loadErr := s.store.GetStageExecution(ctx, stageID)
		if loadErr != nil {
			return planner.ScanSessionStart{}, loadErr
		}
		if execution.PlannerSessionID == nil || execution.PlannerInvocationID == nil {
			return planner.ScanSessionStart{}, fmt.Errorf("scan session identity is unavailable")
		}
		identity.SessionIdentity = planner.SessionIdentity{
			SessionID: *execution.PlannerSessionID, StageExecutionID: stageID,
			InvocationID: *execution.PlannerInvocationID,
		}
	}
	for attempt := 0; attempt < maxADKAppendAttempts; attempt++ {
		session, state, loadErr := s.load(ctx, identity.SessionIdentity)
		if loadErr != nil {
			return planner.ScanSessionStart{}, loadErr
		}
		if state.Status == statusCompleted {
			if state.Scan == nil || state.Completion == nil {
				return planner.ScanSessionStart{}, fmt.Errorf("completed session is not a scan session")
			}
			completion, cloneErr := cloneCompletion(*state.Completion)
			if cloneErr != nil {
				return planner.ScanSessionStart{}, cloneErr
			}
			return planner.ScanSessionStart{Identity: identity, State: cloneScanState(state.Scan.State), Completion: &completion}, nil
		}
		if state.Scan != nil && state.Scan.OwnerClaimID == claimID {
			return planner.ScanSessionStart{}, planner.ErrInvocationInProgress
		}
		if state.Scan == nil && (state.RequestRecorded || state.Plan != nil || state.ADKEventCount != 0) {
			return planner.ScanSessionStart{}, fmt.Errorf("running session is not a scan session")
		}
		next := state
		next.Scan = &scanPersistentState{OwnerClaimID: claimID, State: planner.ScanState{Jobs: []planner.ScanJobRecord{}}}
		if state.Scan != nil {
			next.Scan.State = cloneScanState(state.Scan.State)
		}
		for index := range next.Scan.State.Jobs {
			job := &next.Scan.State.Jobs[index]
			if job.Status == planner.ScanJobStarted {
				job.Status, job.Code = planner.ScanJobUnknown, "scan_outcome_unknown"
			}
		}
		next.NextSequence++
		err = s.appendScan(ctx, identity, session, state.NextSequence, next, scanEvent{Kind: "scan_owner_acquired"})
		if err == nil {
			return planner.ScanSessionStart{Identity: identity, Invoke: true, State: cloneScanState(next.Scan.State)}, nil
		}
		if !errors.Is(err, runstore.ErrConflict) {
			// An ambiguous commit never grants dispatch ownership.
			return planner.ScanSessionStart{}, err
		}
	}
	return planner.ScanSessionStart{}, runstore.ErrConflict
}

func (s *Service) InitializeScan(ctx context.Context, identity planner.ScanSessionIdentity, scan planner.ScanState) error {
	scan = cloneScanState(scan)
	if err := validateScanState(scan); err != nil {
		return err
	}
	if scan.Plan == nil {
		return fmt.Errorf("scan plan artifact is required")
	}
	for _, job := range scan.Jobs {
		if job.Status != planner.ScanJobPending {
			return fmt.Errorf("initial scan jobs must be pending")
		}
	}
	for attempt := 0; attempt < maxADKAppendAttempts; attempt++ {
		session, state, err := s.loadOwnedScan(ctx, identity)
		if err != nil {
			return err
		}
		if state.Scan.State.Plan != nil {
			if reflect.DeepEqual(state.Scan.State, scan) {
				return nil
			}
			return fmt.Errorf("scan plan is immutable: %w", runstore.ErrConflict)
		}
		next := state
		next.Scan = &scanPersistentState{OwnerClaimID: identity.SchedulerClaimID, State: cloneScanState(scan)}
		next.NextSequence++
		event := scanEvent{Kind: "scan_plan_initialized", Plan: scan.Plan, PlanDigest: scan.PlanDigest}
		payload, err := encodeBounded(event)
		if err != nil {
			return err
		}
		next.RequestRecorded, next.RequestDigest = true, jsonDigest(payload)
		if err := s.appendScan(ctx, identity, session, state.NextSequence, next, event); err != nil {
			if errors.Is(err, runstore.ErrConflict) {
				continue
			}
			return err
		}
		return nil
	}
	return runstore.ErrConflict
}

func (s *Service) ClaimScanJob(ctx context.Context, identity planner.ScanSessionIdentity, jobID string) (bool, error) {
	for attempt := 0; attempt < maxADKAppendAttempts; attempt++ {
		session, state, err := s.loadOwnedScan(ctx, identity)
		if err != nil {
			return false, err
		}
		index := scanJobIndex(state.Scan.State, jobID)
		if index < 0 {
			return false, fmt.Errorf("scan job is not in the persisted plan")
		}
		if state.Scan.State.Jobs[index].Status != planner.ScanJobPending {
			return false, nil
		}
		next := state
		next.Scan = &scanPersistentState{OwnerClaimID: identity.SchedulerClaimID, State: cloneScanState(state.Scan.State)}
		next.Scan.State.Jobs[index].Status = planner.ScanJobStarted
		next.NextSequence++
		err = s.appendScan(ctx, identity, session, state.NextSequence, next, scanEvent{
			Kind: "scan_job_started", Job: &next.Scan.State.Jobs[index],
		})
		if err == nil {
			return true, nil
		}
		if !errors.Is(err, runstore.ErrConflict) {
			// Never promote a lost acknowledgement to permission to invoke.
			return false, err
		}
	}
	return false, runstore.ErrConflict
}

func (s *Service) FinishScanJob(ctx context.Context, identity planner.ScanSessionIdentity, job planner.ScanJobRecord) error {
	job = cloneScanJob(job)
	if err := validateScanJob(job); err != nil {
		return err
	}
	if !terminalScanStatus(job.Status) {
		return fmt.Errorf("scan job completion must be terminal")
	}
	for attempt := 0; attempt < maxADKAppendAttempts; attempt++ {
		session, state, err := s.loadOwnedScan(ctx, identity)
		if err != nil {
			return err
		}
		index := scanJobIndex(state.Scan.State, job.ID)
		if index < 0 {
			return fmt.Errorf("scan job is not in the persisted plan")
		}
		previous := state.Scan.State.Jobs[index]
		if previous.Worker != job.Worker || !reflect.DeepEqual(previous.InputArtifacts, job.InputArtifacts) {
			return fmt.Errorf("scan job identity differs: %w", runstore.ErrConflict)
		}
		if terminalScanStatus(previous.Status) {
			if reflect.DeepEqual(previous, job) {
				return nil
			}
			return fmt.Errorf("scan job is already terminal: %w", runstore.ErrConflict)
		}
		if previous.Status == planner.ScanJobPending && job.Status != planner.ScanJobIncomplete && job.Status != planner.ScanJobUnavailable {
			return fmt.Errorf("scan job cannot complete without dispatch intent")
		}
		next := state
		next.Scan = &scanPersistentState{OwnerClaimID: identity.SchedulerClaimID, State: cloneScanState(state.Scan.State)}
		next.Scan.State.Jobs[index] = cloneScanJob(job)
		next.NextSequence++
		err = s.appendScan(ctx, identity, session, state.NextSequence, next, scanEvent{Kind: "scan_job_finished", Job: &job})
		if err == nil {
			return nil
		}
		if !errors.Is(err, runstore.ErrConflict) {
			return err
		}
	}
	return runstore.ErrConflict
}

func (s *Service) CompleteScan(ctx context.Context, identity planner.ScanSessionIdentity, completion planner.Completion) error {
	cloned, err := cloneCompletion(completion)
	if err != nil {
		return err
	}
	for attempt := 0; attempt < maxADKAppendAttempts; attempt++ {
		session, state, err := s.loadScan(ctx, identity)
		if err != nil {
			return err
		}
		if state.Status == statusCompleted {
			if state.Completion != nil && reflect.DeepEqual(*state.Completion, cloned) {
				return nil
			}
			return fmt.Errorf("scan completion already differs: %w", runstore.ErrConflict)
		}
		for _, job := range state.Scan.State.Jobs {
			if !terminalScanStatus(job.Status) {
				return fmt.Errorf("scan jobs remain pending or started")
			}
		}
		next := state
		next.Status, next.Completion = statusCompleted, &cloned
		next.NextSequence++
		if !next.RequestRecorded {
			// Input preparation can fail before a plan exists. Such a session
			// can complete without creating any dispatchable jobs.
			next.RequestRecorded, next.RequestDigest = true, jsonDigest([]byte(`{"kind":"scan_input_unavailable"}`))
		}
		payload, err := encodeCompletionEvent(cloned)
		if err != nil {
			return err
		}
		kind, fields := planner.PlannerEventCompleted, &plannerRunEventData{}
		if cloned.Failure != nil {
			kind = planner.PlannerEventFailed
			fields.Outcome, fields.Code = "failure", cloned.Failure.Code
		} else {
			fields.Outcome = string(cloned.Result.Outcome)
		}
		runData, err := encodePlannerRunEvent(identity.SessionIdentity, kind, fields)
		if err != nil {
			return err
		}
		runKind, err := toRunEventKind(kind)
		if err != nil {
			return err
		}
		err = s.appendScanPayload(ctx, identity, session, state.NextSequence, next, payload, runKind, runData)
		if err == nil {
			return nil
		}
		if !errors.Is(err, runstore.ErrConflict) {
			return err
		}
	}
	return runstore.ErrConflict
}

func (s *Service) loadScan(ctx context.Context, identity planner.ScanSessionIdentity) (runstore.PlannerSession, persistentState, error) {
	if !validScanName(identity.SchedulerClaimID, 256) {
		return runstore.PlannerSession{}, persistentState{}, fmt.Errorf("scan Scheduler claim is invalid")
	}
	session, state, err := s.load(ctx, identity.SessionIdentity)
	if err != nil {
		return session, state, err
	}
	if state.Scan == nil || state.Scan.OwnerClaimID != identity.SchedulerClaimID {
		return session, state, fmt.Errorf("scan session ownership differs: %w", runstore.ErrConflict)
	}
	return session, state, nil
}

func (s *Service) loadOwnedScan(ctx context.Context, identity planner.ScanSessionIdentity) (runstore.PlannerSession, persistentState, error) {
	session, state, err := s.loadScan(ctx, identity)
	if err == nil && state.Status != statusRunning {
		err = fmt.Errorf("scan session is not running")
	}
	return session, state, err
}

func (s *Service) appendScan(ctx context.Context, identity planner.ScanSessionIdentity, session runstore.PlannerSession, sequence int64, state persistentState, event scanEvent) error {
	payload, err := encodeBounded(event)
	if err != nil {
		return err
	}
	// Public events expose a closed action code, never private request data.
	runData, err := encodeBounded(plannerRunEventData{
		StageExecutionID: identity.StageExecutionID, SessionID: identity.SessionID,
		InvocationID: identity.InvocationID,
		Activity:     &adkEventFacts{Kind: event.Kind, Author: "scan_planner", FunctionCalls: []string{}, FunctionResults: []string{}},
	})
	if err != nil {
		return err
	}
	return s.appendScanPayload(ctx, identity, session, sequence, state, payload, runstore.RunEventPlannerActivity, runData)
}

func (s *Service) appendScanPayload(ctx context.Context, identity planner.ScanSessionIdentity, session runstore.PlannerSession, sequence int64, state persistentState, payload []byte, kind runstore.RunEventKind, runData []byte) error {
	encoded, err := encodeState(state)
	if err != nil {
		return err
	}
	eventID, err := s.nextID("planner_event_")
	if err != nil {
		return fmt.Errorf("generate scan event identity: %w", err)
	}
	return s.store.AppendPlannerEvent(ctx, runstore.AppendPlannerEventParams{
		EventID: eventID, SessionID: session.SessionID,
		StageExecutionID: identity.StageExecutionID, InvocationID: identity.InvocationID,
		SchedulerClaimID: identity.SchedulerClaimID, SequenceNumber: sequence,
		EventSchemaVersion: contracts.APIVersion, Event: payload,
		NewStateSchemaVersion: contracts.APIVersion, NewState: encoded,
		RunEvent: runstore.RunEventAppend{EventID: eventID, EventSchemaVersion: contracts.APIVersion, Kind: kind, Data: runData},
	})
}

func validateScanPersistent(value scanPersistentState) error {
	if !validScanName(value.OwnerClaimID, 256) {
		return fmt.Errorf("persisted scan owner is invalid")
	}
	return validateScanState(value.State)
}

func validateScanState(value planner.ScanState) error {
	if len(value.Jobs) > planner.MaxScanJobs {
		return fmt.Errorf("scan job limit exceeded")
	}
	if value.Plan == nil {
		if value.PlanDigest != "" || len(value.Jobs) != 0 {
			return fmt.Errorf("scan jobs require a persisted plan")
		}
		return nil
	}
	if !validScanRef(*value.Plan) || !validDigest(value.PlanDigest) || value.PlanDigest != strings.ToLower(value.PlanDigest) {
		return fmt.Errorf("scan plan reference or digest is invalid")
	}
	seen := map[string]bool{}
	for _, job := range value.Jobs {
		if err := validateScanJob(job); err != nil {
			return err
		}
		if seen[job.ID] {
			return fmt.Errorf("scan job IDs must be unique")
		}
		seen[job.ID] = true
	}
	return nil
}

func validateScanJob(job planner.ScanJobRecord) error {
	if !validScanName(job.ID, 128) || !validScanName(job.Worker, 128) || len(job.InputArtifacts) > 16 {
		return fmt.Errorf("scan job identity or inputs are invalid")
	}
	if job.Status != planner.ScanJobPending && job.Status != planner.ScanJobStarted && !terminalScanStatus(job.Status) {
		return fmt.Errorf("scan job status is invalid")
	}
	if job.Code != "" && !validScanCode(job.Code) {
		return fmt.Errorf("scan job code is invalid")
	}
	if !terminalScanStatus(job.Status) && (job.Code != "" || job.Report != nil) {
		return fmt.Errorf("unresolved scan job contains a result")
	}
	for name, ref := range job.InputArtifacts {
		if !validScanName(name, 128) || !validScanRef(ref) {
			return fmt.Errorf("scan job input reference is invalid")
		}
	}
	if job.Report != nil && !validScanRef(*job.Report) {
		return fmt.Errorf("scan job report reference is invalid")
	}
	return nil
}

func validScanCode(value string) bool {
	if len(value) > 128 || len(value) == 0 || value[0] < 'a' || value[0] > 'z' {
		return false
	}
	for _, char := range value {
		if char < 'a' || char > 'z' {
			if (char < '0' || char > '9') && char != '_' {
				return false
			}
		}
	}
	return true
}

func validScanName(value string, limit int) bool {
	return len(value) <= limit && validPlannerFactKey(value)
}

func validScanRef(ref contracts.ArtifactRef) bool {
	return ref.ValidateExact() == nil && len(ref.Namespace) <= 256 && len(ref.Name) <= 256 && len(*ref.Revision) <= 256
}

func terminalScanStatus(status string) bool {
	switch status {
	case planner.ScanJobCompleted, planner.ScanJobFailed, planner.ScanJobIncomplete, planner.ScanJobUnavailable, planner.ScanJobUnknown:
		return true
	default:
		return false
	}
}

func scanJobIndex(state planner.ScanState, id string) int {
	for index, job := range state.Jobs {
		if job.ID == id {
			return index
		}
	}
	return -1
}

func cloneScanState(state planner.ScanState) planner.ScanState {
	result := state
	if state.Plan != nil {
		ref := cloneScanRef(*state.Plan)
		result.Plan = &ref
	}
	result.Jobs = make([]planner.ScanJobRecord, len(state.Jobs))
	for index, job := range state.Jobs {
		result.Jobs[index] = cloneScanJob(job)
	}
	return result
}

func cloneScanJob(job planner.ScanJobRecord) planner.ScanJobRecord {
	result := job
	result.InputArtifacts = make(map[string]contracts.ArtifactRef, len(job.InputArtifacts))
	for name, ref := range job.InputArtifacts {
		result.InputArtifacts[name] = cloneScanRef(ref)
	}
	if job.Report != nil {
		ref := cloneScanRef(*job.Report)
		result.Report = &ref
	}
	return result
}

func cloneScanRef(ref contracts.ArtifactRef) contracts.ArtifactRef {
	if ref.Revision != nil {
		revision := *ref.Revision
		ref.Revision = &revision
	}
	return ref
}
