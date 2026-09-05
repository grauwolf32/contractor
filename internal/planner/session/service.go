// Package session adapts Contractor's PostgreSQL RunStore into the durable
// session boundary used by Planner implementations.
package session

import (
	"bytes"
	"context"
	cryptorand "crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"reflect"
	"sort"
	"strings"
	"sync"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
)

const (
	maxSessionJSONBytes     = 2 * 1024 * 1024
	maxRecordedPlannerFacts = 256
	maxPlannerFactKeyBytes  = 256
)

type Store interface {
	GetStageExecution(context.Context, string) (runstore.StageExecution, error)
	StartPlanner(context.Context, runstore.StartPlannerParams) error
	GetPlannerSession(context.Context, string) (runstore.PlannerSession, error)
	AppendPlannerEvent(context.Context, runstore.AppendPlannerEventParams) error
}

type Options struct {
	NewID func(string) (string, error)
}

type Service struct {
	store Store
	newID func(string) (string, error)
	idMu  sync.Mutex
}

func New(store Store, options Options) (*Service, error) {
	if store == nil {
		return nil, fmt.Errorf("Planner session store is required")
	}
	if options.NewID == nil {
		options.NewID = randomID
	}
	return &Service{store: store, newID: options.NewID}, nil
}

func (s *Service) Begin(
	ctx context.Context, stageExecutionID string,
) (planner.SessionStart, error) {
	if strings.TrimSpace(stageExecutionID) == "" {
		return planner.SessionStart{}, fmt.Errorf("StageExecution ID is required")
	}
	execution, err := s.store.GetStageExecution(ctx, stageExecutionID)
	if err != nil {
		return planner.SessionStart{}, fmt.Errorf("load StageExecution: %w", err)
	}
	if execution.State == runstore.StagePreparing {
		identity, err := s.newIdentity(stageExecutionID)
		if err != nil {
			return planner.SessionStart{}, err
		}
		initial, err := encodeState(persistentState{Status: statusRunning, NextSequence: 2})
		if err != nil {
			return planner.SessionStart{}, err
		}
		startedEventID, err := s.nextID("planner_event_")
		if err != nil {
			return planner.SessionStart{}, fmt.Errorf("generate Planner started event ID: %w", err)
		}
		startedEvent, err := encodeBounded(startedSessionEvent{Kind: "planner_started"})
		if err != nil {
			return planner.SessionStart{}, err
		}
		startedData, err := encodePlannerRunEvent(identity, planner.PlannerEventStarted, nil)
		if err != nil {
			return planner.SessionStart{}, err
		}
		err = s.store.StartPlanner(ctx, runstore.StartPlannerParams{
			StageExecutionID:   stageExecutionID,
			SessionID:          identity.SessionID,
			InvocationID:       identity.InvocationID,
			StateSchemaVersion: contracts.APIVersion,
			InitialState:       initial,
			EventID:            startedEventID,
			EventSchemaVersion: contracts.APIVersion,
			Event:              startedEvent,
			Reason:             runstore.Reason{Code: "planner_started"},
			RunEvent: runstore.RunEventAppend{
				EventID: startedEventID, EventSchemaVersion: contracts.APIVersion,
				Kind: runstore.RunEventPlannerStarted, Data: startedData,
			},
		})
		if err == nil {
			return planner.SessionStart{Identity: identity, Invoke: true}, nil
		}
		if !errors.Is(err, runstore.ErrConflict) {
			return planner.SessionStart{}, fmt.Errorf("create Planner session: %w", err)
		}
		execution, err = s.store.GetStageExecution(ctx, stageExecutionID)
		if err != nil {
			return planner.SessionStart{}, fmt.Errorf("reload StageExecution: %w", err)
		}
	}
	return s.existing(ctx, execution)
}

func (s *Service) RecordRequest(
	ctx context.Context, identity planner.SessionIdentity, facts planner.RequestFacts,
) error {
	session, state, err := s.load(ctx, identity)
	if err != nil {
		return err
	}
	if state.Status != statusRunning {
		return fmt.Errorf("Planner session is not running")
	}
	payload, err := encodeRequestEvent(facts)
	if err != nil {
		return err
	}
	digest := jsonDigest(payload)
	if state.RequestRecorded {
		if state.RequestDigest == digest {
			return nil
		}
		return fmt.Errorf("Planner request event already differs")
	}
	next := state
	next.RequestRecorded = true
	next.RequestDigest = digest
	next.NextSequence++
	encodedState, err := encodeState(next)
	if err != nil {
		return err
	}
	if err := s.append(
		ctx, session, state.NextSequence, payload, encodedState,
		identity, planner.PlannerEventRequestRecorded, nil,
	); err != nil {
		if errors.Is(err, runstore.ErrConflict) {
			_, recovered, loadErr := s.load(ctx, identity)
			if loadErr == nil && recovered.RequestRecorded && recovered.RequestDigest == digest {
				return nil
			}
		}
		return fmt.Errorf("append Planner request event: %w", err)
	}
	return nil
}

func (s *Service) Complete(
	ctx context.Context, identity planner.SessionIdentity, completion planner.Completion,
) error {
	if err := validateCompletion(completion); err != nil {
		return err
	}
	session, state, err := s.load(ctx, identity)
	if err != nil {
		return err
	}
	if state.Status == statusCompleted {
		if state.Completion != nil && reflect.DeepEqual(*state.Completion, completion) {
			return nil
		}
		return fmt.Errorf("Planner completion already differs")
	}
	if state.Status != statusRunning || !state.RequestRecorded {
		return fmt.Errorf("Planner session cannot complete before its request event")
	}
	payload, err := encodeCompletionEvent(completion)
	if err != nil {
		return err
	}
	next := state
	next.Status = statusCompleted
	next.NextSequence++
	cloned, err := cloneCompletion(completion)
	if err != nil {
		return err
	}
	next.Completion = &cloned
	encodedState, err := encodeState(next)
	if err != nil {
		return err
	}
	eventKind := planner.PlannerEventCompleted
	runFields := &plannerRunEventData{}
	if completion.Failure != nil {
		eventKind = planner.PlannerEventFailed
		runFields = &plannerRunEventData{Outcome: "failure", Code: completion.Failure.Code}
	} else {
		runFields.Outcome = string(completion.Result.Outcome)
	}
	if err := s.append(
		ctx, session, state.NextSequence, payload, encodedState,
		identity, eventKind, runFields,
	); err != nil {
		if errors.Is(err, runstore.ErrConflict) {
			_, recovered, loadErr := s.load(ctx, identity)
			if loadErr == nil && recovered.Status == statusCompleted &&
				recovered.Completion != nil && reflect.DeepEqual(*recovered.Completion, cloned) {
				return nil
			}
		}
		return fmt.Errorf("append Planner completion event: %w", err)
	}
	return nil
}

func (s *Service) RecordPlan(
	ctx context.Context,
	identity planner.SessionIdentity,
	transition planner.PlannerPlanTransition,
) error {
	for attempt := 0; attempt < maxADKAppendAttempts; attempt++ {
		session, state, err := s.load(ctx, identity)
		if err != nil {
			return err
		}
		if state.Status != statusRunning {
			return fmt.Errorf("Planner session is not running")
		}
		projection := clonePlanProjection(transition.Plan)
		payload, err := encodeBounded(planProjectionEvent{
			Kind: string(transition.Kind), ExpectedRevision: transition.ExpectedRevision, Plan: projection,
		})
		if err != nil {
			return err
		}
		transitionDigest := jsonDigest(payload)
		if state.Plan != nil && state.Plan.Revision == transition.Plan.Revision {
			if reflect.DeepEqual(*state.Plan, transition.Plan) &&
				state.LastPlanTransitionDigest == transitionDigest {
				return nil
			}
			return fmt.Errorf("Planner plan revision already differs: %w", runstore.ErrConflict)
		}
		currentRevision := uint64(0)
		if state.Plan != nil {
			currentRevision = state.Plan.Revision
		}
		if transition.ExpectedRevision != currentRevision {
			return fmt.Errorf("Planner plan compare revision differs: %w", runstore.ErrConflict)
		}
		if err := planner.ValidatePlannerPlanTransition(state.Plan, transition.Plan, transition.Kind); err != nil {
			return fmt.Errorf("invalid Planner plan transition: %w", err)
		}
		next := state
		next.NextSequence++
		next.Plan = &projection
		next.LastPlanTransitionDigest = transitionDigest
		encodedState, err := encodeState(next)
		if err != nil {
			return err
		}
		runFields := planTransitionRunFields(state.Plan, projection, transition.Kind)
		err = s.append(
			ctx, session, state.NextSequence, payload, encodedState,
			identity, transition.Kind, &runFields,
		)
		if err == nil {
			return nil
		}
		if !errors.Is(err, runstore.ErrConflict) {
			return fmt.Errorf("append Planner plan transition: %w", err)
		}
	}
	_, recovered, err := s.load(ctx, identity)
	if err == nil && recovered.Plan != nil && reflect.DeepEqual(*recovered.Plan, transition.Plan) {
		projection := clonePlanProjection(transition.Plan)
		payload, encodeErr := encodeBounded(planProjectionEvent{
			Kind: string(transition.Kind), ExpectedRevision: transition.ExpectedRevision, Plan: projection,
		})
		if encodeErr == nil && recovered.LastPlanTransitionDigest == jsonDigest(payload) {
			return nil
		}
	}
	return fmt.Errorf("append Planner plan transition: %w", runstore.ErrConflict)
}

func (s *Service) RecordFact(
	ctx context.Context,
	identity planner.SessionIdentity,
	fact planner.PlannerFact,
) error {
	if err := validatePlannerFact(fact); err != nil {
		return err
	}
	payload, err := encodeBounded(plannerFactEvent{
		Kind: string(fact.Kind), Key: fact.Key, PlanRevision: fact.PlanRevision,
		SubtaskID: fact.SubtaskID, CallID: fact.CallID, WorkerName: fact.WorkerName,
		Outcome: fact.Outcome, Code: fact.Code,
	})
	if err != nil {
		return err
	}
	digest := jsonDigest(payload)
	for attempt := 0; attempt < maxADKAppendAttempts; attempt++ {
		session, state, loadErr := s.load(ctx, identity)
		if loadErr != nil {
			return loadErr
		}
		if state.Status != statusRunning {
			return fmt.Errorf("Planner session is not running")
		}
		if existing, recorded := state.FactDigests[fact.Key]; recorded {
			if existing == digest {
				return nil
			}
			return fmt.Errorf("Planner fact key already differs: %w", runstore.ErrConflict)
		}
		currentRevision := uint64(0)
		if state.Plan != nil {
			currentRevision = state.Plan.Revision
		}
		if fact.PlanRevision != currentRevision {
			return fmt.Errorf("Planner fact plan revision differs: %w", runstore.ErrConflict)
		}
		if len(state.FactDigests) >= maxRecordedPlannerFacts {
			return fmt.Errorf("Planner fact limit reached")
		}
		next := state
		next.NextSequence++
		next.FactDigests = cloneStringMap(state.FactDigests)
		next.FactDigests[fact.Key] = digest
		encodedState, encodeErr := encodeState(next)
		if encodeErr != nil {
			return encodeErr
		}
		revision := fact.PlanRevision
		runFields := plannerRunEventData{
			PlanRevision: &revision, SubtaskID: fact.SubtaskID, CallID: fact.CallID,
			WorkerName: fact.WorkerName, Outcome: fact.Outcome, Code: fact.Code,
		}
		err = s.append(
			ctx, session, state.NextSequence, payload, encodedState,
			identity, fact.Kind, &runFields,
		)
		if err == nil {
			return nil
		}
		if !errors.Is(err, runstore.ErrConflict) {
			return fmt.Errorf("append Planner fact: %w", err)
		}
	}
	_, recovered, err := s.load(ctx, identity)
	if err == nil && recovered.FactDigests[fact.Key] == digest {
		return nil
	}
	return fmt.Errorf("append Planner fact: %w", runstore.ErrConflict)
}

func (s *Service) LoadPlan(
	ctx context.Context,
	identity planner.SessionIdentity,
) (planner.PlannerPlanProjection, bool, error) {
	_, state, err := s.load(ctx, identity)
	if err != nil {
		return planner.PlannerPlanProjection{}, false, err
	}
	if state.Plan == nil {
		return planner.PlannerPlanProjection{}, false, nil
	}
	projection := clonePlanProjection(*state.Plan)
	if err := projection.Validate(); err != nil {
		return planner.PlannerPlanProjection{}, false, fmt.Errorf("persisted Planner plan is invalid: %w", err)
	}
	return projection, true, nil
}

func (s *Service) existing(
	ctx context.Context, execution runstore.StageExecution,
) (planner.SessionStart, error) {
	if execution.PlannerSessionID == nil || execution.PlannerInvocationID == nil {
		return planner.SessionStart{}, fmt.Errorf(
			"StageExecution %q has no recoverable Planner session", execution.StageExecutionID,
		)
	}
	identity := planner.SessionIdentity{
		SessionID: *execution.PlannerSessionID, StageExecutionID: execution.StageExecutionID,
		InvocationID: *execution.PlannerInvocationID,
	}
	_, state, err := s.load(ctx, identity)
	if err != nil {
		return planner.SessionStart{}, err
	}
	switch state.Status {
	case statusRunning:
		return planner.SessionStart{}, planner.ErrInvocationInProgress
	case statusCompleted:
		if state.Completion == nil {
			return planner.SessionStart{}, fmt.Errorf("completed Planner session has no completion")
		}
		completion, err := cloneCompletion(*state.Completion)
		if err != nil {
			return planner.SessionStart{}, err
		}
		return planner.SessionStart{Identity: identity, Completion: &completion}, nil
	default:
		return planner.SessionStart{}, fmt.Errorf("unknown Planner session status %q", state.Status)
	}
}

func (s *Service) load(
	ctx context.Context, identity planner.SessionIdentity,
) (runstore.PlannerSession, persistentState, error) {
	if strings.TrimSpace(identity.SessionID) == "" ||
		strings.TrimSpace(identity.StageExecutionID) == "" ||
		strings.TrimSpace(identity.InvocationID) == "" {
		return runstore.PlannerSession{}, persistentState{}, fmt.Errorf("complete session identity is required")
	}
	session, err := s.store.GetPlannerSession(ctx, identity.SessionID)
	if err != nil {
		return runstore.PlannerSession{}, persistentState{}, fmt.Errorf("load Planner session: %w", err)
	}
	if session.StageExecutionID != identity.StageExecutionID ||
		session.InvocationID != identity.InvocationID ||
		session.StateSchemaVersion != contracts.APIVersion {
		return runstore.PlannerSession{}, persistentState{}, fmt.Errorf("Planner session identity differs")
	}
	state, err := decodeState(session.State)
	if err != nil {
		return runstore.PlannerSession{}, persistentState{}, err
	}
	return session, state, nil
}

func (s *Service) append(
	ctx context.Context,
	session runstore.PlannerSession,
	sequence int64,
	event json.RawMessage,
	state json.RawMessage,
	identity planner.SessionIdentity,
	kind planner.PlannerEventKind,
	runFields *plannerRunEventData,
) error {
	eventID, err := s.nextID("planner_event_")
	if err != nil {
		return fmt.Errorf("generate Planner event ID: %w", err)
	}
	runData, err := encodePlannerRunEvent(identity, kind, runFields)
	if err != nil {
		return err
	}
	runKind, err := toRunEventKind(kind)
	if err != nil {
		return err
	}
	return s.store.AppendPlannerEvent(ctx, runstore.AppendPlannerEventParams{
		EventID: eventID, SessionID: session.SessionID,
		StageExecutionID: identity.StageExecutionID, InvocationID: identity.InvocationID,
		SequenceNumber:     sequence,
		EventSchemaVersion: contracts.APIVersion, Event: event,
		NewStateSchemaVersion: contracts.APIVersion, NewState: state,
		RunEvent: runstore.RunEventAppend{
			EventID: eventID, EventSchemaVersion: contracts.APIVersion,
			Kind: runKind, Data: runData,
		},
	})
}

func (s *Service) newIdentity(stageExecutionID string) (planner.SessionIdentity, error) {
	sessionID, err := s.nextID("planner_session_")
	if err != nil {
		return planner.SessionIdentity{}, fmt.Errorf("generate Planner session ID: %w", err)
	}
	invocationID, err := s.nextID("planner_invocation_")
	if err != nil {
		return planner.SessionIdentity{}, fmt.Errorf("generate Planner invocation ID: %w", err)
	}
	return planner.SessionIdentity{
		SessionID: sessionID, StageExecutionID: stageExecutionID, InvocationID: invocationID,
	}, nil
}

func (s *Service) nextID(prefix string) (string, error) {
	s.idMu.Lock()
	defer s.idMu.Unlock()
	return s.newID(prefix)
}

const (
	statusRunning   = "running"
	statusCompleted = "completed"
)

type persistentState struct {
	Status                   string                         `json:"status"`
	NextSequence             int64                          `json:"nextSequence"`
	RequestRecorded          bool                           `json:"requestRecorded"`
	RequestDigest            string                         `json:"requestDigest,omitempty"`
	ADKEventCount            int64                          `json:"adkEventCount,omitempty"`
	ADKInputTokens           int64                          `json:"adkInputTokens,omitempty"`
	ADKOutputTokens          int64                          `json:"adkOutputTokens,omitempty"`
	Plan                     *planner.PlannerPlanProjection `json:"plan,omitempty"`
	LastPlanTransitionDigest string                         `json:"lastPlanTransitionDigest,omitempty"`
	FactDigests              map[string]string              `json:"factDigests,omitempty"`
	Completion               *planner.Completion            `json:"completion,omitempty"`
}

type planProjectionEvent struct {
	Kind             string                        `json:"kind"`
	ExpectedRevision uint64                        `json:"expectedRevision"`
	Plan             planner.PlannerPlanProjection `json:"plan"`
}

type startedSessionEvent struct {
	Kind string `json:"kind"`
}

type plannerFactEvent struct {
	Kind         string `json:"kind"`
	Key          string `json:"key"`
	PlanRevision uint64 `json:"planRevision"`
	SubtaskID    string `json:"subtaskId,omitempty"`
	CallID       string `json:"callId,omitempty"`
	WorkerName   string `json:"workerName,omitempty"`
	Outcome      string `json:"outcome,omitempty"`
	Code         string `json:"code,omitempty"`
}

type plannerRunEventData struct {
	StageExecutionID string                         `json:"stageExecutionId"`
	SessionID        string                         `json:"sessionId"`
	InvocationID     string                         `json:"invocationId"`
	Plan             *planner.PlannerPlanProjection `json:"plan,omitempty"`
	PlanRevision     *uint64                        `json:"planRevision,omitempty"`
	SubtaskID        string                         `json:"subtaskId,omitempty"`
	CallID           string                         `json:"callId,omitempty"`
	WorkerName       string                         `json:"workerName,omitempty"`
	Outcome          string                         `json:"outcome,omitempty"`
	Code             string                         `json:"code,omitempty"`
	Activity         *adkEventFacts                 `json:"activity,omitempty"`
}

type requestEvent struct {
	Kind               string                           `json:"kind"`
	Binding            string                           `json:"binding,omitempty"`
	Bindings           []string                         `json:"bindings,omitempty"`
	ObjectiveDigest    string                           `json:"objectiveDigest"`
	InstructionsDigest string                           `json:"instructionsDigest"`
	ParameterNames     []string                         `json:"parameterNames"`
	Artifacts          map[string]contracts.ArtifactRef `json:"artifacts"`
}

type completionEvent struct {
	Kind    string                        `json:"kind"`
	Outcome string                        `json:"outcome"`
	Result  *contracts.StageContentResult `json:"result,omitempty"`
	Failure *planner.Failure              `json:"failure,omitempty"`
}

func encodeRequestEvent(facts planner.RequestFacts) (json.RawMessage, error) {
	bindings := append([]string(nil), facts.Bindings...)
	sort.Strings(bindings)
	if len(bindings) == 0 || !validDigest(facts.ObjectiveDigest) ||
		!validDigest(facts.InstructionsDigest) {
		return nil, fmt.Errorf("Planner request facts are invalid")
	}
	for index, binding := range bindings {
		if strings.TrimSpace(binding) == "" || index > 0 && bindings[index-1] == binding {
			return nil, fmt.Errorf("Planner request bindings must be non-empty and unique")
		}
	}
	parameterNames := append([]string{}, facts.ParameterNames...)
	if !sort.StringsAreSorted(parameterNames) {
		return nil, fmt.Errorf("Planner parameter names must be sorted")
	}
	for index, name := range parameterNames {
		if strings.TrimSpace(name) == "" || index > 0 && parameterNames[index-1] == name {
			return nil, fmt.Errorf("Planner parameter names must be non-empty and unique")
		}
	}
	artifacts := make(map[string]contracts.ArtifactRef, len(facts.Artifacts))
	for name, ref := range facts.Artifacts {
		if strings.TrimSpace(name) == "" {
			return nil, fmt.Errorf("Planner request artifact name is required")
		}
		if err := ref.ValidateExact(); err != nil {
			return nil, fmt.Errorf("Planner request artifact %q: %w", name, err)
		}
		artifacts[name] = ref
	}
	event := requestEvent{
		Kind: "planner_request", Bindings: bindings,
		ObjectiveDigest: facts.ObjectiveDigest, InstructionsDigest: facts.InstructionsDigest,
		ParameterNames: parameterNames, Artifacts: artifacts,
	}
	if len(bindings) == 1 {
		// Preserve the v1alpha1 single-Worker audit event shape. Router with
		// multiple fixed Workers uses the plural Planner request form.
		event.Kind = "worker_request"
		event.Binding = bindings[0]
		event.Bindings = nil
	}
	return encodeBounded(event)
}

func encodeCompletionEvent(completion planner.Completion) (json.RawMessage, error) {
	cloned, err := cloneCompletion(completion)
	if err != nil {
		return nil, err
	}
	outcome := "failure"
	if cloned.Result != nil {
		outcome = string(cloned.Result.Outcome)
	}
	return encodeBounded(completionEvent{
		Kind: "planner_completed", Outcome: outcome,
		Result: cloned.Result, Failure: cloned.Failure,
	})
}

func encodeState(state persistentState) (json.RawMessage, error) {
	if state.Status != statusRunning && state.Status != statusCompleted || state.NextSequence <= 0 {
		return nil, fmt.Errorf("Planner state is invalid")
	}
	if state.ADKEventCount < 0 || state.ADKInputTokens < 0 || state.ADKOutputTokens < 0 {
		return nil, fmt.Errorf("Planner ADK counters must be non-negative")
	}
	if state.RequestRecorded != (state.RequestDigest != "") {
		return nil, fmt.Errorf("Planner request state is inconsistent")
	}
	if state.Plan != nil {
		if err := state.Plan.Validate(); err != nil {
			return nil, fmt.Errorf("Planner plan state is invalid: %w", err)
		}
		if !validDigest(state.LastPlanTransitionDigest) {
			return nil, fmt.Errorf("Planner plan transition digest is invalid")
		}
	} else if state.LastPlanTransitionDigest != "" {
		return nil, fmt.Errorf("Planner plan transition digest exists without a plan")
	}
	if len(state.FactDigests) > maxRecordedPlannerFacts {
		return nil, fmt.Errorf("Planner fact state exceeds its limit")
	}
	for key, digest := range state.FactDigests {
		if !validPlannerFactKey(key) || !validDigest(digest) {
			return nil, fmt.Errorf("Planner fact state is invalid")
		}
	}
	if state.Status == statusCompleted {
		if state.Completion == nil || !state.RequestRecorded {
			return nil, fmt.Errorf("completed Planner state is incomplete")
		}
		if err := validateCompletion(*state.Completion); err != nil {
			return nil, err
		}
	} else if state.Completion != nil {
		return nil, fmt.Errorf("running Planner state contains a completion")
	}
	return encodeBounded(state)
}

func decodeState(data json.RawMessage) (persistentState, error) {
	if len(data) == 0 || len(data) > maxSessionJSONBytes {
		return persistentState{}, fmt.Errorf("Planner state exceeds its bounded contract")
	}
	var state persistentState
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&state); err != nil {
		return persistentState{}, fmt.Errorf("decode Planner state: %w", err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		return persistentState{}, fmt.Errorf("Planner state contains trailing JSON")
	}
	if _, err := encodeState(state); err != nil {
		return persistentState{}, err
	}
	return state, nil
}

func validateCompletion(completion planner.Completion) error {
	if (completion.Result == nil) == (completion.Failure == nil) {
		return fmt.Errorf("Planner completion requires exactly one result or failure")
	}
	if completion.Result != nil {
		if err := completion.Result.Validate(); err != nil {
			return fmt.Errorf("Planner result completion is invalid: %w", err)
		}
	} else if strings.TrimSpace(completion.Failure.Code) == "" ||
		strings.TrimSpace(completion.Failure.Message) == "" {
		return fmt.Errorf("Planner failure completion is invalid")
	}
	return nil
}

func cloneCompletion(completion planner.Completion) (planner.Completion, error) {
	encoded, err := json.Marshal(completion)
	if err != nil {
		return planner.Completion{}, fmt.Errorf("encode Planner completion: %w", err)
	}
	if len(encoded) > maxSessionJSONBytes {
		return planner.Completion{}, fmt.Errorf("Planner completion exceeds its bounded contract")
	}
	var result planner.Completion
	decoder := json.NewDecoder(bytes.NewReader(encoded))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&result); err != nil {
		return planner.Completion{}, fmt.Errorf("decode Planner completion: %w", err)
	}
	if err := validateCompletion(result); err != nil {
		return planner.Completion{}, err
	}
	return result, nil
}

func encodeBounded(value any) (json.RawMessage, error) {
	encoded, err := json.Marshal(value)
	if err != nil {
		return nil, fmt.Errorf("encode Planner JSON: %w", err)
	}
	if len(encoded) > maxSessionJSONBytes {
		return nil, fmt.Errorf("Planner JSON exceeds its bounded contract")
	}
	return encoded, nil
}

func jsonDigest(value []byte) string {
	digest := sha256.Sum256(value)
	return "sha256:" + hex.EncodeToString(digest[:])
}

func validDigest(value string) bool {
	if len(value) != len("sha256:")+sha256.Size*2 || !strings.HasPrefix(value, "sha256:") {
		return false
	}
	_, err := hex.DecodeString(strings.TrimPrefix(value, "sha256:"))
	return err == nil
}

func validatePlannerFact(fact planner.PlannerFact) error {
	if !validPlannerFactKey(fact.Key) {
		return fmt.Errorf("Planner fact key is invalid")
	}
	validSubtask := func(value string, allowEmpty bool) bool {
		if value == "" {
			return allowEmpty
		}
		if len(value) > 2 {
			return false
		}
		for _, current := range value {
			if current < '0' || current > '9' {
				return false
			}
		}
		return true
	}
	validCall := strings.HasPrefix(fact.CallID, "dispatch-") && len(fact.CallID) <= 64
	validWorker := strings.TrimSpace(fact.WorkerName) != "" && len(fact.WorkerName) <= 128
	switch fact.Kind {
	case planner.PlannerEventDispatchSelected:
		if !validSubtask(fact.SubtaskID, false) || !validCall || !validWorker ||
			fact.Outcome != "" || fact.Code != "" {
			return fmt.Errorf("dispatch-selected Planner fact is invalid")
		}
	case planner.PlannerEventCurrentChanged:
		if !validSubtask(fact.SubtaskID, true) || fact.CallID != "" || fact.WorkerName != "" ||
			fact.Outcome != "" || fact.Code != "" {
			return fmt.Errorf("current-changed Planner fact is invalid")
		}
	case planner.PlannerEventFinishRequested:
		if fact.SubtaskID != "" || fact.CallID != "" || fact.WorkerName != "" || fact.Code != "" ||
			(fact.Outcome != string(contracts.StageSucceeded) && fact.Outcome != string(contracts.StageFailed)) {
			return fmt.Errorf("finish-requested Planner fact is invalid")
		}
	default:
		return fmt.Errorf("Planner fact kind %q is invalid", fact.Kind)
	}
	return nil
}

func validPlannerFactKey(value string) bool {
	if strings.TrimSpace(value) == "" || len(value) > maxPlannerFactKeyBytes {
		return false
	}
	for _, current := range value {
		if current >= 'a' && current <= 'z' || current >= 'A' && current <= 'Z' ||
			current >= '0' && current <= '9' || strings.ContainsRune("._:-", current) {
			continue
		}
		return false
	}
	return true
}

func clonePlanProjection(value planner.PlannerPlanProjection) planner.PlannerPlanProjection {
	result := value
	result.Subtasks = append([]planner.PlannerSubtask(nil), value.Subtasks...)
	if value.ActiveDispatch != nil {
		active := *value.ActiveDispatch
		result.ActiveDispatch = &active
	}
	return result
}

func cloneStringMap(value map[string]string) map[string]string {
	result := make(map[string]string, len(value)+1)
	for key, current := range value {
		result[key] = current
	}
	return result
}

func planTransitionRunFields(
	previous *planner.PlannerPlanProjection,
	next planner.PlannerPlanProjection,
	kind planner.PlannerEventKind,
) plannerRunEventData {
	projection := clonePlanProjection(next)
	result := plannerRunEventData{Plan: &projection}
	switch kind {
	case planner.PlannerEventDispatchStarted:
		result.SubtaskID = next.ActiveDispatch.SubtaskID
		result.CallID = next.ActiveDispatch.CallID
		result.WorkerName = next.ActiveDispatch.WorkerName
	case planner.PlannerEventDispatchCompleted:
		if previous != nil && previous.ActiveDispatch != nil {
			result.SubtaskID = previous.ActiveDispatch.SubtaskID
			result.CallID = previous.ActiveDispatch.CallID
			result.WorkerName = previous.ActiveDispatch.WorkerName
			for _, subtask := range next.Subtasks {
				if subtask.ID == result.SubtaskID {
					result.Outcome = string(subtask.Status)
					break
				}
			}
		}
	}
	return result
}

func encodePlannerRunEvent(
	identity planner.SessionIdentity,
	kind planner.PlannerEventKind,
	fields *plannerRunEventData,
) (json.RawMessage, error) {
	if strings.TrimSpace(identity.StageExecutionID) == "" || strings.TrimSpace(identity.SessionID) == "" ||
		strings.TrimSpace(identity.InvocationID) == "" {
		return nil, fmt.Errorf("Planner Run event identity is invalid")
	}
	data := plannerRunEventData{}
	if fields != nil {
		data = *fields
		if fields.Plan != nil {
			plan := clonePlanProjection(*fields.Plan)
			data.Plan = &plan
		}
		if fields.Activity != nil {
			activity := *fields.Activity
			activity.FunctionCalls = append([]string(nil), fields.Activity.FunctionCalls...)
			activity.FunctionResults = append([]string(nil), fields.Activity.FunctionResults...)
			data.Activity = &activity
		}
	}
	data.StageExecutionID = identity.StageExecutionID
	data.SessionID = identity.SessionID
	data.InvocationID = identity.InvocationID
	if _, err := toRunEventKind(kind); err != nil {
		return nil, err
	}
	return encodeBounded(data)
}

func toRunEventKind(kind planner.PlannerEventKind) (runstore.RunEventKind, error) {
	mapping := map[planner.PlannerEventKind]runstore.RunEventKind{
		planner.PlannerEventStarted:           runstore.RunEventPlannerStarted,
		planner.PlannerEventRequestRecorded:   runstore.RunEventPlannerRequestRecorded,
		planner.PlannerEventActivity:          runstore.RunEventPlannerActivity,
		planner.PlannerEventPlanChanged:       runstore.RunEventPlannerPlanChanged,
		planner.PlannerEventCurrentChanged:    runstore.RunEventPlannerCurrentChanged,
		planner.PlannerEventDispatchSelected:  runstore.RunEventPlannerDispatchSelected,
		planner.PlannerEventDispatchStarted:   runstore.RunEventPlannerDispatchStarted,
		planner.PlannerEventDispatchCompleted: runstore.RunEventPlannerDispatchCompleted,
		planner.PlannerEventFinishRequested:   runstore.RunEventPlannerFinishRequested,
		planner.PlannerEventCompleted:         runstore.RunEventPlannerCompleted,
		planner.PlannerEventFailed:            runstore.RunEventPlannerFailed,
	}
	value, ok := mapping[kind]
	if !ok {
		return "", fmt.Errorf("unknown Planner Run event kind %q", kind)
	}
	return value, nil
}

func randomID(prefix string) (string, error) {
	bytes := make([]byte, 16)
	if _, err := cryptorand.Read(bytes); err != nil {
		return "", err
	}
	return prefix + hex.EncodeToString(bytes), nil
}
