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

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
)

const maxSessionJSONBytes = 512 * 1024

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
		initial, err := encodeState(persistentState{Status: statusRunning, NextSequence: 1})
		if err != nil {
			return planner.SessionStart{}, err
		}
		err = s.store.StartPlanner(ctx, runstore.StartPlannerParams{
			StageExecutionID:   stageExecutionID,
			SessionID:          identity.SessionID,
			InvocationID:       identity.InvocationID,
			StateSchemaVersion: contracts.APIVersion,
			InitialState:       initial,
			Reason:             runstore.Reason{Code: "planner_started"},
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
	if err := s.append(ctx, session, state.NextSequence, payload, encodedState); err != nil {
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
	if err := s.append(ctx, session, state.NextSequence, payload, encodedState); err != nil {
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
) error {
	eventID, err := s.newID("planner_event_")
	if err != nil {
		return fmt.Errorf("generate Planner event ID: %w", err)
	}
	return s.store.AppendPlannerEvent(ctx, runstore.AppendPlannerEventParams{
		EventID: eventID, SessionID: session.SessionID, SequenceNumber: sequence,
		EventSchemaVersion: contracts.APIVersion, Event: event,
		NewStateSchemaVersion: contracts.APIVersion, NewState: state,
	})
}

func (s *Service) newIdentity(stageExecutionID string) (planner.SessionIdentity, error) {
	sessionID, err := s.newID("planner_session_")
	if err != nil {
		return planner.SessionIdentity{}, fmt.Errorf("generate Planner session ID: %w", err)
	}
	invocationID, err := s.newID("planner_invocation_")
	if err != nil {
		return planner.SessionIdentity{}, fmt.Errorf("generate Planner invocation ID: %w", err)
	}
	return planner.SessionIdentity{
		SessionID: sessionID, StageExecutionID: stageExecutionID, InvocationID: invocationID,
	}, nil
}

const (
	statusRunning   = "running"
	statusCompleted = "completed"
)

type persistentState struct {
	Status          string              `json:"status"`
	NextSequence    int64               `json:"nextSequence"`
	RequestRecorded bool                `json:"requestRecorded"`
	RequestDigest   string              `json:"requestDigest,omitempty"`
	ADKEventCount   int64               `json:"adkEventCount,omitempty"`
	ADKInputTokens  int64               `json:"adkInputTokens,omitempty"`
	ADKOutputTokens int64               `json:"adkOutputTokens,omitempty"`
	Completion      *planner.Completion `json:"completion,omitempty"`
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
		// Preserve the v1alpha1 passthrough audit event shape. Streamline with
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

func randomID(prefix string) (string, error) {
	bytes := make([]byte, 16)
	if _, err := cryptorand.Read(bytes); err != nil {
		return "", err
	}
	return prefix + hex.EncodeToString(bytes), nil
}
