package session

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
	"google.golang.org/adk/session"
)

const (
	maxADKEventFunctions = 32
	maxADKAppendAttempts = 4
)

// ADKOptions binds one ephemeral ADK conversation to the already durable
// Contractor Planner session. AllowedTools is the complete fixed tool set for
// the invocation; unknown provider-controlled names are never persisted.
type ADKOptions struct {
	AppName      string
	UserID       string
	AllowedTools []string
}

// NewADKSession returns the ADK SessionService used by one Streamline Planner
// invocation. Conversation contents stay in memory because partially resuming
// an LLM invocation is deliberately out of scope. Every non-partial event is
// first reduced to bounded, redacted facts and durably appended to Contractor's
// planner_events table.
func (s *Service) NewADKSession(
	ctx context.Context,
	identity planner.SessionIdentity,
	options ADKOptions,
) (session.Service, error) {
	if s == nil || strings.TrimSpace(options.AppName) == "" || strings.TrimSpace(options.UserID) == "" {
		return nil, fmt.Errorf("ADK app and user identities are required")
	}
	_, state, err := s.load(ctx, identity)
	if err != nil {
		return nil, err
	}
	if state.Status != statusRunning {
		return nil, fmt.Errorf("ADK session requires a running Planner session")
	}
	allowed := make(map[string]struct{}, len(options.AllowedTools))
	for _, name := range options.AllowedTools {
		if strings.TrimSpace(name) == "" || len(name) > 128 {
			return nil, fmt.Errorf("ADK allowed tool name is invalid")
		}
		allowed[name] = struct{}{}
	}
	return &adkService{
		memory: session.InMemoryService(), parent: s, identity: identity,
		appName: options.AppName, userID: options.UserID, allowedTools: allowed,
	}, nil
}

type adkService struct {
	memory       session.Service
	parent       *Service
	identity     planner.SessionIdentity
	appName      string
	userID       string
	allowedTools map[string]struct{}
}

func (s *adkService) Create(
	ctx context.Context, request *session.CreateRequest,
) (*session.CreateResponse, error) {
	if request == nil || request.AppName != s.appName || request.UserID != s.userID ||
		request.SessionID != s.identity.SessionID {
		return nil, fmt.Errorf("ADK session identity differs from its Planner session")
	}
	return s.memory.Create(ctx, request)
}

func (s *adkService) Get(
	ctx context.Context, request *session.GetRequest,
) (*session.GetResponse, error) {
	if request == nil || request.AppName != s.appName || request.UserID != s.userID ||
		request.SessionID != s.identity.SessionID {
		return nil, fmt.Errorf("ADK session identity differs from its Planner session")
	}
	return s.memory.Get(ctx, request)
}

func (s *adkService) List(
	ctx context.Context, request *session.ListRequest,
) (*session.ListResponse, error) {
	if request == nil || request.AppName != s.appName || request.UserID != s.userID {
		return nil, fmt.Errorf("ADK session identity differs from its Planner session")
	}
	return s.memory.List(ctx, request)
}

func (s *adkService) Delete(context.Context, *session.DeleteRequest) error {
	return fmt.Errorf("ADK session deletion is not supported during a Planner invocation")
}

func (s *adkService) AppendEvent(
	ctx context.Context, current session.Session, event *session.Event,
) error {
	if current == nil || current.ID() != s.identity.SessionID ||
		current.AppName() != s.appName || current.UserID() != s.userID {
		return fmt.Errorf("ADK event session identity differs")
	}
	if event == nil {
		return fmt.Errorf("ADK event is required")
	}
	if event.Partial {
		return s.memory.AppendEvent(ctx, current, event)
	}
	facts := reduceADKEvent(event, s.allowedTools)
	if err := s.parent.recordADKEvent(ctx, s.identity, facts); err != nil {
		return err
	}
	return s.memory.AppendEvent(ctx, current, event)
}

type adkEventFacts struct {
	Kind              string   `json:"kind"`
	Author            string   `json:"author"`
	FunctionCalls     []string `json:"functionCalls"`
	FunctionResults   []string `json:"functionResults"`
	InputTokens       int64    `json:"inputTokens,omitempty"`
	OutputTokens      int64    `json:"outputTokens,omitempty"`
	SkipSummarization bool     `json:"skipSummarization,omitempty"`
	Escalate          bool     `json:"escalate,omitempty"`
	Truncated         bool     `json:"truncated,omitempty"`
}

func reduceADKEvent(event *session.Event, allowed map[string]struct{}) adkEventFacts {
	facts := adkEventFacts{
		Kind: "adk_event", Author: safeADKAuthor(event.Author),
		FunctionCalls: []string{}, FunctionResults: []string{},
		SkipSummarization: event.Actions.SkipSummarization,
		Escalate:          event.Actions.Escalate,
	}
	if usage := event.UsageMetadata; usage != nil {
		facts.InputTokens = max(0, int64(usage.PromptTokenCount))
		facts.OutputTokens = max(0, int64(usage.CandidatesTokenCount))
	}
	if event.Content != nil {
		for _, part := range event.Content.Parts {
			switch {
			case part != nil && part.FunctionCall != nil:
				facts.FunctionCalls, facts.Truncated = appendAllowedFunction(
					facts.FunctionCalls, part.FunctionCall.Name, allowed, facts.Truncated,
				)
			case part != nil && part.FunctionResponse != nil:
				facts.FunctionResults, facts.Truncated = appendAllowedFunction(
					facts.FunctionResults, part.FunctionResponse.Name, allowed, facts.Truncated,
				)
			}
		}
	}
	return facts
}

func appendAllowedFunction(
	values []string, name string, allowed map[string]struct{}, truncated bool,
) ([]string, bool) {
	if len(values) >= maxADKEventFunctions {
		return values, true
	}
	if _, ok := allowed[name]; !ok {
		return append(values, "unknown"), truncated
	}
	return append(values, name), truncated
}

func safeADKAuthor(author string) string {
	switch author {
	case "user", "streamline_planner", "router_planner":
		return author
	default:
		return "other"
	}
}

func (s *Service) recordADKEvent(
	ctx context.Context, identity planner.SessionIdentity, facts adkEventFacts,
) error {
	payload, err := encodeBounded(facts)
	if err != nil {
		return err
	}
	for attempt := 0; attempt < maxADKAppendAttempts; attempt++ {
		stored, state, err := s.load(ctx, identity)
		if err != nil {
			return err
		}
		if state.Status != statusRunning {
			return fmt.Errorf("Planner session is not running")
		}
		next := state
		next.NextSequence++
		next.ADKEventCount++
		next.ADKInputTokens += facts.InputTokens
		next.ADKOutputTokens += facts.OutputTokens
		encodedState, err := encodeState(next)
		if err != nil {
			return err
		}
		err = s.append(
			ctx, stored, state.NextSequence, payload, encodedState,
			identity, planner.PlannerEventActivity,
			&plannerRunEventData{Activity: &facts},
		)
		if err == nil {
			return nil
		}
		if !errors.Is(err, runstore.ErrConflict) {
			return fmt.Errorf("append redacted ADK event: %w", err)
		}
	}
	return fmt.Errorf("append redacted ADK event: %w", runstore.ErrConflict)
}

var _ session.Service = (*adkService)(nil)
