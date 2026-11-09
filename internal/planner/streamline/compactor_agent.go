package streamline

import (
	"context"
	"encoding/json"
	"iter"

	"github.com/grauwolf32/contractor/internal/planner"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	adksession "google.golang.org/adk/v2/session"
	"google.golang.org/genai"
)

// oneShotCompactor bounds the auxiliary ADK agent to one provider request.
// Keep the provider's raw reply: the compaction owner must charge usage and
// classify truncation, tool output and invalid summaries before any checkpoint.
type oneShotCompactor struct {
	name     string
	delegate compactionModel
	before   func(*model.LLMRequest) error
	called   bool
	reply    compactorReply
}

func (m *oneShotCompactor) Name() string { return m.name }

func (m *oneShotCompactor) GenerateContent(ctx context.Context, request *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		if m.called || stream {
			yield(nil, compactionFailed(compactionReasonInvalid))
			return
		}
		m.called = true
		if err := m.before(request); err != nil {
			yield(nil, err)
			return
		}
		var err error
		m.reply, err = m.delegate.compact(ctx, request)
		if err != nil {
			yield(nil, err)
			return
		}
		// No provider tool call may enter ADK dispatch. The original count and
		// finish reason remain in reply for validateCompactorReply to reject.
		yield(&model.LLMResponse{
			Content:       genai.NewContentFromText(m.reply.text, genai.RoleModel),
			UsageMetadata: m.reply.usage, TurnComplete: true,
		}, nil)
	}
}

func (p *streamlinePlanner) runCompactor(ctx context.Context, selection compactionSelection, before func(*model.LLMRequest) error) (compactorReply, error) {
	encoded, err := json.Marshal(compactorInput{
		Objective: p.request.Objective, PreviousSummary: selection.summary,
		Plan: planView(p.plan.Snapshot()), History: selection.history,
	})
	if err != nil {
		return compactorReply{}, compactionFailed(compactionReasonInputCapacity)
	}
	settings := p.compaction.settings
	oneShot := &oneShotCompactor{name: settings.compactor.Model, delegate: p.compactor, before: before}
	config := &genai.GenerateContentConfig{MaxOutputTokens: int32(settings.compactor.MaxOutputTokens)}
	if selected := settings.compactor.Temperature; selected != nil {
		value := float32(*selected)
		config.Temperature = &value
	}
	root, err := llmagent.New(llmagent.Config{
		// A standalone Runner root uses chat mode in ADK. oneShot, rather than
		// a workflow-node mode, enforces the single provider call.
		Name: "planner_context_compactor", Model: oneShot,
		InstructionProvider: func(agent.ReadonlyContext) (string, error) {
			return compactorContract + "\n\nCompaction guidance:\n" + settings.instructions, nil
		},
		OutputSchema: compactorOutputSchema(), GenerateContentConfig: config,
	})
	initializationFailed := func(err error) (compactorReply, error) {
		return compactorReply{}, planner.NewError("planner_initialization_failed", "Planner compactor could not be initialized", false, err)
	}
	if err != nil {
		return initializationFailed(err)
	}
	// This service belongs only to this call. It cannot read or mutate the main
	// Planner session and is released with the auxiliary agent after the call.
	const appName, userID, sessionID = "planner_compactor", "planner", "compactor"
	sessions := adksession.InMemoryService()
	if _, err := sessions.Create(ctx, &adksession.CreateRequest{AppName: appName, UserID: userID, SessionID: sessionID}); err != nil {
		return initializationFailed(err)
	}
	auxiliary, err := runner.New(runner.Config{AppName: appName, Agent: root, SessionService: sessions})
	if err != nil {
		return initializationFailed(err)
	}
	for _, err := range auxiliary.Run(ctx, userID, sessionID, genai.NewContentFromText(string(encoded), genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			return oneShot.reply, err
		}
	}
	if !oneShot.called {
		return compactorReply{}, compactionFailed(compactionReasonInvalid)
	}
	return oneShot.reply, nil
}
