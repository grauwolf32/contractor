package streamline

// This entire adapter is a spike fixture. Production stays on the current
// non-compacting engine; V420-002 must replace/reuse it at the owning boundary.
import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	plannersession "github.com/grauwolf32/contractor/internal/planner/session"
	"github.com/grauwolf32/contractor/internal/runstore"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	adksession "google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/genai"
)

const spikeTask = "Exact objective; parameter=value; input=source@revision; result=report; continuation=separate"

var errSpikeCheckpoint = errors.New("checkpoint requested at complete boundary")

type spikeProjection struct {
	generation int
	covered    map[string]bool
	markers    map[string]bool
	memory     string
	sessionID  string
}

func spikeGroups(events []*adksession.Event) ([][]*adksession.Event, error) {
	var groups [][]*adksession.Event
	var active []*adksession.Event
	pending, seen := map[string]bool{}, map[string]bool{}
	for _, event := range events {
		if event.ID == "" || seen[event.ID] || event.Partial || event.ErrorCode != "" || event.Branch != "" || event.IsolationScope != "" || event.Actions.Compaction != nil {
			return nil, errors.New("invalid source scope/identity")
		}
		seen[event.ID] = true
		if event.Content == nil {
			continue
		}
		for _, part := range event.Content.Parts {
			if part.FunctionCall != nil {
				if len(active) > 0 || part.FunctionCall.ID == "" || pending[part.FunctionCall.ID] {
					return nil, errors.New("interleaved call")
				}
				pending[part.FunctionCall.ID] = true
			}
		}
		if len(pending) > 0 && len(active) == 0 {
			active = append(active, event)
			continue
		}
		hasResponse := false
		for _, part := range event.Content.Parts {
			if reply := part.FunctionResponse; reply != nil {
				hasResponse = true
				if !pending[reply.ID] {
					return nil, errors.New("orphan response")
				}
				delete(pending, reply.ID)
			}
		}
		if hasResponse {
			active = append(active, event)
			if len(pending) == 0 {
				groups = append(groups, active)
				active = nil
			}
		} else if len(active) > 0 {
			return nil, errors.New("incomplete group")
		} else {
			groups = append(groups, []*adksession.Event{event})
		}
	}
	if len(active) > 0 {
		return nil, errors.New("pending group")
	}
	return groups, nil
}

func (p *spikeProjection) source(current adksession.Session) []*adksession.Event {
	var result []*adksession.Event
	for event := range current.Events().All() {
		if event.Author != "user" && !p.markers[event.ID] {
			result = append(result, event)
		}
	}
	return result
}

func (p *spikeProjection) selectPrefix(current adksession.Session, keep int) ([]string, error) {
	if keep < 1 || keep > 16 {
		return nil, errors.New("tail limit")
	}
	groups, err := spikeGroups(p.source(current))
	if err != nil {
		return nil, err
	}
	if len(groups) <= keep {
		return nil, errors.New("ineffective")
	}
	var selected []string
	for _, group := range groups[:len(groups)-keep] {
		for _, e := range group {
			selected = append(selected, e.ID)
		}
	}
	return selected, nil
}

func (p *spikeProjection) apply(current adksession.Session) ([]*genai.Content, error) {
	if p.sessionID != "" && p.sessionID != current.ID() {
		return nil, errors.New("foreign session")
	}
	var retained []*adksession.Event
	for _, event := range p.source(current) {
		if !p.covered[event.ID] {
			retained = append(retained, event)
		}
	}
	if _, err := spikeGroups(retained); err != nil {
		return nil, err
	}
	contents := []*genai.Content{genai.NewContentFromText(spikeTask, genai.RoleUser)}
	if p.memory != "" {
		contents = append(contents, genai.NewContentFromText(p.memory, genai.RoleModel))
	}
	for _, event := range retained {
		if event.Content != nil {
			contents = append(contents, event.Content)
		}
	}
	serialized, err := json.Marshal(contents)
	if err != nil {
		return nil, err
	}
	if p.generation > 0 && len(serialized) > 1024*1024 {
		return nil, errors.New("planner_compaction_limit_exceeded")
	}
	return contents, nil
}

func (p *spikeProjection) commit(ctx context.Context, service adksession.Service, current adksession.Session, selected []string, generation int, memory string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if p.sessionID != "" && p.sessionID != current.ID() {
		return errors.New("foreign session")
	}
	p.sessionID = current.ID()
	next, err := p.selectPrefix(current, 1)
	if err != nil {
		return err
	}
	if generation != p.generation || !slices.Equal(next, selected) {
		return errors.New("stale generation/source")
	}
	if len(strings.TrimSpace(memory)) == 0 || len(memory) > 16*1024 {
		return errors.New("planner_compaction_failed")
	}
	marker := &adksession.Event{ID: fmt.Sprintf("checkpoint-%d", generation+1), Timestamp: time.Unix(int64(100+generation), 0), Author: "streamline_planner"}
	if err := service.AppendEvent(ctx, current, marker); err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	p.covered = map[string]bool{}
	for _, id := range selected {
		p.covered[id] = true
	}
	if p.markers == nil {
		p.markers = map[string]bool{}
	}
	p.markers[marker.ID] = true
	p.memory = memory
	p.generation++
	return nil
}

func TestPostgresCompactionSpikeTwoCheckpoints(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool := isolatedStreamlinePool(t, ctx)
	store := runstore.NewPostgresStore(pool)
	createPreparingStreamlineStage(t, ctx, store)
	sessions, err := plannersession.New(store, plannersession.Options{})
	if err != nil {
		t.Fatal(err)
	}
	started, err := sessions.Begin(ctx, "stage-1")
	if err != nil {
		t.Fatal(err)
	}
	service, err := sessions.NewADKSession(ctx, started.Identity, plannersession.ADKOptions{AppName: "spike", UserID: "owner", AllowedTools: []string{"execute"}})
	if err != nil {
		t.Fatal(err)
	}
	current, err := service.Create(ctx, &adksession.CreateRequest{AppName: "spike", UserID: "owner", SessionID: started.Identity.SessionID})
	if err != nil {
		t.Fatal(err)
	}
	projection := &spikeProjection{}
	var payloads []chatRequest
	var gatewayMu sync.Mutex
	mainCalls, summaryCalls, effects, total := 0, 0, 0, int64(0)
	workerHandle, planIdentity := new(int), new(int)
	originalHandle, originalPlan := workerHandle, planIdentity
	deadline, _ := ctx.Deadline()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gatewayMu.Lock()
		defer gatewayMu.Unlock()
		var payload chatRequest
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Error(err)
			return
		}
		payloads = append(payloads, payload)
		var message any
		finish := "stop"
		if payload.Model == "compactor" {
			summaryCalls++
			if len(payload.Tools) > 0 {
				t.Error("compactor has tools")
			}
			message = map[string]any{"content": fmt.Sprintf(`{"summary":"rolling-memory-%d"}`, summaryCalls)}
		} else {
			mainCalls++
			if mainCalls <= 4 {
				finish = "tool_calls"
				message = map[string]any{"content": nil, "tool_calls": []any{map[string]any{"id": fmt.Sprintf("call-%d", mainCalls), "type": "function", "function": map[string]any{"name": "execute", "arguments": fmt.Sprintf(`{"value":"worker-%d"}`, mainCalls)}}}}
			} else {
				message = map[string]any{"content": "finish"}
			}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": []any{map[string]any{"finish_reason": finish, "message": message}}, "usage": map[string]int{"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}})
	}))
	defer server.Close()
	main, err := NewOpenAICompatibleModel(GatewaySettings{URL: server.URL, Model: "main", HTTPClient: server.Client()})
	if err != nil {
		t.Fatal(err)
	}
	compactor, err := NewOpenAICompatibleModel(GatewaySettings{URL: server.URL, Model: "compactor", MaxOutputTokens: 1024, HTTPClient: server.Client()})
	if err != nil {
		t.Fatal(err)
	}
	execute, err := functiontool.New(functiontool.Config{Name: "execute", Description: "Return one exact validated fixture Worker result."}, func(_ agent.Context, args struct {
		Value string `json:"value"`
	}) (map[string]any, error) {
		effects++
		return map[string]any{"value": args.Value, "result": strings.Repeat("x", 2048), "ref": "report@exact-revision"}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	callbacks, models, used := 0, 0, int64(0)
	root, err := llmagent.New(llmagent.Config{Name: "streamline_planner", Model: main, Tools: []tool.Tool{execute}, Instruction: "Preserve the protected task.",
		BeforeModelCallbacks: []llmagent.BeforeModelCallback{func(c agent.Context, request *model.LLMRequest) (*model.LLMResponse, error) {
			callbacks++
			if d, _ := c.Deadline(); !d.Equal(deadline) || workerHandle != originalHandle || planIdentity != originalPlan {
				return nil, errors.New("owner reset")
			}
			if models == 2 || models == 4 {
				if projection.generation == models/2-1 {
					return nil, errSpikeCheckpoint
				}
			}
			var err error
			loaded, loadErr := service.Get(c, &adksession.GetRequest{AppName: "spike", UserID: "owner", SessionID: current.Session.ID()})
			if loadErr != nil {
				return nil, loadErr
			}
			request.Contents, err = projection.apply(loaded.Session)
			if err != nil {
				return nil, err
			}
			models++
			return nil, nil
		}}, AfterModelCallbacks: []llmagent.AfterModelCallback{func(_ agent.Context, response *model.LLMResponse, responseErr error) (*model.LLMResponse, error) {
			if responseErr != nil {
				return nil, responseErr
			}
			used += int64(response.UsageMetadata.TotalTokenCount)
			return nil, nil
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	run, err := runner.New(runner.Config{AppName: "spike", Agent: root, SessionService: service})
	if err != nil {
		t.Fatal(err)
	}
	for turn := 0; turn < 3; turn++ {
		var iterationErr error
		for _, err := range run.Run(ctx, "owner", current.Session.ID(), genai.NewContentFromText(spikeTask, genai.RoleUser), agent.RunConfig{}) {
			if err != nil {
				iterationErr = err
				break
			}
		}
		if turn == 2 {
			if iterationErr != nil {
				t.Fatal(iterationErr)
			}
			break
		}
		if !errors.Is(iterationErr, errSpikeCheckpoint) {
			t.Fatalf("missing checkpoint boundary: %v", iterationErr)
		}
		loaded, err := service.Get(ctx, &adksession.GetRequest{AppName: "spike", UserID: "owner", SessionID: current.Session.ID()})
		if err != nil {
			t.Fatal(err)
		}
		selected, err := projection.selectPrefix(loaded.Session, 1)
		if err != nil {
			t.Fatal(err)
		}
		input := []*genai.Content{genai.NewContentFromText(spikeTask+projection.memory, genai.RoleUser)}
		for _, event := range projection.source(loaded.Session) {
			if slices.Contains(selected, event.ID) && !projection.covered[event.ID] {
				input = append(input, event.Content)
			}
		}
		request := &model.LLMRequest{Contents: input}
		serialized, err := encodeChatRequest(request, "compactor", 1024)
		if err != nil {
			t.Fatal(err)
		}
		if err := (spikeCompactorPolicy{window: 128 * 1024, output: 1024, calls: 1, total: 200000}).admit(serialized); err != nil {
			t.Fatal(err)
		}
		memory, known, err := spikeSummarize(ctx, compactor, request, 1024*1024)
		total += known
		if err != nil {
			t.Fatal(err)
		}
		if err := projection.commit(ctx, service, loaded.Session, selected, projection.generation, memory); err != nil {
			t.Fatal(err)
		}
	}
	gatewayMu.Lock()
	defer gatewayMu.Unlock()
	if effects != 4 || mainCalls != 5 || summaryCalls != 2 || models != 5 || projection.generation != 2 || used+total != 126 || callbacks != 7 {
		t.Fatalf("owner counts effects/main/summary/models/generation/tokens/callbacks=%d/%d/%d/%d/%d/%d/%d", effects, mainCalls, summaryCalls, models, projection.generation, used+total, callbacks)
	}
	last, _ := json.Marshal(payloads[len(payloads)-1])
	text := string(last)
	for _, wanted := range []string{spikeTask, "rolling-memory-2", "call-4", "report@exact-revision"} {
		if !strings.Contains(text, wanted) {
			t.Fatalf("missing protected/tail value %s", wanted)
		}
	}
	for _, removed := range []string{"rolling-memory-1", "call-1", "call-2", "call-3"} {
		if strings.Contains(text, removed) {
			t.Fatalf("old context still projected: %s", removed)
		}
	}
	var raw string
	if err := pool.QueryRow(ctx, "SELECT string_agg(data::text,'') FROM workflow_run_events").Scan(&raw); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(raw, spikeTask) || strings.Contains(raw, "rolling-memory") || strings.Contains(raw, "exact-revision") {
		t.Fatal("raw context reached durable reduced facts")
	}
}

// Rejected writes below are actual PostgreSQL constraint/SQLSTATE rejections.
func TestPostgresCompactionSpikeMarkerFailureFences(t *testing.T) {
	for _, fault := range []string{"invalid", "unavailable"} {
		t.Run(fault, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
			defer cancel()
			pool := isolatedStreamlinePool(t, ctx)
			store := runstore.NewPostgresStore(pool)
			createPreparingStreamlineStage(t, ctx, store)
			sessions, err := plannersession.New(store, plannersession.Options{})
			if err != nil {
				t.Fatal(err)
			}
			started, err := sessions.Begin(ctx, "stage-1")
			if err != nil {
				t.Fatal(err)
			}
			service, err := sessions.NewADKSession(ctx, started.Identity, plannersession.ADKOptions{AppName: "spike", UserID: "owner"})
			if err != nil {
				t.Fatal(err)
			}
			current, err := service.Create(ctx, &adksession.CreateRequest{AppName: "spike", UserID: "owner", SessionID: started.Identity.SessionID})
			if err != nil {
				t.Fatal(err)
			}
			for i := 0; i < 3; i++ {
				event := &adksession.Event{ID: fmt.Sprint(i), Author: "streamline_planner", LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("large source"+strings.Repeat("x", 1024), genai.RoleModel)}}
				if err := service.AppendEvent(ctx, current.Session, event); err != nil {
					t.Fatal(err)
				}
			}
			// No-content markers are reduced as planner.activity with this author.
			if fault == "invalid" {
				if _, err := pool.Exec(ctx, `ALTER TABLE workflow_run_events ADD CONSTRAINT spike_marker_rejected CHECK (NOT (kind = 'planner.activity' AND data #>> '{activity,author}' = 'streamline_planner')) NOT VALID`); err != nil {
					t.Fatal(err)
				}
			} else {
				installRequiredEventGuard(t, ctx, pool, "ordinary", fault)
			}
			p := &spikeProjection{}
			selected, err := p.selectPrefix(current.Session, 1)
			if err != nil {
				t.Fatal(err)
			}
			err = p.commit(ctx, service, current.Session, selected, 0, "memory")
			var write *plannersession.ADKEventWriteError
			if !errors.As(err, &write) || p.generation != 0 || p.memory != "" {
				t.Fatalf("failed marker admitted continuation: %v", err)
			}
			if fault == "invalid" && !errors.Is(err, runstore.ErrInvalid) {
				t.Fatal("invalid lost class")
			}
		})
	}
}
