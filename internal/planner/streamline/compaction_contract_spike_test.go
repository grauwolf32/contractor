package streamline

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"io"
	"net/http"
	"slices"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/strictjson"
	"google.golang.org/adk/v2/model"
	adksession "google.golang.org/adk/v2/session"
	"google.golang.org/genai"
)

// Proposed authoring shape only. No production parser accepts this fixture.
type spikePolicy struct {
	Strategy     string `json:"strategy"`
	ModelPolicy  string `json:"modelPolicy,omitempty"`
	Instructions *struct {
		Ref string `json:"ref"`
	} `json:"instructions,omitempty"`
	Ratio  *float64 `json:"contextWindowRatio,omitempty"`
	Keep   *int     `json:"keepRecentGroups,omitempty"`
	Cycles *int     `json:"maxCompactions,omitempty"`
}

func spikeResolvePolicy(data string, profile string, modeled bool, window int) (spikePolicy, error) {
	p := spikePolicy{Strategy: "none"}
	if data != "" {
		p.Strategy = ""
		var members map[string]json.RawMessage
		if err := strictjson.DecodeNonNull([]byte(data), &members); err != nil {
			return p, err
		}
		for _, value := range members {
			if string(bytes.TrimSpace(value)) == "null" {
				return p, errors.New("null policy field")
			}
		}
		if err := strictjson.DecodeNonNull([]byte(data), &p); err != nil {
			return p, err
		}
	}
	switch p.Strategy {
	case "none":
		if p.ModelPolicy != "" || p.Instructions != nil || p.Ratio != nil || p.Keep != nil || p.Cycles != nil {
			return p, errors.New("mixed none")
		}
	case "compact@1":
		if !modeled || (profile != "streamline@1" && profile != "router@1") || window <= 0 || p.ModelPolicy == "" || !strings.Contains(p.ModelPolicy, "@") || p.Instructions == nil || p.Instructions.Ref == "" {
			return p, errors.New("ineligible")
		}
		if p.Ratio == nil {
			v := 0.75
			p.Ratio = &v
		}
		if p.Keep == nil {
			v := 4
			p.Keep = &v
		}
		if p.Cycles == nil {
			v := 8
			p.Cycles = &v
		}
		if *p.Ratio <= 0 || *p.Ratio >= 1 || *p.Keep < 1 || *p.Keep > 16 || *p.Cycles < 1 || *p.Cycles > 16 {
			return p, errors.New("policy limits")
		}
	default:
		return p, errors.New("strategy")
	}
	return p, nil
}

func TestCompactionSpikePolicyMatrix(t *testing.T) {
	compact := `{"strategy":"compact@1","modelPolicy":"compact@1","instructions":{"ref":"instructions/compact.md"}}`
	for _, test := range []struct {
		name, body, profile string
		modeled             bool
		window              int
		valid               bool
	}{
		{"omitted", "", "passthrough@1", false, 0, true},
		{"streamline", compact, "streamline@1", true, 10000, true},
		{"router", compact, "router@1", true, 10000, true},
		{"passthrough", compact, "passthrough@1", false, 0, false},
		{"model-free", compact, "streamline@1", false, 10000, false},
		{"missing-window", compact, "router@1", true, 0, false},
		{"terminal", `{"strategy":"terminal@1"}`, "streamline@1", true, 10000, false},
		{"summarizer", `{"strategy":"none","summarizerModelPolicy":"s@1"}`, "streamline@1", true, 10000, false},
		{"none-mixed", `{"strategy":"none","maxCompactions":1}`, "streamline@1", true, 10000, false},
		{"unknown", strings.Replace(compact, `"modelPolicy"`, `"modelURL"`, 1), "router@1", true, 10000, false},
		{"ratio0", strings.TrimSuffix(compact, "}") + `,"contextWindowRatio":0}`, "streamline@1", true, 10000, false},
		{"ratio1", strings.TrimSuffix(compact, "}") + `,"contextWindowRatio":1}`, "streamline@1", true, 10000, false},
		{"cycles17", strings.TrimSuffix(compact, "}") + `,"maxCompactions":17}`, "streamline@1", true, 10000, false},
		{"recent17", strings.TrimSuffix(compact, "}") + `,"keepRecentGroups":17}`, "streamline@1", true, 10000, false},
		{"missing-strategy", `{}`, "streamline@1", true, 10000, false},
		{"null-ratio", strings.TrimSuffix(compact, "}") + `,"contextWindowRatio":null}`, "streamline@1", true, 10000, false},
		{"null", `null`, "streamline@1", true, 10000, false},
		{"duplicate", `{"strategy":"none","strategy":"compact@1"}`, "streamline@1", true, 10000, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			p, err := spikeResolvePolicy(test.body, test.profile, test.modeled, test.window)
			if (err == nil) != test.valid {
				t.Fatalf("valid=%t err=%v", test.valid, err)
			}
			if err == nil && p.Strategy == "compact@1" && (*p.Ratio != 0.75 || *p.Keep != 4 || *p.Cycles != 8) {
				t.Fatal("defaults changed")
			}
		})
	}
	// Worker choices are independent; an escalation's default is explicitly none.
	for _, worker := range []string{"none", "terminal@1", "compact@1"} {
		if _, err := spikeResolvePolicy(compact, "streamline@1", true, 10000); err != nil {
			t.Fatalf("independent Worker %s: %v", worker, err)
		}
	}
}

type spikeCompactorPolicy struct{ window, output, calls, total, tools, workers int }

func (p spikeCompactorPolicy) admit(serialized []byte) error {
	if p.window <= 0 || p.output <= 0 || p.output >= p.window || p.calls != 1 || p.total <= 0 || p.tools != 0 || p.workers != 0 {
		return errors.New("incompatible policy")
	}
	// Byte-fallback admission is intentionally route-restricted in the design:
	// a byte-level tokenizer cannot produce more than one token per UTF-8 byte;
	// the complete serialized request is counted, plus 1024 protocol tokens.
	// Routes without this operator guarantee require an exact tokenizer instead.
	var envelope struct {
		Messages []json.RawMessage `json:"messages"`
	}
	_ = json.Unmarshal(serialized, &envelope)
	framing := 1024
	if len(envelope.Messages) > 8 {
		framing += 128 * len(envelope.Messages)
	}
	if len(serialized) > 4*1024*1024 || len(serialized)+framing+p.output > p.window || len(serialized)+framing+p.output > p.total {
		return errors.New("input capacity")
	}
	return nil
}
func TestCompactionSpikeCompactorCapacityBeforeTransport(t *testing.T) {
	request, _ := json.Marshal(map[string]any{"instructions": strings.Repeat("i", 1024), "groups": []string{strings.Repeat("я", 1024)}, "summary": "old"})
	valid := spikeCompactorPolicy{window: 16384, output: 1024, calls: 1, total: 20000}
	if err := valid.admit(request); err != nil {
		t.Fatal(err)
	}
	for _, change := range []func(*spikeCompactorPolicy){func(p *spikeCompactorPolicy) { p.window = 2000 }, func(p *spikeCompactorPolicy) { p.calls = 2 }, func(p *spikeCompactorPolicy) { p.tools = 1 }, func(p *spikeCompactorPolicy) { p.workers = 1 }, func(p *spikeCompactorPolicy) { p.total = 0 }, func(p *spikeCompactorPolicy) { p.output = p.window }} {
		invalid := valid
		change(&invalid)
		if err := invalid.admit(request); err == nil {
			t.Fatal("invalid route admitted transport")
		}
	}
	if err := valid.admit(bytes.Repeat([]byte("x"), 4*1024*1024+1)); err == nil {
		t.Fatal("oversized indivisible input admitted")
	}
}

// The archive stores canonical JSON backing once, with immutable offsets into
// it; no full plan/result history is injected into a prompt. A cursor is an
// opaque random server-side capability, bound to every owner/revision/view.
type spikeArchive struct {
	session              string
	generation, revision int
	worker, view         string
	closed               bool
	data                 [][]byte
	bytes                int
	cursors              map[string]spikeCursor
	visits               int
}
type spikeCursor struct {
	session, worker, view               string
	generation, revision, index, offset int
}
type spikeFragment struct {
	Index  int    `json:"index"`
	Offset int    `json:"offset"`
	Data   string `json:"text"`
}
type spikePage struct {
	Items []spikeFragment `json:"items"`
	Next  string          `json:"next,omitempty"`
}

func (a *spikeArchive) add(value []byte) error {
	if len(a.data) >= 4096 || 2*(a.bytes+len(value))+(len(a.data)+1)*1024 > 32*1024*1024 {
		return errors.New("planner_compaction_limit_exceeded")
	}
	a.data = append(a.data, value)
	a.bytes += len(value)
	a.revision++
	a.cursors = nil
	return nil
}
func (a *spikeArchive) page(session, worker, view, cursor string) (spikePage, error) {
	empty := spikePage{}
	if a.closed || session != a.session || worker != a.worker || view != a.view {
		return empty, errors.New("history_cursor_invalid")
	}
	position := spikeCursor{session: a.session, worker: a.worker, view: a.view, generation: a.generation, revision: a.revision}
	if cursor != "" {
		var ok bool
		position, ok = a.cursors[cursor]
		if !ok || position.session != session || position.worker != worker || position.view != view || position.generation != a.generation || position.revision != a.revision {
			return empty, errors.New("history_cursor_invalid")
		}
	}
	result := spikePage{}
	for position.index < len(a.data) && len(result.Items) < 8 {
		raw := a.data[position.index]
		end := min(len(raw), position.offset+16*1024)
		for !utf8.Valid(raw[position.offset:end]) {
			end--
		}
		fragment := spikeFragment{Index: position.index, Offset: position.offset, Data: string(raw[position.offset:end])}
		candidate := spikePage{Items: append(slices.Clone(result.Items), fragment), Next: strings.Repeat("x", 32)}
		encoded, _ := json.Marshal(candidate)
		if len(encoded) > 32*1024 {
			break
		}
		a.visits++
		result.Items = append(result.Items, fragment)
		position.offset = end
		if end == len(raw) {
			position.index++
			position.offset = 0
		}
	}
	if cursor != "" {
		delete(a.cursors, cursor)
	}
	if position.index < len(a.data) {
		if len(a.cursors) >= 8 {
			return empty, errors.New("cursor limit")
		}
		key := make([]byte, 16)
		if _, err := rand.Read(key); err != nil {
			return empty, err
		}
		result.Next = fmt.Sprintf("%x", key)
		if a.cursors == nil {
			a.cursors = map[string]spikeCursor{}
		}
		a.cursors[result.Next] = position
	}
	return result, nil
}
func TestCompactionSpikeExactHistoryPagesAtMaximum(t *testing.T) {
	a := &spikeArchive{session: "logical", worker: "builder", view: "validated-results"}
	for i := 0; i < 4096; i++ {
		raw, _ := json.Marshal(map[string]any{"ordinal": i, "result": strings.Repeat("r", 3072), "ref": fmt.Sprintf("report@exact-%d", i)})
		if err := a.add(raw); err != nil {
			t.Fatal(err)
		}
	}
	if err := a.add([]byte("overflow")); err == nil {
		t.Fatal("group maximum")
	}
	cursor, read, encodedBytes := "", 0, 0
	for {
		page, err := a.page("logical", "builder", "validated-results", cursor)
		if err != nil {
			t.Fatal(err)
		}
		encoded, _ := json.Marshal(page)
		if len(encoded) > 32*1024 || len(page.Items) > 8 || len(page.Items) == 0 {
			t.Fatal("page bound/progress")
		}
		encodedBytes += len(encoded)
		for _, item := range page.Items {
			raw := []byte(item.Data)
			if !bytes.Equal(raw, a.data[item.Index]) || item.Offset != 0 {
				t.Fatal("exact archive authority lost")
			}
			read++
		}
		cursor = page.Next
		if cursor == "" {
			break
		}
	}
	if read != 4096 || a.visits != 4096 || encodedBytes < a.bytes {
		t.Fatal("unbounded rescan/missing data")
	}
	// Raw ADK payload plus canonical index plus a 4-MiB request, 1-MiB tail,
	// 16-KiB summary and 32-KiB page remain below accounted 64 MiB here.
	peak := 2*a.bytes + 4*1024*1024 + 1024*1024 + 16*1024 + 32*1024 + 4096*1024
	if a.bytes > 32*1024*1024 || peak > 64*1024*1024 {
		t.Fatal("retention/peak accounting")
	}
}
func TestCompactionSpikeLargeExactResultFragmentsAndCursorFences(t *testing.T) {
	a := &spikeArchive{session: "logical", worker: "builder", view: "validated-results"}
	raw, _ := json.Marshal(map[string]any{"result": strings.Repeat("я", 32*1024), "payload": strings.Repeat("p", 256*1024), "refs": make([]string, 128)})
	if err := a.add(raw); err != nil {
		t.Fatal(err)
	}
	var recovered []byte
	cursor := ""
	for {
		page, err := a.page("logical", "builder", "validated-results", cursor)
		if err != nil {
			t.Fatal(err)
		}
		encoded, _ := json.Marshal(page)
		if len(encoded) > 32*1024 || len(page.Items) == 0 {
			t.Fatal("fragment bound")
		}
		for _, item := range page.Items {
			part := []byte(item.Data)
			if item.Offset != len(recovered) {
				t.Fatal("offset")
			}
			recovered = append(recovered, part...)
		}
		cursor = page.Next
		if cursor == "" {
			break
		}
	}
	if !bytes.Equal(raw, recovered) {
		t.Fatal("lossy reference/result recovery")
	}
	first, _ := a.page("logical", "builder", "validated-results", "")
	for _, args := range [][4]string{{"foreign", "builder", "validated-results", first.Next}, {"logical", "other", "validated-results", first.Next}, {"logical", "builder", "raw", first.Next}, {"logical", "builder", "validated-results", "invalid"}} {
		page, err := a.page(args[0], args[1], args[2], args[3])
		if err == nil || len(page.Items) != 0 {
			t.Fatal("foreign cursor leaked data")
		}
	}
	a.generation++
	if page, err := a.page("logical", "builder", "validated-results", first.Next); err == nil || len(page.Items) > 0 {
		t.Fatal("stale generation")
	}
	a.generation--
	a.revision++
	if _, err := a.page("logical", "builder", "validated-results", first.Next); err == nil {
		t.Fatal("stale index")
	}
	a.closed = true
	if _, err := a.page("logical", "builder", "validated-results", ""); err == nil {
		t.Fatal("expired owner")
	}
}

func TestCompactionSpikeSelectionRejectsUnsafeAndStale(t *testing.T) {
	for _, scenario := range []string{"equal-time", "out-of-order", "partial", "error", "foreign", "isolation", "incomplete", "orphan", "duplicate", "stale", "native"} {
		t.Run(scenario, func(t *testing.T) {
			ctx := t.Context()
			service := adksession.InMemoryService()
			created, err := service.Create(ctx, &adksession.CreateRequest{AppName: "spike", UserID: "user", SessionID: "session"})
			if err != nil {
				t.Fatal(err)
			}
			for i := 0; i < 17; i++ {
				timestamp := time.Unix(1, 0)
				if scenario == "out-of-order" {
					timestamp = time.Unix(int64(17-i), 0)
				}
				event := &adksession.Event{ID: fmt.Sprint(i), Timestamp: timestamp, Author: "worker", LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("raw", genai.RoleModel)}}
				if i == 0 {
					switch scenario {
					case "native":
						event.Actions.Compaction = &adksession.EventCompaction{StartTimestamp: time.Unix(1, 0), EndTimestamp: time.Unix(2, 0), CompactedContent: genai.NewContentFromText("memory", genai.RoleModel)}
					case "partial":
						event.Partial = true
					case "error":
						event.ErrorCode = "fixture"
					case "foreign":
						event.Branch = "foreign"
					case "isolation":
						event.IsolationScope = "foreign"
					case "incomplete":
						event.Content = &genai.Content{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "pending", Name: "execute"}}}}
					case "orphan":
						event.Content = &genai.Content{Role: "user", Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{ID: "pending", Name: "execute", Response: map[string]any{}}}}}
					}
				}
				if scenario == "duplicate" && i == 1 {
					event.ID = "0"
				}
				if err := service.AppendEvent(ctx, created.Session, event); err != nil {
					t.Fatal(err)
				}
			}
			p := &spikeProjection{}
			selected, err := p.selectPrefix(created.Session, 16)
			// Partial events are intentionally absent from ADK storage; production must
			// fence a pending partial in its ingest owner, not trust a stored-only scan.
			if scenario == "partial" {
				if created.Session.Events().Len() != 16 {
					t.Fatal("partial storage assumption")
				}
				return
			}
			valid := scenario == "equal-time" || scenario == "out-of-order" || scenario == "stale"
			if valid {
				if err != nil || len(selected) != 1 {
					t.Fatalf("exact ordinals fail: %v", err)
				}
			} else {
				if err == nil {
					t.Fatal("unsafe selected")
				}
				return
			}
			if scenario == "stale" {
				all, _ := p.selectPrefix(created.Session, 1)
				p.generation++
				before := created.Session.Events().Len()
				if err := p.commit(ctx, service, created.Session, all, 0, "memory"); err == nil || created.Session.Events().Len() != before {
					t.Fatal("stale marker mutated history")
				}
			}
		})
	}
}

func TestCompactionSpikeHistoryReaderIsCallableThroughRunner(t *testing.T) {
	a := &spikeArchive{session: "logical", worker: "builder", view: "validated-results"}
	exact := []byte(`{"result":"old validated result","ref":"report@exact-old"}`)
	if err := a.add(exact); err != nil {
		t.Fatal(err)
	}
	calls := 0
	read, err := functiontool.New(functiontool.Config{Name: "read_context_history", Description: "Read exact earlier validated Worker results of this live Planner."}, func(_ agent.Context, args struct {
		Cursor string `json:"cursor"`
	}) (spikePage, error) {
		calls++
		return a.page("logical", "builder", "validated-results", args.Cursor)
	})
	if err != nil {
		t.Fatal(err)
	}
	llm := &scriptedModel{steps: []modelStep{
		functionStep("read_context_history", map[string]any{"cursor": ""}),
		func(request *model.LLMRequest) (*model.LLMResponse, error) {
			encoded, _ := json.Marshal(request.Contents)
			if !strings.Contains(string(encoded), "report@exact-old") {
				return nil, errors.New("reader response missing from actual next model request")
			}
			return textStep("done")(request)
		},
	}}
	sessions := adksession.InMemoryService()
	if _, err := sessions.Create(t.Context(), &adksession.CreateRequest{AppName: "migration_history", UserID: "fixture-user", SessionID: "fixture-session"}); err != nil {
		t.Fatal(err)
	}
	run := migrationRunner(t, sessions, llm, []tool.Tool{read})
	for _, err := range run.Run(t.Context(), "fixture-user", "fixture-session", genai.NewContentFromText("Lossy memory falsely claims report@wrong; retrieve exact source.", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			t.Fatal(err)
		}
	}
	if calls != 1 || llm.callCount() != 2 {
		t.Fatal("reader call/continuation repeated")
	}
}

type spikeLostAck struct {
	adksession.Service
	appends int
	lost    bool
}

func (s *spikeLostAck) AppendEvent(ctx context.Context, current adksession.Session, event *adksession.Event) error {
	s.appends++
	if err := s.Service.AppendEvent(ctx, current, event); err != nil {
		return err
	}
	if s.lost {
		return io.ErrUnexpectedEOF
	}
	return nil
}
func TestCompactionSpikeAmbiguousMarkerAndCancellationFence(t *testing.T) {
	for _, phase := range []string{"lost-ack", "cancelled", "deadline"} {
		t.Run(phase, func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			memory := adksession.InMemoryService()
			current, err := memory.Create(ctx, &adksession.CreateRequest{AppName: "spike", UserID: "user", SessionID: "session"})
			if err != nil {
				t.Fatal(err)
			}
			for i := 0; i < 3; i++ {
				event := &adksession.Event{ID: fmt.Sprint(i), Author: "worker", LLMResponse: model.LLMResponse{Content: genai.NewContentFromText(strings.Repeat("raw", 1024), genai.RoleModel)}}
				if err := memory.AppendEvent(ctx, current.Session, event); err != nil {
					t.Fatal(err)
				}
			}
			p := &spikeProjection{}
			selected, err := p.selectPrefix(current.Session, 1)
			if err != nil {
				t.Fatal(err)
			}
			service := &spikeLostAck{Service: memory, lost: phase == "lost-ack"}
			if phase == "cancelled" {
				cancel()
			}
			if phase == "deadline" {
				var stop context.CancelFunc
				ctx, stop = context.WithDeadline(ctx, time.Unix(1, 0))
				defer stop()
			}
			err = p.commit(ctx, service, current.Session, selected, 0, "memory")
			if err == nil || p.generation != 0 || p.memory != "" {
				t.Fatal("uncertain checkpoint admitted continuation")
			}
			if phase == "lost-ack" {
				if service.appends != 1 || current.Session.Events().Len() != 4 || !errors.Is(err, io.ErrUnexpectedEOF) {
					t.Fatal("ambiguous append repeated/rolled back")
				}
			} else if service.appends != 0 {
				t.Fatal("expired checkpoint appended")
			}
		})
	}
}

func spikeSummarize(ctx context.Context, llm model.LLM, request *model.LLMRequest, sourceBytes int) (string, int64, error) {
	var memory struct {
		Summary string `json:"summary"`
	}
	var known int64
	var result *model.LLMResponse
	for response, err := range llm.GenerateContent(ctx, request, false) {
		if err != nil {
			var rejected *gatewayResponseRejected
			if errors.As(err, &rejected) && rejected.usage != nil {
				known += int64(rejected.usage.TotalTokenCount)
			}
			return "", known, err
		}
		if response == nil || response.UsageMetadata == nil {
			return "", known, errors.New("planner_gateway_invalid_response")
		}
		known += int64(response.UsageMetadata.TotalTokenCount)
		if result != nil {
			return "", known, errors.New("multiple responses")
		}
		result = response
	}
	if result == nil || result.Content == nil {
		return "", known, errors.New("planner_compaction_failed")
	}
	text := ""
	for _, part := range result.Content.Parts {
		if part.FunctionCall != nil || part.FunctionResponse != nil || part.Thought {
			return "", known, errors.New("tool/thought-bearing")
		}
		text += part.Text
	}
	if err := strictjson.DecodeNonNull([]byte(text), &memory); err != nil {
		return "", known, err
	}
	if strings.TrimSpace(memory.Summary) == "" || len(memory.Summary) > 16*1024 || len(memory.Summary) >= sourceBytes {
		return "", known, errors.New("planner_compaction_failed")
	}
	return memory.Summary, known, nil
}
func TestCompactionSpikeInvalidSummaryNeverCommits(t *testing.T) {
	for _, scenario := range []string{"empty", "none", "tool", "truncated", "oversized", "schema", "duplicate", "ineffective", "missing-usage"} {
		t.Run(scenario, func(t *testing.T) {
			content := `{"summary":"memory"}`
			finish := "stop"
			switch scenario {
			case "empty":
				content = `{"summary":""}`
			case "none":
				content = ""
			case "truncated":
				finish = "length"
			case "oversized":
				content = `{"summary":"` + strings.Repeat("x", 16*1024+1) + `"}`
			case "schema":
				content = `{"summary":"memory","extra":true}`
			case "duplicate":
				content = `{"summary":"first","summary":"second"}`
			case "ineffective":
				content = `{"summary":"` + strings.Repeat("x", 1024) + `"}`
			}
			transport := spikeTransport(func(request *http.Request) (*http.Response, error) {
				message := map[string]any{"content": content}
				if scenario == "tool" {
					message["tool_calls"] = []any{map[string]any{"id": "bad", "type": "function", "function": map[string]any{"name": "execute", "arguments": "{}"}}}
				}
				body := map[string]any{"choices": []any{map[string]any{"finish_reason": finish, "message": message}}, "usage": map[string]int{"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}}
				if scenario == "missing-usage" {
					delete(body, "usage")
				}
				encoded, _ := json.Marshal(body)
				return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(bytes.NewReader(encoded))}, nil
			})
			llm, err := NewOpenAICompatibleModel(GatewaySettings{URL: "https://fixture.example/v1", Model: "compactor", HTTPClient: &http.Client{Transport: transport}})
			if err != nil {
				t.Fatal(err)
			}
			memory, known, err := spikeSummarize(t.Context(), llm, &model.LLMRequest{Contents: []*genai.Content{genai.NewContentFromText("bounded input", genai.RoleUser)}}, 512)
			if err == nil || memory != "" {
				t.Fatal("invalid summary eligible for commit")
			}
			expected := int64(18)
			if scenario == "missing-usage" {
				expected = 0
			}
			if known != expected {
				t.Fatalf("lost known charge: %d", known)
			}
		})
	}
}

type spikeTransport func(*http.Request) (*http.Response, error)

func (f spikeTransport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestCompactionSpikePostcommitIneffectiveStopsBeforeDispatch(t *testing.T) {
	ctx := t.Context()
	service := adksession.InMemoryService()
	current, err := service.Create(ctx, &adksession.CreateRequest{AppName: "spike", UserID: "user", SessionID: "session"})
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 3; i++ {
		if err := service.AppendEvent(ctx, current.Session, &adksession.Event{ID: fmt.Sprint(i), Author: "worker", LLMResponse: model.LLMResponse{Content: genai.NewContentFromText(strings.Repeat("raw", 1024), genai.RoleModel)}}); err != nil {
			t.Fatal(err)
		}
	}
	p := &spikeProjection{}
	selected, err := p.selectPrefix(current.Session, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := p.commit(ctx, service, current.Session, selected, 0, "accepted-memory"); err != nil {
		t.Fatal(err)
	}
	effects, calls, charged := 0, 1, int64(18) // One accepted compactor response.
	execute, err := functiontool.New(functiontool.Config{Name: "execute", Description: "Forbidden late Worker effect."}, func(_ agent.Context, _ struct{}) (map[string]any, error) { effects++; return map[string]any{}, nil })
	if err != nil {
		t.Fatal(err)
	}
	llm := &scriptedModel{steps: []modelStep{func(request *model.LLMRequest) (*model.LLMResponse, error) {
		response, err := functionStep("execute", map[string]any{})(request)
		response.UsageMetadata = &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 900, CandidatesTokenCount: 7, TotalTokenCount: 907}
		return response, err
	}}}
	root, err := llmagent.New(llmagent.Config{Name: "worker", Model: llm, Tools: []tool.Tool{execute},
		BeforeModelCallbacks: []llmagent.BeforeModelCallback{func(_ agent.Context, request *model.LLMRequest) (*model.LLMResponse, error) {
			calls++
			var err error
			request.Contents, err = p.apply(current.Session)
			return nil, err
		}},
		AfterModelCallbacks: []llmagent.AfterModelCallback{func(_ agent.Context, response *model.LLMResponse, err error) (*model.LLMResponse, error) {
			if err != nil {
				return nil, err
			}
			charged += int64(response.UsageMetadata.TotalTokenCount)
			if response.UsageMetadata.PromptTokenCount >= 750 {
				return nil, errors.New("planner_compaction_failed: ineffective")
			}
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
	var stopped error
	for _, err := range run.Run(ctx, "user", "session", genai.NewContentFromText(spikeTask, genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			stopped = err
			break
		}
	}
	if stopped == nil || effects != 0 || calls != 2 || charged != 925 || p.generation != 1 || p.memory != "accepted-memory" {
		t.Fatal("ineffective response reset authority/charges or dispatched a Worker")
	}
}
