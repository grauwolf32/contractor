package session

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/grauwolf32/contractor/internal/runstore"
	"google.golang.org/adk/v2/model"
	adksession "google.golang.org/adk/v2/session"
	"google.golang.org/genai"
)

// New SDK workflow fields are invocation-local data, not Contractor timeline
// facts. One real PostgreSQL fixture checks every field independently, while
// raw JSON is a positive control that each canary really reached the SDK.
func TestPostgresADKEventFieldsRemainInRAMOnly(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool := isolatedPlannerPool(t, ctx)
	store := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	createPlannerStage(t, ctx, store)
	service, err := New(store, Options{})
	if err != nil {
		t.Fatal(err)
	}
	start, err := service.Begin(ctx, "stage-planner")
	if err != nil {
		t.Fatal(err)
	}
	adk, err := service.NewADKSession(ctx, start.Identity, ADKOptions{AppName: "migration_fields", UserID: "fixture-user", AllowedTools: []string{"finish"}})
	if err != nil {
		t.Fatal(err)
	}
	created, err := adk.Create(ctx, &adksession.CreateRequest{AppName: "migration_fields", UserID: "fixture-user", SessionID: start.Identity.SessionID})
	if err != nil {
		t.Fatal(err)
	}
	fields := []struct {
		name  string
		apply func(*adksession.Event, string)
	}{
		{"output", func(e *adksession.Event, canary string) { e.Output = map[string]any{"nested": []any{canary, true}} }},
		{"node_info", func(e *adksession.Event, canary string) {
			e.NodeInfo = &adksession.NodeInfo{Path: canary, MessageAsOutput: true, OutputFor: []string{canary + "/ancestor"}}
		}},
		{"isolation_scope", func(e *adksession.Event, canary string) { e.IsolationScope = canary }},
		{"requested_input", func(e *adksession.Event, canary string) {
			e.RequestedInput = &adksession.RequestInput{InterruptID: canary + "-interrupt", Message: canary + "-message", Payload: map[string]any{"private": canary}, ResponseSchema: &jsonschema.Schema{Type: "object", Description: canary + "-schema", Properties: map[string]*jsonschema.Schema{"answer": {Type: "string", Description: canary + "-property"}}}}
		}},
		{"state_delta", func(e *adksession.Event, canary string) {
			e.Actions.StateDelta["migration_private_state"] = map[string]any{"private": canary}
		}},
		{"llm_response_json", func(e *adksession.Event, canary string) {
			e.CustomMetadata = map[string]any{"private": canary}
			e.ErrorMessage = canary + "-provider-error"
			e.SessionResumptionHandle = canary + "-resume"
		}},
	}
	for index, field := range fields {
		t.Run(field.name, func(t *testing.T) {
			canary := "private-adk26-" + field.name + "-canary"
			event := adksession.NewEvent(ctx, "fixture-invocation")
			event.Author = "streamline_planner"
			event.LLMResponse = model.LLMResponse{Content: genai.NewContentFromText("private-adk26-distinct-content-canary", genai.RoleModel), UsageMetadata: &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 11, CandidatesTokenCount: 7}}
			field.apply(event, canary)
			raw, err := json.Marshal(event)
			if err != nil || !strings.Contains(string(raw), canary) {
				t.Fatal("raw Event serialization did not carry the private fixture")
			}
			assertMigrationFieldJSON(t, raw, field.name, canary)
			if field.name == "llm_response_json" {
				response, err := json.Marshal(event.LLMResponse)
				if err != nil || !strings.Contains(string(response), `"errorMessage"`) || !strings.Contains(string(response), `"customMetadata"`) || !strings.Contains(string(response), canary) || strings.Contains(string(response), `"Content"`) {
					t.Fatal("direct LLMResponse JSON dropped data or used the old field names")
				}
			}
			if err := adk.AppendEvent(ctx, created.Session, event); err != nil {
				t.Fatal(err)
			}
			loaded, err := adk.Get(ctx, &adksession.GetRequest{AppName: "migration_fields", UserID: "fixture-user", SessionID: start.Identity.SessionID})
			if err != nil {
				t.Fatal(err)
			}
			for _, ram := range []adksession.Session{created.Session, loaded.Session} {
				if ram.Events().Len() != index+1 {
					t.Fatal("RAM history lost an event")
				}
				got, err := json.Marshal(ram.Events().At(index))
				if err != nil || string(got) != string(raw) {
					t.Fatal("SDK RAM roundtrip lost or changed event fields")
				}
				if !reflect.DeepEqual(migrationEventField(event, field.name), migrationEventField(ram.Events().At(index), field.name)) {
					t.Fatal("SDK RAM roundtrip lost the target field's structured value")
				}
				if field.name == "state_delta" {
					value, err := ram.State().Get("migration_private_state")
					if err != nil || !reflect.DeepEqual(value, map[string]any{"private": canary}) {
						t.Fatal("ordinary StateDelta did not update invocation-local state")
					}
				}
			}
			page, err := store.ListRunEvents(ctx, "run-planner", 0, 30)
			if err != nil || page.More {
				t.Fatal("bounded durable event page could not be read")
			}
			publicSource, err := json.Marshal(page)
			if err != nil || strings.Contains(string(publicSource), "private-adk26-") {
				t.Fatal("private SDK data escaped into the durable/public Run event source")
			}
			stored, err := store.GetPlannerSession(ctx, start.Identity.SessionID)
			if err != nil || strings.Contains(string(stored.State), "private-adk26-") || strings.Contains(string(stored.State), "migration_private_state") {
				t.Fatal("private SDK state escaped into the durable Planner projection")
			}
			activities := 0
			for _, durable := range page.Events {
				if durable.Kind != runstore.RunEventPlannerActivity {
					continue
				}
				activities++
				var data struct {
					Activity map[string]any `json:"activity"`
				}
				if err := json.Unmarshal(durable.Data, &data); err != nil {
					t.Fatal(err)
				}
				// The existing Run-event encoder clones empty name slices to nil;
				// their durable JSON is null. Preserve that exact projection.
				want := map[string]any{"kind": "adk_event", "author": "streamline_planner", "functionCalls": nil, "functionResults": nil, "inputTokens": float64(11), "outputTokens": float64(7)}
				if !reflect.DeepEqual(data.Activity, want) {
					t.Fatal("durable ADK facts changed their explicit allowlist")
				}
			}
			if activities != index+1 {
				t.Fatal("a required reduced event was dropped or repeated")
			}
		})
	}
}

func migrationEventField(event *adksession.Event, name string) any {
	switch name {
	case "output":
		return event.Output
	case "node_info":
		return event.NodeInfo
	case "isolation_scope":
		return event.IsolationScope
	case "requested_input":
		return event.RequestedInput
	case "state_delta":
		return event.Actions.StateDelta
	default:
		return event.LLMResponse
	}
}

func assertMigrationFieldJSON(t *testing.T, raw []byte, name, canary string) {
	t.Helper()
	var shape map[string]json.RawMessage
	if err := json.Unmarshal(raw, &shape); err != nil {
		t.Fatal(err)
	}
	keys := map[string]string{"output": "output", "node_info": "nodeInfo", "isolation_scope": "isolationScope", "requested_input": "requestedInput", "state_delta": "stateDelta"}
	if name == "state_delta" {
		if err := json.Unmarshal(shape["actions"], &shape); err != nil {
			t.Fatal(err)
		}
	}
	selected := []string{keys[name]}
	if name == "llm_response_json" {
		selected = []string{"errorMessage", "customMetadata", "sessionResumptionHandle"}
	}
	for _, key := range selected {
		if !strings.Contains(string(shape[key]), canary) {
			t.Fatalf("target JSON field %s dropped its private fixture", key)
		}
	}
}
