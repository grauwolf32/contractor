package streamline

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/planner"
	"google.golang.org/adk/v2/model"
	adksession "google.golang.org/adk/v2/session"
	"google.golang.org/genai"
)

func TestCompactionGroupBoundary(t *testing.T) {
	c := newContextCompaction(compactionSettings{keepRecent: 1, maxCompactions: 2, threshold: 1})
	events := []*adksession.Event{{ID: "protected", Author: "user", LLMResponse: model.LLMResponse{
		Content: genai.NewContentFromText("protected", genai.RoleUser),
	}}}
	for index := range maxCompactionGroups {
		events = append(events, &adksession.Event{ID: fmt.Sprint(index), LLMResponse: model.LLMResponse{
			Content: genai.NewContentFromText("turn", genai.RoleModel),
		}})
	}
	if err := c.prepareMain(fixedSession(events), planner.PlannerPlan{}, &model.LLMRequest{}); err != nil {
		t.Fatalf("4096 complete source groups rejected: %v", err)
	}
	events = append(events, &adksession.Event{ID: "one-over", LLMResponse: model.LLMResponse{
		Content: genai.NewContentFromText("turn", genai.RoleModel),
	}})
	err := c.prepareMain(fixedSession(events), planner.PlannerPlan{}, &model.LLMRequest{})
	assertPlannerCode(t, err, "planner_compaction_limit_exceeded")
	if !strings.Contains(err.Error(), "group") {
		t.Fatalf("wrong bound: %v", err)
	}
}

func TestCompactionTailBoundaryBeforeAndAfterCheckpoint(t *testing.T) {
	for _, extra := range []int{0, 1} {
		t.Run(fmt.Sprintf("extra-%d", extra), func(t *testing.T) {
			c := newContextCompaction(compactionSettings{keepRecent: 1, maxCompactions: 2, threshold: 1})
			content := genai.NewContentFromText("x", genai.RoleModel)
			encoded, err := json.Marshal(content)
			if err != nil {
				t.Fatal(err)
			}
			content.Parts[0].Text = strings.Repeat("x", maxCompactionTailBytes-len(encoded)+1+extra)
			events := []*adksession.Event{
				{ID: "protected", Author: "user", LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("protected", genai.RoleUser)}},
				{ID: "older", LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("older", genai.RoleModel)}},
				{ID: "tail", LLMResponse: model.LLMResponse{Content: content}},
			}
			selection, failure := c.selectCheckpoint(fixedSession(events))
			if extra == 1 {
				assertPlannerCode(t, failure, "planner_compaction_limit_exceeded")
				if !strings.Contains(failure.Error(), "tail") {
					t.Fatalf("wrong bound: %v", failure)
				}
				return
			}
			if failure != nil {
				t.Fatalf("1 MiB tail rejected: %v", failure)
			}
			if failure := c.publish(selection, "memory"); failure != nil {
				t.Fatal(failure)
			}
			request := &model.LLMRequest{}
			if err := c.prepareMain(fixedSession(events), planner.PlannerPlan{}, request); err != nil {
				t.Fatalf("1 MiB projected tail rejected: %v", err)
			}
			// A low main prompt measurement does not permit the retained tail
			// to grow beyond its byte ceiling between checkpoints.
			if failure := c.observeMain(&genai.GenerateContentResponseUsageMetadata{}); failure != nil {
				t.Fatal(failure)
			}
			events = append(events, &adksession.Event{ID: "later", LLMResponse: model.LLMResponse{
				Content: genai.NewContentFromText("later", genai.RoleModel),
			}})
			err = c.prepareMain(fixedSession(events), planner.PlannerPlan{}, &model.LLMRequest{})
			assertPlannerCode(t, err, "planner_compaction_limit_exceeded")
			if !strings.Contains(err.Error(), "tail") {
				t.Fatalf("wrong bound: %v", err)
			}
		})
	}
}
