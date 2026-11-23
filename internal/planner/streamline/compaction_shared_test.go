package streamline

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"google.golang.org/adk/v2/model"
	adksession "google.golang.org/adk/v2/session"
	"google.golang.org/genai"
)

func TestSharedCompactionBoundaries(t *testing.T) {
	var fixture struct {
		Tails     []struct{ Groups, Keep, Covered, Eligible int }
		Summaries []struct {
			Previous              string
			Repeat, ExtraRawBytes int
			Valid                 bool
		}
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "compaction-boundary-cases.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	event := func(id, text string) *adksession.Event {
		return &adksession.Event{ID: id, LLMResponse: model.LLMResponse{Content: genai.NewContentFromText(text, genai.RoleModel)}}
	}
	protected := func() []*adksession.Event {
		first := event("protected", "task")
		first.Author = "user"
		return []*adksession.Event{first}
	}
	for index, test := range fixture.Tails {
		t.Run(fmt.Sprintf("tail-%d", index), func(t *testing.T) {
			c := newContextCompaction(compactionSettings{keepRecent: test.Keep, maxCompactions: 3})
			events := protected()
			for i := range test.Groups {
				events = append(events, event(fmt.Sprint(i), "step"))
				if test.Covered > 0 && i+1 == test.Covered+test.Keep {
					selection, failure := c.selectCheckpoint(fixedSession(events))
					if failure != nil {
						t.Fatal(failure)
					}
					if failure := c.publish(selection, "memory"); failure != nil {
						t.Fatal(failure)
					}
				}
			}
			selection, failure := c.selectCheckpoint(fixedSession(events))
			if test.Eligible == 0 {
				assertPlannerCode(t, failure, "planner_compaction_failed")
			} else if failure != nil || selection.end-test.Covered != test.Eligible {
				t.Fatalf("selected %d groups: %v", selection.end-test.Covered, failure)
			}
		})
	}
	policy := compactionInvocation("streamline").Stage.ExecutionConfig.Planner.CompactorModelPolicy
	for index, test := range fixture.Summaries {
		t.Run(fmt.Sprintf("summary-%d", index), func(t *testing.T) {
			c := newContextCompaction(compactionSettings{keepRecent: 1, maxCompactions: 3})
			events := append(protected(), event("large", strings.Repeat("a", 2000)), event("small", "new fact"))
			first, failure := c.selectCheckpoint(fixedSession(events))
			if failure != nil {
				t.Fatal(failure)
			}
			previous := strings.Repeat(test.Previous, test.Repeat)
			reply, _ := summaryStep(previous)(nil)
			accepted, failure := validateCompactorReply(reply, *policy, first.sourceBytes)
			if failure != nil {
				t.Fatal(failure)
			}
			if failure := c.publish(first, accepted); failure != nil {
				t.Fatal(failure)
			}
			events = append(events, event("newest", "protected tail"))
			second, failure := c.selectCheckpoint(fixedSession(events))
			if failure != nil {
				t.Fatal(failure)
			}
			raw, err := json.Marshal(events[2].Content)
			if err != nil {
				t.Fatal(err)
			}
			candidate := strings.Repeat("x", len(raw)+test.ExtraRawBytes)
			reply, _ = summaryStep(candidate)(nil)
			_, failure = validateCompactorReply(reply, *policy, second.sourceBytes)
			if (failure == nil) != test.Valid {
				t.Fatalf("source=%d old=%d raw=%d candidate=%d: %v", second.sourceBytes, len(previous), len(raw), len(candidate), failure)
			}
		})
	}
}
