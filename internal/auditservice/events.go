package auditservice

import (
	"context"
	"encoding/json"
	"math"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
)

type Event struct {
	AuditID        string         `json:"auditId"`
	Sequence       uint64         `json:"sequence"`
	Kind           string         `json:"kind"`
	EntityID       string         `json:"entityId"`
	EntityRevision *uint64        `json:"entityRevision,omitempty"`
	Summary        map[string]any `json:"summary"`
	CreatedAt      time.Time      `json:"createdAt"`
}

type EventPage struct {
	Items           []Event
	ThroughSequence uint64
	Total           int
}

// ListEventsPage reads an owner-fenced immutable prefix in one short read-only
// snapshot. Unlike mutable finding pages, continuations survive new revisions.
// A page reads at most p.Limit events and no count: events are numbered 1, 2,
// 3 and so on without gaps (AUD-R223), so the prefix through ThroughSequence
// holds exactly that many events, and counting them on every page would make
// reading N events cost O(N^2/limit).
func (s *Service) ListEventsPage(ctx context.Context, p auditstore.EventListParams) (EventPage, error) {
	through, events, err := s.reads().ListEventsPage(ctx, p)
	if err != nil {
		return EventPage{}, err
	}
	result := EventPage{ThroughSequence: through, Total: int(through), Items: make([]Event, 0, len(events))}
	for _, event := range events {
		result.Items = append(result.Items, Event{
			AuditID: event.AuditID, Sequence: event.Sequence, Kind: event.Kind, EntityID: event.EntityID,
			EntityRevision: event.EntityRevision, Summary: safeEventSummary(event.Summary),
			CreatedAt: event.CreatedAt,
		})
	}
	return result, nil
}

// Event summaries are a public projection, not arbitrary stored JSON. Never
// expose free-form messages, rationale, request digests or future nested data.
func safeEventSummary(raw json.RawMessage) map[string]any {
	result := make(map[string]any)
	var source map[string]json.RawMessage
	if json.Unmarshal(raw, &source) != nil {
		return result
	}
	for key, value := range source {
		switch key {
		case "round", "items", "reviews", "members", "count":
			var number int64
			if json.Unmarshal(value, &number) == nil && number >= 0 && number <= math.MaxInt32 {
				result[key] = number
			}
		case "runId", "subjectId", "findingId", "workflowRole":
			var text string
			if json.Unmarshal(value, &text) == nil && contracts.ValidResourceID(text) {
				result[key] = text
			}
		default:
			var text string
			if json.Unmarshal(value, &text) == nil && eventSummaryEnum(key, text) {
				result[key] = text
			}
		}
	}
	return result
}

func eventSummaryEnum(key, value string) bool {
	switch key {
	case "state", "from", "to":
		return auditstore.AuditState(value).Valid() || auditstore.RoundState(value).Valid()
	case "role":
		return auditstore.ExecutionRole(value).Valid()
	case "outcome":
		return auditstore.TerminalOutcome(value).Valid()
	case "disposition":
		return auditstore.CollectionDisposition(value).Valid()
	case "subjectKind":
		return auditstore.ReviewSubjectKind(value).Valid()
	case "kind":
		return value == auditstore.FindingReviewKind || value == auditstore.ReportAcceptanceReviewKind || auditstore.ItemApprovalKind(value).Valid()
	case "verdict":
		return auditstore.AnalystVerdict(value).Valid()
	case "action":
		return auditstore.ReviewAction(value).Valid()
	}
	return false
}
