package auditservice

import (
	"context"
	"encoding/json"
	"math"
	"regexp"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/jackc/pgx/v5"
)

var eventResourceID = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$`)

type EventListParams struct {
	OwnerID, AuditID string
	// A continuation freezes the immutable prefix; new events appear on refresh.
	ThroughSequence *uint64
	BeforeSequence  *uint64
	Limit           int
}

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
func (s *Service) ListEventsPage(ctx context.Context, p EventListParams) (EventPage, error) {
	var result EventPage
	if !validReviewIdentity(p.OwnerID, 256) || !validReviewIdentity(p.AuditID, 256) ||
		p.Limit < 1 || p.Limit > auditstore.MaxPageSize+1 ||
		(p.ThroughSequence == nil) != (p.BeforeSequence == nil) ||
		(p.ThroughSequence != nil && (*p.ThroughSequence > math.MaxInt64 ||
			*p.BeforeSequence == 0 || *p.BeforeSequence > *p.ThroughSequence)) {
		return result, auditstore.ErrInvalid
	}
	tx, err := s.pool.BeginTx(ctx, pgx.TxOptions{IsoLevel: pgx.RepeatableRead, AccessMode: pgx.ReadOnly})
	if err != nil {
		return result, err
	}
	defer func() { _ = tx.Rollback(ctx) }()
	audit, err := auditstore.NewPostgresStore(tx).Get(ctx, p.OwnerID, p.AuditID)
	if err != nil {
		return result, err
	}
	result.ThroughSequence = audit.EventSequence
	if p.ThroughSequence != nil {
		if *p.ThroughSequence > audit.EventSequence {
			return result, auditstore.ErrInvalid
		}
		result.ThroughSequence = *p.ThroughSequence
	}
	err = tx.QueryRow(ctx, `SELECT count(*) FROM audit_events WHERE audit_id=$1 AND sequence_number <= $2`,
		p.AuditID, int64(result.ThroughSequence)).Scan(&result.Total)
	if err != nil {
		return result, err
	}
	var before *int64
	if p.BeforeSequence != nil {
		value := int64(*p.BeforeSequence)
		before = &value
	}
	rows, err := tx.Query(ctx, `
SELECT audit_id, sequence_number, kind, entity_id, entity_revision, summary, created_at
 FROM audit_events WHERE audit_id=$1 AND sequence_number <= $2
 AND ($3::bigint IS NULL OR sequence_number < $3)
 ORDER BY sequence_number DESC LIMIT $4`, p.AuditID, int64(result.ThroughSequence), before, p.Limit)
	if err != nil {
		return result, err
	}
	result.Items = make([]Event, 0, p.Limit)
	for rows.Next() {
		var event Event
		var summary json.RawMessage
		if err := rows.Scan(&event.AuditID, &event.Sequence, &event.Kind, &event.EntityID,
			&event.EntityRevision, &summary, &event.CreatedAt); err != nil {
			rows.Close()
			return result, err
		}
		event.Summary = safeEventSummary(summary)
		result.Items = append(result.Items, event)
	}
	err = rows.Err()
	rows.Close()
	if err != nil {
		return result, err
	}
	return result, tx.Commit(ctx)
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
			if json.Unmarshal(value, &text) == nil && eventResourceID.MatchString(text) {
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
		return ReviewSubjectKind(value).Valid()
	case "kind":
		return value == FindingReviewKind || value == "report-acceptance" || auditstore.ItemApprovalKind(value).Valid()
	case "verdict":
		return AnalystVerdict(value).Valid()
	case "action":
		return ReviewAction(value).Valid()
	}
	return false
}
