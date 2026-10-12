package auditstore

import "strings"

// Audit events are numbered 1..N per Audit without gaps (AUD-R223). The
// database function contractor_append_audit_events is the only writer of
// audit_events; Go code reaches it only through an auditEventBatch.
//
// A statement that records events describes them as one batch and uses it in
// the single UPDATE of its audits row, which aliases the table as audit:
//
//	UPDATE audits AS audit
//	   SET ..., <batch.advance()>
//	  ...
//	RETURNING audit.*, <batch.appended()>
//
// advance reserves the batch: the revision and next_event_sequence each
// advance once per event. RETURNING runs once for the updated row, after the
// update, and hands the same events to the function, which numbers them
// directly after the Audit's last event and rejects a reservation that does
// not continue the sequence (SQLSTATE P0004). Event expressions therefore see
// the updated row, so an Audit's own event carries its new revision. They
// may read the UPDATE's FROM items and earlier CTEs, never later ones.
type auditEventBatch []auditEventSource

// auditEventSource is one part of a batch: count is a bigint expression for
// how many events it adds and events a jsonb array expression of exactly that
// many event objects, in order.
type auditEventSource struct {
	count  string
	events string
}

func auditEvents(sources ...auditEventSource) auditEventBatch {
	return auditEventBatch(sources)
}

// auditEvent is one event. Arguments are SQL expressions; a bind parameter
// needs an explicit cast because jsonb_build_object accepts any type, and an
// event without an entity revision passes NULL.
func auditEvent(kind, entityID, entityRevision, summary string) auditEventSource {
	return auditEventSource{
		count:  "1",
		events: "jsonb_build_array(" + auditEventObject(kind, entityID, entityRevision, summary) + ")",
	}
}

// auditEventsFrom is one event per row of relation, ordered by order. Count
// and events read the same relation, so they cannot disagree.
func auditEventsFrom(relation, order, kind, entityID, entityRevision, summary string) auditEventSource {
	return auditEventSource{
		count: "(SELECT count(*) FROM " + relation + ")",
		events: "COALESCE((SELECT jsonb_agg(" +
			auditEventObject(kind, entityID, entityRevision, summary) +
			" ORDER BY " + order + ") FROM " + relation + "), '[]'::jsonb)",
	}
}

// auditEventArray takes events already encoded as a jsonb array of event
// objects.
func auditEventArray(array string) auditEventSource {
	return auditEventSource{count: "jsonb_array_length(" + array + ")", events: array}
}

func auditEventObject(kind, entityID, entityRevision, summary string) string {
	return "jsonb_build_object('kind', " + kind + ", 'entityId', " + entityID +
		", 'entityRevision', " + entityRevision + ", 'summary', " + summary + ")"
}

// size is the number of events in the batch.
func (b auditEventBatch) size() string {
	counts := make([]string, len(b))
	for index, source := range b {
		counts[index] = source.count
	}
	return "(" + strings.Join(counts, " + ") + ")"
}

// advance is the SET item that reserves the batch on the updated audits row.
func (b auditEventBatch) advance() string {
	return "revision = audit.revision + " + b.size() +
		",\n           next_event_sequence = audit.next_event_sequence + " + b.size()
}

// created is the revision and next_event_sequence of an Audit inserted with
// the batch as its first events.
func (b auditEventBatch) created() (revision, nextEventSequence string) {
	return b.size(), "1 + " + b.size()
}

// appended is the RETURNING item that appends the reserved batch. Its value,
// last_event_sequence, is the sequence of the batch's last event.
func (b auditEventBatch) appended() string {
	arrays := make([]string, len(b))
	for index, source := range b {
		arrays[index] = source.events
	}
	return "contractor_append_audit_events(audit.audit_id, " + strings.Join(arrays, " || ") +
		") AS last_event_sequence"
}
