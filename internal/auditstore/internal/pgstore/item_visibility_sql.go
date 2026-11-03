package pgstore

// acceptedItemSQL restricts a read to immutable published Round membership.
// Proposed rows are private construction state, including their reviews.
// The owning Round is one indexed lookup, independent of other Audit items.
func acceptedItemSQL(item string) string {
	return `EXISTS (SELECT 1 FROM audit_rounds AS visible_round
 WHERE visible_round.audit_id = ` + item + `.audit_id
   AND visible_round.round_id = ` + item + `.round_id
   AND visible_round.state <> 'proposed')`
}

// visibleItemReviewSQL uses the request alias shared by review projections.
// Finding/report reviews retain their own lifecycle; only item-action requests
// depend on accepted Round membership.
const visibleItemReviewSQL = `(request.subject_kind <> 'audit-item-action' OR EXISTS (
 SELECT 1 FROM audit_items AS visible_item
 JOIN audit_rounds AS visible_round USING (round_id, audit_id)
 WHERE visible_item.audit_id = request.audit_id AND visible_item.item_id = request.subject_id
   AND visible_round.state <> 'proposed'))`
