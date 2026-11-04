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
