package auditstore

// ReviewDecisionWriteParams records an item/report action. Finding edits use
// the ordinary Project command and common mutation receipt.
type ReviewDecisionWriteParams struct {
	DecisionID, RequestID, AuditID, ActorID, Action         string
	Rationale, SubjectDigest, IdempotencyKey, RequestDigest string
	SubjectRevision                                         uint64
}
