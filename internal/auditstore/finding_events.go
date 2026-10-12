package auditstore

type RejectedFindingProposalParams struct {
	AuditID   string
	ReceiptID string
	RunID     string
	Reason    string
}
