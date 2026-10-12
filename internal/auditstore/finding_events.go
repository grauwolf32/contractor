package auditstore

type RejectedFindingProposalParams struct {
	AuditID   string
	ReceiptID string
	RunID     string
	Reason    string
}

type DirectFindingAssessmentParams struct {
	AuditID         string
	FindingID       string
	FindingRevision int64
	AssessmentID    string
	RetainedBytes   int64
}
