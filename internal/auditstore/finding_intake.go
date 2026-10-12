package auditstore

// DirectFindingAssessment is a direct verification that finding intake
// accepted for the finding a receipt contributed to in an Audit: the
// assessment, the retained verification result and the contract it was
// checked against, linked to the Audit under their logical keys, and the
// evidence bytes they retain.
type DirectFindingAssessment struct {
	AuditID            string
	ReceiptID          string
	AssessmentID       string
	SemanticAssessment string
	Result             ArtifactLink
	Contract           ArtifactLink
	RetainedBytes      int64
}
