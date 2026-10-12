package auditstore

type ProjectDeletionClaim struct {
	ProjectID string
	OwnerID   string
	ClaimID   string
	Phase     string
}
