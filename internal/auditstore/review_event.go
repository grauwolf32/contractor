package auditstore

// ReviewEventParams describes an event emitted by a human review mutation.
type ReviewEventParams struct {
	AuditID        string
	Kind           string
	EntityID       string
	EntityRevision *uint64
	Summary        map[string]any
}
