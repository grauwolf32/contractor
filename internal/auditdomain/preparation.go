package auditdomain

// AuditPhase identifies available work independently of pause/cancel/terminal
// lifecycle state. Preparing and inventory have no current Round.
type AuditPhase string

const (
	AuditPhaseNotStarted AuditPhase = "not-started"
	AuditPhasePreparing  AuditPhase = "preparing"
	AuditPhaseInventory  AuditPhase = "inventory"
	AuditPhaseRounds     AuditPhase = "rounds"
)

type PreparationStatus string

const (
	PreparationPending  PreparationStatus = "pending"
	PreparationRunning  PreparationStatus = "running"
	PreparationAccepted PreparationStatus = "accepted"
	PreparationFailed   PreparationStatus = "failed"
)
