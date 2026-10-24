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

func (p AuditPhase) Valid() bool {
	return p == AuditPhaseNotStarted || p == AuditPhasePreparing || p == AuditPhaseInventory || p == AuditPhaseRounds
}

// ShowsPreparation reports whether an Audit in phase p shows its
// preparation: it is preparing or inventory. The database requires an Audit
// in either phase to pin a prepare role.
func (p AuditPhase) ShowsPreparation() bool {
	return p == AuditPhasePreparing || p == AuditPhaseInventory
}

type PreparationStatus string

const (
	PreparationPending  PreparationStatus = "pending"
	PreparationRunning  PreparationStatus = "running"
	PreparationAccepted PreparationStatus = "accepted"
	PreparationFailed   PreparationStatus = "failed"
)

func (s PreparationStatus) Valid() bool {
	return s == PreparationPending || s == PreparationRunning || s == PreparationAccepted || s == PreparationFailed
}
