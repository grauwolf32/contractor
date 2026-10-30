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

// PrecedesRounds reports whether an Audit in phase p has started and has no
// Round yet: it is preparing or building its inventory. Neither phase has a
// current Round (AUD-R194), and the database requires an Audit in either
// phase to pin a baseline and a profile with a prepare role. This is the one
// definition of that phase set: TestPrecedesRoundsHasOneDefinition fails on
// a copy in Go, and a PostgreSQL test holds the audits_phase_shape
// constraint and the preparation baseline trigger to it.
func (p AuditPhase) PrecedesRounds() bool {
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
