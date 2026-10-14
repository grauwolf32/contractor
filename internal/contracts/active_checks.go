package contracts

// ActiveChecksPolicy controls admission of active Audit checks. A Project owns
// approval policy; an Audit pins its effective value when it starts.
type ActiveChecksPolicy string

const (
	ActiveChecksAutomatic        ActiveChecksPolicy = "automatic"
	ActiveChecksApprovalRequired ActiveChecksPolicy = "approval-required"
	ActiveChecksProhibited       ActiveChecksPolicy = "prohibited"
)

func (p ActiveChecksPolicy) Valid() bool {
	return p == ActiveChecksAutomatic || p == ActiveChecksApprovalRequired || p == ActiveChecksProhibited
}
