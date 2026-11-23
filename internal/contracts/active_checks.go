package contracts

// ActiveChecksPolicy controls admission of active Audit checks. A Project owns
// operation policy; an Audit pins its effective value when it starts.
type ActiveChecksPolicy string

const (
	ActiveChecksAutomatic  ActiveChecksPolicy = "automatic"
	ActiveChecksProhibited ActiveChecksPolicy = "prohibited"
)

func (p ActiveChecksPolicy) Valid() bool {
	return p == ActiveChecksAutomatic || p == ActiveChecksProhibited
}
