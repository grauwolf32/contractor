// Package added is the declared fixture with one more State constant.
package added

type State string

const (
	Open   State = "open"
	Held   State = "held"
	Closed State = "closed"
	Paused State = "paused"
)
