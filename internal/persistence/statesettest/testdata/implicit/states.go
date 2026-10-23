// Package implicit repeats a State constant implicitly.
package implicit

type State string

const (
	Open State = "open"
	Held
)
