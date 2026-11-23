// Package localtype aliases State inside a function.
package localtype

type State string

const Open State = "open"

func hidden() State {
	type parked = State
	return parked("parked")
}
