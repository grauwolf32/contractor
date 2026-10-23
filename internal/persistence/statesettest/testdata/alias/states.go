// Package alias declares a State constant through an alias of State.
package alias

type State string

const Open State = "open"

type parked = State

const Parked parked = "parked"
