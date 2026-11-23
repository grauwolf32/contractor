// Package declared is a fixture for DeclaredValues: State has three
// constants written in each supported form, next to values of other types
// that reuse the same words.
package declared

type State string

const (
	Open State = "open"
	Held       = State("held")
)

const Closed State = "closed"

// Other reuses a State word; it is a different set.
type Other string

const OtherOpen Other = "open"

const untyped = "archived"

var variable State = "paused"
