// Package local declares a State constant inside a function.
package local

type State string

const Open State = "open"

func hidden() State {
	const Hidden State = "hidden"
	return Hidden
}
