// Package computed declares a State whose value is not a string literal.
package computed

type State string

const Open State = "op" + "en"
