// Package credentialerrors holds lookup outcomes shared across credential
// providers and RuntimeConfig validation without a package dependency cycle.
package credentialerrors

import "errors"

var (
	LLMNotFound     = errors.New("LLM credential not found")
	RuntimeInvalid  = errors.New("invalid Runtime adapter credential")
	RuntimeNotFound = errors.New("Runtime adapter credential not found")
)
