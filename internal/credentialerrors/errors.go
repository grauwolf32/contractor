// Package credentialerrors holds lookup outcomes shared across credential
// providers and RuntimeConfig validation without a package dependency cycle.
package credentialerrors

import "github.com/grauwolf32/contractor/internal/apierror"

var (
	LLMNotFound    = apierror.New(apierror.NotFound, "LLM credential not found")
	RuntimeInvalid = apierror.New(apierror.RuntimeConfigInvalid, "invalid Runtime adapter credential",
		apierror.WithMessage("Runtime credential request is invalid"))
	RuntimeNotFound = apierror.New(apierror.NotFound, "Runtime adapter credential not found")
)
