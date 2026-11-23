package a2a

import (
	"fmt"
	"net/http"
)

// New builds an Invoker on an injected plain-HTTP client for tests; private
// production traffic uses NewMTLS.
func New(httpClient *http.Client, options Options) (*Invoker, error) {
	if httpClient == nil {
		return nil, fmt.Errorf("A2A HTTP client is required")
	}
	pollInterval, err := validatedPollInterval(options)
	if err != nil {
		return nil, err
	}
	return &Invoker{
		build: sdkClientBuilder(cloneBoundedHTTPClient(httpClient)), pollInterval: pollInterval,
	}, nil
}
