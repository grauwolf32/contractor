package controlplane

import (
	"context"
	"errors"
	"testing"

	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

func TestCandidateResolutionUnavailableDoesNotHideInfrastructureErrors(t *testing.T) {
	t.Parallel()
	for name, err := range map[string]error{
		"typed resolution":    &runtimeconfig.ResolutionError{},
		"missing config":      runtimeconfig.ErrNotFound,
		"missing LLM key":     credentials.ErrNotFound,
		"missing Runtime key": credentials.ErrRuntimeCredentialNotFound,
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			if !candidateResolutionUnavailable(err) {
				t.Fatalf("candidate error %v was not classified as an ineligible edge", err)
			}
		})
	}
	for name, err := range map[string]error{
		"database":       errors.New("database unavailable"),
		"provider clash": credentials.ErrConflict,
		"cancelled":      context.Canceled,
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			if candidateResolutionUnavailable(err) {
				t.Fatalf("infrastructure error %v was hidden as capacity", err)
			}
		})
	}
}
