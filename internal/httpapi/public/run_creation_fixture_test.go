package public

import (
	"context"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/runservice"
)

// Test embeddings compose the same explicit service boundary as the app.
type RunWriter = runservice.PublicRunWriter
type UnitOfWork interface {
	Do(context.Context, func(RunWriter, *artifacts.Service) error) error
}

func newTestRunCreator(t *testing.T, services testServices, runs runservice.RunReader, transaction UnitOfWork, skillsAvailable bool) *runservice.Service {
	t.Helper()
	creator, err := runservice.New(runservice.Options{
		Runs: runs, Workflows: services.Config,
		LLMCredentials: services.Credentials, CredentialGuard: services.ManagedCredentials,
		RuntimeCredentials: services.RuntimeCredentials, Projects: services.Projects,
		SkillInitializationAvailable: skillsAvailable, PublicTransaction: transaction.Do,
	})
	if err != nil {
		t.Fatal(err)
	}
	return creator
}
