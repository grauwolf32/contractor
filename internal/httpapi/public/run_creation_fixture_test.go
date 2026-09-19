package public

import (
	"context"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/runservice"
	"testing"
)

// Test embeddings compose the same explicit service boundary as the app.
type RunWriter = runservice.PublicRunWriter
type UnitOfWork interface {
	Do(context.Context, func(RunWriter, *artifacts.Service) error) error
}

func newTestRunCreator(t *testing.T, dependencies Dependencies, runs runservice.RunReader, transaction UnitOfWork, skillsAvailable bool) *runservice.Service {
	t.Helper()
	creator, err := runservice.New(runservice.Options{
		Runs: runs, Workflows: dependencies.Config,
		LLMCredentials: dependencies.Credentials, CredentialGuard: dependencies.ManagedCredentials,
		RuntimeCredentials: dependencies.RuntimeCredentials, Projects: dependencies.Projects,
		SkillInitializationAvailable: skillsAvailable, PublicTransaction: transaction.Do,
	})
	if err != nil {
		t.Fatal(err)
	}
	return creator
}
