package scheduler

import (
	"context"
	"testing"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestToolWorkerMaterializesSettingsWithoutGatewayOrCredentialServices(t *testing.T) {
	snapshot, err := config.Load("../../configs/scan", config.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	workflow, err := snapshot.Workflow("nuclei-target@1")
	if err != nil {
		t.Fatal(err)
	}
	scheduler := &Scheduler{options: Options{RuntimeSettings: contracts.RuntimeSettings{
		ArtifactAPIURL: "https://artifacts.example", RequestTimeoutSeconds: 30,
	}}}
	settings, err := scheduler.workerExecutionSettings(context.Background(), workflow.Stages["scan"])
	if err != nil {
		t.Fatal(err)
	}
	value := settings["scanner"]
	if !value.ModelPolicy.IsZero() || value.RuntimeSettings.LLMGatewayURL != "" || value.RuntimeSettings.LLMGatewayToken != nil || value.ResolvedRuntimeConfigProvenance.LLMGatewayConfig != nil {
		t.Fatalf("model dependency: %+v", value)
	}
}
