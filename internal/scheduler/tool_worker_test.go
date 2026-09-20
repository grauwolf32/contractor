package scheduler

import (
	"context"
	"testing"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
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
	stage := workflow.Stages["scan"]
	settings, err := scheduler.workerExecutionSettingsForRun(context.Background(), runstore.WorkflowRun{}, stage, schedulerTestReservations(t, stage))
	if err != nil {
		t.Fatal(err)
	}
	value := settings["scanner"]
	if !value.ModelPolicy.IsZero() || value.RuntimeSettings.LLMGatewayURL != "" || value.RuntimeSettings.LLMGatewayToken != nil || value.ResolvedRuntimeConfigProvenance.LLMGatewayConfig != nil {
		t.Fatalf("model dependency: %+v", value)
	}
}
