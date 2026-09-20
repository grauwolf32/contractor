package auditservice

import (
	"errors"
	"os"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestInventoryConstructionDoesNotChooseScanExecutor(t *testing.T) {
	snapshot, err := config.Load("../../configs", config.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	profile, err := snapshot.AuditProfile("openapi-sqlmap-scan@1")
	if err != nil {
		t.Fatal(err)
	}
	// A task executor is selected explicitly, independently of the entry Stage.
	binding := profile.Workflows["scan"]
	scan := binding.Workflow.Stages["scan"]
	delete(binding.Workflow.Stages, "scan")
	binding.Workflow.Stages["custom-executor"] = scan
	binding.Workflow.AuditTask.Stage = "custom-executor"
	binding.Workflow.EntryStage = "prepare-context"
	ordinary, err := snapshot.Workflow("artifact-copy@2")
	if err != nil {
		t.Fatal(err)
	}
	before := ordinary.Stages[ordinary.EntryStage]
	before.WorkflowOutputs = map[string]string{}
	before.On.Succeeded = config.TransitionAction{Kind: config.TransitionNext, NextStage: "custom-executor"}
	before.Context = config.StageContext{Artifacts: map[string]config.ContextArtifact{
		"openapi": {Namespace: "inputs", Name: "openapi", Required: true},
	}}
	binding.Workflow.Stages["prepare-context"] = before
	profile.Workflows["scan"] = binding
	if err := config.ValidateWorkflowGraph(binding.Workflow); err != nil {
		t.Fatal(err)
	}
	if err := config.ValidateAuditTaskProfile(profile); err != nil {
		t.Fatal(err)
	}
	for _, scanner := range []string{"sqlmap", "nuclei"} {
		t.Run(scanner, func(t *testing.T) {
			inputs := make(map[string]artifacts.ReadResult)
			for name, file := range map[string]string{"openapi": "openapi.json", "settings": scanner + "-settings.json"} {
				data, err := os.ReadFile("../../configs/scan/examples/audit-openapi-scan/" + file)
				if err != nil {
					t.Fatal(err)
				}
				revision := "exact-" + name
				inputs[name] = artifacts.ReadResult{
					Ref:     contracts.ArtifactRef{Namespace: "project", Name: name, Revision: &revision},
					Payload: artifacts.Payload{MediaType: auditdomain.JSONMediaType, Data: data},
				}
			}
			inventory, err := buildInventory(profile, DraftSelection{}, inputs, nil)
			if err != nil || len(inventory.Tasks) != 1 || inventory.Tasks[0].Document.Scan.Scanner != scanner {
				t.Fatalf("builder consulted executor instead of settings: %v", err)
			}
			wantMismatch := scanner != scan.AuditScan.Scanner
			for step, err := range map[string]error{
				"before dispatch": validateInventoryTaskExecution(profile, inventory),
				"input preview":   ValidateInputPreview(profile, Scope{}, inputs, nil),
			} {
				if (err != nil) != wantMismatch || wantMismatch && !errors.Is(err, ErrInvalid) {
					t.Errorf("%s compatibility: %v, mismatch expected: %t", step, err, wantMismatch)
				}
			}
		})
	}
}
