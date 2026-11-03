// Run from the repository root. This validates design fixtures, not live data.
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/engagementcontract"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

func run() error {
	root := "docs/design/ui/experiments/new-check-opendesign-2026-11-02"
	data, err := os.ReadFile(filepath.Join(root, "scope-fixtures.json"))
	if err != nil {
		return err
	}
	var fixtures map[string]json.RawMessage
	if err := strictjson.Decode(data, &fixtures); err != nil {
		return err
	}
	document := func(name string) (json.RawMessage, error) {
		var value struct {
			Artifact  contracts.ArtifactRef `json:"artifact"`
			MediaType string                `json:"mediaType"`
			Document  json.RawMessage       `json:"document"`
		}
		if err := strictjson.Decode(fixtures[name], &value); err != nil {
			return nil, err
		}
		return value.Document, nil
	}
	revision := "scope-r1"
	options := auditdomain.InventoryOptions{Round: 1, WorkflowRole: "check", SourceInputName: "source", SourceRef: contracts.ArtifactRef{Namespace: "inputs", Name: "scope-example", Revision: &revision}, ApprovalRequirement: auditdomain.ApprovalNone}
	api, err := document("openapi")
	if err != nil {
		return err
	}
	inventory, err := auditdomain.BuildOpenAPIInventory(api, "application/yaml", options)
	if err != nil {
		return fmt.Errorf("OpenAPI: %w", err)
	}
	if len(inventory.Worklist.Items) != 6 {
		return fmt.Errorf("OpenAPI count = %d", len(inventory.Worklist.Items))
	}
	checklist, err := document("checklist")
	if err != nil {
		return err
	}
	check, err := auditdomain.BuildChecklistInventory(checklist, "application/json", options)
	if err != nil {
		return fmt.Errorf("checklist: %w", err)
	}
	if len(check.Worklist.Items) != 3 {
		return fmt.Errorf("checklist count = %d", len(check.Worklist.Items))
	}
	engagement, err := document("engagement")
	if err != nil {
		return err
	}
	if _, err := engagementcontract.Decode(engagement); err != nil {
		return fmt.Errorf("engagement: %w", err)
	}
	generatedPath := "ui/test-results/new-check-design/scope-generated.json"
	generated, err := os.ReadFile(generatedPath)
	if err != nil {
		return err
	}
	var documents []json.RawMessage
	if err := strictjson.Decode(generated, &documents); err != nil {
		return err
	}
	if len(documents) < 3 {
		return fmt.Errorf("missing browser-generated checklist documents")
	}
	for index, payload := range documents {
		if _, err := auditdomain.BuildChecklistInventory(payload, "application/json", options); err != nil {
			return fmt.Errorf("browser document %d: %w", index, err)
		}
	}
	fmt.Printf("Validated 6 OpenAPI operations, checklist fixture, engagement and %d browser-authored checklist documents with existing domain owners.\n", len(documents))
	return nil
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
