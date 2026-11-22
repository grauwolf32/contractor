//go:build e2e

package e2e

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"go.yaml.in/yaml/v4"
)

func TestWorkflowMaterialSetsAcrossProductionProcesses(t *testing.T) {
	ordinary := domainGatewayStage{name: "ordinary-material-set", tools: []string{"read_artifact", "write_text_artifact"}}
	ordinary.steps = materialSetReadSteps()
	ordinary.steps = append(ordinary.steps, toolGatewayStep("write_text_artifact", fixedArguments(map[string]any{"name": "report", "media_type": "text/plain", "text": "Read exact members from one ordinary set"})), finalGatewayStep("Materials read", map[string]domainArtifactBinding{"report": {namespace: "reader", name: "report"}}))
	generator := domainGatewayStage{name: "initial-material-set", tools: []string{"read_artifact", "list_project_findings", "read_project_finding", "select_project_findings", "read_finding_selection", "submit_hypothesis_proposals"}}
	generator.steps = materialSetReadSteps()
	generator.steps = append(generator.steps, toolGatewayStep("submit_hypothesis_proposals", fixedArguments(map[string]any{"hypotheses_json": `{"next":{"statement":"An alternate actor can access the selected record","scope":{"actor":"alternate reader"},"grounds":"The selected architecture materials leave this actor untested"}}`, "work_json": "[]"})), finalGatewayStep("Hypothesis proposal retained", map[string]domainArtifactBinding{"proposals": {namespace: "generator", name: "proposals"}}))
	h := startConfiguredFindingsHarness(t, []domainGatewayStage{ordinary, generator}, func(root string) { installMaterialSetProgram(t, root) })
	project := createProjectResource(t, h.client, h.baseURL)
	members := make([]contracts.MaterialDescriptor, 17)
	for i := range members {
		payload := []byte(fmt.Sprint("Selected architecture material ", i))
		ref := uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "architecture", fmt.Sprintf("report-%02d", i), "text/plain", payload)
		members[i] = contracts.MaterialDescriptor{ScopeKind: "project", ScopeID: project.ProjectID, Ref: contracts.ArtifactRef{Namespace: ref.Namespace, Name: ref.Name, Revision: ref.Revision}, Digest: contentdigest.Bytes(payload), MediaType: "text/plain", SizeBytes: int64(len(payload))}
	}
	raw, _ := json.Marshal(contracts.MaterialSet{Schema: contracts.MaterialSetSchema, Materials: members})
	selection := uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "selections", "materials", "application/json", raw)
	body := mustBasicJSON(t, map[string]any{"workflow": "material-set-reader@1", "artifacts": map[string]artifactRef{"materials": selection}})
	ordinaryRun := postProjectRun(t, h.client, h.baseURL, project.ProjectID, "ordinary-material-set", body, false)
	waitForFindingsRun(t, h, ordinaryRun, "succeeded")
	draft, _ := basicAuditRequest(t, h, http.MethodPost, "/v1/projects/"+project.ProjectID+"/audits", map[string]any{"profile": map[string]string{"name": "material-set-process", "version": "1"}, "inputs": map[string]artifactRef{"materials": selection}}, "material-set-draft", 0, http.StatusCreated)
	id := draft["auditId"].(string)
	basicAuditRequest(t, h, http.MethodPost, "/v1/audits/"+id+"/start", nil, "material-set-start", uint64(draft["revision"].(float64)), http.StatusOK)
	waitBasicTask(t, h, id, 1, "settled")
	for basicAuditGet(t, h, id)["state"] != "completed" {
		basicPoll(t, h)
	}
	hypotheses, _ := basicAuditRequest(t, h, http.MethodGet, "/v1/audits/"+id+"/hypotheses?limit=64", nil, "", 0, http.StatusOK)
	if len(hypotheses["items"].([]any)) != 1 || h.gateway.CompletedStages() != 2 || len(h.gateway.Failures()) > 0 {
		t.Fatal("material-set consumer/proposal failed", hypotheses, h.gateway.Failures())
	}
}

func materialSetReadSteps() []domainGatewayStep {
	steps := []domainGatewayStep{toolGatewayStep("read_artifact", stageReadArguments("materials", nil))}
	for _, index := range []int{0, 16} {
		steps = append(steps, toolGatewayStep("read_artifact", func(request map[string]any) (map[string]any, error) {
			responses := findingsToolResponses(request, "read_artifact")
			if len(responses) == 0 {
				return nil, fmt.Errorf("missing set descriptor")
			}
			raw, e := artifactTextBytes(responses[0]["text"])
			if e != nil {
				return nil, e
			}
			set, e := contracts.DecodeMaterialSet(raw)
			if e != nil || len(set.Materials) != 17 {
				return nil, fmt.Errorf("wrong material set: %v", e)
			}
			selected := set.Materials[index]
			if selected.ScopeKind != "run" || selected.Ref.Namespace != "material-inputs" {
				return nil, fmt.Errorf("member lacks ordinary Run binding")
			}
			return map[string]any{"namespace": selected.Ref.Namespace, "name": selected.Ref.Name, "revision": *selected.Ref.Revision}, nil
		}))
	}
	return steps
}

func installMaterialSetProgram(t *testing.T, root string) {
	t.Helper()
	installGenerationAuditProgram(t, root, true)
	var profile struct {
		Spec auditcontract.Program `yaml:"spec"`
	}
	raw, e := os.ReadFile(filepath.Join(root, "audit-profiles/basic_process.yaml"))
	if e != nil {
		t.Fatal(e)
	}
	if e = yaml.Unmarshal(raw, &profile); e != nil {
		t.Fatal(e)
	}
	p := profile.Spec
	p.Inputs = map[string]auditcontract.ProgramInput{"materials": {Required: true, MediaTypes: []string{"application/json"}}}
	p.Workflows = map[string]auditcontract.ProgramWorkflow{"generate": {Ref: "audit-hypothesis-generation@4", Inputs: map[string]auditcontract.InputMapping{"materials": {Source: "audit-input", Name: "materials"}, "context": {Source: "task-context"}, "analysis": {Source: "prior-results"}, "results": {Source: "task-results"}}, Parameters: map[string]string{}, Outputs: map[string]string{"proposals": "proposals"}}}
	p.Continuation = nil
	value := map[string]any{"apiVersion": "contractor/v1alpha1", "kind": "AuditProfile", "metadata": map[string]string{"name": "material-set-process", "version": "1"}, "spec": p}
	if e = os.WriteFile(filepath.Join(root, "audit-profiles/basic_process.yaml"), mustBasicJSON(t, value), 0600); e != nil {
		t.Fatal(e)
	}
	files := map[string]string{"instructions/material-set-reader.md": "Read the configured exact material set with ordinary Artifact tools and retain a concise report.", "agent-templates/material_set_reader.yaml": `apiVersion: contractor/v1alpha1
kind: AgentTemplate
metadata: {name: material_set_reader, version: "1"}
spec:
  description: Read explicit exact Workflow materials
  runtime: adk@1
  instructions: {ref: instructions/material-set-reader.md}
  toolsets:
    - {ref: run-artifacts@1, tools: [read_artifact]}
    - {ref: text-artifacts@1, tools: [write_text_artifact]}
  sandboxProfile: local-workdir@1
`, "workflows/material_set_reader.yaml": `apiVersion: contractor/v1alpha1
kind: Workflow
metadata: {name: material-set-reader, version: "1"}
spec:
  parameters: {}
  inputs:
    materials: {required: true, materialSet: true, mediaTypes: [text/plain]}
  outputs:
    report: {required: true, primary: true, mediaTypes: [text/plain]}
  executionConfig:
    workers: {llmGateway: local-litellm@1, credential: development-worker, modelPolicy: worker@2}
  entryStage: read
  stages:
    read:
      objective: Read two exact selected materials
      instructions: {ref: instructions/material-set-reader.md}
      planner: passthrough@1
      agents: {reader: {template: material_set_reader@1, namespace: reader}}
      context: {artifacts: {materials: {namespace: inputs, name: materials, required: true}}}
      result: {artifacts: {report: {required: true, mediaTypes: [text/plain], from: {namespace: reader, name: report}}}}
      workflowOutputs: {report: report}
      on:
        succeeded: {succeed: {}}
        failed: {fail: {}}
        interrupted: {fail: {}}
`}
	for path, text := range files {
		if e = os.WriteFile(filepath.Join(root, path), []byte(text), 0600); e != nil {
			t.Fatal(e)
		}
	}
}
