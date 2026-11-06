//go:build e2e

package e2e

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"maps"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"go.yaml.in/yaml/v4"
)

func TestAuditHypothesisContinuationAcrossProductionProcesses(t *testing.T) {
	for _, initial := range []bool{false, true} {
		t.Run(fmt.Sprint("initial=", initial), func(t *testing.T) {
			exerciseAuditHypothesisContinuation(t, initial, false)
		})
	}
}

// The production browser resumes the same retained scenario through session/CSRF APIs.
func TestAuditResearchUIAcrossProductionProcesses(t *testing.T) {
	exerciseAuditHypothesisContinuation(t, false, true)
}

func exerciseAuditHypothesisContinuation(t *testing.T, initial, browser bool) {
	t.Helper()
	stages := []domainGatewayStage{}
	if !initial {
		stages = append(stages, hypothesisFindingSeedStage(), generationCheckStage("refuted"))
	}
	stages = append(stages, generationProposalStage(!initial, false, !initial), generationCheckStage("inconclusive"), generationProposalStage(true, true, !initial))
	h := startFindingsHarnessForBrowser(t, stages, browser, func(root string) {
		installGenerationAuditProgram(t, root, initial)
		installHypothesisFindingSeed(t, root)
	})
	project := createProjectResource(t, h.client, h.baseURL)
	source := uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "fixture", "source", "application/zip", auditProgramZip(t, map[string][]byte{"app.py": []byte(findingsSource)}))
	report := uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "fixture", "report", "text/plain", []byte(findingsEvidence))
	inputs := map[string]artifactRef{"source": source, "report": report}
	if !initial {
		seedBody := mustBasicJSON(t, map[string]any{"workflow": "hypothesis-finding-seed@1", "artifacts": map[string]any{}})
		seedRun := postProjectRun(t, h.client, h.baseURL, project.ProjectID, "seed-hypothesis-findings", seedBody, false)
		waitForFindingsRun(t, h, seedRun, "succeeded")
		members := []contracts.MaterialDescriptor{}
		for i := range 17 {
			payload := []byte(fmt.Sprint("Architecture boundary ", i))
			ref := uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "architecture", fmt.Sprintf("report-%02d", i), "text/plain", payload)
			members = append(members, contracts.MaterialDescriptor{ScopeKind: "project", ScopeID: project.ProjectID, Ref: contracts.ArtifactRef{Namespace: ref.Namespace, Name: ref.Name, Revision: ref.Revision}, Digest: contentdigest.Bytes(payload), MediaType: "text/plain", SizeBytes: int64(len(payload))})
		}
		raw := mustBasicJSON(t, contracts.MaterialSet{Schema: contracts.MaterialSetSchema, Materials: members})
		inputs["materials"] = uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "selections", "architecture", "application/json", raw)
	}
	draft, _ := basicAuditRequest(t, h, http.MethodPost, "/v1/projects/"+project.ProjectID+"/audits", map[string]any{"profile": map[string]string{"name": "generation-process", "version": "1"}, "inputs": inputs}, "generation-draft", 0, http.StatusCreated)
	id := draft["auditId"].(string)
	blocked := h.gateway.blockNextRequest()
	basicAuditRequest(t, h, http.MethodPost, "/v1/audits/"+id+"/start", nil, "generation-start", uint64(draft["revision"].(float64)), http.StatusOK)
	select {
	case <-blocked:
	case <-h.ctx.Done():
		t.Fatal("initial ordinary work did not reach Runtime")
	}
	current := basicAuditGet(t, h, id)
	basicAuditRequest(t, h, http.MethodPost, "/v1/audits/"+id+"/pause", nil, "generation-pause", uint64(current["revision"].(float64)), http.StatusOK)
	h.gateway.releaseBlockedRequest()
	count := 1
	if initial {
		count = 2
	}
	pausedTasks := waitBasicTask(t, h, id, count, map[bool]string{true: "ready", false: "settled"}[initial])
	current = basicAuditGet(t, h, id)
	if current["state"] != "paused" || current["submittedRunCount"].(float64) != 1 {
		t.Fatalf("pause dispatched continuation: state=%v submitted=%v", current["state"], current["submittedRunCount"])
	}
	h.restartServer()
	current = basicAuditGet(t, h, id)
	if browser {
		runResearchBrowser(t, h, project.ProjectID, id)
	} else {
		basicAuditRequest(t, h, http.MethodPost, "/v1/audits/"+id+"/resume", nil, "generation-resume", uint64(current["revision"].(float64)), http.StatusOK)
	}
	total := 4
	if initial {
		total = 3
	}
	tasks := waitBasicTask(t, h, id, total, "settled")
	for basicAuditGet(t, h, id)["state"] != "completed" {
		basicPoll(t, h)
	}
	if h.gateway.CompletedStages() != total+map[bool]int{true: 0, false: 1}[initial] || len(h.gateway.Failures()) > 0 {
		t.Fatalf("duplicate or failed generation: completed=%d failures=%v", h.gateway.CompletedStages(), h.gateway.Failures())
	}
	if tasks[0].Document.TaskID != pausedTasks[0].Document.TaskID {
		t.Fatal("restart changed accepted work identity")
	}
	checkIndex := 2
	if initial {
		checkIndex = 1
	}
	check := tasks[checkIndex]
	if check.Document.Hypothesis == nil || check.Document.Hypothesis.Revision != 1 {
		t.Fatal("generated check has no exact system hypothesis pin")
	}
	result := basicTaskArtifact(t, h, id, check, "result")
	if !strings.Contains(string(result), `"verdict":"inconclusive"`) || !strings.Contains(string(result), check.Document.Hypothesis.ID) {
		t.Fatal("check output lost inconclusive result or exact assertion")
	}
	hypotheses, _ := basicAuditRequest(t, h, http.MethodGet, "/v1/audits/"+id+"/hypotheses?limit=64", nil, "", 0, http.StatusOK)
	expected := 3
	if initial {
		expected = 1
	}
	if len(hypotheses["items"].([]any)) != expected {
		t.Fatal("proposal replay duplicated assertions")
	}
	if !initial {
		generated := 0
		for _, raw := range hypotheses["items"].([]any) {
			document := raw.(map[string]any)["document"].(map[string]any)
			if document["origin"].(map[string]any)["kind"] != "run" {
				continue
			}
			generated++
			sources := document["sources"].([]any)
			if len(sources) != 1 && len(sources) != 3 {
				t.Fatal("missing exact grounds", document)
			}
			for _, source := range sources {
				m := source.(map[string]any)
				if m["scope_kind"] != "audit" || m["scope_id"] != id {
					t.Fatal("grounds were not retained by Audit", m)
				}
			}
		}
		if generated != 2 {
			t.Fatal("wrong generated hypothesis count", generated)
		}
		findings, _ := basicAuditRequest(t, h, http.MethodGet, "/v1/projects/"+project.ProjectID+"/findings?limit=100", nil, "", 0, http.StatusOK)
		if len(findings["items"].([]any)) != 1 {
			t.Fatal("hypothesis checks created/resolved problems implicitly", findings)
		}
	}
}

func generationCheckStage(verdict string) domainGatewayStage {
	stage := basicConclusionStage(false)
	stage.steps = []domainGatewayStep{toolGatewayStep("read_audit_task", fixedArguments(map[string]any{})), toolGatewayStep("submit_check_result", fixedArguments(map[string]any{"verdict": verdict, "rationale": "The assertion applies only to the selected actor and exact fixture", "tested_conditions": []string{"selected fixture and actor"}, "limitations": []string{"Another actor remains untested"}, "evidence": []string{"evidence-000", "evidence-001"}})), finalGatewayStep("Scoped check result retained", nil)}
	return stage
}

func generationProposalStage(prior, empty, exploratory bool) domainGatewayStage {
	contextSlot, resultSlot, resultName := "analysis", "proposals", "proposals"
	if exploratory {
		contextSlot, resultSlot, resultName = "research", "draft_report", "drafts"
	}
	stage := domainGatewayStage{name: fmt.Sprintf("generation/empty=%t", empty), tools: []string{"read_artifact", "read_text_artifact", "list_project_findings", "read_project_finding", "select_project_findings", "read_finding_selection", "submit_hypothesis_proposals"}}
	stage.steps = append(stage.steps, toolGatewayStep("read_artifact", stageRefArguments("context", nil)))
	stage.steps = append(stage.steps, toolGatewayStep("read_artifact", stageRefArguments(contextSlot, nil)))
	if prior {
		stage.steps = append(stage.steps, toolGatewayStep("read_artifact", func(request map[string]any) (map[string]any, error) {
			reads := findingsToolResponses(request, "read_artifact")
			if len(reads) != 2 {
				return nil, fmt.Errorf("missing proposal context")
			}
			raw, err := base64.StdEncoding.DecodeString(fmt.Sprint(reads[1]["dataBase64"]))
			if err != nil {
				return nil, err
			}
			var selected auditcontract.ProposalContext
			if json.Unmarshal(raw, &selected) != nil || len(selected.Observations) != 1 {
				return nil, fmt.Errorf("missing exact observation")
			}
			ref := selected.Materials[selected.Observations[0].Key].Ref
			return map[string]any{"namespace": ref.Namespace, "name": ref.Name, "revision": *ref.Revision}, nil
		}))
	}
	if prior && !empty {
		stage.steps = append(stage.steps, toolGatewayStep("read_artifact", func(request map[string]any) (map[string]any, error) {
			reads := findingsToolResponses(request, "read_artifact")
			if len(reads) != 3 {
				return nil, fmt.Errorf("missing exact initial reads")
			}
			raw, err := base64.StdEncoding.DecodeString(fmt.Sprint(reads[1]["dataBase64"]))
			if err != nil {
				return nil, err
			}
			var selected auditcontract.ProposalContext
			if json.Unmarshal(raw, &selected) != nil {
				return nil, fmt.Errorf("invalid proposal context")
			}
			for _, key := range slices.Sorted(maps.Keys(selected.Materials)) {
				m := selected.Materials[key]
				if m.Ref.Namespace == "material-inputs" && m.MediaType == "text/plain" {
					return map[string]any{"namespace": m.Ref.Namespace, "name": m.Ref.Name, "revision": *m.Ref.Revision}, nil
				}
			}
			return nil, fmt.Errorf("missing native architecture input")
		}))
		stage.steps = append(stage.steps,
			toolGatewayStep("list_project_findings", fixedArguments(map[string]any{"limit": 100})),
			toolGatewayStep("read_project_finding", func(request map[string]any) (map[string]any, error) {
				pages := findingsToolResponses(request, "list_project_findings")
				if len(pages) != 1 || len(pages[0]["items"].([]any)) != 1 {
					return nil, fmt.Errorf("missing live Project finding")
				}
				item := pages[0]["items"].([]any)[0].(map[string]any)
				return map[string]any{"finding_id": item["findingId"], "revision": item["revision"]}, nil
			}),
			toolGatewayStep("select_project_findings", func(request map[string]any) (map[string]any, error) {
				item := findingsToolResponses(request, "list_project_findings")[0]["items"].([]any)[0].(map[string]any)
				return map[string]any{"selection_key": "selected-grounds", "findings": []any{map[string]any{"findingId": item["findingId"], "revision": item["revision"]}}}, nil
			}),
			toolGatewayStep("read_finding_selection", func(request map[string]any) (map[string]any, error) {
				selected := findingsToolResponses(request, "select_project_findings")
				if len(selected) != 1 {
					return nil, fmt.Errorf("missing exact Finding selection")
				}
				return map[string]any{"collection_ref": selected[0]["artifact"].(map[string]any)["ref"]}, nil
			}),
		)
	}
	submit := func(request map[string]any) (map[string]any, error) {
		reads := findingsToolResponses(request, "read_artifact")
		if len(reads) < 2 {
			return nil, fmt.Errorf("missing exact generation context")
		}
		data, err := base64.StdEncoding.DecodeString(fmt.Sprint(reads[1]["dataBase64"]))
		if err != nil {
			return nil, err
		}
		assignment, err := base64.StdEncoding.DecodeString(fmt.Sprint(reads[0]["dataBase64"]))
		if err != nil || !strings.Contains(string(assignment), `"objective"`) {
			return nil, fmt.Errorf("generator task objective missing")
		}
		var selected auditcontract.ProposalContext
		if json.Unmarshal(data, &selected) != nil || selected.Schema != auditcontract.ProposalContextSchema || selected.Program.Workflows["check"].Ref != "basic-check@1" {
			return nil, fmt.Errorf("wrong pinned program")
		}
		if prior {
			if len(reads) != map[bool]int{true: 3, false: 4}[empty] || len(selected.Observations) != 1 {
				return nil, fmt.Errorf("missing selected ordinary result")
			}
			negative, err := base64.StdEncoding.DecodeString(fmt.Sprint(reads[2]["dataBase64"]))
			if err != nil || !strings.Contains(string(negative), `"verdict"`) {
				return nil, fmt.Errorf("prior result omitted")
			}
		}
		hypotheses, work := "{}", "[]"
		if !empty {
			hypotheses = `{"next":{"statement":"An alternate actor can access the selected record","scope":{"actor":"alternate reader"},"grounds":"The selected materials leave the alternate actor untested"}}`
			work = `[{"key":"check-next","workflow":"check","objective":"Check the alternate actor because it remains outside the tested conditions","hypothesis":"next"}]`
		}
		args := map[string]any{"hypotheses_json": hypotheses, "work_json": work}
		if prior && !empty {
			if len(selected.Materials) != 19 || selected.Observations[0].Hypothesis == nil {
				return nil, fmt.Errorf("missing architecture set or exact checked hypothesis: materials=%d", len(selected.Materials))
			}
			selection := findingsToolResponses(request, "select_project_findings")[0]["artifact"].(map[string]any)["ref"]
			raw, _ := json.Marshal(map[string]any{"finding": selection})
			args["selected_materials_json"] = string(raw)
			var content map[string]any
			_ = json.Unmarshal([]byte(hypotheses), &content)
			sources := []string{"finding", selected.Observations[0].Key}
			architecture, err := base64.StdEncoding.DecodeString(fmt.Sprint(reads[3]["dataBase64"]))
			if err != nil || !strings.HasPrefix(string(architecture), "Architecture boundary ") {
				return nil, fmt.Errorf("architecture payload was not consumed")
			}
			for _, handle := range slices.Sorted(maps.Keys(selected.Materials)) {
				m := selected.Materials[handle]
				if m.Ref.Namespace == "material-inputs" && m.MediaType == "text/plain" {
					sources = append(sources, handle)
					break
				}
			}
			content["next"].(map[string]any)["sources"] = sources
			content["later"] = map[string]any{"statement": "A second boundary can expose another actor's record", "scope": map[string]string{"actor": "second reader"}, "grounds": "The exact Finding snapshot motivates later investigation", "sources": []string{"finding"}}
			raw, _ = json.Marshal(content)
			args["hypotheses_json"] = string(raw)
		}
		return args, nil
	}
	// Replay the exact local publication to exercise acknowledgement recovery.
	stage.steps = append(stage.steps, toolGatewayStep("submit_hypothesis_proposals", submit), toolGatewayStep("submit_hypothesis_proposals", submit), finalGatewayStep("Proposal decision retained", map[string]domainArtifactBinding{resultSlot: {namespace: "generator", name: resultName}}))
	return stage
}

func installGenerationAuditProgram(t *testing.T, root string, initial bool) {
	t.Helper()
	installBasicAuditProgram(t, root)
	var profile struct {
		Spec auditcontract.Program `yaml:"spec"`
	}
	if err := yaml.Unmarshal([]byte(basicProcessProgram), &profile); err != nil {
		t.Fatal(err)
	}
	p := profile.Spec
	delete(p.Workflows, "copy")
	p.Inputs["materials"] = auditcontract.ProgramInput{Required: false, MediaTypes: []string{"application/json"}}

	check := p.Workflows["check"]
	check.Inputs["material"] = auditcontract.InputMapping{Source: "audit-input", Name: "report"}
	p.Workflows["check"] = check
	inputs := map[string]auditcontract.InputMapping{"context": {Source: "task-context"}, "analysis": {Source: "prior-results"}, "results": {Source: "task-results"}, "materials": {Source: "audit-input", Name: "materials"}, "material": {Source: "audit-input", Name: "report"}}
	p.Workflows["generate"] = auditcontract.ProgramWorkflow{Ref: "audit-hypothesis-generation@1", Inputs: inputs, Parameters: map[string]string{}, Outputs: map[string]string{"proposals": "proposals"}}
	p.Continuation = &auditcontract.Continuation{Workflow: "generate"}
	p.InitialWork = []auditcontract.InitialWork{{Key: "initial-generation", Workflow: "generate", Objective: "Propose a scoped check from selected materials"}}
	if !initial {
		p.Hypotheses = map[string]auditcontract.HypothesisSeed{"initial": {Statement: "The selected actor can access another record", Scope: map[string]string{"actor": "selected reader"}, Grounds: "Explicit initial suspicion"}}
		p.InitialWork = []auditcontract.InitialWork{{Key: "first-check", Workflow: "check", Objective: "Check the first actor", Hypothesis: "initial"}}
	}
	p.Limits.MaxTasks = 8
	p.Limits.MaxTasksPerIteration = 8
	p.Limits.MaxSubmittedRuns = 12
	p.Limits.MaxHypotheses = 8
	p.Limits.MaxHypothesisRevisions = 8
	p.Limits.MaxHypothesisBytes = 1 << 20
	p.Limits.MaxEvidenceBytes = 8 << 20
	if !initial {
		// A second ordinary Workflow uses different identity, input and output
		// names. The protocol resolves roles exclusively from its pinned graph.
		raw, err := os.ReadFile(filepath.Join(root, "workflows/audit_hypothesis_generation.yaml"))
		if err != nil {
			t.Fatal(err)
		}
		var wf map[string]any
		if err = json.Unmarshal(raw, &wf); err != nil {
			t.Fatal(err)
		}
		wf["metadata"].(map[string]any)["name"] = "architecture-exploration"
		spec := wf["spec"].(map[string]any)
		inputs := spec["inputs"].(map[string]any)
		inputs["research"] = inputs["analysis"]
		delete(inputs, "analysis")
		outputs := spec["outputs"].(map[string]any)
		outputs["notes"] = outputs["proposals"]
		delete(outputs, "proposals")
		stage := spec["stages"].(map[string]any)["generate"].(map[string]any)
		context := stage["context"].(map[string]any)["artifacts"].(map[string]any)
		context["research"] = context["analysis"]
		delete(context, "analysis")
		context["research"].(map[string]any)["name"] = "research"
		result := stage["result"].(map[string]any)["artifacts"].(map[string]any)
		result["draft_report"] = result["proposals"]
		delete(result, "proposals")
		result["draft_report"].(map[string]any)["from"].(map[string]any)["name"] = "drafts"
		stage["workflowOutputs"] = map[string]string{"notes": "draft_report"}
		if err = os.WriteFile(filepath.Join(root, "workflows/architecture_exploration.yaml"), mustBasicJSON(t, wf), 0600); err != nil {
			t.Fatal(err)
		}
		binding := p.Workflows["generate"]
		delete(p.Workflows, "generate")
		binding.Ref = "architecture-exploration@1"
		binding.Inputs["research"] = binding.Inputs["analysis"]
		delete(binding.Inputs, "analysis")
		binding.Outputs["proposals"] = "notes"
		p.Workflows["explore"] = binding
		p.Continuation.Workflow = "explore"
	}
	value := map[string]any{"apiVersion": "contractor/v1alpha1", "kind": "AuditProfile", "metadata": map[string]string{"name": "generation-process", "version": "1"}, "spec": p}
	if err := os.WriteFile(filepath.Join(root, "audit-profiles/basic_process.yaml"), mustBasicJSON(t, value), 0600); err != nil {
		t.Fatal(err)
	}
}

func hypothesisFindingSeedStage() domainGatewayStage {
	stage := domainGatewayStage{name: "hypothesis-project-finding", tools: withMemoryTools([]string{"finding", "read_artifact", "write_text_artifact"})}
	stage.steps = []domainGatewayStep{
		toolGatewayStep("write_text_artifact", fixedArguments(map[string]any{"name": "proof", "text": "A pre-existing exact Project problem observation", "media_type": "text/plain"})),
		toolGatewayStep("finding", func(request map[string]any) (map[string]any, error) {
			ref, found := lastExactArtifact(request, "findings-review", "proof")
			if !found {
				return nil, fmt.Errorf("missing Project proof")
			}
			return map[string]any{"title": "Observed boundary problem", "description": "An evidenced observed problem under the selected conditions", "evidence_refs": []any{ref}, "publish_problem": true}, nil
		}),
		toolGatewayStep("write_text_artifact", fixedArguments(map[string]any{"name": "report", "text": "Project problem retained", "media_type": "text/markdown"})),
		finalGatewayStep("Project problem saved", map[string]domainArtifactBinding{"report": {namespace: "findings-review", name: "report"}}),
	}
	return stage
}

func installHypothesisFindingSeed(t *testing.T, root string) {
	t.Helper()
	source, err := os.ReadFile(filepath.Join(root, "workflows/project_findings_review.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	workflow := strings.Replace(string(source), "name: project-findings-review", "name: hypothesis-finding-seed", 1)
	workflow = strings.Replace(workflow, "template: project_findings_analyst@1", "template: hypothesis_finding_seed@1", 1)
	if err = os.WriteFile(filepath.Join(root, "workflows/hypothesis_finding_seed.yaml"), []byte(workflow), 0600); err != nil {
		t.Fatal(err)
	}
	source, err = os.ReadFile(filepath.Join(root, "agent-templates/project_findings_analyst.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	template := strings.Replace(string(source), "name: project_findings_analyst", "name: hypothesis_finding_seed", 1)
	template = strings.Replace(template, "tools: [list_project_findings, read_project_finding, select_project_findings, read_finding_selection]", "tools: [finding]", 1)
	if err = os.WriteFile(filepath.Join(root, "agent-templates/hypothesis_finding_seed.yaml"), []byte(template), 0600); err != nil {
		t.Fatal(err)
	}
}
