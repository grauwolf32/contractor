//go:build e2e

package e2e

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"go.yaml.in/yaml/v4"
)

func TestAuditHypothesisContinuationAcrossProductionProcesses(t *testing.T) {
	for _, initial := range []bool{false, true} {
		t.Run(fmt.Sprint("initial=", initial), func(t *testing.T) {
			stages := []domainGatewayStage{}
			if !initial {
				stages = append(stages, generationCheckStage("refuted"))
			}
			stages = append(stages, generationProposalStage(!initial, false), generationCheckStage("inconclusive"), generationProposalStage(true, true))
			h := startConfiguredFindingsHarness(t, stages, func(root string) { installGenerationAuditProgram(t, root, initial) })
			project := createProjectResource(t, h.client, h.baseURL)
			source := uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "fixture", "source", "application/zip", auditProgramZip(t, map[string][]byte{"app.py": []byte(findingsSource)}))
			report := uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "fixture", "report", "text/plain", []byte(findingsEvidence))
			draft, _ := basicAuditRequest(t, h, http.MethodPost, "/v1/projects/"+project.ProjectID+"/audits", map[string]any{"profile": map[string]string{"name": "generation-process", "version": "1"}, "inputs": map[string]artifactRef{"source": source, "report": report}}, "generation-draft", 0, http.StatusCreated)
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
			basicAuditRequest(t, h, http.MethodPost, "/v1/audits/"+id+"/resume", nil, "generation-resume", uint64(current["revision"].(float64)), http.StatusOK)
			total := 4
			if initial {
				total = 3
			}
			tasks := waitBasicTask(t, h, id, total, "settled")
			for basicAuditGet(t, h, id)["state"] != "completed" {
				basicPoll(t, h)
			}
			if h.gateway.CompletedStages() != total || len(h.gateway.Failures()) > 0 {
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
			expected := 2
			if initial {
				expected = 1
			}
			if len(hypotheses["items"].([]any)) != expected {
				t.Fatal("proposal replay duplicated assertions")
			}
		})
	}
}

func generationCheckStage(verdict string) domainGatewayStage {
	stage := basicConclusionStage(false)
	stage.steps = []domainGatewayStep{toolGatewayStep("read_audit_task", fixedArguments(map[string]any{})), toolGatewayStep("submit_check_result", fixedArguments(map[string]any{"verdict": verdict, "rationale": "The assertion applies only to the selected actor and exact fixture", "tested_conditions": []string{"selected fixture and actor"}, "limitations": []string{"Another actor remains untested"}, "evidence": []string{"evidence-000", "evidence-001"}})), finalGatewayStep("Scoped check result retained", nil)}
	return stage
}

func generationProposalStage(prior, empty bool) domainGatewayStage {
	stage := domainGatewayStage{name: fmt.Sprintf("generation/empty=%t", empty), tools: []string{"read_artifact", "read_text_artifact", "read_finding_selection", "submit_hypothesis_proposals"}}
	stage.steps = append(stage.steps, toolGatewayStep("read_artifact", stageRefArguments("context", nil)))
	stage.steps = append(stage.steps, toolGatewayStep("read_artifact", stageRefArguments("generation_context", nil)))
	if prior {
		stage.steps = append(stage.steps, toolGatewayStep("read_artifact", stageRefArguments("source_01", nil)))
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
			if len(reads) != 3 || len(selected.Observations) != 1 {
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
		return map[string]any{"hypotheses_json": hypotheses, "work_json": work}, nil
	}
	// Replay the exact local publication to exercise acknowledgement recovery.
	stage.steps = append(stage.steps, toolGatewayStep("submit_hypothesis_proposals", submit), toolGatewayStep("submit_hypothesis_proposals", submit), finalGatewayStep("Proposal decision retained", map[string]domainArtifactBinding{"proposals": {namespace: "generator", name: "proposals"}}))
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
	check := p.Workflows["check"]
	check.Inputs["material"] = auditcontract.InputMapping{Source: "audit-input", Name: "report"}
	p.Workflows["check"] = check
	inputs := map[string]auditcontract.InputMapping{"context": {Source: "task-context"}, "generation_context": {Source: "prior-results"}, "material": {Source: "audit-input", Name: "report"}}
	for i := 1; i <= 16; i++ {
		inputs[fmt.Sprintf("source_%02d", i)] = auditcontract.InputMapping{Source: "task-input"}
	}
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
	p.Limits.MaxHypotheses = 4
	p.Limits.MaxHypothesisRevisions = 4
	p.Limits.MaxHypothesisBytes = 1 << 20
	p.Limits.MaxEvidenceBytes = 8 << 20
	value := map[string]any{"apiVersion": "contractor/v1alpha1", "kind": "AuditProfile", "metadata": map[string]string{"name": "generation-process", "version": "1"}, "spec": p}
	if err := os.WriteFile(filepath.Join(root, "audit-profiles/basic_process.yaml"), mustBasicJSON(t, value), 0600); err != nil {
		t.Fatal(err)
	}
}
