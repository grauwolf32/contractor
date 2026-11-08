//go:build e2e

package e2e

import (
	"encoding/json"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestTraceFindingCoordinatesAcrossProcesses(t *testing.T) {
	stage := ordinaryFindingsProducerStage("trace-workspace")
	stage.tools = append(stage.tools, "annotate_trace")
	stage.steps = []domainGatewayStep{
		toolGatewayStep("annotate_trace", fixedArguments(map[string]any{"path": "app.py", "symbol": "lookup", "target": "route"})),
		toolGatewayStep("finding", fixedArguments(map[string]any{
			"title": "Uncontrolled query", "description": "The inspected source passes input to the database.",
			"file": "app.py", "line": 7, "publish_problem": true,
		})),
		toolGatewayStep("write_text_artifact", fixedArguments(map[string]any{"name": "result", "text": "Trace finding recorded", "media_type": "text/plain"})),
		finalGatewayStep("Trace finding complete", map[string]domainArtifactBinding{"result": {namespace: "audit-check", name: "result"}}),
	}
	h := startConfiguredFindingsHarness(t, []domainGatewayStage{stage}, func(root string) {
		path := filepath.Join(root, "agent-templates", "fixture-ordinary-findings.yaml")
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		text := strings.ReplaceAll(string(raw), "security-findings-code@1", "security-findings-trace@1")
		text = strings.Replace(text, "  toolsets:\n", "  toolsets:\n    - ref: taint-annotations@1\n      tools: [annotate_trace]\n", 1)
		if err := os.WriteFile(path, []byte(text), 0600); err != nil {
			t.Fatal(err)
		}
	})
	project := createProjectResource(t, h.client, h.baseURL)
	payload := auditProgramZip(t, map[string][]byte{"app.py": []byte("#" + strings.Repeat("x", 128*1024) + "\n" + "def handle(value):\n    return lookup(value)\n\ndef lookup(value):\n    return db.execute(value)\n")})
	source := uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "sources", "trace-source", "application/zip", payload)
	run := runOrdinaryFindingProducer(t, h, project.ProjectID, source, payload, artifactRef{}, nil)
	receipts := findingsReceipts(t, h, run)
	if len(receipts) != 1 || receipts[0].FindingID == nil || len(receipts[0].ProjectEvidence) != 2 {
		t.Fatalf("trace finding publication = %+v", receipts)
	}
	receipt := receipts[0]
	if len(receipt.Document.Locations) != 1 || receipt.Document.Locations[0].File != "app.py" || receipt.Document.Locations[0].Line == nil || *receipt.Document.Locations[0].Line != 6 {
		t.Fatalf("original source coordinate = %+v", receipt.Document.Locations)
	}
	// Project-held evidence must survive ordinary Run deletion.
	deleteASVSBacktraceRun(t, h.ctx, h.client, h.baseURL, run)
	foundProof := false
	for _, evidence := range receipt.ProjectEvidence {
		ref := evidence.Ref
		raw, mime := download(t, h.client, h.baseURL+"/v1/projects/"+project.ProjectID+"/artifacts/"+ref.Namespace+"/"+ref.Name+"?revision="+url.QueryEscape(*ref.Revision))
		if mime != "application/json" {
			continue
		}
		var proof struct {
			Source struct {
				StartLine int `json:"startLine"`
			} `json:"source"`
			Workspace struct {
				StartLine        int    `json:"startLine"`
				SnippetStartLine int    `json:"snippetStartLine"`
				Text             string `json:"text"`
			} `json:"workspace"`
		}
		if err := json.Unmarshal(raw, &proof); err != nil {
			t.Fatal(err)
		}
		if proof.Source.StartLine != 6 || proof.Workspace.StartLine != 7 || proof.Workspace.SnippetStartLine != 2 || !strings.Contains(proof.Workspace.Text, "# @trace target=route") {
			t.Fatalf("retained trace proof = %s", raw)
		}
		foundProof = true
	}
	if !foundProof || len(h.gateway.Failures()) != 0 {
		t.Fatalf("trace evidence missing or gateway failed: %v", h.gateway.Failures())
	}
}
