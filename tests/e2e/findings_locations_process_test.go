//go:build e2e

package e2e

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/findingintake"
)

// Exercise the actual source facade, durable intake, public API, collection
// materialization and Python reader in independent production processes.
func TestFindingLocationsAcrossProcesses(t *testing.T) {
	h := startFindingsHarness(t, []domainGatewayStage{locatedFindingProducer(), locatedFindingReader()})
	project := createProjectResource(t, h.client, h.baseURL)
	source := uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID,
		"sources", "located-source", "application/zip", auditProgramZip(t, map[string][]byte{"app.py": []byte(findingsSource)}))
	body, err := json.Marshal(map[string]any{"workflow": "source-findings-review@1", "artifacts": map[string]artifactRef{"source": source}})
	if err != nil {
		t.Fatal(err)
	}
	runID := postProjectRun(t, h.client, h.baseURL, project.ProjectID, "located-producer", body, false)
	waitForFindingsRun(t, h, runID, "succeeded")
	receipts := findingsReceipts(t, h, runID)
	if len(receipts) != 1 {
		t.Fatalf("receipts = %d", len(receipts))
	}
	proposal := receipts[0].Document
	if err := checkLocatedProposal(proposal); err != nil {
		t.Fatal(err)
	}
	if receipts[0].Origin.RunID != runID || receipts[0].Origin.Audit != nil {
		t.Fatal("wrong ordinary Run provenance")
	}
	collection := publishFindingsCollection(t, h, "located-snapshot", []findingintake.CollectionSelection{{
		Kind: "run", ID: runID, ReceiptIDs: []string{receipts[0].ReceiptID}, Findings: []findingintake.CollectionFindingSelection{},
	}}, false)
	deleteASVSBacktraceRun(t, h.ctx, h.client, h.baseURL, runID)
	readerID := runFindingsReader(t, h, "located-reader", collection.Artifact.Ref)
	report, _ := download(t, h.client, h.baseURL+"/v1/runs/"+readerID+"/outputs/report")
	if !strings.Contains(string(report), "app.py:4–6 CWE-89") {
		t.Fatalf("reader report = %s", report)
	}
	if h.gateway.CompletedStages() != 2 || len(h.gateway.Failures()) != 0 {
		t.Fatalf("gateway failures: %v", h.gateway.Failures())
	}
}

func checkLocatedProposal(proposal auditdomain.FindingProposal) error {
	if proposal.Schema != auditdomain.FindingProposalSchema || proposal.Subject != nil ||
		len(proposal.Locations) != 1 || proposal.Locations[0].File != "app.py" || proposal.Locations[0].Range == nil ||
		*proposal.Locations[0].Range != (auditdomain.FindingLineRange{StartLine: 4, EndLine: 6}) ||
		len(proposal.StandardRefs) != 1 || proposal.StandardRefs[0].RequirementID != "CWE-89" {
		return fmt.Errorf("source finding coordinates or classification changed: %+v", proposal)
	}
	return nil
}

func locatedFindingProducer() domainGatewayStage {
	return domainGatewayStage{name: "located/producer", tools: []string{
		"open_source_archive", "list_source_files", "read_source", "search_source", "finding", "write_text_artifact",
	}, steps: []domainGatewayStep{
		toolGatewayStep("open_source_archive", stageRefArguments("source", nil)),
		toolGatewayStep("read_source", fixedArguments(map[string]any{"path": "app.py", "start_line": 1, "max_lines": 100})),
		toolGatewayStep("finding", fixedArguments(map[string]any{
			"title": "Query concatenation", "description": findingsEvidence, "file": "app.py",
			"range": map[string]int{"start_line": 4, "end_line": 6}, "cwe": "CWE-89",
		})),
		toolGatewayStep("write_text_artifact", func(request map[string]any) (map[string]any, error) {
			responses := findingsToolResponses(request, "finding")
			if len(responses) != 1 || responses[0]["receipt_id"] == nil || responses[0]["client_key"] == nil {
				return nil, fmt.Errorf("missing facade receipt: %+v", responses)
			}
			return map[string]any{"name": "report", "media_type": "text/markdown", "text": findingsEvidence}, nil
		}),
		finalGatewayStep("Source review complete", map[string]domainArtifactBinding{"report": {namespace: "source-findings", name: "report"}}),
	}}
}

func locatedFindingReader() domainGatewayStage {
	return domainGatewayStage{name: "located/reader", tools: withMemoryTools([]string{"list_findings", "read_artifact", "write_text_artifact"}), steps: []domainGatewayStep{
		toolGatewayStep("list_findings", fixedArguments(map[string]any{})),
		toolGatewayStep("read_artifact", findingsDocumentArguments(0, false)),
		toolGatewayStep("write_text_artifact", func(request map[string]any) (map[string]any, error) {
			reads := findingsToolResponses(request, "read_artifact")
			if len(reads) != 1 {
				return nil, fmt.Errorf("reader did not open the retained proposal")
			}
			data, err := base64.StdEncoding.DecodeString(fmt.Sprint(reads[0]["dataBase64"]))
			if err != nil {
				return nil, err
			}
			proposal, err := auditdomain.DecodeFindingProposal(data)
			if err != nil {
				return nil, err
			}
			if err := checkLocatedProposal(proposal); err != nil {
				return nil, err
			}
			return map[string]any{"name": "report", "media_type": "text/markdown", "text": "app.py:4–6 CWE-89"}, nil
		}),
		finalGatewayStep("Located collection read", map[string]domainArtifactBinding{"report": {namespace: "findings-review", name: "report"}}),
	}}
}
