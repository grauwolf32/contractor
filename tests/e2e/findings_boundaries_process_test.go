//go:build e2e

package e2e

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestFindingsReaderBoundariesAcrossProcesses(t *testing.T) {
	collection, contents := findingsCollectionFixture(t)
	payload, err := auditdomain.BuildFindingCollectionPackage(collection, contents)
	if err != nil {
		t.Fatal(err)
	}
	targets, err := auditdomain.FindingCollectionTargets(payload)
	if err != nil {
		t.Fatal(err)
	}
	var emptyRun, partialRun, conflictRun, invalidRun string
	var partialRef, conflictRef contracts.ArtifactRef
	first := collection.Documents[0]
	h := startFindingsHarness(t, []domainGatewayStage{findingsBoundaryReaderStage(), findingsBoundaryReaderStage()}, func(h *findingsHarness) {
		// Start Runs while no Runtime is registered, and seed precisely the durable
		// state left by interrupted preparation through the production Artifact store.
		empty := collection
		empty.Entries = []auditdomain.FindingCollectionEntry{}
		empty.Documents = []auditdomain.FindingCollectionDocument{}
		emptyPayload, err := auditdomain.BuildFindingCollectionPackage(empty, map[string][]byte{})
		if err != nil {
			t.Fatal(err)
		}
		emptyInput := uploadProjectArtifact(t, h.client, h.baseURL, "findings-empty", auditdomain.FindingCollectionMediaType, emptyPayload)
		fullInput := uploadProjectArtifact(t, h.client, h.baseURL, "findings-full", auditdomain.FindingCollectionMediaType, payload)
		badInput := uploadProjectArtifact(t, h.client, h.baseURL, "findings-invalid", auditdomain.FindingCollectionMediaType, []byte("incomplete ZIP"))
		emptyRun = createFindingsReader(t, h, "empty-reader", emptyInput)
		partialRun = createFindingsReader(t, h, "partial-reader", fullInput)
		conflictRun = createFindingsReader(t, h, "conflict-reader", fullInput)
		invalidRun = createFindingsReader(t, h, "invalid-reader", badInput)
		pool, err := pgxpool.New(h.ctx, h.databaseURL)
		if err != nil {
			t.Fatal(err)
		}
		defer pool.Close()
		service := artifacts.NewService(artifacts.NewPostgresRepository(pool))
		for _, test := range []struct {
			run  string
			data []byte
			ref  *contracts.ArtifactRef
		}{
			{partialRun, contents[first.ID], &partialRef}, {conflictRun, []byte("conflicting retained bytes"), &conflictRef},
		} {
			scoped, err := service.Run(test.run)
			if err != nil {
				t.Fatal(err)
			}
			written, err := scoped.Write(h.ctx, targets[first.ID], artifacts.Payload{MediaType: first.MediaType, Data: test.data}, nil)
			if err != nil {
				t.Fatal(err)
			}
			*test.ref = written.Ref
		}
		t.Run("inaccessible-input", func(t *testing.T) {
			missingRevision := "missing-revision"
			body, _ := json.Marshal(map[string]any{"workflow": "findings-review@1", "artifacts": map[string]artifactRef{
				"findings": {Namespace: "projects", Name: "not-owned-or-missing", Revision: &missingRevision},
			}})
			request, _ := http.NewRequest(http.MethodPost, h.baseURL+"/v1/runs", bytes.NewReader(body))
			request.Header.Set("Authorization", "Bearer "+publicToken)
			request.Header.Set("Content-Type", "application/json")
			request.Header.Set("Idempotency-Key", "missing-collection")
			response := do(t, h.client, request, http.StatusNotFound)
			response.Body.Close()
		})
	})
	t.Run("empty-is-explicit-success", func(t *testing.T) {
		waitForFindingsRun(t, h, emptyRun, "succeeded")
		raw, _ := download(t, h.client, h.baseURL+"/v1/runs/"+emptyRun+"/outputs/report")
		var page struct {
			Items  []any   `json:"items"`
			Cursor *string `json:"next_cursor"`
		}
		if err := json.Unmarshal(raw, &page); err != nil || page.Items == nil || len(page.Items) != 0 || page.Cursor != nil {
			t.Fatalf("empty reader fabricated contents: %s, %v", raw, err)
		}
	})
	t.Run("interrupted-preparation-reuses-exact-ref", func(t *testing.T) {
		waitForFindingsRun(t, h, partialRun, "succeeded")
		raw, _ := download(t, h.client, h.baseURL+"/v1/runs/"+partialRun+"/outputs/report")
		if !bytes.Contains(raw, []byte(*partialRef.Revision)) {
			t.Fatal("reader did not reuse the actual retained receipt")
		}
		for _, document := range collection.Documents {
			target := targets[document.ID]
			data, mime := download(t, h.client, h.baseURL+"/v1/runs/"+partialRun+"/artifacts/"+target.Namespace+"/"+target.Name)
			if !bytes.Equal(data, contents[document.ID]) || mime != document.MediaType {
				t.Fatalf("materialized %s differs", document.ID)
			}
		}
	})
	for name, runID := range map[string]string{"conflicting-document-fails": conflictRun, "invalid-zip-fails": invalidRun} {
		t.Run(name, func(t *testing.T) {
			result := waitForFindingsRun(t, h, runID, "failed")
			if len(result.Outputs) != 0 || len(findingsReceipts(t, h, runID)) != 0 {
				t.Fatal("failed preparation exposed a successful report/proposal")
			}
		})
	}
	data, _ := download(t, h.client, h.baseURL+"/v1/runs/"+conflictRun+"/artifacts/"+conflictRef.Namespace+"/"+conflictRef.Name+"?revision="+url.QueryEscape(*conflictRef.Revision))
	if string(data) != "conflicting retained bytes" {
		t.Fatal("preparation overwrote conflicting evidence")
	}
	if h.gateway.CompletedStages() != 2 || len(h.gateway.Failures()) != 0 {
		t.Fatalf("partial/failed preparation exposed tools: %d %v", h.gateway.CompletedStages(), h.gateway.Failures())
	}
}

func findingsBoundaryReaderStage() domainGatewayStage {
	return domainGatewayStage{name: "findings/boundaries", tools: []string{"list_findings", "read_artifact", "write_text_artifact"}, steps: []domainGatewayStep{
		toolGatewayStep("list_findings", fixedArguments(map[string]any{})),
		toolGatewayStep("write_text_artifact", func(request map[string]any) (map[string]any, error) {
			responses := findingsToolResponses(request, "list_findings")
			if len(responses) != 1 || responses[0]["next_cursor"] != nil {
				return nil, fmt.Errorf("boundary reader inventory is incomplete")
			}
			for _, raw := range findingsPageItems(responses) {
				item := raw.(map[string]any)
				document := item["proposal"].(map[string]any)
				ref := document["ref"].(map[string]any)
				if !strings.HasPrefix(fmt.Sprint(ref["namespace"]), "findings-") || ref["revision"] == nil {
					return nil, fmt.Errorf("unprepared proposal exposed")
				}
			}
			text, err := json.Marshal(responses[0])
			if err != nil {
				return nil, err
			}
			return map[string]any{"name": "report", "media_type": "text/markdown", "text": string(text)}, nil
		}),
		finalGatewayStep("Reader boundary fixture completed", map[string]domainArtifactBinding{"report": {namespace: "findings-review", name: "report"}}),
	}}
}

func findingsCollectionFixture(t *testing.T) (auditdomain.FindingCollection, map[string][]byte) {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(repoRoot(t), "internal/auditdomain/testdata/finding-collection-v1.fixture.json"))
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Collection auditdomain.FindingCollection `json:"collection"`
		Contents   map[string]string             `json:"contents"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	contents := map[string][]byte{}
	for key, value := range fixture.Contents {
		contents[key] = []byte(value)
	}
	return fixture.Collection, contents
}
